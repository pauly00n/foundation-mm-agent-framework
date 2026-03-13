"""
train.py — THE FILE THE AGENT ITERATES ON.

The agent may freely modify this file between runs to test hypotheses.
prepare.py is fixed and must never be modified.

Baseline architecture: 4-block 3D CNN
  Conv3d → BN → ReLU → MaxPool3d  (×4)
  → GlobalAvgPool → Dropout → Linear(256, NUM_CLASSES)

Training budget: exactly 3 minutes wall-clock time per seed.
The standard run command executes 3 seeds and averages the results:

    for SEED in 42 7 13; do SEED=$SEED uv run src/train.py; done

Results are appended to outputs/results.jsonl after every seed.
The agent reads the last 3 entries and averages val_acc / test_acc as truth.
"""

import os
import sys
import time
import json
import uuid
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import prepare.py from src/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prepare import get_dataloader, NUM_CLASSES  # noqa: E402

# ===========================================================================
# ★  HYPERPARAMETERS — agent modifies this block between experiments  ★
# ===========================================================================

LR           = 5e-4        # AdamW LR (best known)
BATCH_SIZE   = 8           # samples per GPU step
DROPOUT      = 0.5         # dropout probability
WEIGHT_DECAY = 1e-1        # WD=0.1

# Architecture notes (free-text, logged to results.jsonl for the agent)
ARCH_NOTES = (
    "MRI+Clinical fusion: ResNet+SE (1→16→32→64→128) + ClinicalEncoder MLP(5→64→128). "
    "Gated fusion: gate=sigmoid(Linear(128,128)) applied to MRI feat, concat(gated_mri, clinical)→Linear(256,5). "
    "120 epochs each, ensemble=3. CosineAnnealingLR T_max=120. "
    "DROPOUT=0.5. WD=0.1. H+V flip. Standard CE. TTA=8 passes. LR=5e-4. BS=8. "
    "Clinical z-score normalization (5 features)."
)

# Hard cap on epochs per model in ensemble
MAX_EPOCHS = 120   # sweet spot for N_ENSEMBLE=3
N_ENSEMBLE = 3     # ensemble of 3 models
MIXUP_ALPHA = 0.0  # Mixup disabled

# Training budget (seconds) — do NOT change this
BUDGET_SECONDS = 180  # 3 minutes

# Mixed precision — set False if you hit numerical issues
USE_AMP = True

# Number of DataLoader workers
NUM_WORKERS = 4

# ===========================================================================
# Paths
# ===========================================================================
OUTPUTS_DIR  = REPO_ROOT / "outputs"
RESULTS_FILE = OUTPUTS_DIR / "results.jsonl"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Logging
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ===========================================================================
# Device
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)
if DEVICE.type == "cuda":
    log.info("GPU: %s  (%.1f GB)", torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)


# ===========================================================================
# Model definition
# ===========================================================================

class FocalLoss(nn.Module):
    """Focal Loss: focuses training on hard examples."""
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class ConvBlock3D(nn.Module):
    """Conv3d → BatchNorm3d → ReLU → MaxPool3d."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale


class ResBlock3D(nn.Module):
    """Residual block with SE + stochastic depth."""

    def __init__(self, ch: int, drop_path: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch),
        )
        self.se        = SEBlock3D(ch)
        self.relu      = nn.ReLU(inplace=True)
        self.drop_path = drop_path

    def forward(self, x):
        if self.training and self.drop_path > 0 and torch.rand(1).item() < self.drop_path:
            return x  # skip this block entirely (stochastic depth)
        return self.relu(x + self.se(self.block(x)))


class CardiacCNN3D(nn.Module):
    """
    4-block ResNet+SE 3D CNN with larger first kernel (7×1×7).
    Input:  (B, 1, D=16, H=128, W=128)
    Output: (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBlock3D(1,   16,  pool=True),  ResBlock3D(16))
        self.stage2 = nn.Sequential(ConvBlock3D(16,  32,  pool=True),  ResBlock3D(32))
        self.stage3 = nn.Sequential(ConvBlock3D(32,  64,  pool=True),  ResBlock3D(64))
        self.stage4 = nn.Sequential(ConvBlock3D(64,  128, pool=True),  ResBlock3D(128))

        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)   # (B, 128)
        return self.dropout(x)


class ClinicalEncoder(nn.Module):
    """
    MLP encoder for tabular clinical features.
    Input:  (B, 5)  — [Height, Weight, EDV, ESV, EF]
    Output: (B, 128)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 128)


class MultiModalCardiacNet(nn.Module):
    """
    Gated fusion: clinical gate weights MRI embedding element-wise.
    Concat(gated_mri, clinical_feat) → Linear(256, NUM_CLASSES).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.mri_encoder      = CardiacCNN3D(num_classes=num_classes, dropout=dropout)
        self.clinical_encoder = ClinicalEncoder()
        self.gate             = nn.Linear(128, 128)
        self.classifier       = nn.Linear(128 + 128, num_classes)
        nn.init.xavier_uniform_(self.gate.weight);       nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def _mri_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mri_encoder.stage1(x)
        x = self.mri_encoder.stage2(x)
        x = self.mri_encoder.stage3(x)
        x = self.mri_encoder.stage4(x)
        x = self.mri_encoder.gap(x).flatten(1)
        return self.mri_encoder.dropout(x)

    def forward(self, volumes: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        mri_feat      = self._mri_embed(volumes)
        clinical_feat = self.clinical_encoder(clinical)
        gate          = torch.sigmoid(self.gate(clinical_feat))
        fused         = torch.cat([mri_feat * gate, clinical_feat], dim=1)
        return self.classifier(fused)


# ===========================================================================
# Training loop
# ===========================================================================

# Global clinical normalization stats (set in main() from training data)
_CLINICAL_MEAN: torch.Tensor = None
_CLINICAL_STD:  torch.Tensor = None


def normalize_clinical(clinical: torch.Tensor) -> torch.Tensor:
    """Z-score normalize clinical features using training-set stats."""
    if _CLINICAL_MEAN is not None and _CLINICAL_STD is not None:
        return (clinical - _CLINICAL_MEAN) / (_CLINICAL_STD + 1e-8)
    return clinical


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    deadline: float,
    scheduler=None,
) -> tuple:
    """
    Train for one epoch or until *deadline* (wall-clock time.time()).

    Returns (avg_loss, n_correct, n_total, timed_out).
    timed_out=True means the budget expired mid-epoch.
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        if time.time() >= deadline:
            return total_loss, total_correct, total_samples, True

        volumes, clinical, labels = batch
        volumes  = volumes.to(DEVICE, non_blocking=True)
        clinical = clinical.to(DEVICE, non_blocking=True)
        labels   = labels.to(DEVICE, non_blocking=True)

        # Z-score normalize clinical features
        clinical = normalize_clinical(clinical)

        # Augmentation: H+V flips only
        B = volumes.size(0)
        for i in range(B):
            if torch.rand(1).item() < 0.5:
                volumes[i] = torch.flip(volumes[i], dims=[-1])   # H flip
            if torch.rand(1).item() < 0.5:
                volumes[i] = torch.flip(volumes[i], dims=[-2])   # V flip

        # Mixup augmentation
        if MIXUP_ALPHA > 0 and B > 1:
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            idx = torch.randperm(B, device=DEVICE)
            volumes  = lam * volumes  + (1 - lam) * volumes[idx]
            clinical = lam * clinical + (1 - lam) * clinical[idx]
            labels_b = labels[idx]
        else:
            lam, labels_b = 1.0, labels

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP):
            logits = model(volumes, clinical)
            if MIXUP_ALPHA > 0 and B > 1:
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * labels.size(0)
        preds          = logits.detach().argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss, total_correct, total_samples, False


def main():
    # ------------------------------------------------------------------
    # Seed — read from SEED env var so the 3-seed loop can pass different
    # values without modifying this file:
    #   for SEED in 42 7 13; do SEED=$SEED uv run src/train.py; done
    # ------------------------------------------------------------------
    seed = int(os.environ.get("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiment_id = str(uuid.uuid4())[:8]
    log.info("=== Experiment %s  (seed=%d) ===", experiment_id, seed)
    log.info("LR=%.2e  BS=%d  DROPOUT=%.2f  AMP=%s", LR, BATCH_SIZE, DROPOUT, USE_AMP)
    log.info("Arch: %s", ARCH_NOTES)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader = get_dataloader(
        "train", batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = get_dataloader(
        "val", batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    log.info(
        "Dataset: %d train  %d val batches (BS=%d)",
        len(train_loader), len(val_loader), BATCH_SIZE,
    )

    # ------------------------------------------------------------------
    # Compute clinical feature normalization stats from training set
    # Features: [Height, Weight, EDV, ESV, EF]
    # ------------------------------------------------------------------
    global _CLINICAL_MEAN, _CLINICAL_STD
    all_clinical = []
    for _, clinical, _ in train_loader:
        all_clinical.append(clinical)
    all_clinical = torch.cat(all_clinical, dim=0)  # (N_train, 5)
    _CLINICAL_MEAN = all_clinical.mean(dim=0).to(DEVICE)
    _CLINICAL_STD  = all_clinical.std(dim=0).to(DEVICE)
    log.info("Clinical mean: %s", _CLINICAL_MEAN.cpu().numpy().round(2))
    log.info("Clinical std:  %s", _CLINICAL_STD.cpu().numpy().round(2))

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    _tmp = MultiModalCardiacNet(num_classes=NUM_CLASSES, dropout=DROPOUT)
    n_params = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
    log.info("Model parameters per member: %s  (ensemble size: %d)", f"{n_params:,}", N_ENSEMBLE)
    del _tmp

    # ------------------------------------------------------------------
    # 3-minute budget: train N_ENSEMBLE models, ensemble at inference
    # ------------------------------------------------------------------
    t_start      = time.time()
    deadline     = t_start + BUDGET_SECONDS
    ensemble_models = []
    total_epochs = 0
    timed_out    = False

    for m_idx in range(N_ENSEMBLE):
        if time.time() >= deadline:
            log.info("Budget exhausted before training member %d.", m_idx + 1)
            break

        log.info("=== Training ensemble member %d/%d (%.1fs elapsed) ===",
                 m_idx + 1, N_ENSEMBLE, time.time() - t_start)

        model     = MultiModalCardiacNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler    = GradScaler(enabled=USE_AMP)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

        epoch = 0
        while time.time() < deadline and epoch < MAX_EPOCHS:
            epoch += 1
            total_epochs += 1

            loss_sum, n_correct, n_total, timed_out = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, deadline, scheduler
            )

            if n_total > 0 and epoch % 10 == 0:
                log.info("  M%d E%d/%d  train_loss=%.4f  train_acc=%.4f",
                         m_idx+1, epoch, MAX_EPOCHS,
                         loss_sum/n_total, n_correct/n_total)

            if timed_out:
                break

            if scheduler is not None:
                scheduler.step()

        ensemble_models.append(model)
        log.info("  Member %d trained (%d epochs)", m_idx + 1, epoch)

    log.info("Ensemble: %d models trained, %d total epochs", len(ensemble_models), total_epochs)

    # Use last model reference for logging (actual eval uses ensemble below)
    epoch = total_epochs

    total_time = time.time() - t_start
    log.info("Training finished after %.1fs", total_time)

    # ------------------------------------------------------------------
    # Final evaluation on validation set (ensemble)
    # ------------------------------------------------------------------
    log.info("Running ensemble validation (%d models) …", len(ensemble_models))

    @torch.no_grad()
    def ensemble_evaluate(models, loader):
        """Evaluate with ensemble + TTA (all H/V/D flip combinations).
        Returns val_acc, val_loss, per-class accuracy dict, and confusion matrix."""
        for m in models:
            m.eval()
            m.to(DEVICE)
        crit = nn.CrossEntropyLoss()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        # Confusion matrix: rows=true, cols=pred
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
        tta_flips = [
            [], [[-1]], [[-2]], [[-3]],
            [[-1], [-2]], [[-1], [-3]], [[-2], [-3]], [[-1], [-2], [-3]],
        ]
        for batch in loader:
            volumes, clinical, labels = batch
            volumes  = volumes.to(DEVICE, non_blocking=True)
            clinical = clinical.to(DEVICE, non_blocking=True)
            labels   = labels.to(DEVICE, non_blocking=True)
            clinical = normalize_clinical(clinical)
            B = volumes.size(0)
            probs_sum = torch.zeros(B, NUM_CLASSES, device=DEVICE)
            n_passes = 0
            for m in models:
                for flips in tta_flips:
                    aug = volumes.clone()
                    for dims in flips:
                        aug = torch.flip(aug, dims=dims)
                    probs_sum += torch.softmax(m(aug, clinical), dim=1)
                    n_passes += 1
            avg_probs = probs_sum / n_passes
            preds = avg_probs.argmax(dim=1)
            # Loss on the ensemble average logits (not just models[0])
            with torch.no_grad():
                logits_sum = sum(m(volumes, clinical) for m in models)
                ensemble_logits = logits_sum / len(models)
            loss = crit(ensemble_logits, labels)
            total_loss    += loss.item() * B
            total_correct += (preds == labels).sum().item()
            total_samples += B
            # Accumulate confusion matrix
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf_matrix[int(t), int(p)] += 1

        # Per-class accuracy from confusion matrix diagonal
        from prepare import IDX_TO_LABEL
        per_class_acc = {}
        for cls_idx in range(NUM_CLASSES):
            cls_total = conf_matrix[cls_idx].sum()
            cls_correct = conf_matrix[cls_idx, cls_idx]
            per_class_acc[IDX_TO_LABEL[cls_idx]] = (
                round(float(cls_correct) / float(cls_total), 4) if cls_total > 0 else 0.0
            )

        return {
            "val_acc":       total_correct / max(total_samples, 1),
            "val_loss":      total_loss    / max(total_samples, 1),
            "per_class_acc": per_class_acc,
            "conf_matrix":   conf_matrix,
        }

    metrics = ensemble_evaluate(ensemble_models, val_loader)
    log.info(
        "RESULTS → val_acc=%.4f  val_loss=%.4f",
        metrics["val_acc"], metrics["val_loss"],
    )
    log.info("Per-class accuracy: %s", metrics["per_class_acc"])

    # Save confusion matrix as PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from prepare import IDX_TO_LABEL as _IDX_TO_LABEL
        conf = metrics["conf_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        class_names = [_IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(class_names)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {experiment_id}\n(mri+clinical, val set)")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                        color="white" if conf[i, j] > conf.max() / 2 else "black")
        plt.tight_layout()
        cm_path = OUTPUTS_DIR / f"confusion_matrix_{experiment_id}.png"
        plt.savefig(cm_path, dpi=120)
        plt.close(fig)
        log.info("Confusion matrix saved → %s", cm_path)
    except Exception as exc:
        log.warning("Could not save confusion matrix: %s", exc)

    # ------------------------------------------------------------------
    # Optional evaluation on held-out test set (ACDC_testing)
    # ------------------------------------------------------------------
    test_metrics: dict = {}
    try:
        test_loader = get_dataloader(
            "test", batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        log.info("Running ensemble test evaluation (%d samples) …", len(test_loader.dataset))
        _test = ensemble_evaluate(ensemble_models, test_loader)
        log.info(
            "TEST → test_acc=%.4f  test_loss=%.4f",
            _test["val_acc"], _test["val_loss"],
        )
        test_metrics = {
            "test_acc":  round(_test["val_acc"],  6),
            "test_loss": round(_test["val_loss"], 6),
        }
        log.info("Test per-class accuracy: %s", _test["per_class_acc"])
    except RuntimeError:
        log.info("No test split found — skipping test evaluation.")

    # ------------------------------------------------------------------
    # Append to outputs/results.jsonl
    # ------------------------------------------------------------------
    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "seed":          seed,
        "modality":      "mri+clinical",
        "val_acc":       round(metrics["val_acc"],  6),
        "val_loss":      round(metrics["val_loss"], 6),
        "per_class_acc": metrics["per_class_acc"],
        **test_metrics,
        "epochs_run":    epoch,
        "wall_time_s":   round(total_time, 1),
        "config": {
            "lr":           LR,
            "batch_size":   BATCH_SIZE,
            "dropout":      DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "use_amp":      USE_AMP,
            "arch_notes":   ARCH_NOTES,
        },
    }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Result appended to %s", RESULTS_FILE)
    log.info("=== Experiment %s complete ===", experiment_id)

    # Print a clean summary line for easy parsing
    test_line = ""
    if test_metrics:
        test_line = (
            f"  test_acc      : {test_metrics['test_acc']:.4f}\n"
            f"  test_loss     : {test_metrics['test_loss']:.4f}\n"
        )
    print(
        f"\n{'='*60}\n"
        f"  experiment_id : {experiment_id}\n"
        f"  seed          : {seed}\n"
        f"  val_acc       : {metrics['val_acc']:.4f}\n"
        f"  val_loss      : {metrics['val_loss']:.4f}\n"
        f"{test_line}"
        f"  epochs_run    : {epoch}\n"
        f"  wall_time_s   : {total_time:.1f}\n"
        f"{'='*60}\n"
    )


if __name__ == "__main__":
    main()
