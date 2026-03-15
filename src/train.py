"""
train.py — THE FILE THE AGENT ITERATES ON.

The agent may freely modify this file between runs to test hypotheses.
prepare.py is fixed and must never be modified.

Architecture: 4-block ResNet+SE 3D CNN with clinical gated fusion.
Training: 5-fold stratified cross-validation on all 100 patients (80/20).
Each fold trains one model, validates on the held-out 20.
Final metric = mean val_acc across 5 folds.

Run command:
    uv run src/train.py

Results are appended to outputs/results.jsonl after each experiment.
"""

import os
import sys
import time
import json
import math
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import prepare.py from src/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prepare import NUM_CLASSES, IDX_TO_LABEL, DATA_PROC  # noqa: E402

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
    "5-fold CV on 100 patients. CosineAnnealingLR T_max=MAX_EPOCHS. "
    "DROPOUT=0.5. WD=0.1. H+V flip. Standard CE. TTA=8 passes. LR=5e-4. BS=8. "
    "Clinical z-score normalization (5 features). MAX_EPOCHS=60. Plain CE. H+V flips. LR=5e-4. Original arch. CosineAnnealingLR. FiLM fusion: clinical→gamma+beta modulate MRI embedding."
)

MAX_EPOCHS = 60
MIXUP_ALPHA = 0.0  # Mixup disabled

# Training budget (seconds) per fold — do NOT change this
BUDGET_SECONDS = 180  # 3 minutes per fold

# Mixed precision — set False if you hit numerical issues
USE_AMP = True

# Number of DataLoader workers
NUM_WORKERS = 4

# 5-fold CV settings
N_FOLDS = 5
FOLD_SEEDS = [42, 7, 13, 99, 0]  # one fixed seed per fold

# ===========================================================================
# Paths
# ===========================================================================
OUTPUTS_DIR  = REPO_ROOT / "outputs"
RESULTS_FILE = OUTPUTS_DIR / "results.jsonl"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit_hash() -> str:
    """Return the current git commit hash (short, 12 chars). Falls back to timestamp."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

# ===========================================================================
# Logging — all output goes to run.log (overwritten each run)
# ===========================================================================
LOG_FILE = REPO_ROOT / "run.log"
_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.handlers.clear()
_fh = logging.FileHandler(LOG_FILE, mode="w")
_fh.setFormatter(_log_formatter)
_root_logger.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_log_formatter)
_root_logger.addHandler(_sh)
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
# Dataset for loading individual .pt files
# ===========================================================================

class PTFileDataset(Dataset):
    """Load a list of .pt files as a dataset."""

    def __init__(self, pt_files: list):
        self.files = pt_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample   = torch.load(self.files[idx], weights_only=True)
        volume   = sample["volume"].float()                          # (1, D, H, W)
        label    = torch.tensor(sample["label"], dtype=torch.long)
        clinical = sample.get("clinical", torch.zeros(5)).float()   # (5,)
        return volume, clinical, label


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
        return self.net(x)  # (B, 256)


class MultiModalCardiacNet(nn.Module):
    """
    FiLM fusion: clinical features generate gamma+beta to modulate MRI embedding.
    FiLM(mri_feat) = gamma * mri_feat + beta, then concat with clinical_feat → classifier.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.mri_encoder      = CardiacCNN3D(num_classes=num_classes, dropout=dropout)
        self.clinical_encoder = ClinicalEncoder()
        # FiLM: predict gamma and beta from clinical features
        self.film_gamma = nn.Linear(128, 128)
        self.film_beta  = nn.Linear(128, 128)
        self.classifier = nn.Linear(128 + 128, num_classes)
        for layer in [self.film_gamma, self.film_beta, self.classifier]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _mri_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mri_encoder.stage1(x)
        x = self.mri_encoder.stage2(x)
        x = self.mri_encoder.stage3(x)
        x = self.mri_encoder.stage4(x)
        x = self.mri_encoder.gap(x).flatten(1)
        return self.mri_encoder.dropout(x)

    def forward(self, volumes: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        mri_feat      = self._mri_embed(volumes)          # (B, 128)
        clinical_feat = self.clinical_encoder(clinical)   # (B, 128)
        gamma = self.film_gamma(clinical_feat)             # (B, 128)
        beta  = self.film_beta(clinical_feat)              # (B, 128)
        modulated = gamma * mri_feat + beta                # FiLM modulation
        fused = torch.cat([modulated, clinical_feat], dim=1)  # (B, 256)
        return self.classifier(fused)


# ===========================================================================
# Training loop
# ===========================================================================

# Global clinical normalization stats (reset per fold from that fold's training data)
_CLINICAL_MEAN: torch.Tensor = None
_CLINICAL_STD:  torch.Tensor = None


def augment_clinical(clinical: torch.Tensor) -> torch.Tensor:
    """Add derived features: BMI = Weight/Height^2, SV = EDV - ESV.
    Input: (B, 5) [Height, Weight, EDV, ESV, EF]
    Output: (B, 7) [Height, Weight, EDV, ESV, EF, BMI, SV]
    """
    height = clinical[:, 0]  # cm
    weight = clinical[:, 1]  # kg
    edv    = clinical[:, 2]
    esv    = clinical[:, 3]
    # BMI = weight(kg) / (height(m))^2
    height_m = height / 100.0
    bmi = weight / (height_m * height_m + 1e-8)
    # Stroke volume
    sv = edv - esv
    return torch.cat([clinical, bmi.unsqueeze(1), sv.unsqueeze(1)], dim=1)


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


@torch.no_grad()
def evaluate_with_tta(model, loader):
    """Evaluate a single model with TTA (all H/V/D flip combinations).
    Returns val_acc, val_loss, per-class accuracy dict, and confusion matrix."""
    model.eval()
    model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
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
        for flips in tta_flips:
            aug = volumes.clone()
            for dims in flips:
                aug = torch.flip(aug, dims=dims)
            probs_sum += torch.softmax(model(aug, clinical), dim=1)
            n_passes += 1
        avg_probs = probs_sum / n_passes
        preds = avg_probs.argmax(dim=1)
        # Loss on the base (non-augmented) logits
        logits = model(volumes, clinical)
        loss = crit(logits, labels)
        total_loss    += loss.item() * B
        total_correct += (preds == labels).sum().item()
        total_samples += B
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf_matrix[int(t), int(p)] += 1

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


def collect_all_pt_files():
    """Collect all 100 .pt files from train/val/test dirs, return sorted list + labels."""
    all_files = []
    for split in ["train", "val", "test"]:
        split_dir = DATA_PROC / split
        if split_dir.exists():
            all_files.extend(sorted(split_dir.glob("*.pt")))
    # Sort by patient name for determinism
    all_files.sort(key=lambda p: p.stem)

    # Extract labels
    labels = []
    for f in all_files:
        sample = torch.load(f, weights_only=True)
        labels.append(int(sample["label"]))

    return all_files, labels


def make_stratified_folds(all_files, labels, n_folds=5, random_state=42):
    """Create n_folds stratified splits. Returns list of (train_files, val_files)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(all_files, labels):
        train_files = [all_files[i] for i in train_idx]
        val_files   = [all_files[i] for i in val_idx]
        folds.append((train_files, val_files))
    return folds


def main():
    experiment_id = get_git_commit_hash()
    log.info("=== Experiment %s  (5-fold CV) ===", experiment_id)
    log.info("LR=%.2e  BS=%d  DROPOUT=%.2f  AMP=%s", LR, BATCH_SIZE, DROPOUT, USE_AMP)
    log.info("Arch: %s", ARCH_NOTES)

    # ------------------------------------------------------------------
    # Collect all 100 patients and create 5 stratified folds
    # ------------------------------------------------------------------
    all_files, all_labels = collect_all_pt_files()
    log.info("Total patients: %d", len(all_files))
    folds = make_stratified_folds(all_files, all_labels, n_folds=N_FOLDS, random_state=42)

    n_params = sum(p.numel() for p in MultiModalCardiacNet().parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # Train one model per fold
    # ------------------------------------------------------------------
    global _CLINICAL_MEAN, _CLINICAL_STD
    fold_results = []
    overall_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    experiment_start = time.time()

    for fold_idx, (train_files, val_files) in enumerate(folds):
        fold_seed = FOLD_SEEDS[fold_idx]
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

        log.info("=== Fold %d/%d  (seed=%d, train=%d, val=%d) ===",
                 fold_idx + 1, N_FOLDS, fold_seed, len(train_files), len(val_files))

        # Build dataloaders for this fold
        train_dataset = PTFileDataset(train_files)
        val_dataset   = PTFileDataset(val_files)
        train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)

        # Compute clinical normalization stats from this fold's training set
        all_clinical = []
        for _, clinical, _ in train_loader:
            all_clinical.append(clinical)
        all_clinical = torch.cat(all_clinical, dim=0)
        _CLINICAL_MEAN = all_clinical.mean(dim=0).to(DEVICE)
        _CLINICAL_STD  = all_clinical.std(dim=0).to(DEVICE)
        log.info("  Clinical mean: %s", _CLINICAL_MEAN.cpu().numpy().round(2))
        log.info("  Clinical std:  %s", _CLINICAL_STD.cpu().numpy().round(2))

        # Model, optimizer, loss
        model     = MultiModalCardiacNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler    = GradScaler(enabled=USE_AMP)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        # Train with budget
        t_fold_start = time.time()
        deadline = t_fold_start + BUDGET_SECONDS
        epoch = 0
        timed_out = False

        while time.time() < deadline and epoch < MAX_EPOCHS:
            epoch += 1
            loss_sum, n_correct, n_total, timed_out = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, deadline, scheduler
            )
            if n_total > 0 and epoch % 20 == 0:
                log.info("  Fold %d E%d/%d  train_loss=%.4f  train_acc=%.4f",
                         fold_idx + 1, epoch, MAX_EPOCHS,
                         loss_sum / n_total, n_correct / n_total)
            if timed_out:
                break
            if scheduler is not None:
                scheduler.step()

        fold_time = time.time() - t_fold_start
        log.info("  Fold %d: %d epochs in %.1fs", fold_idx + 1, epoch, fold_time)

        # Evaluate this fold
        metrics = evaluate_with_tta(model, val_loader)
        log.info("  Fold %d RESULTS → val_acc=%.4f  val_loss=%.4f",
                 fold_idx + 1, metrics["val_acc"], metrics["val_loss"])
        log.info("  Fold %d per-class: %s", fold_idx + 1, metrics["per_class_acc"])

        fold_results.append({
            "fold": fold_idx + 1,
            "seed": fold_seed,
            "val_acc": metrics["val_acc"],
            "val_loss": metrics["val_loss"],
            "per_class_acc": metrics["per_class_acc"],
            "epochs": epoch,
            "wall_time_s": round(fold_time, 1),
            "val_patients": [f.stem for f in val_files],
        })
        overall_conf_matrix += metrics["conf_matrix"]

        # Free GPU memory
        del model, optimizer, scaler, scheduler
        torch.cuda.empty_cache()

    total_time = time.time() - experiment_start

    # ------------------------------------------------------------------
    # Aggregate results across folds
    # ------------------------------------------------------------------
    fold_accs  = [r["val_acc"] for r in fold_results]
    fold_losses = [r["val_loss"] for r in fold_results]
    mean_acc  = np.mean(fold_accs)
    std_acc   = np.std(fold_accs)
    mean_loss = np.mean(fold_losses)

    # Per-class accuracy from the aggregated confusion matrix (every patient evaluated once)
    overall_per_class = {}
    for cls_idx in range(NUM_CLASSES):
        cls_total = overall_conf_matrix[cls_idx].sum()
        cls_correct = overall_conf_matrix[cls_idx, cls_idx]
        overall_per_class[IDX_TO_LABEL[cls_idx]] = (
            round(float(cls_correct) / float(cls_total), 4) if cls_total > 0 else 0.0
        )
    overall_acc = overall_conf_matrix.diagonal().sum() / overall_conf_matrix.sum()

    log.info("=" * 60)
    log.info("5-FOLD CV RESULTS:")
    log.info("  Per-fold val_acc: %s", [round(a, 4) for a in fold_accs])
    log.info("  Mean val_acc: %.4f ± %.4f", mean_acc, std_acc)
    log.info("  Overall acc (all 100 patients): %.4f", overall_acc)
    log.info("  Per-class accuracy: %s", overall_per_class)
    log.info("  Total wall time: %.1fs", total_time)

    # Save confusion matrix as PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf = overall_conf_matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        class_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(class_names)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {experiment_id}\n(5-fold CV, all 100 patients)")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                        color="white" if conf[i, j] > conf.max() / 2 else "black")
        plt.tight_layout()
        cm_dir = OUTPUTS_DIR / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        cm_path = cm_dir / f"confusion_matrix_{experiment_id}.png"
        plt.savefig(cm_path, dpi=120)
        plt.close(fig)
        log.info("Confusion matrix saved → %s", cm_path)
    except Exception as exc:
        log.warning("Could not save confusion matrix: %s", exc)

    # ------------------------------------------------------------------
    # Append to outputs/results.jsonl
    # ------------------------------------------------------------------
    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "modality":      "mri+clinical",
        "cv_folds":      N_FOLDS,
        "val_acc":       round(mean_acc, 6),
        "val_acc_std":   round(std_acc, 6),
        "val_loss":      round(mean_loss, 6),
        "overall_acc":   round(float(overall_acc), 6),
        "per_class_acc": overall_per_class,
        "per_fold_acc":  [round(a, 4) for a in fold_accs],
        "wall_time_s":   round(total_time, 1),
        "config": {
            "lr":           LR,
            "batch_size":   BATCH_SIZE,
            "dropout":      DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "use_amp":      USE_AMP,
            "max_epochs":   MAX_EPOCHS,
            "budget_per_fold_s": BUDGET_SECONDS,
            "arch_notes":   ARCH_NOTES,
        },
    }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Result appended to %s", RESULTS_FILE)
    log.info("=== Experiment %s complete ===", experiment_id)

    # Print a clean summary
    print(
        f"\n{'='*60}\n"
        f"  experiment_id : {experiment_id}\n"
        f"  cv_folds      : {N_FOLDS}\n"
        f"  val_acc (mean) : {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"  overall_acc    : {overall_acc:.4f}\n"
        f"  per_class_acc  : {overall_per_class}\n"
        f"  wall_time_s    : {total_time:.1f}\n"
        f"{'='*60}\n"
    )


if __name__ == "__main__":
    main()
