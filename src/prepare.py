"""
prepare.py — FIXED. The agent must never modify this file.

Responsibilities:
  1. Download the raw ACDC dataset (100 labelled patients, 5 classes)
  2. Extract and parse NIfTI volumes with nibabel
  3. Normalize + resize each volume to (1, D=16, H=128, W=128)
  4. Split patients 60/20/20 stratified by class into train/val/test
  5. Save processed tensors to data/processed/{split}/
  6. Expose eval(model, val_loader) -> {"val_acc": float, "val_loss": float}

Dataset: ACDC (Automated Cardiac Diagnosis Challenge)
  Source: https://humanheart-project.creatis.insa-lyon.fr/database/
          #collection/637218c173e9f0047faa00fb
  100 labelled patients from ACDC_training, 5 balanced classes (20 each):
    NOR=0  DCM=1  HCM=2  MINF=3  RV=4
  Ground-truth pathology is read from per-patient Info.cfg ("Group" field).
  Segmentation masks (_gt.nii.gz) use label 3 = LV cavity for EDV/ESV/EF.

Split (all from ACDC_training, stratified, random_state=42):
  train : 60 patients (12 per class)
  val   : 20 patients  (4 per class)  ← agent tunes against this
  test  : 20 patients  (4 per class)  ← locked held-out metric

The official ACDC_testing folder has only 1 publicly labelled patient
and is not used.

Download fallback: if ACDC_training.zip is already present at
data/raw/ACDC_training.zip the download step is skipped automatically.
"""

import os
import sys
import json
import zipfile
import hashlib
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (all relative to repo root, resolved at runtime)
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW    = REPO_ROOT / "data" / "raw"
DATA_PROC   = REPO_ROOT / "data" / "processed"
ACDC_ZIP      = DATA_RAW / "ACDC_training.zip"
ACDC_DIR      = DATA_RAW / "ACDC_training"
ACDC_TEST_DIR = DATA_RAW / "ACDC_testing"

# Medical Decathlon fallback (Task02_Heart — segmentation only, no pathology labels)
DECATHLON_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar"
)
DECATHLON_TAR = DATA_RAW / "Task02_Heart.tar"
DECATHLON_DIR = DATA_RAW / "Task02_Heart"

# ACDC direct download (public mirror hosted by Creatis)
ACDC_URL = (
    "https://acdc.creatis.insa-lyon.fr/description/databases/ACDC_training.zip"
)

# ---------------------------------------------------------------------------
# Volume shape after preprocessing
# ---------------------------------------------------------------------------
TARGET_D = 16    # depth  (temporal / slice axis)
TARGET_H = 128   # height
TARGET_W = 128   # width

# ---------------------------------------------------------------------------
# Label mapping: ACDC pathology string → integer class index
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[str, int] = {
    "NOR":  0,
    "DCM":  1,
    "HCM":  2,
    "MINF": 3,
    "RV":   4,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  Download helpers
# ===========================================================================

def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    import requests  # lazy import — only needed during data prep

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    log.info("Download complete: %s", dest)


def download_acdc(force: bool = False) -> None:
    """
    Download the raw ACDC training set.

    Priority:
      1. If ACDC_DIR already exists and contains patient folders → skip.
      2. If ACDC_ZIP already exists → just extract.
      3. Attempt download from ACDC_URL.
      4. If download fails (server requires registration) → print manual
         instructions and exit with a helpful error message.
    """
    if ACDC_DIR.exists() and any(ACDC_DIR.glob("patient*")):
        log.info("ACDC data already extracted at %s — skipping download.", ACDC_DIR)
        return

    if not ACDC_ZIP.exists() or force:
        log.info("Attempting to download ACDC dataset from %s", ACDC_URL)
        try:
            _download_file(ACDC_URL, ACDC_ZIP, desc="ACDC_training.zip")
        except Exception as exc:
            log.error(
                "Download failed: %s\n\n"
                "The ACDC dataset may require manual registration.\n"
                "Please download ACDC_training.zip from:\n"
                "  https://humanheart-project.creatis.insa-lyon.fr/database/"
                "#collection/637218c173e9f0047faa00fb\n"
                "and place it at:\n"
                "  %s\n"
                "Then re-run prepare.py.",
                exc,
                ACDC_ZIP,
            )
            sys.exit(1)

    log.info("Extracting %s → %s", ACDC_ZIP, DATA_RAW)
    with zipfile.ZipFile(ACDC_ZIP, "r") as zf:
        zf.extractall(DATA_RAW)

    # The zip may unpack as "training/" or "ACDC_training/" — normalise
    for candidate in [DATA_RAW / "training", DATA_RAW / "ACDC_training"]:
        if candidate.exists() and candidate != ACDC_DIR:
            candidate.rename(ACDC_DIR)
            break

    log.info("ACDC extracted to %s", ACDC_DIR)


# ===========================================================================
# 2.  Parse ACDC patient metadata
# ===========================================================================

def parse_patient_label(patient_dir: Path) -> int:
    """
    Read the Info.cfg file inside a patient directory and return the
    integer class index for the 'Group' field.

    Info.cfg format (example):
        ED: 1
        ES: 12
        Group: DCM
        Height: 184.0
        NbFrame: 30
        Weight: 95.0
    """
    cfg_path = patient_dir / "Info.cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Info.cfg not found in {patient_dir}")

    with open(cfg_path) as f:
        for line in f:
            key, _, value = line.partition(":")
            if key.strip() == "Group":
                group = value.strip().upper()
                if group not in LABEL_MAP:
                    raise ValueError(
                        f"Unknown group '{group}' in {cfg_path}. "
                        f"Expected one of {list(LABEL_MAP.keys())}"
                    )
                return LABEL_MAP[group]

    raise ValueError(f"'Group' key not found in {cfg_path}")


def _parse_info_cfg(patient_dir: Path) -> Dict[str, str]:
    """Parse all key-value pairs from Info.cfg into a dict."""
    cfg_path = patient_dir / "Info.cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Info.cfg not found in {patient_dir}")
    result: Dict[str, str] = {}
    with open(cfg_path) as f:
        for line in f:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _compute_lv_volume_ml(gt_path: Path) -> float:
    """
    Compute LV cavity volume in mL from a segmentation mask NIfTI file.

    Label 3 = LV cavity (ACDC convention).
    Volume = (number of label-3 voxels) × (voxel volume in mm³) / 1000
    """
    img = nib.load(str(gt_path))
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()  # (dx, dy, dz) in mm; may be 3- or 4-tuple
    voxel_vol_mm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
    n_lv_voxels = int((data == 3).sum())
    return n_lv_voxels * voxel_vol_mm3 / 1000.0  # mL


def parse_clinical_features(patient_dir: Path) -> torch.Tensor:
    """
    Extract 5 clinical/functional features for a patient and return them
    as a float32 Tensor of shape (5,):

        [Height (cm), Weight (kg), EDV (mL), ESV (mL), EF (fraction)]

    EDV and ESV are computed from the LV cavity (label=3) in the GT
    segmentation masks at the ED and ES frames respectively.
    EF = (EDV - ESV) / EDV  (clamped to [0, 1]).

    If any feature cannot be computed (missing file, zero EDV, etc.) the
    corresponding value is set to 0.0 so the tensor is always valid.
    """
    cfg = _parse_info_cfg(patient_dir)
    patient_id = patient_dir.name

    height = float(cfg.get("Height", 0.0) or 0.0)
    weight = float(cfg.get("Weight", 0.0) or 0.0)

    ed_frame = int(cfg.get("ED", 1))
    es_frame = int(cfg.get("ES", 1))

    # Locate GT masks for ED and ES frames
    gt_ed_path = patient_dir / f"{patient_id}_frame{ed_frame:02d}_gt.nii.gz"
    gt_es_path = patient_dir / f"{patient_id}_frame{es_frame:02d}_gt.nii.gz"

    edv, esv, ef = 0.0, 0.0, 0.0
    try:
        if gt_ed_path.exists():
            edv = _compute_lv_volume_ml(gt_ed_path)
        if gt_es_path.exists():
            esv = _compute_lv_volume_ml(gt_es_path)
        if edv > 0.0:
            ef = max(0.0, min(1.0, (edv - esv) / edv))
    except Exception as exc:
        log.warning("Clinical feature computation failed for %s: %s", patient_id, exc)

    return torch.tensor([height, weight, edv, esv, ef], dtype=torch.float32)


def get_ed_nifti(patient_dir: Path) -> Path:
    """
    Return the path to the end-diastolic (ED) NIfTI frame for a patient.

    ACDC naming convention:
        patientXXX_frame01.nii.gz   ← ED frame (frame number from Info.cfg)
        patientXXX_frame12.nii.gz   ← ES frame
    We use the ED frame as the canonical volume for classification.
    """
    cfg_path = patient_dir / "Info.cfg"
    ed_frame = None
    with open(cfg_path) as f:
        for line in f:
            key, _, value = line.partition(":")
            if key.strip() == "ED":
                ed_frame = int(value.strip())
                break

    if ed_frame is None:
        raise ValueError(f"'ED' key not found in {cfg_path}")

    patient_id = patient_dir.name  # e.g. "patient001"
    nii_path = patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz"
    if not nii_path.exists():
        # Fallback: pick the first .nii.gz that is NOT the gt file
        candidates = sorted(
            p for p in patient_dir.glob("*.nii.gz") if "_gt" not in p.name
        )
        if not candidates:
            raise FileNotFoundError(f"No NIfTI volumes found in {patient_dir}")
        nii_path = candidates[0]
        log.warning("ED frame not found; using fallback %s", nii_path.name)

    return nii_path


# ===========================================================================
# 3.  Volume preprocessing
# ===========================================================================

def load_and_preprocess(nii_path: Path) -> torch.Tensor:
    """
    Load a NIfTI volume and return a float32 tensor of shape (1, D, H, W).

    Steps:
      1. Load with nibabel → numpy array (H, W, D) or (H, W, D, T)
      2. If 4-D (time series), take the first volume (already ED from filename)
      3. Resize spatial dims to (TARGET_H, TARGET_W) via area-averaging
      4. Resize depth to TARGET_D via linear interpolation along slice axis
      5. Normalise to [0, 1] using per-volume min-max
      6. Add channel dim → (1, D, H, W)
    """
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)  # (H, W, D) or (H, W, D, T)

    # Drop time axis if present
    if vol.ndim == 4:
        vol = vol[..., 0]

    # vol shape: (H, W, D)
    H, W, D = vol.shape

    # --- Resize H, W to (TARGET_H, TARGET_W) using torch interpolation ---
    # Work in (1, 1, H, W) for F.interpolate
    vol_t = torch.from_numpy(vol)  # (H, W, D)
    # Permute to (D, H, W) then treat D as batch
    vol_t = vol_t.permute(2, 0, 1).unsqueeze(1).float()  # (D, 1, H, W)
    vol_t = torch.nn.functional.interpolate(
        vol_t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False
    )  # (D, 1, TARGET_H, TARGET_W)
    vol_t = vol_t.squeeze(1)  # (D, TARGET_H, TARGET_W)

    # --- Resize D to TARGET_D ---
    # Treat as (1, 1, D, H, W) for trilinear interpolation
    vol_t = vol_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    vol_t = torch.nn.functional.interpolate(
        vol_t,
        size=(TARGET_D, TARGET_H, TARGET_W),
        mode="trilinear",
        align_corners=False,
    )  # (1, 1, TARGET_D, TARGET_H, TARGET_W)
    vol_t = vol_t.squeeze(0)  # (1, TARGET_D, TARGET_H, TARGET_W)

    # --- Per-volume min-max normalisation to [0, 1] ---
    vmin = vol_t.min()
    vmax = vol_t.max()
    if vmax - vmin > 1e-6:
        vol_t = (vol_t - vmin) / (vmax - vmin)
    else:
        vol_t = torch.zeros_like(vol_t)

    return vol_t  # (1, TARGET_D, TARGET_H, TARGET_W)


# ===========================================================================
# 4.  Build dataset splits and save .pt files
# ===========================================================================

def build_splits(acdc_dir: Path) -> None:
    """
    Scan acdc_dir (ACDC_training) for patient folders, parse labels,
    split 60/20/20 stratified by class into train/val/test, preprocess
    each volume, and save tensors to data/processed/{train,val,test}/.

    All three splits come from ACDC_training (100 labelled patients):
      - train: 60 patients (12 per class)
      - val:   20 patients (4 per class)  ← agent tunes against this
      - test:  20 patients (4 per class)  ← locked held-out metric

    The official ACDC_testing folder has only 1 publicly labelled patient
    and is not used.

    Saved files:
        data/processed/{split}/patientXXX.pt  → {"volume": Tensor, "label": int}
    """
    patient_dirs = sorted(acdc_dir.glob("patient*"))
    if not patient_dirs:
        raise RuntimeError(
            f"No patient directories found in {acdc_dir}. "
            "Check that the ACDC data was extracted correctly."
        )

    log.info("Found %d patient directories in training set.", len(patient_dirs))

    # Collect (path, label, nii) triples — skip patients with missing data
    samples: List[Tuple[Path, int]] = []
    for pdir in patient_dirs:
        try:
            label = parse_patient_label(pdir)
            nii   = get_ed_nifti(pdir)
            samples.append((pdir, label, nii))
        except Exception as exc:
            log.warning("Skipping %s: %s", pdir.name, exc)

    if not samples:
        raise RuntimeError("No valid patient samples found.")

    log.info("Valid samples: %d", len(samples))

    # Stratified split: 60 / 20 / 20  (train / val / test)
    indices = list(range(len(samples)))
    labels  = [s[1] for s in samples]

    # First carve off 40% (val+test), then split that 50/50
    idx_train, idx_tmp, _, lbl_tmp = train_test_split(
        indices, labels, test_size=0.40, stratify=labels, random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, stratify=lbl_tmp, random_state=42
    )

    split_map = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    for split, idxs in split_map.items():
        out_dir = DATA_PROC / split
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Processing split='%s' (%d samples) …", split, len(idxs))

        for i in tqdm(idxs, desc=split):
            pdir, label, nii_path = samples[i]
            try:
                volume   = load_and_preprocess(nii_path)
                clinical = parse_clinical_features(pdir)
                out_path = out_dir / f"{pdir.name}.pt"
                torch.save({"volume": volume, "label": label, "clinical": clinical}, out_path)
            except Exception as exc:
                log.warning("Failed to process %s: %s", pdir.name, exc)

    # Save split metadata for reproducibility
    meta = {
        "train": [samples[i][0].name for i in idx_train],
        "val":   [samples[i][0].name for i in idx_val],
        "test":  [samples[i][0].name for i in idx_test],
        "label_map": LABEL_MAP,
        "volume_shape": [1, TARGET_D, TARGET_H, TARGET_W],
        "note": "all splits from ACDC_training (60/20/20 stratified by class)",
    }
    with open(DATA_PROC / "splits.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "Splits saved → train=%d  val=%d  test=%d",
        len(idx_train), len(idx_val), len(idx_test),
    )


def build_test_split(acdc_test_dir: Path = ACDC_TEST_DIR) -> None:
    """
    No-op.  The official ACDC_testing folder contains only 1 publicly
    labelled patient — far too few for a reliable metric.  The held-out
    test split is instead carved from ACDC_training inside build_splits()
    (20 patients, 4 per class, stratified).
    """
    log.info(
        "build_test_split: skipped — test set is already built from "
        "ACDC_training by build_splits() (20 patients, 4 per class)."
    )


# ===========================================================================
# 5.  PyTorch Dataset
# ===========================================================================

class ACDCDataset(Dataset):
    """
    Loads preprocessed .pt files from data/processed/{split}/.

    Each item: (volume: FloatTensor[1,D,H,W], label: LongTensor scalar)
    """

    def __init__(self, split: str, data_dir: Path = DATA_PROC):
        self.split_dir = data_dir / split
        self.files = sorted(self.split_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(
                f"No .pt files found in {self.split_dir}. "
                "Run prepare.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample   = torch.load(self.files[idx], weights_only=True)
        volume   = sample["volume"].float()                          # (1, D, H, W)
        label    = torch.tensor(sample["label"], dtype=torch.long)
        # Backward-compatible: older .pt files without clinical features get zeros
        clinical = sample.get("clinical", torch.zeros(5)).float()   # (5,)
        return volume, clinical, label


def get_dataloader(
    split: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    data_dir: Path = DATA_PROC,
) -> DataLoader:
    """Convenience factory for train/val/test DataLoaders."""
    dataset = ACDCDataset(split=split, data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ===========================================================================
# 6.  Evaluation function (called by train.py at end of each run)
# ===========================================================================

@torch.no_grad()
def eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    clinical_mean: torch.Tensor = None,
    clinical_std: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Evaluate *model* on *val_loader*.

    Args:
        clinical_mean: Training-set clinical feature means for z-score normalization.
        clinical_std:  Training-set clinical feature stds for z-score normalization.
            Both must be provided for models trained with normalized clinical features.

    Returns:
        {"val_acc": float, "val_loss": float}

    The model is set to eval mode and restored to its previous training
    state afterwards.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        # Support both old (volume, label) and new (volume, clinical, label) loaders
        if len(batch) == 3:
            volumes, clinical, labels = batch
            clinical = clinical.to(device, non_blocking=True)
            # Apply z-score normalization if stats were provided
            if clinical_mean is not None and clinical_std is not None:
                mean = clinical_mean.to(device)
                std = clinical_std.to(device)
                clinical = (clinical - mean) / (std + 1e-8)
        else:
            volumes, labels = batch
            clinical = None

        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        # Support both single-modality and multi-modal model signatures
        if clinical is not None:
            try:
                logits = model(volumes, clinical)
            except TypeError:
                logits = model(volumes)
        else:
            logits = model(volumes)

        loss   = criterion(logits, labels)

        total_loss    += loss.item() * labels.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    if was_training:
        model.train()

    val_loss = total_loss / max(total_samples, 1)
    val_acc  = total_correct / max(total_samples, 1)

    return {"val_acc": val_acc, "val_loss": val_loss}


# ===========================================================================
# 7.  CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ACDC cardiac MRI data for classification."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if the zip already exists.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; assume data/raw/ACDC_training/ already exists.",
    )
    args = parser.parse_args()

    # Step 1: Download
    if not args.skip_download:
        download_acdc(force=args.force_download)
    else:
        log.info("--skip-download set; assuming %s exists.", ACDC_DIR)

    if not ACDC_DIR.exists():
        log.error(
            "ACDC directory not found at %s.\n"
            "Either run without --skip-download or place the extracted\n"
            "ACDC_training folder at that path manually.",
            ACDC_DIR,
        )
        sys.exit(1)

    # Step 2: Build train/val splits from ACDC_training
    build_splits(ACDC_DIR)

    # Step 3: Build test split from ACDC_testing (may be small / absent)
    build_test_split(ACDC_TEST_DIR)

    log.info("Data preparation complete. Processed files in %s", DATA_PROC)


if __name__ == "__main__":
    main()
