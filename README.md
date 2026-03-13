# blackbox-mm-prototype

Autonomous ML experiment framework for **cardiac MRI classification** on the
[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
an AI agent autonomously iterates on `train.py`, runs fixed-budget (1-min)
experiments, checks `val_acc`, and repeats — logging every hypothesis and
result to `outputs/`.

```
blackbox-mm-prototype/
├── src/
│   ├── prepare.py      ← FIXED  — data pipeline, clinical features, eval()
│   └── train.py        ← AGENT ITERATES ON THIS
├── data/
│   ├── raw/            ← downloaded ACDC zip + extracted folders
│   └── processed/      ← .pt tensors (train / val / test)
├── outputs/
│   ├── results.jsonl              ← one JSON line per experiment
│   ├── confusion_matrix_<id>.png  ← per-run confusion matrix (mri+clinical runs)
│   └── research_log.md            ← agent's running hypothesis log
├── program.md          ← instructions for the AI agent
├── pyproject.toml
└── requirements.txt
```

---

## Task

5-class cardiac pathology classification from 3D MRI volumes **fused with
clinical tabular features**:

| Index | Label | Description |
|-------|-------|-------------|
| 0 | NOR  | Normal |
| 1 | DCM  | Dilated Cardiomyopathy |
| 2 | HCM  | Hypertrophic Cardiomyopathy |
| 3 | MINF | Myocardial Infarction |
| 4 | RV   | Right Ventricular abnormality |

MRI input shape: `(1, 16, 128, 128)` — single channel, D×H×W.  
Clinical input shape: `(5,)` — `[Height, Weight, EDV, ESV, EF]`.

---

## Setup

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA GPU (tested on H100 80 GB; any CUDA-capable GPU works)

### Install dependencies

```bash
# From repo root
uv sync
```

This creates a `.venv/` and installs all dependencies from `pyproject.toml`.

---

## Step 1 — Prepare data

```bash
uv run src/prepare.py
```

What this does:
1. Downloads the raw ACDC training set (~1.5 GB) from the Creatis server.
2. Parses each patient's `Info.cfg` to extract the pathology label, Height, and Weight.
3. Loads the end-diastolic NIfTI frame with `nibabel`.
4. Resizes to `(1, 16, 128, 128)` and normalises to `[0, 1]`.
5. Computes **EDV, ESV, EF** from the `_gt.nii.gz` segmentation masks (LV cavity = label 3):
   - `EDV` = LV voxel count at ED frame × voxel volume (mm³→mL)
   - `ESV` = LV voxel count at ES frame × voxel volume (mm³→mL)
   - `EF`  = `(EDV − ESV) / EDV`, clamped to `[0, 1]`
6. Splits patients **60/20/20** (stratified by class) into train/val/test.
7. Saves `.pt` tensors to `data/processed/{train,val,test}/`.

Each `.pt` file contains:
```python
{
    "volume":   FloatTensor[1, 16, 128, 128],  # preprocessed MRI
    "clinical": FloatTensor[5],                # [Height, Weight, EDV, ESV, EF]
    "label":    int,                           # 0–4
}
```

> **Manual download fallback:** If the automatic download fails (the ACDC
> server occasionally requires registration), download `ACDC_training.zip`
> from https://humanheart-project.creatis.insa-lyon.fr/database/ and place
> it at `data/raw/ACDC_training.zip`, then re-run `prepare.py`.

Options:
```bash
uv run src/prepare.py --skip-download   # if data/raw/ACDC_training/ already exists
uv run src/prepare.py --force-download  # re-download even if zip exists
```

---

## Step 2 — Run an experiment

```bash
uv run src/train.py
```

- Trains for exactly **1 minute** wall-clock time (`BUDGET_SECONDS = 60`).
- Prints a summary at the end:
  ```
  ============================================================
    experiment_id : dbdb0fe5
    val_acc       : 0.7000
    val_loss      : 1.0570
    test_acc      : 0.5500
    test_loss     : 1.2254
    epochs_run    : 240
    wall_time_s   : 53.5
  ============================================================
  ```
- Appends one JSON line to `outputs/results.jsonl` (includes `modality`,
  `per_class_acc`, and all config fields).
- Saves a confusion matrix PNG to `outputs/confusion_matrix_<id>.png`.

---

## Step 3 — Launch the autonomous agent

Point the Blackbox CLI at `program.md` to start the autonomous research loop:

```bash
blackbox program.md
```

The agent will:
1. Read `outputs/results.jsonl` to understand prior experiments.
2. Form a hypothesis (e.g. "clinical EF feature will break HCM/MINF confusion").
3. Edit `src/train.py` with exactly one change.
4. Run `uv run src/train.py` (1-min budget).
5. Log the result + interpretation to `outputs/research_log.md`.
6. Repeat until `val_acc ≥ 0.90` or 20 experiments are done.

---

## Viewing results

```bash
# Pretty-print all experiment results (with modality and per-class accuracy)
cat outputs/results.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    pca = r.get('per_class_acc', {})
    mod = r.get('modality', 'mri')
    print(f\"{r['experiment_id']}  [{mod}]  val_acc={r['val_acc']:.4f}  \", end='')
    if pca:
        print('  '.join(f\"{k}={v:.2f}\" for k, v in pca.items()))
    else:
        print(r['config']['arch_notes'][:60])
"

# View the agent's research log
cat outputs/research_log.md
```

---

## Architecture

### MRI Encoder (`CardiacCNN3D`) — unchanged

```
Input (B, 1, 16, 128, 128)
  │
  ├─ Stage1: ConvBlock3D(1→16)  + ResBlock3D(16)   → (B, 16, 16, 64, 64)
  ├─ Stage2: ConvBlock3D(16→32) + ResBlock3D(32)   → (B, 32, 16, 32, 32)
  ├─ Stage3: ConvBlock3D(32→64) + ResBlock3D(64)   → (B, 64, 16, 16, 16)
  ├─ Stage4: ConvBlock3D(64→128)+ ResBlock3D(128)  → (B, 128, 16, 8, 8)
  │
  └─ GlobalAvgPool3d → Dropout → (B, 128)  ← MRI embedding
```

Each `ResBlock3D` includes a **Squeeze-and-Excitation** channel attention
module and optional stochastic depth.

### Clinical Encoder (`ClinicalEncoder`) — new

```
Input (B, 5)  ← [Height, Weight, EDV, ESV, EF]
  │
  ├─ Linear(5→64)  → BN → ReLU
  └─ Linear(64→128)→ BN → ReLU  → (B, 128)  ← clinical embedding
```

### Fusion Head (`MultiModalCardiacNet`) — new

```
MRI embedding    (B, 128)  ┐
                            ├─ cat → (B, 256) → Linear(256, 5) → logits
Clinical embedding (B, 128) ┘
```

Total parameters per ensemble member: **~1.5 M**

---

## Data splits

All splits are drawn from the 100 labelled patients in `ACDC_training`
(20 per class), stratified by pathology class:

| Split | Patients | Per class |
|-------|----------|-----------|
| train | 60       | 12        |
| val   | 20       | 4         |
| test  | 20       | 4         |

The test split is **locked** — `train.py` never trains on it; it is only
evaluated at the end of each run. Split membership is recorded in
`data/processed/splits.json`.

---

## Results log schema (`results.jsonl`)

Each line is a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | str | UTC ISO-8601 |
| `experiment_id` | str | 8-char hex |
| `modality` | str | `"mri+clinical"` for new runs; absent for old MRI-only runs |
| `val_acc` | float | Validation accuracy (primary metric) |
| `val_loss` | float | Validation cross-entropy loss |
| `per_class_acc` | dict | `{"NOR": f, "DCM": f, "HCM": f, "MINF": f, "RV": f}` — new runs only |
| `test_acc` | float | Held-out test accuracy |
| `test_loss` | float | Held-out test loss |
| `epochs_run` | int | Total epochs across all ensemble members |
| `wall_time_s` | float | Total wall-clock seconds |
| `config` | dict | `lr`, `batch_size`, `dropout`, `weight_decay`, `use_amp`, `arch_notes` |

> Old MRI-only entries (before the multimodal upgrade) are preserved exactly
> as written — no backfill. The `modality` and `per_class_acc` fields are
> absent on those lines.

---

## Hyperparameters (top of `train.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `LR` | `5e-4` | AdamW learning rate |
| `BATCH_SIZE` | `8` | Samples per GPU step |
| `DROPOUT` | `0.5` | Dropout before classifier |
| `WEIGHT_DECAY` | `0.05` | AdamW weight decay |
| `USE_AMP` | `True` | Mixed precision (fp16) |
| `MAX_EPOCHS` | `120` | Epochs per ensemble member |
| `N_ENSEMBLE` | `2` | Number of independently trained models |
| `BUDGET_SECONDS` | `60` | **Fixed — do not change** |

---

## GPU

Tested on **NVIDIA H100 80 GB**.  
The code uses `device = torch.device("cuda")` automatically when a GPU is
available, falling back to CPU for debugging.

---

## License

MIT
