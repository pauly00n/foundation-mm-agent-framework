# automl-cardiac

Autonomous ML experiment framework for **cardiac MRI classification** on the
[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
an AI agent autonomously iterates on `train.py`, runs fixed-budget (3-min per fold)
experiments, checks `val_acc`, and repeats — logging every hypothesis and
result to `outputs/`.

```
blackbox-mm-prototype/
├── src/
│   ├── prepare.py      ← FIXED  — data pipeline, clinical features
│   └── train.py        ← AGENT ITERATES ON THIS
├── data/
│   ├── raw/            ← downloaded ACDC zip + extracted folders
│   └── processed/      ← .pt tensors per patient
├── outputs/
│   ├── results.jsonl              ← one JSON line per experiment (this run)
│   ├── confusion_matrices/        ← per-run confusion matrix PNGs
│   └── research_log.md            ← agent's running hypothesis log
├── program.md          ← instructions for the AI agent
├── run.log             ← training output from last run (gitignored)
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
uv sync
```

---

## Step 1 — Prepare data

```bash
uv run src/prepare.py
```

What this does:
1. Downloads the raw ACDC training set (~1.5 GB) from the Creatis server.
2. Parses each patient's `Info.cfg` for pathology label, Height, Weight.
3. Loads the end-diastolic NIfTI frame with `nibabel`.
4. Resizes to `(1, 16, 128, 128)` and normalises to `[0, 1]`.
5. Computes **EDV, ESV, EF** from segmentation masks (LV cavity = label 3).
6. Saves one `.pt` file per patient to `data/processed/`.

Each `.pt` file contains:
```python
{
    "volume":   FloatTensor[1, 16, 128, 128],  # preprocessed MRI
    "clinical": FloatTensor[5],                # [Height, Weight, EDV, ESV, EF]
    "label":    int,                           # 0–4
}
```

> **Manual download fallback:** If the automatic download fails, download
> `ACDC_training.zip` from https://humanheart-project.creatis.insa-lyon.fr/database/
> and place it at `data/raw/ACDC_training.zip`, then re-run `prepare.py`.

```bash
uv run src/prepare.py --skip-download   # if data/raw/ACDC_training/ already exists
uv run src/prepare.py --force-download  # re-download even if zip exists
```

---

## Step 2 — Run a single experiment (manual)

```bash
uv run src/train.py
```

- Runs **5-fold stratified cross-validation** on all 100 patients (80 train / 20 val per fold).
- Each fold trains for up to `MAX_EPOCHS` within a **3-minute wall-clock budget**.
- Total wall time: ~15 minutes (5 folds × 3 minutes).
- Appends one JSON line to `outputs/results.jsonl`.
- Saves a confusion matrix PNG to `outputs/confusion_matrices/`.
- All training output is written to `run.log` (overwritten each run).

Example summary output:
```
============================================================
  experiment_id : e64fb90fe628
  cv_folds      : 5
  val_acc (mean) : 0.6300 ± 0.1166
  overall_acc    : 0.6300
  per_class_acc  : {'NOR': 0.55, 'DCM': 0.70, 'HCM': 0.65, 'MINF': 0.45, 'RV': 0.80}
  wall_time_s    : 166.4
============================================================
```

---

## Evaluation setup: 5-fold cross-validation

All 100 patients are split into 5 stratified folds of 20 (4 per class per fold).
Each fold trains on the remaining 80 patients and evaluates on its 20.
Every patient is evaluated exactly once. The primary metric is **mean val_acc
across all 5 folds**.

There is no separate held-out test set — with 100 patients, 5-fold CV is the
standard evaluation protocol. Since no early stopping is used (the budget just
runs to completion), there is no val leakage.

---

## Results log schema (`results.jsonl`)

Each line is a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | str | UTC ISO-8601 |
| `experiment_id` | str | Git commit hash (12-char) |
| `modality` | str | `"mri+clinical"` |
| `cv_folds` | int | Always 5 |
| `val_acc` | float | Mean val accuracy across 5 folds |
| `val_acc_std` | float | Std of val accuracy across 5 folds |
| `val_loss` | float | Mean val loss across 5 folds |
| `overall_acc` | float | Accuracy over all 100 patients combined |
| `per_class_acc` | dict | `{"NOR": f, "DCM": f, "HCM": f, "MINF": f, "RV": f}` |
| `per_fold_acc` | list | `[fold1_acc, fold2_acc, ..., fold5_acc]` |
| `wall_time_s` | float | Total wall-clock seconds |
| `config` | dict | `lr`, `batch_size`, `dropout`, `weight_decay`, `use_amp`, `max_epochs`, `arch_notes` |

---

## Hyperparameters (top of `train.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `LR` | `5e-4` | AdamW learning rate |
| `BATCH_SIZE` | `8` | Samples per GPU step |
| `DROPOUT` | `0.5` | Dropout before classifier |
| `WEIGHT_DECAY` | `0.1` | AdamW weight decay |
| `MAX_EPOCHS` | `120` | Epochs per fold (tunable) |
| `USE_AMP` | `True` | Mixed precision (fp16) |
| `BUDGET_SECONDS` | `180` | **Fixed — do not change** |

---


## Starting an autonomous research run

Each research run lives on its own git branch (`autoresearch/<tag>`).
The experiment loop is defined in `program.md`.

**To start a run, open the Blackbox CLI in this repo and send this prompt:**

```
Read program.md and execute the instructions step by step, starting with the SETUP section.
```

The agent will:
1. Read `README.md`, `src/prepare.py`, and `src/train.py` to understand the codebase
2. Ask you for a run tag (e.g. `baseline-cv`, `augmentation-sweep`)
3. Create branch `autoresearch/<tag>`, clear `results.jsonl`
4. Autonomously iterate: form hypothesis → commit `train.py` → run experiment → log results → repeat
5. Stop when `val_acc ≥ 0.85` or after 20 experiments

Results accumulate in `outputs/results.jsonl` and `outputs/research_log.md` on the branch.
Each experiment is a git commit — `git log --oneline` gives the full history.

---


## GPU

Tested on **NVIDIA H100 80 GB**.  
Uses `torch.device("cuda")` automatically, falling back to CPU.

---

## License

MIT
