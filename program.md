# program.md — Instructions for the Autonomous ML Agent

You are an autonomous ML research agent. Your sole objective is to
**maximise mean `val_acc` on the ACDC cardiac MRI 5-class classification task**
across all 5 folds of cross-validation, within a fixed 3-minute wall-clock
training budget per fold (15 minutes total per experiment).

---

## 0. Ground Rules

| Rule | Detail |
|------|--------|
| **Only modify `src/train.py`** | `src/prepare.py` is fixed infrastructure — never touch it. |
| **NEVER modify `program.md`** | This file is fixed — never touch it. |
| **One hypothesis per experiment** | Change exactly one thing at a time so results are interpretable. |
| **Always read history first** | Before proposing a change, read `outputs/results.jsonl` in full. |
| **Log everything** | Append to `outputs/research_log.md` after every experiment (template below). |
| **Budget is sacred** | `BUDGET_SECONDS = 180` in `train.py` must never be changed. |
| **No external data** | Only the ACDC dataset prepared by `prepare.py`. |
| **Always run full 5-fold CV** | One experiment = one `uv run src/train.py` call. Never report a single-fold result as truth. |
| **Commit before running** | Commit `src/train.py` before every experiment — the commit hash is the experiment ID. |

---

## SETUP — Do This Once Before the First Experiment

Before running any experiments, complete this setup sequence **once**:

### Step 1 — Read the codebase
Read these three files in full to understand the project:
```bash
cat README.md
cat src/prepare.py
cat src/train.py
```

### Step 2 — Agree on a run tag with the user
Ask the user: **"What tag should I use for this research run?"**

The tag should be a short descriptive slug, e.g. `baseline-cv`, `augmentation-sweep`, `fusion-search`.
Wait for the user's response before proceeding.

### Step 3 — Verify the branch does not exist
```bash
git branch --list autoresearch/<tag>
```
If the branch already exists, tell the user and ask for a different tag.

### Step 4 — Create and switch to the branch
```bash
git checkout -b autoresearch/<tag>
```

### Step 5 — Clear results.jsonl for this fresh run
```bash
> outputs/results.jsonl
```
This ensures `results.jsonl` only contains experiments from this run.
(History from previous runs is preserved in their own branches.)

### Step 6 — Begin the experiment loop
Proceed to §1.

---

## 1. The Experiment Loop

```
LOOP:
  1. Read outputs/results.jsonl          ← understand what has been tried this run
  2. Read outputs/research_log.md        ← understand why things were tried
  3. Form ONE hypothesis                 ← e.g. "reducing MAX_EPOCHS will reduce overfitting"
  4. Edit src/train.py                   ← implement exactly that change
  5. Commit the change BEFORE running:
       git add src/train.py
       git commit -m "<one-line hypothesis summary>"
  6. Run the experiment:
       uv run src/train.py
     (all training output goes to run.log — do NOT read run.log, it is for debugging only)
  7. Read the last entry in results.jsonl  ← the experiment_id is the commit hash
  8. Append to outputs/research_log.md
  9. Commit the results:
       git add outputs/results.jsonl outputs/research_log.md
       git commit -m "results: <experiment_id> val_acc=X.XX"
  10. GOTO 1
```

One experiment = ~15 minutes wall-clock (5 folds × 3 minutes each).

**Important:** `run.log` is overwritten on every run and is gitignored.
Do NOT read it as part of the loop — it floods context. Only read it if
a run crashes and you need to debug.

---

## 2. Success Metric

- **Primary:**   mean `val_acc` across all 5 folds  (target ≥ 0.85)
- **Stability:** `val_acc_std` across folds — lower is better; a result with
  high mean but std > 0.10 is unreliable (high variance across data partitions)
- **Per-class:** `per_class_acc` — pay special attention to `MINF` and `RV`,
  which are the historically hardest classes

All metrics are written to `outputs/results.jsonl` by `train.py`.
A confusion matrix PNG is saved to `outputs/confusion_matrices/` after every run.

### How to read the last experiment result

```bash
tail -1 outputs/results.jsonl | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
print(f'experiment_id : {r[\"experiment_id\"]}')
print(f'val_acc (mean): {r[\"val_acc\"]:.4f} ± {r[\"val_acc_std\"]:.4f}')
print(f'overall_acc   : {r[\"overall_acc\"]:.4f}')
print(f'per_fold_acc  : {r[\"per_fold_acc\"]}')
print(f'per_class_acc : {r[\"per_class_acc\"]}')
print(f'wall_time_s   : {r[\"wall_time_s\"]}')
"
```

---

## 3. What the Data Looks Like

Each sample in the DataLoader is a 3-tuple:

```python
volumes,  # FloatTensor (B, 1, 16, 128, 128) — preprocessed MRI
clinical, # FloatTensor (B, 5)               — [Height, Weight, EDV, ESV, EF]
labels    # LongTensor  (B,)                 — class index 0–4
```

`prepare.py` computes the clinical features from the raw ACDC files:
- **Height** (cm), **Weight** (kg) — from `Info.cfg`
- **EDV** (mL) — LV cavity (label=3) voxel count × voxel volume at ED frame
- **ESV** (mL) — same at ES frame
- **EF** — `(EDV − ESV) / EDV`, clamped to `[0, 1]`

These features are physiologically discriminative:
- DCM: high EDV, low EF (~0.15–0.30)
- HCM: normal/low EDV, **preserved EF** (~0.60–0.75)
- MINF: variable EDV, **reduced EF** (~0.30–0.45)
- NOR: normal EDV, normal EF (~0.55–0.70)
- RV: enlarged RV (not captured by LV EF — rely on MRI for this class)

### 5-Fold CV Setup

All 100 patients are split into 5 stratified folds of 20 (4 per class per fold).
Each fold: train on the other 80 patients, evaluate on this fold's 20.
Clinical normalization stats (mean/std) are computed from the 80 training
patients of each fold independently — no leakage.

---

## 4. Current Architecture (`MultiModalCardiacNet`)

```
MRI branch:
  Input (B, 1, 16, 128, 128)
  → Stage1–4: ResNet+SE 3D CNN → GAP → Dropout → (B, 128)

Clinical branch:
  Input (B, 5)
  → Linear(5→64) → BN → ReLU → Linear(64→128) → BN → ReLU → (B, 128)

Fusion (gated):
  gate = sigmoid(Linear(128, 128)(clinical_feat))
  fused = cat[mri_feat * gate, clinical_feat]  → (B, 256)
  → Linear(256, 5) → logits
```

The agent may modify:
- `ClinicalEncoder` (e.g. deeper MLP, add dropout, feature normalisation)
- `MultiModalCardiacNet` (e.g. attention-based fusion instead of gated)
- `CardiacCNN3D` (e.g. wider channels, deeper stages)
- `train_one_epoch` (e.g. augmentation, loss function)
- `main()` (e.g. scheduler, MAX_EPOCHS, TTA passes)

The agent must **never** modify:
- `BUDGET_SECONDS = 180`
- The 5-fold CV loop structure in `main()`
- The `results.jsonl` logging block at the end of `main()`
- `src/prepare.py`

---

## 5. Directions to Explore (ordered by expected impact)

Work through these roughly in order, but always let the data guide you.
If an experiment hurts performance, revert that specific change before
trying the next hypothesis.

### 5.1 Epochs & Regularisation
The current `MAX_EPOCHS = 120` was inherited from the old setup.
With 80 training patients per fold, overfitting is a real risk (train_acc
often reaches 0.95+ while val_acc lags). Consider:
- **Reduce MAX_EPOCHS** (try 60, 80) — stop before overfitting
- **Increase DROPOUT** (try 0.6, 0.7)
- **Increase WEIGHT_DECAY** (try 0.2, 0.3)

### 5.2 Data Augmentation (MRI branch only)
All augmentations must be applied on-GPU (use `torch` ops).

| Augmentation | Implementation hint |
|---|---|
| Random horizontal flip | `torch.flip(vol, dims=[-1])` with p=0.5 |
| Random vertical flip   | `torch.flip(vol, dims=[-2])` with p=0.5 |
| Random depth flip      | `torch.flip(vol, dims=[-3])` with p=0.5 |
| Intensity jitter       | `vol * (1 + 0.1*torch.randn_like(vol))` |
| Gaussian noise         | `vol + 0.02*torch.randn_like(vol)` |

### 5.3 Loss Function
- **Label smoothing**: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Class-weighted CE**: upweight MINF and RV (the hard classes)
  ```python
  weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], device=DEVICE)
  criterion = nn.CrossEntropyLoss(weight=weights)
  ```
- **Focal loss**: focus on hard examples (already defined in train.py)

### 5.4 Learning Rate & Schedule
- **LR sweep**: try 1e-3, 2e-4
- **Warmup + cosine**: linear warmup for first 5% of steps, then cosine decay
- **OneCycleLR**: `OneCycleLR(optimizer, max_lr=LR, ...)`

### 5.5 Fusion Strategy
- **Cross-attention**: clinical embedding attends over MRI spatial features
- **FiLM conditioning**: modulate MRI feature maps with clinical scalars
- **Simple concat** (revert gated): sometimes simpler is better

### 5.6 Clinical Feature Engineering
- **Add derived features**: BMI = Weight/Height², SV = EDV−ESV (stroke volume)
- **Log-transform** EDV/ESV (right-skewed distributions)

### 5.7 Architecture Changes
- Wider MRI encoder: `1→32→64→128→256`
- Deeper clinical MLP: `5→64→128→128`
- Larger batch size: `BATCH_SIZE = 16`

---

## 6. Hyperparameter Block

```python
# ===========================================================================
# ★  HYPERPARAMETERS — agent modifies this block between experiments  ★
# ===========================================================================
LR           = 5e-4
BATCH_SIZE   = 8
DROPOUT      = 0.5
WEIGHT_DECAY = 0.1
ARCH_NOTES   = "..."
MAX_EPOCHS   = 120   # epochs per fold — tunable
```

---

## 7. Research Log Format

After every experiment, append the following block to `outputs/research_log.md`:

```markdown
---
## Experiment <N> — <timestamp>
**Experiment ID (commit hash):** <experiment_id>

**Hypothesis:** <one sentence: what you expected and why>

**Change made:**
```diff
- old line(s) from train.py
+ new line(s) from train.py
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | X.XXXX |
| val_acc (std)  | X.XXXX |
| per_fold_acc   | [X.XX, X.XX, X.XX, X.XX, X.XX] |
| per_class_acc  | NOR=X.XX  DCM=X.XX  HCM=X.XX  MINF=X.XX  RV=X.XX |
| prev best      | X.XXXX |

**Interpretation:** <2–4 sentences: did it work? why? what does this suggest?>

**Next hypothesis:** <what to try next based on this result>
```

---

## 8. Stopping Criteria

Stop iterating when **any** of the following is true:
- Mean `val_acc` across 5 folds ≥ 0.85 — excellent result, stop and report
- You have run **20 experiments** — summarise findings
- Three consecutive experiments show no improvement > 0.02 in mean `val_acc`
  — try a fundamentally different direction before concluding

---

## 9. Final Report

When stopping, append a `## Final Summary` section to `outputs/research_log.md`:

```markdown
## Final Summary

**Best experiment (commit hash):** <experiment_id>
**Best mean val_acc (5 folds):** X.XXXX ± X.XXXX
**Best overall_acc:** X.XXXX
**Best config:** <describe the winning train.py configuration>

**Key findings:**
1. ...
2. ...
3. ...

**Hard classes analysis:**
- MINF: ...
- RV: ...

**Recommended next steps (beyond 3-min budget):**
- ...
```

Then make a final commit:
```bash
git add outputs/research_log.md
git commit -m "final summary: best val_acc=X.XX on autoresearch/<tag>"
```

---

## 10. Quick Reference

```bash
# Run data preparation (once, or after any change to prepare.py)
uv run src/prepare.py --skip-download

# Standard experiment run — always use this single command
uv run src/train.py

# Read the last experiment result
tail -1 outputs/results.jsonl | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
print(f'experiment_id : {r[\"experiment_id\"]}')
print(f'val_acc (mean): {r[\"val_acc\"]:.4f} ± {r[\"val_acc_std\"]:.4f}')
print(f'overall_acc   : {r[\"overall_acc\"]:.4f}')
print(f'per_fold_acc  : {r[\"per_fold_acc\"]}')
print(f'per_class_acc : {r[\"per_class_acc\"]}')
print(f'wall_time_s   : {r[\"wall_time_s\"]}')
"

# View all experiment results this run
cat outputs/results.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    pca = r.get('per_class_acc', {})
    print(f\"{r['experiment_id']}  val_acc={r['val_acc']:.4f} ± {r.get('val_acc_std',0):.4f}\", end='')
    if pca:
        print('  ' + '  '.join(f\"{k}={v:.2f}\" for k, v in pca.items()))
    else:
        print()
"

# View research log
cat outputs/research_log.md

# Check latest confusion matrix
ls -t outputs/confusion_matrices/*.png | head -1

# View git log for this run
git log --oneline autoresearch/<tag>
```

---

*This file is read-only for the agent. Do not modify program.md.*
