# program.md — Instructions for the Autonomous ML Agent

You are an autonomous ML research agent.  Your sole objective is to
**maximise mean `val_acc` on the ACDC cardiac MRI 5-class classification task**
across 3 random seeds, within a fixed 3-minute wall-clock training budget per seed.

---

## 0. Ground Rules

| Rule | Detail |
|------|--------|
| **Only modify `src/train.py`** | `src/prepare.py` is fixed infrastructure — never touch it. |
| **One hypothesis per experiment** | Change exactly one thing at a time so results are interpretable. |
| **Always read history first** | Before proposing a change, read `outputs/results.jsonl` in full. |
| **Log everything** | Append to `outputs/research_log.md` after every experiment (template below). |
| **Budget is sacred** | `BUDGET_SECONDS = 180` in `train.py` must never be changed. |
| **No external data** | Only the ACDC dataset prepared by `prepare.py`. |
| **Always run 3 seeds** | Every experiment = 3 runs. Never report a single-seed result as truth. |

---

## 1. The Experiment Loop

```
LOOP:
  1. Read outputs/results.jsonl          ← understand what has been tried
  2. Read outputs/research_log.md        ← understand why things were tried
  3. Form ONE hypothesis                 ← e.g. "cosine LR will help"
  4. Edit src/train.py                   ← implement exactly that change
  5. Run 3 seeds:
       for SEED in 42 7 13; do SEED=$SEED uv run src/train.py; done
  6. Read the last 3 entries in results.jsonl
  7. Compute mean ± std of val_acc and test_acc across the 3 seeds
  8. Append to outputs/research_log.md  ← record hypothesis, change, result, interpretation
  9. GOTO 1
```

One experiment = 9 minutes wall-clock (3 seeds × 3 minutes each).

---

## 2. Success Metric

- **Primary:**   mean `val_acc` across seeds 42, 7, 13  (target ≥ 0.80)
- **Secondary:** mean `test_acc` across seeds — treat this as ground truth;
  `val_acc` is a noisy proxy (only 20 samples), `test_acc` is the honest signal
- **Stability:** std of `val_acc` across seeds — lower is better; a result with
  high mean but std > 0.08 is unreliable
- **Per-class:** `per_class_acc` — pay special attention to `HCM` and `MINF`,
  which are the historically confused pair

All metrics are written to `outputs/results.jsonl` by `train.py`.
A confusion matrix PNG is saved to `outputs/confusion_matrix_<id>.png` after
every seed run.

### How to compute the 3-seed average

```bash
cat outputs/results.jsonl | python3 -c "
import sys, json
runs = [json.loads(l) for l in sys.stdin]
last3 = runs[-3:]
import statistics
val_accs  = [r['val_acc']  for r in last3]
test_accs = [r.get('test_acc', 0) for r in last3]
print(f'seeds:     {[r[\"seed\"] for r in last3]}')
print(f'val_acc:   mean={statistics.mean(val_accs):.4f}  std={statistics.stdev(val_accs):.4f}')
print(f'test_acc:  mean={statistics.mean(test_accs):.4f}  std={statistics.stdev(test_accs):.4f}')
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

---

## 4. Current Architecture (`MultiModalCardiacNet`)

```
MRI branch:
  Input (B, 1, 16, 128, 128)
  → Stage1–4: ResNet+SE 3D CNN → GAP → Dropout → (B, 128)

Clinical branch:
  Input (B, 5)
  → Linear(5→64) → BN → ReLU → Linear(64→128) → BN → ReLU → (B, 128)

Fusion:
  cat[(B,128), (B,128)] → (B, 256) → Linear(256, 5) → logits
```

The agent may modify:
- `ClinicalEncoder` (e.g. deeper MLP, add dropout, feature normalisation)
- `MultiModalCardiacNet` (e.g. attention-based fusion instead of concat)
- `CardiacCNN3D` (e.g. wider channels, deeper stages)
- `train_one_epoch` (e.g. augmentation, loss function)
- `main()` (e.g. scheduler, ensemble size, TTA passes)

The agent must **never** modify:
- `BUDGET_SECONDS = 180`
- `MAX_EPOCHS = 400`
- The seed block at the top of `main()`
- The `results.jsonl` logging block at the end of `main()`
- `src/prepare.py`

---

## 5. Directions to Explore (ordered by expected impact)

Work through these roughly in order, but always let the data guide you.
If an experiment hurts performance, revert that specific change before
trying the next hypothesis.

### 5.1 Clinical Feature Engineering
The raw `[Height, Weight, EDV, ESV, EF]` features have very different scales
(Height ~150–190, EF ~0.1–0.8). Consider:
- **Z-score normalisation** per feature using training-set mean/std
- **Add derived features**: BMI = Weight/Height², SV = EDV−ESV (stroke volume)
- **Log-transform** EDV/ESV (right-skewed distributions)

### 5.2 Fusion Strategy
The current concat+linear fusion is simple. Alternatives:
- **Gated fusion**: learn a gate `g = sigmoid(W·clinical)` to weight MRI features
- **Cross-attention**: clinical embedding attends over MRI spatial features
- **FiLM conditioning**: modulate MRI feature maps with clinical scalars

### 5.3 Learning Rate Schedule
- **Cosine annealing:** `CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)`
- **Warmup + cosine:** linear warmup for first 5% of steps, then cosine decay
- **OneCycleLR:** `OneCycleLR(optimizer, max_lr=LR, ...)`

### 5.4 Data Augmentation (MRI branch only)
All augmentations must be applied on-GPU (use `torch` ops).

| Augmentation | Implementation hint |
|---|---|
| Random horizontal flip | `torch.flip(vol, dims=[-1])` with p=0.5 |
| Random vertical flip   | `torch.flip(vol, dims=[-2])` with p=0.5 |
| Random depth flip      | `torch.flip(vol, dims=[-3])` with p=0.5 |
| Intensity jitter       | `vol * (1 + 0.1*torch.randn_like(vol))` |
| Gaussian noise         | `vol + 0.02*torch.randn_like(vol)` |

### 5.5 Loss Function
- **Class-weighted CE**: upweight HCM and MINF (the confused pair)
  ```python
  weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0], device=DEVICE)
  criterion = nn.CrossEntropyLoss(weight=weights)
  ```
- **Label smoothing**: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Focal loss**: focus on hard examples

### 5.6 Regularisation
- Increase / decrease `DROPOUT` (try 0.1, 0.3, 0.5)
- `WEIGHT_DECAY` sweep (1e-3, 1e-2, 5e-2)
- Add dropout inside `ClinicalEncoder`

### 5.7 Architecture Changes
- Wider MRI encoder: `1→32→64→128→256`
- Deeper clinical MLP: `5→64→128→128`
- Larger ensemble: `N_ENSEMBLE = 3` or `4`

### 5.8 Batch Size & Optimiser
- `BATCH_SIZE = 4` (more gradient steps per epoch: 14 vs 7)
- `torch.optim.SGD` with momentum=0.9 and cosine LR

---

## 6. Hyperparameter Block

```python
# ===========================================================================
# ★  HYPERPARAMETERS — agent modifies this block between experiments  ★
# ===========================================================================
LR           = 5e-4
BATCH_SIZE   = 8
DROPOUT      = 0.5
WEIGHT_DECAY = 0.05
ARCH_NOTES   = "..."
MAX_EPOCHS   = 400   # ← do NOT change; calibrated to 3-min budget
N_ENSEMBLE   = 2
```

---

## 7. Research Log Format

After every 3-seed experiment, append the following block to `outputs/research_log.md`:

```markdown
---
## Experiment <N> — <timestamp>
**Seeds:** 42, 7, 13
**Experiment IDs:** <id_seed42>, <id_seed7>, <id_seed13>

**Hypothesis:** <one sentence: what you expected and why>

**Change made:**
```diff
- old line(s) from train.py
+ new line(s) from train.py
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | X.XXXX  | X.XXXX   | X.XXXX   |
| 7    | X.XXXX  | X.XXXX   | X.XXXX   |
| 13   | X.XXXX  | X.XXXX   | X.XXXX   |
| **mean** | **X.XXXX** | **X.XXXX** | **X.XXXX** |
| **std**  | **X.XXXX** | **X.XXXX** | **X.XXXX** |

- per_class_acc (mean across seeds): NOR=X.XX  DCM=X.XX  HCM=X.XX  MINF=X.XX  RV=X.XX
- prev best mean val_acc: X.XXXX

**Interpretation:** <2–4 sentences: did it work? why? what does this suggest?>

**Next hypothesis:** <what to try next based on this result>
```

---

## 8. Stopping Criteria

Stop iterating when **any** of the following is true:
- Mean `val_acc` across 3 seeds ≥ 0.85 — excellent result, stop and report
- You have run **15 experiments** (45 seed-runs) — summarise findings
- Three consecutive experiments show no improvement > 0.02 in mean `val_acc`
  — try a fundamentally different direction before concluding

---

## 9. Final Report

When stopping, append a `## Final Summary` section to `outputs/research_log.md`:

```markdown
## Final Summary

**Best experiment:** <experiment_id>
**Best mean val_acc (3 seeds):** X.XXXX ± X.XXXX
**Best mean test_acc (3 seeds):** X.XXXX ± X.XXXX
**Best config:** <describe the winning train.py configuration>

**Key findings:**
1. ...
2. ...
3. ...

**HCM vs MINF confusion analysis:**
- Before clinical features: ...
- After clinical features: ...

**Recommended next steps (beyond 3-min budget):**
- ...
```

---

## 10. Quick Reference

```bash
# Run data preparation (once, or after any change to prepare.py)
uv run src/prepare.py --skip-download

# Standard experiment run — always use this, never run a single seed
for SEED in 42 7 13; do SEED=$SEED uv run src/train.py; done

# Compute 3-seed average for the last experiment
cat outputs/results.jsonl | python3 -c "
import sys, json, statistics
runs = [json.loads(l) for l in sys.stdin]
last3 = runs[-3:]
val_accs  = [r['val_acc']  for r in last3]
test_accs = [r.get('test_acc', 0) for r in last3]
print(f'seeds:     {[r[\"seed\"] for r in last3]}')
print(f'val_acc:   mean={statistics.mean(val_accs):.4f}  std={statistics.stdev(val_accs):.4f}')
print(f'test_acc:  mean={statistics.mean(test_accs):.4f}  std={statistics.stdev(test_accs):.4f}')
"

# View all results — modality-aware
cat outputs/results.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    pca = r.get('per_class_acc', {})
    mod = r.get('modality', 'mri')
    seed = r.get('seed', '?')
    print(f\"{r['experiment_id']}  [seed={seed}]  [{mod}]  val_acc={r['val_acc']:.4f}  test_acc={r.get('test_acc',0):.4f}\", end='')
    if pca:
        print('  ' + '  '.join(f\"{k}={v:.2f}\" for k, v in pca.items()))
    else:
        print('  ' + r['config']['arch_notes'][:50])
"

# View research log
cat outputs/research_log.md

# Check confusion matrix for latest run
ls -t outputs/confusion_matrix_*.png | head -1
```

---

*This file is read-only for the agent. Do not modify program.md.*
