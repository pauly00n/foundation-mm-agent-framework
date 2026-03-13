# program.md — Instructions for the Autonomous ML Agent

You are an autonomous ML research agent.  Your sole objective is to
**maximise `val_acc` on the ACDC cardiac MRI 5-class classification task**
within a fixed 10-minute wall-clock training budget per experiment.

---

## 0. Ground Rules

| Rule | Detail |
|------|--------|
| **Only modify `src/train.py`** | `src/prepare.py` is fixed infrastructure — never touch it. |
| **One hypothesis per run** | Change exactly one thing at a time so results are interpretable. |
| **Always read history first** | Before proposing a change, read `outputs/results.jsonl` in full. |
| **Log everything** | Append to `outputs/research_log.md` after every run (template below). |
| **Budget is sacred** | `BUDGET_SECONDS = 600` in `train.py` must never be changed. |
| **No external data** | Only the ACDC dataset prepared by `prepare.py`. |

---

## 1. The Experiment Loop

```
LOOP:
  1. Read outputs/results.jsonl          ← understand what has been tried
  2. Read outputs/research_log.md        ← understand why things were tried
  3. Form ONE hypothesis                 ← e.g. "cosine LR will help"
  4. Edit src/train.py                   ← implement exactly that change
  5. Run:  uv run src/train.py           ← 10-minute budget fires automatically
  6. Read the printed summary + results.jsonl
  7. Append to outputs/research_log.md  ← record hypothesis, change, result, interpretation
  8. GOTO 1
```

---

## 2. Success Metric

- **Primary:**   `val_acc`  (higher is better; target ≥ 0.80)
- **Secondary:** `val_loss` (lower is better; use to break ties)

Both are logged automatically to `outputs/results.jsonl` by `train.py`.

---

## 3. Directions to Explore (ordered by expected impact)

Work through these roughly in order, but always let the data guide you.
If an experiment hurts performance, revert that specific change before
trying the next hypothesis.

### 3.1 Learning Rate Schedule
- **Cosine annealing:** `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)`
- **Warmup + cosine:** linear warmup for first 5% of steps, then cosine decay
- **OneCycleLR:** `torch.optim.lr_scheduler.OneCycleLR`
- Tip: estimate `T_max` from the number of epochs completed in the baseline run.

### 3.2 Data Augmentation (apply only to training set)
Add transforms inside `train_one_epoch` or wrap the DataLoader.
All augmentations must be applied on-GPU (use `torch` ops, not PIL/albumentations).

| Augmentation | Implementation hint |
|---|---|
| Random horizontal flip | `torch.flip(vol, dims=[-1])` with p=0.5 |
| Random vertical flip   | `torch.flip(vol, dims=[-2])` with p=0.5 |
| Random depth flip      | `torch.flip(vol, dims=[-3])` with p=0.5 |
| Intensity jitter       | `vol * (1 + 0.1*torch.randn_like(vol))` |
| Gaussian noise         | `vol + 0.02*torch.randn_like(vol)` |
| Random crop + resize   | crop to 0.8–1.0 of spatial size, resize back |

### 3.3 Architecture Changes
Modify `CardiacCNN3D` in `train.py`.

| Idea | How |
|---|---|
| Wider channels | 1→64→128→256→512 |
| Deeper network | Add a 5th ConvBlock |
| Residual connections | Add skip connections between blocks (ResNet-style) |
| Squeeze-and-Excitation | Add SE block after each ConvBlock |
| Depth-wise separable conv | Replace Conv3d with depthwise+pointwise |
| Larger kernel in first layer | kernel_size=7 for first conv |

### 3.4 Regularisation
- Increase / decrease `DROPOUT` (try 0.1, 0.2, 0.4, 0.5)
- Add `WEIGHT_DECAY` sweep (1e-5, 1e-3)
- Label smoothing: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- Stochastic depth (drop entire blocks with probability p)

### 3.5 Batch Size & Optimiser
- Try `BATCH_SIZE = 4` (more gradient steps per minute) or `16`
- Try `torch.optim.SGD` with momentum=0.9 and cosine LR
- Try `torch.optim.Adam` (no weight decay) vs `AdamW`

### 3.6 Mixed Precision
- `USE_AMP = True` is the default — verify it's actually helping throughput
- If you see NaN losses, set `USE_AMP = False`

### 3.7 Class Imbalance
- Check class distribution in `data/processed/splits.json`
- If imbalanced: use `torch.nn.CrossEntropyLoss(weight=class_weights)`
- Compute class weights: `n_total / (n_classes * n_per_class)`

---

## 4. What to Change in `train.py`

The agent-editable section is clearly marked:

```python
# ===========================================================================
# ★  HYPERPARAMETERS — agent modifies this block between experiments  ★
# ===========================================================================
LR           = 3e-4
BATCH_SIZE   = 8
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4
ARCH_NOTES   = "..."
```

Beyond hyperparameters, the agent may also modify:
- The `CardiacCNN3D` class definition
- The `train_one_epoch` function (e.g. to add augmentation or LR scheduling)
- The `main()` function (e.g. to add a scheduler step after each epoch)

The agent must **never** modify:
- `BUDGET_SECONDS = 600`
- The `results.jsonl` logging block at the end of `main()`
- `src/prepare.py`

---

## 5. Research Log Format

After every run, append the following block to `outputs/research_log.md`:

```markdown
---
## Run <N> — <experiment_id> — <timestamp>

**Hypothesis:** <one sentence: what you expected and why>

**Change made:**
```diff
- old line(s) from train.py
+ new line(s) from train.py
```

**Result:**
- val_acc:  X.XXXX  (prev best: X.XXXX)
- val_loss: X.XXXX
- epochs_run: N
- wall_time_s: NNN

**Interpretation:** <2–4 sentences: did it work? why? what does this suggest?>

**Next hypothesis:** <what to try next based on this result>
```

---

## 6. Stopping Criteria

Stop iterating when **any** of the following is true:
- `val_acc >= 0.90` — excellent result, stop and report
- You have run **20 experiments** — summarise findings in `research_log.md`
- Three consecutive experiments show no improvement > 0.005 in `val_acc`
  — try a fundamentally different direction before concluding

---

## 7. Final Report

When stopping, append a `## Final Summary` section to `outputs/research_log.md`:

```markdown
## Final Summary

**Best experiment:** <experiment_id>
**Best val_acc:** X.XXXX
**Best val_loss:** X.XXXX
**Best config:** <describe the winning train.py configuration>

**Key findings:**
1. ...
2. ...
3. ...

**Recommended next steps (beyond 10-min budget):**
- ...
```

---

## 8. Quick Reference

```bash
# Run data preparation (once)
uv run src/prepare.py

# Run one experiment (10-minute budget)
uv run src/train.py

# View all results so far
cat outputs/results.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r['experiment_id']}  val_acc={r['val_acc']:.4f}  val_loss={r['val_loss']:.4f}  {r['config']['arch_notes'][:60]}\")
"

# View research log
cat outputs/research_log.md
```

---

*This file is read-only for the agent. Do not modify program.md.*
