---
## Experiment 1 — 2026-03-14T23:43:35Z
**Experiment ID (commit hash):** 3b50037375e7

**Hypothesis:** The current configuration is a reasonable starting point; running it establishes the baseline mean val_acc to guide all future changes.

**Change made:**
```diff
  (no change — baseline run)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6300 |
| val_acc (std)  | 0.1166 |
| per_fold_acc   | [0.80, 0.70, 0.45, 0.60, 0.60] |
| per_class_acc  | NOR=0.55  DCM=0.70  HCM=0.65  MINF=0.45  RV=0.80 |
| prev best      | N/A |

**Interpretation:** Baseline achieves 0.63 mean val_acc with very high variance (std=0.1166). Fold 3 collapsed to 0.45, suggesting the model is sensitive to data partition. Train_acc reaches 0.95+ while val_acc lags badly — clear overfitting. MINF is the hardest class (0.45). The 120 epochs complete in only ~33s per fold (well within 3-min budget), meaning we could run many more epochs or try other changes. High variance suggests regularisation is insufficient.

**Next hypothesis:** Reduce MAX_EPOCHS to 60 to stop before overfitting kicks in — train_acc was already 0.875 at E60 while val_acc was still reasonable.

---
## Experiment 2 — 2026-03-14T23:45:49Z
**Experiment ID (commit hash):** 216828724be0

**Hypothesis:** Reducing MAX_EPOCHS from 120 to 60 will reduce overfitting (train_acc was 0.95+ while val_acc lagged badly in baseline).

**Change made:**
```diff
- MAX_EPOCHS = 120
+ MAX_EPOCHS = 60
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.7000 |
| val_acc (std)  | 0.0949 |
| per_fold_acc   | [0.85, 0.70, 0.60, 0.60, 0.75] |
| per_class_acc  | NOR=0.70  DCM=0.75  HCM=0.55  MINF=0.70  RV=0.80 |
| prev best      | 0.6300 |

**Interpretation:** Strong improvement (+0.07 mean val_acc). MINF jumped from 0.45→0.70, confirming overfitting was hurting the hard classes most. Variance also reduced (0.1166→0.0949). Total wall time only 85.5s — we're using less than 10% of the budget. Fold 3 still weak at 0.60. HCM is now the weakest class at 0.55. Since we're only using ~17s per fold, we can afford many more epochs — try 200 epochs to see if more training with the cosine schedule helps.

**Next hypothesis:** Try MAX_EPOCHS=200 — with only 17s per fold we have budget headroom, and the cosine schedule may benefit from a longer warmup/decay cycle.
