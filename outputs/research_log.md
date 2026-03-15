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

---
## Experiment 3 — 2026-03-14T23:53:04Z
**Experiment ID (commit hash):** 10456cd00875

**Hypothesis:** MAX_EPOCHS=200 will allow the cosine schedule to decay more gradually and improve convergence.

**Change made:**
```diff
- MAX_EPOCHS = 60
+ MAX_EPOCHS = 200
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6200 |
| val_acc (std)  | 0.0980 |
| per_fold_acc   | [0.80, 0.60, 0.60, 0.50, 0.60] |
| per_class_acc  | NOR=0.40  DCM=0.75  HCM=0.65  MINF=0.50  RV=0.80 |
| prev best      | 0.7000 |

**Interpretation:** Worse than E2 (0.62 vs 0.70). More epochs caused overfitting — train_acc reached 1.0 in some folds while val_acc dropped. NOR collapsed to 0.40. MAX_EPOCHS=60 was the sweet spot. The model clearly overfits with 200 epochs. Reverting to 60 epochs and trying a different direction.

**Next hypothesis:** Keep MAX_EPOCHS=60 and add label smoothing (0.1) to reduce overconfidence and improve generalization on hard classes.

---
## Experiment 4 — 2026-03-14T23:56:06Z
**Experiment ID (commit hash):** d9aacf611276

**Hypothesis:** Label smoothing=0.1 will reduce overconfidence and improve generalization on hard classes.

**Change made:**
```diff
- criterion = nn.CrossEntropyLoss()
+ criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6800 |
| val_acc (std)  | 0.1030 |
| per_fold_acc   | [0.85, 0.70, 0.55, 0.60, 0.70] |
| per_class_acc  | NOR=0.60  DCM=0.75  HCM=0.55  MINF=0.70  RV=0.80 |
| prev best      | 0.7000 |

**Interpretation:** Slightly worse than E2 (0.68 vs 0.70). Label smoothing didn't help here — it may be slowing convergence within the 60-epoch budget. Fold 3 still weak at 0.55. HCM remains the hardest class. Reverting label smoothing and trying class-weighted CE to specifically boost MINF and RV.

**Next hypothesis:** Remove label smoothing, keep MAX_EPOCHS=60, add class weights [1,1,1,2,2] to upweight MINF and RV (the historically hard classes).

---
## Experiment 5 — 2026-03-14T23:58:33Z
**Experiment ID (commit hash):** cbbfb9d12f7e

**Hypothesis:** Class-weighted CE [1,1,1,2,2] will upweight MINF and RV, improving accuracy on hard classes.

**Change made:**
```diff
- criterion = nn.CrossEntropyLoss()
+ class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], device=DEVICE)
+ criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6800 |
| val_acc (std)  | 0.0927 |
| per_fold_acc   | [0.85, 0.70, 0.60, 0.60, 0.65] |
| per_class_acc  | NOR=0.60  DCM=0.75  HCM=0.55  MINF=0.70  RV=0.80 |
| prev best      | 0.7000 |

**Interpretation:** Same as E4 (0.68), still below E2 (0.70). Class weighting didn't help — the model already gets MINF=0.70 and RV=0.80 with plain CE. HCM remains the bottleneck at 0.55. The loss function changes (E4, E5) haven't beaten plain CE. Reverting to plain CE + MAX_EPOCHS=60 and trying richer augmentation (depth flip + intensity jitter + gaussian noise) to improve generalization.

**Next hypothesis:** Keep MAX_EPOCHS=60 + plain CE, add depth flip (p=0.5), intensity jitter, and gaussian noise augmentations to improve generalization.

---
## Experiment 6 — 2026-03-15T00:00:55Z
**Experiment ID (commit hash):** 0eb4d3102b89

**Hypothesis:** Adding depth flip + intensity jitter + gaussian noise will improve generalization.

**Change made:**
```diff
+ if torch.rand(1).item() < 0.5:
+     volumes[i] = torch.flip(volumes[i], dims=[-3])   # D flip
+ volumes = volumes * (1 + 0.1 * torch.randn_like(volumes)).clamp(0,1)
+ volumes = volumes + 0.02 * torch.randn_like(volumes).clamp(0,1)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6600 |
| val_acc (std)  | 0.1068 |
| per_fold_acc   | [0.80, 0.50, 0.60, 0.65, 0.75] |
| per_class_acc  | NOR=0.95  DCM=0.80  HCM=0.25  MINF=0.80  RV=0.50 |
| prev best      | 0.7000 |

**Interpretation:** Worse than E2 (0.66 vs 0.70). Heavy augmentation hurt badly — HCM collapsed to 0.25 and RV to 0.50. The intensity jitter + noise is too aggressive for only 60 epochs; the model can't converge properly. Fold 2 dropped to 0.50. Reverting to H+V flips only and trying a fundamentally different direction: enriching the clinical feature set with derived features (BMI, stroke volume).

**Next hypothesis:** Revert augmentation to H+V flips only, keep MAX_EPOCHS=60, and add derived clinical features BMI=Weight/Height² and SV=EDV-ESV to the ClinicalEncoder input (7 features instead of 5).

---
## Experiment 7 — 2026-03-15T00:04:14Z
**Experiment ID (commit hash):** cd4b2209b76c

**Hypothesis:** Adding derived clinical features BMI and SV will give the model more discriminative power.

**Change made:**
```diff
- nn.Linear(5, 64)  # ClinicalEncoder input
+ nn.Linear(7, 64)  # ClinicalEncoder input
+ augment_clinical(): adds BMI=W/H^2, SV=EDV-ESV
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6900 |
| val_acc (std)  | 0.0860 |
| per_fold_acc   | [0.80, 0.75, 0.55, 0.70, 0.65] |
| per_class_acc  | NOR=0.60  DCM=0.75  HCM=0.70  MINF=0.60  RV=0.80 |
| prev best      | 0.7000 |

**Interpretation:** Close to E2 (0.69 vs 0.70) but with lower variance (0.0860 vs 0.0949). HCM improved significantly (0.55→0.70). Fold 2 improved (0.70→0.75). However MINF dropped (0.70→0.60) and fold 3 still weak at 0.55. The derived features help HCM but hurt MINF. Combining derived features with a higher LR may help convergence.

**Next hypothesis:** Keep derived features (7), MAX_EPOCHS=60, and increase LR from 5e-4 to 1e-3 to see if faster convergence improves results.
