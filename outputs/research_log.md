# Research Log — ACDC Cardiac MRI 5-Fold CV

---
## Experiment 1 — 2026-03-16T04:13Z
**Experiment ID (commit hash):** 60f2bc54d74a

**Hypothesis:** Establish baseline with current config: wider CNN (1→32→64→128→256, 10.7M params), cross-attention fusion, deeper ClinicalEncoder (5→64→128→128), TTA=8, label_smoothing=0.1, WD=0.05, MAX_EPOCHS=200, 5-fold CV on all 100 patients.

**Change made:**
```diff
First experiment — no changes from starting config.
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.5900 |
| val_acc (std)  | 0.0583 |
| per_fold_acc   | [0.65, 0.65, 0.55, 0.60, 0.50] |
| per_class_acc  | NOR=0.60  DCM=0.65  HCM=0.50  MINF=0.50  RV=0.70 |
| prev best      | N/A (first experiment) |

**Interpretation:** The wider CNN (10.7M params) is severely overfitting on 80 training patients — train_acc reaches 0.95+ but val_acc is only 0.59. HCM and MINF are at chance (0.50). The cross-attention fusion with this large model is not working well. Need to dramatically reduce model size.

**Next hypothesis:** Revert to smaller CNN (1→16→32→64→128, ~1.16M params) with gated fusion (simpler than cross-attention), reduce MAX_EPOCHS to 80, increase DROPOUT to 0.6, and increase WEIGHT_DECAY to 0.1 to combat overfitting.

---
## Experiment 2 — 2026-03-16T04:17Z
**Experiment ID (commit hash):** 87bcd87ed9c7

**Hypothesis:** Smaller CNN (1→16→32→64→128, 1.5M params) with gated fusion, DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80 will reduce overfitting and improve val_acc.

**Change made:**
```diff
- CardiacCNN3D: 1→32→64→128→256 (10.7M params), cross-attention fusion
+ CardiacCNN3D: 1→16→32→64→128 (1.5M params), gated fusion
- DROPOUT=0.5, WD=0.05, MAX_EPOCHS=200
+ DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80
- ClinicalEncoder: 3-layer (5→64→128→128 with dropout)
+ ClinicalEncoder: 2-layer (5→64→128)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6900 |
| val_acc (std)  | 0.1068 |
| per_fold_acc   | [0.85, 0.70, 0.60, 0.55, 0.75] |
| per_class_acc  | NOR=0.85  DCM=0.70  HCM=0.45  MINF=0.80  RV=0.65 |
| prev best      | 0.5900 |

**Interpretation:** Big improvement (+0.10). Smaller model generalizes much better. Fold 1 hit 0.85! But high variance (0.1068) and HCM is weak (0.45). The model only uses ~23s per fold (well under 180s budget) — 80 epochs is too few. Need more epochs to use the full budget. MINF jumped to 0.80 — the gated fusion with z-score normalization is working well for MINF.

**Next hypothesis:** Increase MAX_EPOCHS to 200 to use more of the 180s budget. Keep DROPOUT=0.6 and WD=0.1 to prevent overfitting with more epochs.

---
## Experiment 3 — 2026-03-16T04:23Z
**Experiment ID (commit hash):** 392d1827d683

**Hypothesis:** MAX_EPOCHS=200 will use more of the 180s budget and improve convergence.

**Change made:**
```diff
- MAX_EPOCHS = 80
+ MAX_EPOCHS = 200
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6700 |
| val_acc (std)  | 0.1166 |
| per_fold_acc   | [0.80, 0.65, 0.60, 0.50, 0.80] |
| per_class_acc  | NOR=0.80  DCM=0.65  HCM=0.50  MINF=0.75  RV=0.65 |
| prev best      | 0.6900 |

**Interpretation:** Worse than Exp 2 (0.67 vs 0.69). More epochs leads to overfitting — train_acc reaches 0.97+ but val_acc drops. The model uses ~57s per fold with 200 epochs. The sweet spot was 80 epochs (~23s). Should use remaining budget for ensembling multiple models rather than training one model longer.

**Next hypothesis:** Revert to MAX_EPOCHS=80. Try DROPOUT=0.5 (slightly less) and WD=0.05 (less aggressive) — the current regularization may be too strong, preventing the model from learning enough in 80 epochs.

---
## Experiment 4 — 2026-03-16T04:25Z
**Experiment ID (commit hash):** c197d67db38f

**Hypothesis:** DROPOUT=0.5, WD=0.05, MAX_EPOCHS=80 — less regularization will let the model learn more.

**Change made:**
```diff
- DROPOUT=0.6, WD=0.1, MAX_EPOCHS=200
+ DROPOUT=0.5, WD=0.05, MAX_EPOCHS=80
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6900 |
| val_acc (std)  | 0.1020 |
| per_fold_acc   | [0.85, 0.65, 0.65, 0.55, 0.75] |
| per_class_acc  | NOR=0.80  DCM=0.70  HCM=0.50  MINF=0.80  RV=0.65 |
| prev best      | 0.6900 |

**Interpretation:** Same as Exp 2 (0.69). The model uses only ~23s per fold — 157s of budget is wasted. HCM is stuck at 0.50. The key insight: with 80 epochs taking only 23s, I can fit ~7 models per fold in the 180s budget. Ensembling multiple diverse models should reduce variance and improve accuracy.

**Next hypothesis:** Implement within-fold ensembling: train N_ENSEMBLE=5 models per fold (each with different random seed), average their predictions at inference. This uses ~115s of the 180s budget and provides ensemble diversity.

---
## Experiment 5 — 2026-03-16T04:37Z
**Experiment ID (commit hash):** 28e906e6e0d8

**Hypothesis:** Within-fold ensembling (N_ENSEMBLE=5, 80 epochs each) will reduce variance and improve accuracy through model diversity.

**Change made:**
```diff
+ N_ENSEMBLE = 5 models per fold
+ evaluate_ensemble_with_tta(): averages TTA predictions across all ensemble models
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6600 |
| val_acc (std)  | 0.0860 |
| per_fold_acc   | [0.80, 0.65, 0.60, 0.55, 0.70] |
| per_class_acc  | NOR=0.75  DCM=0.70  HCM=0.40  MINF=0.85  RV=0.60 |
| prev best      | 0.6900 |

**Interpretation:** Worse than single model (0.66 vs 0.69). The 5-model ensemble didn't help — HCM dropped to 0.40. Models are too similar (same arch, same data, different seeds only). MINF=0.85 is excellent though! The ensemble is averaging out correct HCM predictions. Need to go back to single model approach but with more epochs.

**Next hypothesis:** Revert to single model (N_ENSEMBLE=1). Try MAX_EPOCHS=120 with DROPOUT=0.5, WD=0.05 — a middle ground between 80 (too few) and 200 (overfitting). Also remove label smoothing (use standard CE) since it may be hurting HCM discrimination.

---
## Experiment 6 — 2026-03-16T04:41Z
**Experiment ID (commit hash):** 1be371a21730

**Hypothesis:** Single model, MAX_EPOCHS=120, standard CE (no label smoothing) will improve HCM discrimination.

**Change made:**
```diff
- N_ENSEMBLE=5, MAX_EPOCHS=80, label_smoothing=0.1
+ N_ENSEMBLE=1, MAX_EPOCHS=120, standard CE (no label smoothing)
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6200 |
| val_acc (std)  | 0.1030 |
| per_fold_acc   | [0.75, 0.65, 0.50, 0.50, 0.70] |
| per_class_acc  | NOR=0.70  DCM=0.65  HCM=0.50  MINF=0.70  RV=0.55 |
| prev best      | 0.6900 |

**Interpretation:** Worse (0.62 vs 0.69). Removing label smoothing hurt significantly. Label smoothing was providing important regularization. The best config remains Exp 2 (DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80, label_smoothing=0.1).

**Next hypothesis:** Revert to Exp 2 config (DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80, label_smoothing=0.1). Try N_ENSEMBLE=3 (fewer ensemble members, more diverse) with class-weighted CE to boost HCM and RV.

---
## Experiment 7 — 2026-03-16T04:48Z
**Experiment ID (commit hash):** 1d0f9ccbe8e3

**Hypothesis:** N_ENSEMBLE=3 with class-weighted CE (HCM=2.0, RV=1.5) will boost hard classes while ensemble reduces variance.

**Change made:**
```diff
+ N_ENSEMBLE=3, DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80
+ class_weights = [1.0, 1.0, 2.0, 1.0, 1.5] + label_smoothing=0.1
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6700 |
| val_acc (std)  | 0.0748 |
| per_fold_acc   | [0.80, 0.70, 0.60, 0.60, 0.65] |
| per_class_acc  | NOR=0.70  DCM=0.60  HCM=0.60  MINF=0.75  RV=0.70 |
| prev best      | 0.6900 |

**Interpretation:** Lower variance (0.075) but lower mean (0.67 vs 0.69). HCM improved to 0.60 (from 0.45-0.50 in Exp 2). Class weights are helping HCM but ensemble is averaging out some correct predictions. The best approach is single model with the right hyperparameters.

**Next hypothesis:** Single model (N_ENSEMBLE=1), keep class-weighted CE (HCM=2.0, RV=1.5) + label_smoothing=0.1, DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80. Also try LR=1e-3 (higher) with OneCycleLR for faster convergence in 80 epochs.

---
## Experiment 8 — 2026-03-16T04:51Z
**Experiment ID (commit hash):** 058018ccc705

**Hypothesis:** LR=1e-3 with OneCycleLR will converge faster in 80 epochs.

**Change made:**
```diff
- LR=5e-4, CosineAnnealingLR, N_ENSEMBLE=3
+ LR=1e-3, OneCycleLR, N_ENSEMBLE=1
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6500 |
| val_acc (std)  | 0.0949 |
| per_fold_acc   | [0.80, 0.70, 0.65, 0.55, 0.55] |
| per_class_acc  | NOR=0.50  DCM=0.75  HCM=0.60  MINF=0.75  RV=0.65 |
| prev best      | 0.6900 |

**Interpretation:** Worse (0.65 vs 0.69). OneCycleLR with LR=1e-3 is too aggressive. NOR dropped to 0.50. The best config remains Exp 2 (CosineAnnealingLR, LR=5e-4, single model, 80 epochs).

**Next hypothesis:** Revert to exact Exp 2 config (LR=5e-4, CosineAnnealingLR, DROPOUT=0.6, WD=0.1, MAX_EPOCHS=80, label_smoothing=0.1, no class weights). But set MAX_EPOCHS=500 and let the budget (180s) naturally stop training — this way the model trains as many epochs as possible within the budget, using CosineAnnealingWarmRestarts for cyclic LR.

---
## Experiment 9 — 2026-03-16T05:04Z
**Experiment ID (commit hash):** 3c8128ce554b

**Hypothesis:** MAX_EPOCHS=500 with CosineAnnealingWarmRestarts T_0=40 will use the full 180s budget and find better optima through cyclic LR.

**Change made:**
```diff
- MAX_EPOCHS=80, CosineAnnealingLR
+ MAX_EPOCHS=500, CosineAnnealingWarmRestarts T_0=40, T_mult=2
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6300 |
| val_acc (std)  | 0.0812 |
| per_fold_acc   | [0.75, 0.70, 0.60, 0.55, 0.55] |
| per_class_acc  | NOR=0.70  DCM=0.70  HCM=0.55  MINF=0.65  RV=0.55 |
| prev best      | 0.6900 |

**Interpretation:** Worse (0.63 vs 0.69). 500 epochs overfits despite warm restarts. The model completes all 500 epochs in ~143s. The best config remains 80 epochs (Exp 2). Training longer hurts.

**Next hypothesis:** Revert to Exp 2 config exactly (80 epochs, CosineAnnealingLR, LR=5e-4, DROPOUT=0.6, WD=0.1, label_smoothing=0.1). Try adding Stochastic Weight Averaging (SWA) over the last 20 epochs to smooth the loss landscape and improve generalization.

---
## Experiment 10 — 2026-03-16T05:08Z
**Experiment ID (commit hash):** 77752faf0f06

**Hypothesis:** SWA from epoch 60 (averaging last 21 checkpoints) will smooth the loss landscape and improve generalization.

**Change made:**
```diff
+ SWA: average model weights from epoch 60 to 80 (21 checkpoints)
+ BN stats update after SWA weight averaging
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.6600 |
| val_acc (std)  | 0.1020 |
| per_fold_acc   | [0.85, 0.65, 0.60, 0.55, 0.65] |
| per_class_acc  | NOR=0.55  DCM=0.75  HCM=0.55  MINF=0.75  RV=0.70 |
| prev best      | 0.6900 |

**Interpretation:** SWA didn't help (0.66 vs 0.69). Averaging over 21 checkpoints smooths out discriminative features. Fold 1 still hits 0.85 but other folds are weak. The high variance across folds is the main issue.

**Next hypothesis:** Remove SWA. Try DROPOUT=0.7 (more aggressive) with WD=0.15 — the model is clearly overfitting (train_acc=0.95+ vs val_acc=0.55-0.65 on weak folds). More aggressive regularization may help the weak folds without hurting the strong ones.
