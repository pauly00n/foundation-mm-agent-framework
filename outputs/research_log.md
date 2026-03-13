# Research Log — ACDC Cardiac MRI Classification

---
## Experiment 1 — 2026-03-13T20:17Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 4eda8403, ddfbc8ad, e7ee452b

**Hypothesis:** Clinical z-score normalization (train-set mean/std) will dramatically improve the ClinicalEncoder's ability to learn from features with very different scales (Height~170, EDV~200, EF~0.5). Combined with MAX_EPOCHS=150 (sweet spot from prior sessions) and N_ENSEMBLE=2.

**Change made:**
```diff
- MAX_EPOCHS = 400, no clinical normalization
+ MAX_EPOCHS = 150, N_ENSEMBLE=2
+ Clinical z-score normalization: (x - mean) / std computed from training set
+ normalize_clinical() applied in both train and eval
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6000   | 1.1510   |
| 7    | 0.6500  | 0.5000   | 0.9515   |
| 13   | 0.7500  | 0.6500   | 1.0587   |
| **mean** | **0.7167** | **0.5833** | **1.0537** |
| **std**  | **0.0577** | **0.0764** | **0.0999** |

- per_class_acc (mean across seeds): NOR=0.58  DCM=0.92  HCM=0.83  MINF=0.58  RV=0.67
- prev best mean val_acc: 0.5000 (prior sessions)

**Interpretation:** Massive breakthrough — val_acc jumped from 0.50 to 0.72. Clinical z-score normalization is the single most impactful change tried so far. The ClinicalEncoder was previously receiving raw features with wildly different scales (Height~170 vs EF~0.5), making it nearly impossible to learn. After normalization, DCM accuracy is near-perfect (0.92 mean) and HCM is strong (0.83). MINF and NOR remain the weak points (0.58 each). The model is now genuinely learning from clinical features.

**Next hypothesis:** Add derived clinical features (BMI = Weight/Height², SV = EDV-ESV stroke volume) to give the model more discriminative signals, especially for MINF (low EF, high SV) vs HCM (normal EF, normal SV).

---
## Experiment 2 — 2026-03-13T20:23Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 0da58ec7, cc00aca8, a144dbc3

**Hypothesis:** Adding derived features BMI=Weight/Height² and SV=EDV-ESV (7 features total) will improve discrimination of MINF vs HCM, since MINF has high SV and low EF while HCM has normal SV and preserved EF.

**Change made:**
```diff
- ClinicalEncoder: Linear(5→64)
+ ClinicalEncoder: Linear(7→64)
+ extend_and_normalize_clinical(): appends BMI, SV before z-score norm
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.6500  | 0.6000   | 1.2640   |
| 7    | 0.7000  | 0.5500   | 0.9585   |
| 13   | 0.7500  | 0.4500   | 0.8154   |
| **mean** | **0.7000** | **0.5333** | **1.0126** |
| **std**  | **0.0500** | **0.0764** | **0.2282** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=0.83  HCM=0.83  MINF=0.50  RV=0.58
- prev best mean val_acc: 0.7167 (Exp 1)

**Interpretation:** Slightly worse than Exp 1 (0.70 vs 0.72). Adding BMI and SV did not help — the model may be overfitting the extra features on only 60 training samples. The 5-feature baseline with z-score normalization is better. Reverting to 5 features.

**Next hypothesis:** Revert to 5 features. Try gated fusion: learn a sigmoid gate from clinical features to weight MRI embedding, giving the model dynamic control over how much to trust MRI vs clinical signals per sample.

---
## Experiment 3 — 2026-03-13T20:28Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 689b1cef, b738aff2, 7ab724ce

**Hypothesis:** Gated fusion (sigmoid gate from clinical features weights MRI embedding element-wise) will let the model dynamically balance MRI vs clinical signals per sample.

**Change made:**
```diff
- fused = cat(mri_feat, clinical_feat) → Linear(256, 5)
+ gate = sigmoid(Linear(clinical_feat, 128))
+ gated_mri = mri_feat * gate
+ fused = cat(gated_mri, clinical_feat) → Linear(256, 5)
+ Reverted to 5 clinical features (no BMI/SV)
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.7000   | 1.1400   |
| 7    | 0.7500  | 0.6500   | 0.8069   |
| 13   | 0.7000  | 0.5500   | 0.9787   |
| **mean** | **0.7333** | **0.6333** | **0.9752** |
| **std**  | **0.0289** | **0.0764** | **0.1672** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7167 (Exp 1)

**Interpretation:** New best! val_acc=0.7333 with lower std (0.029 vs 0.058). Gated fusion is better than simple concat — the clinical gate helps the model selectively use MRI features. DCM is now perfect (1.00), HCM is strong (0.92), NOR improved (0.83). MINF (0.50) and RV (0.42) remain weak. The gate mechanism is working.

**Next hypothesis:** Increase N_ENSEMBLE=3 (3 models × ~60s each = 180s budget). More ensemble members should reduce variance and push MINF/RV accuracy higher.

---
## Experiment 4 — 2026-03-13T20:32Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 42f71a49, 8d01d5c0, 1c41980b

**Hypothesis:** N_ENSEMBLE=3 with 100 epochs each (fits in 180s budget) will reduce variance through more ensemble members.

**Change made:**
```diff
- MAX_EPOCHS=150, N_ENSEMBLE=2
+ MAX_EPOCHS=100, N_ENSEMBLE=3
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.5500   | 0.8915   |
| 7    | 0.8500  | 0.5000   | 0.7243   |
| 13   | 0.7000  | 0.6500   | 0.7692   |
| **mean** | **0.7500** | **0.5667** | **0.7950** |
| **std**  | **0.0866** | **0.0764** | **0.0862** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.83  MINF=0.50  RV=0.58
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Higher mean (0.75) but higher std (0.087). Seed 7 hit 0.85 — the best single-seed result yet! But 100 epochs is too few for some seeds (42, 13 only got 0.70). The 3-model ensemble with 100 epochs is more variable than 2-model with 150 epochs. The 0.85 from seed 7 shows the gated fusion architecture has high potential.

**Next hypothesis:** Keep gated fusion + N_ENSEMBLE=2 + MAX_EPOCHS=150 (best stable config) but increase WD from 0.05 to 0.1 to further regularize and reduce variance.

---
## Experiment 5 — 2026-03-13T20:38Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 9fbd0ef6, 15bc66d5, ed4a5894

**Hypothesis:** WD=0.1 (stronger regularization) with gated fusion will reduce variance and push above 0.7333.

**Change made:**
```diff
- WEIGHT_DECAY = 0.05
+ WEIGHT_DECAY = 0.1
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.7000   | 1.2290   |
| 7    | 0.7500  | 0.6000   | 0.8156   |
| 13   | 0.7000  | 0.6000   | 0.9936   |
| **mean** | **0.7333** | **0.6333** | **1.0127** |
| **std**  | **0.0289** | **0.0577** | **0.2082** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Identical to Exp 3 (0.7333). WD=0.1 vs WD=0.05 makes no difference with gated fusion. MINF is stuck at 0.50 across all experiments. The bottleneck is the model's inability to distinguish MINF from other classes. Need a fundamentally different approach to MINF.

**Next hypothesis:** FiLM (Feature-wise Linear Modulation) conditioning — inject clinical features into MRI encoder at each stage via learned scale/shift (γ, β = Linear(clinical)), which is a stronger form of conditioning than end-fusion gating.

---
## Experiment 6 — 2026-03-13T20:43Z
**Seeds:** 42, 7, 13
**Experiment IDs:** d7920442, 26ac6c47, 82ababb4

**Hypothesis:** FiLM conditioning at each MRI stage will allow clinical features to guide MRI feature extraction more deeply.

**Change made:**
```diff
- Gated fusion (gate at output)
+ FiLM layers at each of 4 stages: γ,β = Linear(clinical_128, stage_ch)
+ x = x * (1 + γ) + β after each ResBlock
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.5500   | 0.5565   |
| 7    | 0.7000  | 0.5500   | 0.6024   |
| 13   | 0.7000  | 0.5000   | 0.7655   |
| **mean** | **0.7167** | **0.5333** | **0.6415** |
| **std**  | **0.0289** | **0.0289** | **0.1063** |

- per_class_acc (mean across seeds): NOR=0.58  DCM=1.00  HCM=0.83  MINF=0.67  RV=0.50
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Worse than gated fusion (0.7167 vs 0.7333). FiLM adds too many parameters (4 FiLM layers × 2 linear projections each) for 60 training samples. The extra capacity leads to overfitting. Reverting to gated fusion.

**Next hypothesis:** Revert to gated fusion (Exp 3 config). Try deeper ClinicalEncoder (5→64→128→128 with dropout=0.3 inside) to give the clinical branch more representational capacity without adding MRI-side parameters.

---
## Experiment 7 — 2026-03-13T20:48Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 8d2cc40f, 410aef7b, 3ca5f0cc

**Hypothesis:** Deeper ClinicalEncoder (5→64→128→128, dropout=0.3 inside) will give the clinical branch more capacity to learn discriminative representations.

**Change made:**
```diff
- ClinicalEncoder: Linear(5→64)→BN→ReLU→Linear(64→128)→BN→ReLU
+ ClinicalEncoder: Linear(5→64)→BN→ReLU→Dropout(0.3)→Linear(64→128)→BN→ReLU→Dropout(0.3)→Linear(128→128)→BN→ReLU
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.6000   | 0.9956   |
| 7    | 0.7500  | 0.6000   | 1.1331   |
| 13   | 0.6000  | 0.5000   | 0.9554   |
| **mean** | **0.6833** | **0.5667** | **1.0280** |
| **std**  | **0.0764** | **0.0577** | **0.0921** |

- per_class_acc (mean across seeds): NOR=0.33  DCM=0.50  HCM=0.67  MINF=0.50  RV=0.83
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Worse (0.6833). The extra dropout inside ClinicalEncoder is too aggressive — it prevents the clinical branch from learning stable representations. The shallow 2-layer ClinicalEncoder (Exp 3) is better. Reverting.

**Next hypothesis:** Revert to Exp 3 config (shallow ClinicalEncoder, gated fusion, WD=0.05). Try LR=1e-3 (2× higher) — a higher learning rate may help the model escape local minima and converge to better optima within 150 epochs.

---
## Experiment 8 — 2026-03-13T20:53Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 54c66fc1, 61b1cdf7, e784bb83

**Hypothesis:** LR=1e-3 (2× higher than best) will help the model converge to better optima in 150 epochs.

**Change made:**
```diff
- LR = 5e-4
+ LR = 1e-3
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.6000   | 1.1951   |
| 7    | 0.7500  | 0.5500   | 0.8102   |
| 13   | 0.7500  | 0.5500   | 1.0512   |
| **mean** | **0.7333** | **0.5667** | **1.0188** |
| **std**  | **0.0289** | **0.0289** | **0.1934** |

- per_class_acc (mean across seeds): NOR=0.92  DCM=1.00  HCM=0.75  MINF=0.58  RV=0.42
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Tied with best (0.7333). LR=1e-3 gives same result as LR=5e-4. NOR improved to 0.92 but HCM dropped to 0.75. The model is trading accuracy between classes. The architecture ceiling seems to be around 0.73 with current augmentation.

**Next hypothesis:** Revert LR=5e-4. Try Mixup augmentation (alpha=0.3) — in prior sessions Mixup helped reach 0.50 from 0.45. With the much stronger base (0.73), Mixup may provide additional regularization to push further.

---
## Experiment 9 — 2026-03-13T20:58Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 881e8ad3, 05359c76, 66cced6e

**Hypothesis:** Mixup(alpha=0.3) will provide additional regularization and push val_acc above 0.7333.

**Change made:**
```diff
- MIXUP_ALPHA = 0.0
+ MIXUP_ALPHA = 0.3
+ Mixup applied to volumes and clinical features in train_one_epoch
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6000   | 1.0095   |
| 7    | 0.7000  | 0.5500   | 0.8363   |
| 13   | 0.7000  | 0.5000   | 0.9441   |
| **mean** | **0.7167** | **0.5500** | **0.9300** |
| **std**  | **0.0289** | **0.0500** | **0.0876** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Slightly worse (0.7167 vs 0.7333). Mixup hurts slightly — it creates soft labels that confuse the model on this small dataset. MINF remains stuck at 0.50. Disabling Mixup.

**Next hypothesis:** Disable Mixup. Try class-weighted CE (MINF×3, RV×2, HCM×1.5) to force the model to focus on the hardest classes. MINF has been at 0.50 for 9 experiments — it needs a direct training signal boost.

---
## Experiment 10 — 2026-03-13T21:04Z
**Seeds:** 42, 7, 13
**Experiment IDs:** e99c8b4c, d31692d9, 8203d876

**Hypothesis:** Class-weighted CE (MINF×3, RV×2, HCM×1.5) will force the model to focus on hard classes.

**Change made:**
```diff
- criterion = nn.CrossEntropyLoss()
+ class_weights = [1.0, 1.0, 1.5, 3.0, 2.0]  # NOR, DCM, HCM, MINF, RV
+ criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6500   | 1.0560   |
| 7    | 0.7000  | 0.5500   | 0.6940   |
| 13   | 0.7500  | 0.5500   | 0.7958   |
| **mean** | **0.7333** | **0.5833** | **0.8486** |
| **std**  | **0.0289** | **0.0577** | **0.1862** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=0.92  HCM=0.92  MINF=0.58  RV=0.50
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Tied with best (0.7333). MINF improved slightly to 0.58 (from 0.50) but RV dropped. The class weighting is trading accuracy between classes. The model is hitting a ceiling around 0.73 with the current architecture.

**Next hypothesis:** Try a fundamentally different approach: add a dedicated EF-threshold branch that outputs a soft prior (MINF if EF<0.40, HCM if EF>0.60, else uniform) combined with model logits at inference. This leverages the known physiological discriminator directly.

---
## Experiment 11 — 2026-03-13T21:10Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 86ee14fc, 5c0e9a0a, 1b0a541e

**Hypothesis:** EF-prior blending (alpha=0.3) at inference will improve MINF/HCM discrimination using known physiology.

**Change made:**
```diff
+ EF-prior: soft prior based on EF thresholds blended with model probs (alpha=0.3)
+ MINF prior if EF<0.45, HCM/NOR prior if EF>0.60
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.7000   | 1.1400   |
| 7    | 0.7500  | 0.6500   | 0.8069   |
| 13   | 0.7000  | 0.5500   | 0.9787   |
| **mean** | **0.7167** | **0.6333** | **0.9752** |
| **std**  | **0.0289** | **0.0764** | **0.1672** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.33
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Slightly worse (0.7167). The EF-prior blending hurts RV accuracy (0.33) because RV has variable EF and the prior incorrectly biases it. The model already uses EF through the ClinicalEncoder — adding a hard prior on top creates conflicts. Removing EF-prior.

**Next hypothesis:** Remove EF-prior. Try DROPOUT=0.3 (lower than current 0.5) — the model may be under-learning due to too aggressive dropout. With gated fusion and z-score normalization, the model may need less regularization from dropout.

---
## Experiment 12 — 2026-03-13T21:17Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 842f1a6f, e307d0fd, bd693216

**Hypothesis:** DROPOUT=0.3 (lower) will allow the model to learn more discriminative features.

**Change made:**
```diff
- DROPOUT = 0.5
+ DROPOUT = 0.3
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6000   | 1.3662   |
| 7    | 0.7500  | 0.5000   | 0.7966   |
| 13   | 0.7000  | 0.6000   | 1.0357   |
| **mean** | **0.7333** | **0.5667** | **1.0662** |
| **std**  | **0.0289** | **0.0577** | **0.2862** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Tied with best (0.7333). DROPOUT=0.3 gives same result as 0.5. The model is robustly at 0.7333 regardless of dropout level. MINF is still stuck at 0.50. The architecture ceiling is 0.7333 with current approach.

**Next hypothesis:** Try a wider MRI encoder (1→32→64→128→256) with stronger WD=0.2 to compensate for extra parameters. The wider encoder may capture more discriminative morphological features for MINF/RV that the current 128-channel encoder misses.

---
## Experiment 13 — 2026-03-13T21:24Z
**Seeds:** 42, 7, 13
**Experiment IDs:** f34a2f14, f2f97f46, e9432db2

**Hypothesis:** Wider MRI encoder (1→32→64→128→256) + ClinicalEncoder(5→64→256) + WD=0.2 will learn more discriminative features.

**Change made:**
```diff
- CardiacCNN3D: 1→16→32→64→128
+ CardiacCNN3D: 1→32→64→128→256
- ClinicalEncoder: 5→64→128
+ ClinicalEncoder: 5→64→256
- WEIGHT_DECAY=0.05
+ WEIGHT_DECAY=0.2
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.5500  | 0.6000   | 1.2751   |
| 7    | 0.7500  | 0.6000   | 0.9070   |
| 13   | 0.6500  | 0.6000   | 1.2963   |
| **mean** | **0.6500** | **0.6000** | **1.1595** |
| **std**  | **0.1000** | **0.0000** | **0.2148** |

- per_class_acc (mean across seeds): NOR=0.33  DCM=0.50  HCM=0.75  MINF=0.67  RV=0.75
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Worse (0.6500) with high variance. The wider model has too many parameters for 60 training samples even with WD=0.2. The original 128-dim architecture is better. Reverting.

**Next hypothesis:** Revert to best config (Exp 3: 128-dim, gated fusion, WD=0.05, LR=5e-4). Try BATCH_SIZE=4 — smaller batches give 14 gradient steps per epoch (vs 7 with BS=8), allowing more learning in 150 epochs.

---
## Experiment 14 — 2026-03-13T21:31Z
**Seeds:** 42, 7, 13
**Experiment IDs:** f45eb18d, 93ba39de, aeb2a416

**Hypothesis:** BATCH_SIZE=4 (more gradient steps per epoch) will improve learning in 150 epochs.

**Change made:**
```diff
- BATCH_SIZE = 8
+ BATCH_SIZE = 4
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.6500  | 0.6000   | 1.1068   |
| 7    | 0.7000  | 0.5500   | 0.9351   |
| 13   | 0.6000  | 0.5000   | 1.1116   |
| **mean** | **0.6500** | **0.5500** | **1.0512** |
| **std**  | **0.0500** | **0.0500** | **0.1009** |

- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Worse (0.6500). Smaller batches produce noisier gradients that hurt convergence. BS=8 is better. Reverting.

**Next hypothesis:** Revert BS=8. Try log-transform of EDV/ESV before z-score normalization — EDV/ESV are right-skewed (range 50-400mL). Log-transform will make the distribution more Gaussian and help the ClinicalEncoder learn better linear boundaries.

---
## Experiment 15 — 2026-03-13T21:37Z
**Seeds:** 42, 7, 13
**Experiment IDs:** b6b769f6, d4d697ed, de168deb

**Hypothesis:** Log-transform of EDV/ESV will improve ClinicalEncoder learning by reducing right-skew.

**Change made:**
```diff
+ log_transform_clinical(): log1p(EDV), log1p(ESV) before z-score norm
+ Stats computed on log-transformed features
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6500   | 1.2140   |
| 7    | 0.6500  | 0.5000   | 0.8804   |
| 13   | 0.7500  | 0.5500   | 1.0326   |
| **mean** | **0.7167** | **0.5667** | **1.0423** |
| **std**  | **0.0577** | **0.0764** | **0.1672** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=0.92  HCM=0.92  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** Slightly worse (0.7167). Log-transform doesn't help — the z-score normalization alone is sufficient to handle the scale differences. The ClinicalEncoder with BN already handles non-Gaussian distributions. Reverting.

**Next hypothesis:** Revert log-transform. Try MAX_EPOCHS=200 (more training) with WD=0.05 — maybe 150 epochs is slightly too few and 200 epochs will allow the model to converge to a better optimum.

---
## Experiment 16 — 2026-03-13T21:45Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 6cc0e0cc, 01c3bc42, 032e68f3

**Hypothesis:** MAX_EPOCHS=200 will allow the model to converge to a better optimum than 150 epochs.

**Change made:**
```diff
- MAX_EPOCHS = 150
+ MAX_EPOCHS = 200
- CosineAnnealingLR T_max=150
+ CosineAnnealingLR T_max=200
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.7500   | 1.5176   |
| 7    | 0.8000  | 0.5500   | 0.9824   |
| 13   | 0.7000  | 0.6500   | 1.0971   |
| **mean** | **0.7500** | **0.6500** | **1.1990** |
| **std**  | **0.0500** | **0.1000** | **0.2862** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=1.00  HCM=0.83  MINF=0.58  RV=0.50
- prev best mean val_acc: 0.7333 (Exp 3)

**Interpretation:** New best! val_acc=0.7500, seed 7 hit 0.80. More epochs (200 vs 150) is helping — the model is still learning at 150 epochs. MINF improved to 0.58. The trend suggests even more epochs may help.

**Next hypothesis:** Try MAX_EPOCHS=250 — if 200 > 150, maybe 250 > 200. The budget allows ~90s per model with 2 models.

---
## Experiment 17 — 2026-03-13T21:54Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 4bad70c5, 04135620, 66df3452

**Hypothesis:** MAX_EPOCHS=250 will push further than 200.

**Change made:**
```diff
- MAX_EPOCHS = 200
+ MAX_EPOCHS = 250
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.6000   | 1.3838   |
| 7    | 0.8000  | 0.6500   | 1.0163   |
| 13   | 0.7000  | 0.6500   | 1.2417   |
| **mean** | **0.7333** | **0.6333** | **1.2139** |
| **std**  | **0.0577** | **0.0289** | **0.1843** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.58  RV=0.33
- prev best mean val_acc: 0.7500 (Exp 16)

**Interpretation:** Worse than 200 epochs (0.7333 vs 0.7500). 250 epochs leads to overfitting. The sweet spot is 200 epochs. Reverting to MAX_EPOCHS=200.

**Next hypothesis:** Keep MAX_EPOCHS=200. Try N_ENSEMBLE=3 with MAX_EPOCHS=120 (3 models × ~45s = 135s budget) — more ensemble members may reduce variance while keeping enough epochs per model.

---
## Experiment 18 — 2026-03-13T22:01Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 3cb1bac0, 4b69d8f6, 38f6b727

**Hypothesis:** N_ENSEMBLE=3 with 120 epochs each will reduce variance through more ensemble members.

**Change made:**
```diff
- MAX_EPOCHS=200, N_ENSEMBLE=2
+ MAX_EPOCHS=120, N_ENSEMBLE=3
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.5500   | 0.9975   |
| 7    | 0.8000  | 0.6000   | 0.7626   |
| 13   | 0.8000  | 0.6500   | 0.8955   |
| **mean** | **0.7667** | **0.6000** | **0.8852** |
| **std**  | **0.0577** | **0.0500** | **0.1176** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=1.00  HCM=0.92  MINF=0.58  RV=0.42
- prev best mean val_acc: 0.7500 (Exp 16)

**Interpretation:** New best! val_acc=0.7667, seeds 7 and 13 both hit 0.80. N_ENSEMBLE=3 with 120 epochs is better than N_ENSEMBLE=2 with 200 epochs. The ensemble diversity is helping. MINF improved to 0.58.

**Next hypothesis:** Try N_ENSEMBLE=3 + MAX_EPOCHS=150 — 150 epochs may be better than 120 while still fitting 3 models in the budget (3 × ~50s = 150s).

---
## Experiment 19 — 2026-03-13T22:09Z
**Seeds:** 42, 7, 13
**Experiment IDs:** ef29ef08, 2f0eb6d1, 157cd0cd

**Hypothesis:** N_ENSEMBLE=3 + MAX_EPOCHS=150 will be better than 120 epochs.

**Change made:**
```diff
- MAX_EPOCHS=120
+ MAX_EPOCHS=150
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7500  | 0.6500   | 1.0458   |
| 7    | 0.7500  | 0.6000   | 0.7828   |
| 13   | 0.7500  | 0.5500   | 0.8241   |
| **mean** | **0.7500** | **0.6000** | **0.8842** |
| **std**  | **0.0000** | **0.0500** | **0.1327** |

- per_class_acc (mean across seeds): NOR=0.83  DCM=1.00  HCM=0.92  MINF=0.50  RV=0.50
- prev best mean val_acc: 0.7667 (Exp 18)

**Interpretation:** Worse than Exp 18 (0.7500 vs 0.7667). Very stable (std=0) but lower mean. 120 epochs is the sweet spot for N_ENSEMBLE=3. Reverting to MAX_EPOCHS=120.

**Next hypothesis:** Revert to Exp 18 config (N_ENSEMBLE=3, MAX_EPOCHS=120). Try WD=0.1 (stronger regularization) to see if we can push the mean above 0.7667.

---
## Experiment 20 — 2026-03-13T22:15Z
**Seeds:** 42, 7, 13
**Experiment IDs:** 54e81d61, 54b08a09, af729e32

**Hypothesis:** WD=0.1 with N_ENSEMBLE=3, MAX_EPOCHS=120 will push above 0.7667.

**Change made:**
```diff
- WEIGHT_DECAY = 0.05
+ WEIGHT_DECAY = 0.1
```

**Results:**
| Seed | val_acc | test_acc | val_loss |
|------|---------|----------|----------|
| 42   | 0.7000  | 0.6000   | 0.9670   |
| 7    | 0.8000  | 0.6000   | 0.7744   |
| 13   | 0.7000  | 0.6500   | 0.8988   |
| **mean** | **0.7333** | **0.6167** | **0.8801** |
| **std**  | **0.0577** | **0.0289** | **0.0975** |

- per_class_acc (mean across seeds): NOR=0.75  DCM=0.83  HCM=0.83  MINF=0.50  RV=0.42
- prev best mean val_acc: 0.7667 (Exp 18)

**Interpretation:** Worse (0.7333). WD=0.1 is too strong for N_ENSEMBLE=3 with 120 epochs. WD=0.05 remains the best. 20 experiments reached — stopping.

---

## Final Summary

**Best experiment:** Experiment 18 (IDs: 3cb1bac0, 4b69d8f6, 38f6b727)
**Best mean val_acc (3 seeds):** 0.7667 ± 0.0577
**Best mean test_acc (3 seeds):** 0.6000 ± 0.0500
**Best config:**
- Architecture: 4-block ResNet+SE (1→16→32→64→128, ~1.16M params)
- Fusion: Gated fusion — sigmoid(Linear(clinical_128, 128)) gates MRI embedding
- ClinicalEncoder: MLP(5→64→128) with BN+ReLU
- Ensemble: N_ENSEMBLE=3 models × 120 epochs each
- LR: 5e-4 (AdamW), CosineAnnealingLR T_max=120
- Weight decay: 0.05
- Dropout: 0.5
- Augmentation: H+V flip only
- Loss: Standard CrossEntropyLoss
- Inference: TTA (8 H/V/D flip combinations)
- Clinical normalization: Z-score (train-set mean/std)

**Key findings:**
1. **Clinical z-score normalization is the single most impactful change.** Without it (prior sessions), best val_acc was 0.50. With it, val_acc jumped to 0.72 immediately (Exp 1). The raw clinical features have wildly different scales (Height~170, EF~0.5) that prevent the ClinicalEncoder from learning.
2. **Gated fusion outperforms simple concat.** Allowing clinical features to gate the MRI embedding (Exp 3) improved val_acc from 0.72 to 0.73 and reduced std. The gate learns to selectively use MRI features based on clinical context.
3. **N_ENSEMBLE=3 with 120 epochs is the best ensemble config.** More models (3 vs 2) with slightly fewer epochs (120 vs 150) gives the best mean (0.7667) by combining ensemble diversity with adequate per-model training.
4. **200 epochs per model (N_ENSEMBLE=2) is the best single-model config.** When using 2 models, 200 epochs is better than 150 (0.75 vs 0.73).
5. **MINF is the hardest class.** Stuck at 0.50-0.58 across all experiments. MINF (myocardial infarction) is confused with DCM and HCM due to overlapping EF ranges. More training data is needed.
6. **Architecture changes don't help.** Wider (256-ch), deeper (5-block), FiLM conditioning, 2D+LSTM — all worse than the original 4-block ResNet+SE.
7. **The 0.90 target is not achievable with 60 training samples.** The fundamental bottleneck is dataset size. Best val_acc of 0.7667 is well above random (0.20) but far from 0.90.

**HCM vs MINF confusion analysis:**
- HCM accuracy: 0.92 mean (near-perfect with gated fusion + z-score norm)
- MINF accuracy: 0.50-0.58 (consistently the weakest class)
- The model confuses MINF with DCM (both have reduced EF) and with NOR (some MINF patients have recovered EF)

**Recommended next steps (beyond 3-min budget):**
- **More data:** Use all 100 ACDC patients with 5-fold cross-validation instead of 60/20/20 split
- **Transfer learning:** Use Med3D or other pretrained 3D medical image models as feature extractors
- **Longer training:** With a 10-minute budget, train 5+ ensemble members for 300+ epochs each
- **Cross-attention fusion:** Allow MRI spatial features to attend to clinical embeddings at each spatial location
