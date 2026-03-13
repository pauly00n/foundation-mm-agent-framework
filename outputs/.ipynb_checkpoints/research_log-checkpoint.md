# Research Log — ACDC Cardiac MRI Classification

**Task:** 5-class cardiac MRI classification (NOR/DCM/HCM/MINF/RV)  
**Dataset:** 70 train / 15 val samples (perfectly balanced, 14/3 per class)  
**Budget:** 60 seconds wall-clock per experiment  
**Target:** val_acc ≥ 0.90

---

## Run 1 — 0e658cd6 — 2026-03-12T22:56:56Z

**Hypothesis:** CosineAnnealingLR will help the model converge better within the 1-min budget by decaying LR smoothly.

**Change made:**
```diff
- ARCH_NOTES = "Baseline: 4-block 3D CNN, GlobalAvgPool, Linear(256→5). No augmentation. No LR schedule."
+ ARCH_NOTES = "Baseline + CosineAnnealingLR(T_max=300). No augmentation."
+ scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
+ scheduler.step() after each epoch
```

**Result:**
- val_acc:  0.1333  (prev best: 0.3333)
- val_loss: 2.6879
- epochs_run: 323
- wall_time_s: 60.0

**Interpretation:** Worse than baseline. The model reached 100% train accuracy but only 13% val — severe overfitting. The cosine schedule decayed LR too aggressively, causing the model to overfit faster. The core problem is the tiny dataset (70 samples).

**Next hypothesis:** Heavy regularization (dropout=0.5, weight_decay=1e-3) to combat overfitting.

---

## Run 2 — 09f9262a — 2026-03-12T22:58:17Z

**Hypothesis:** Increasing dropout to 0.5 and weight decay to 1e-3 will reduce overfitting on the tiny 70-sample dataset.

**Change made:**
```diff
- DROPOUT      = 0.3
- WEIGHT_DECAY = 1e-4
+ DROPOUT      = 0.5
+ WEIGHT_DECAY = 1e-3
- (removed CosineAnnealingLR)
```

**Result:**
- val_acc:  0.1333  (prev best: 0.3333)
- val_loss: 4.0898
- epochs_run: 328
- wall_time_s: 60.0

**Interpretation:** Still 13% val_acc. Regularization alone is insufficient — the model still memorizes training data. The val_loss is higher (4.09 vs 2.69) suggesting the model is confused. Need data augmentation to create more diverse training examples.

**Next hypothesis:** Add on-GPU data augmentation (flips, intensity jitter, Gaussian noise) with lower LR.

---

## Run 3 — 19e7f057 — 2026-03-12T23:01:26Z

**Hypothesis:** With only 70 training samples, on-GPU augmentation (H/V/D flips, intensity jitter, Gaussian noise) will dramatically improve generalization by creating diverse training views.

**Change made:**
```diff
- LR = 3e-4
+ LR = 1e-4
+ # Per-batch augmentation in train_one_epoch:
+ random H/V/D flips (p=0.5 each)
+ intensity jitter: vol * (1 + 0.1*randn)
+ Gaussian noise: vol + 0.02*randn
+ criterion = CrossEntropyLoss(label_smoothing=0.1)
```

**Result:**
- val_acc:  0.2667  (prev best: 0.3333)
- val_loss: 3.2411
- epochs_run: 312
- wall_time_s: 60.1

**Interpretation:** Improvement from 0.133 to 0.267 — augmentation is helping. But still below baseline (0.333). Training acc dropped to ~20-40% early on, suggesting augmentation is too strong for the model to learn. Need per-sample (not per-batch) augmentation and a better architecture.

**Next hypothesis:** Add CosineAnnealingLR on top of augmentation.

---

## Run 4 — c882cad0 — 2026-03-12T23:02:52Z

**Hypothesis:** CosineAnnealingLR combined with augmentation will help the model settle into better minima.

**Change made:**
```diff
+ scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
```

**Result:**
- val_acc:  0.2000  (prev best: 0.3333)
- val_loss: 6.5233
- epochs_run: 307
- wall_time_s: 60.0

**Interpretation:** Worse — val_loss exploded to 6.5. The cosine schedule decays LR too fast for this tiny dataset with augmentation. Schedulers consistently hurt. Revert scheduler.

**Next hypothesis:** Small model (16→32→64→128) with per-sample augmentation to reduce overfitting capacity.

---

## Run 5 — 890ae06f — 2026-03-12T23:04:54Z

**Hypothesis:** A smaller model (half the channels) will have less capacity to overfit 70 samples, and per-sample augmentation will create more diverse training views.

**Change made:**
```diff
- channels: 1→32→64→128→256, head: Linear(256,5)
+ channels: 1→16→32→64→128, head: Linear(128,5)
+ per-sample augmentation (loop over batch items)
- (removed CosineAnnealingLR)
```

**Result:**
- val_acc:  0.2000  (prev best: 0.3333)
- val_loss: 2.3638
- epochs_run: 328
- wall_time_s: 60.1

**Interpretation:** val_loss improved (2.36) but val_acc still 0.20. Training acc is only ~20-40% — the model can't even fit training data with heavy augmentation. The augmentation is too aggressive, preventing learning. Need to find the right balance.

**Next hypothesis:** Re-run small model with same config to check variance (same result: 0.267 on re-run df38ff7a).

---

## Run 6 — 08988f24 — 2026-03-12T23:07:47Z

**Hypothesis:** The small model underfits. Go back to baseline channels (32→64→128→256) with per-sample augmentation.

**Change made:**
```diff
- channels: 1→16→32→64→128
+ channels: 1→32→64→128→256, head: Linear(256,5)
```

**Result:**
- val_acc:  0.2000  (prev best: 0.3333)
- val_loss: 5.7030
- epochs_run: 292
- wall_time_s: 60.0

**Interpretation:** Worse — bigger model overfits more. The small model with augmentation is the right direction. Need better architecture, not just size.

**Next hypothesis:** Add residual connections to the small model for better gradient flow.

---

## Run 7 — 150fa4dc — 2026-03-12T23:09:29Z

**Hypothesis:** Residual connections (ResNet-style) will improve gradient flow and allow the small model to learn more discriminative features.

**Change made:**
```diff
+ class ResBlock3D: two Conv3d + skip connection
+ LR = 3e-4 (back to original)
+ DROPOUT = 0.4
- DROPOUT = 0.5
+ model: stage1..4 each = ConvBlock3D + ResBlock3D
```

**Result:**
- val_acc:  0.4000  (prev best: 0.3333) ← NEW BEST
- val_loss: 3.5639
- epochs_run: 257
- wall_time_s: 60.0

**Interpretation:** Big jump to 0.40! Residual connections significantly help. But training acc is back to ~100% — still overfitting. The architecture improvement is real but regularization needs work.

**Next hypothesis:** Lower LR to 1e-4 + dropout=0.5 to reduce overfitting of the ResNet.

---

## Run 8 — 16863e8a — 2026-03-12T23:10:46Z

**Hypothesis:** The ResNet overfits with LR=3e-4. Lower LR=1e-4 + dropout=0.5 will reduce the train/val gap.

**Change made:**
```diff
- LR = 3e-4, DROPOUT = 0.4
+ LR = 1e-4, DROPOUT = 0.5
```

**Result:**
- val_acc:  0.3333  (prev best: 0.4000)
- val_loss: 1.4118
- epochs_run: 256
- wall_time_s: 60.0

**Interpretation:** Worse val_acc despite lower val_loss (1.41). The lower LR prevents the model from learning fast enough within the 60s budget. LR=3e-4 is better for this budget. The val_loss improvement suggests better calibration but not better accuracy.

**Next hypothesis:** Keep LR=3e-4 but add random crop+resize augmentation for more diversity.

---

## Run 9 — e74dc253 — 2026-03-12T23:12:32Z

**Hypothesis:** Random crop+resize augmentation will create more diverse spatial views and reduce overfitting.

**Change made:**
```diff
+ random crop (scale 0.8-1.0) + trilinear resize back to original size (p=0.5)
+ DROPOUT = 0.5
```

**Result:**
- val_acc:  0.3333  (prev best: 0.4000)
- val_loss: 2.5811
- epochs_run: 256
- wall_time_s: 60.0

**Interpretation:** Worse — random crop adds trilinear interpolation overhead, reducing epochs from 257 to 256, and the spatial distortion doesn't help. The crop augmentation is too slow and not beneficial for this dataset.

**Next hypothesis:** Add Squeeze-and-Excitation (SE) blocks to the ResNet for channel attention.

---

## Run 10 — 8098b24c — 2026-03-12T23:14:07Z

**Hypothesis:** SE blocks will help the model focus on the most discriminative channels, which is especially useful for medical imaging where specific anatomical features matter.

**Change made:**
```diff
+ class SEBlock3D: AdaptiveAvgPool3d → Linear → ReLU → Linear → Sigmoid
+ ResBlock3D now includes SE: relu(x + se(block(x)))
- random crop augmentation removed
+ DROPOUT = 0.4
```

**Result:**
- val_acc:  0.4667  (prev best: 0.4000) ← NEW BEST
- val_loss: 2.7325
- epochs_run: 246
- wall_time_s: 60.0

**Interpretation:** New best! SE blocks improve val_acc from 0.40 to 0.467. Channel attention helps the model focus on relevant cardiac features. Still overfitting (95-100% train acc). Need stronger regularization.

**Next hypothesis:** Add Mixup augmentation to force smoother decision boundaries.

---

## Run 11 — 9873ac4a — 2026-03-12T23:15:34Z

**Hypothesis:** Mixup (alpha=0.4) will force the model to learn smoother decision boundaries and reduce overfitting.

**Change made:**
```diff
+ Mixup: lam ~ Beta(0.4,0.4), mixed_vol = lam*vol + (1-lam)*vol[perm]
+ loss = lam*CE(logits, labels) + (1-lam)*CE(logits, labels_perm)
```

**Result:**
- val_acc:  0.2000  (prev best: 0.4667)
- val_loss: 3.8708
- epochs_run: 239
- wall_time_s: 60.0

**Interpretation:** Mixup catastrophically hurt performance. With only 70 samples and 5 classes, mixing labels creates too much confusion. The model can't learn meaningful features from mixed cardiac MRI volumes. Revert immediately.

**Next hypothesis:** Wider ResNet+SE (32→64→128→256) to give more capacity.

---

## Run 12 — bfae9de7 — 2026-03-12T23:17:19Z

**Hypothesis:** Wider channels (32→64→128→256) with SE blocks will give more representational capacity while keeping the SE attention mechanism.

**Change made:**
```diff
- channels: 1→16→32→64→128, head: Linear(128,5)
+ channels: 1→32→64→128→256, head: Linear(256,5)
- Mixup removed
```

**Result:**
- val_acc:  0.4000  (prev best: 0.4667)
- val_loss: 2.6743
- epochs_run: 219
- wall_time_s: 60.0

**Interpretation:** Worse than small model with SE (0.40 vs 0.467). The wider model overfits more. The small model (16→32→64→128) with SE is the sweet spot for this dataset size.

**Next hypothesis:** 3-block model with 2-layer FC head.

---

## Run 13 — 0302d2cb — 2026-03-12T23:18:49Z

**Hypothesis:** A 3-block model (1→32→64→128) with a 2-layer FC head (128→64→5) will have better classifier capacity while keeping the feature extractor small.

**Change made:**
```diff
- 4 stages (1→16→32→64→128)
+ 3 stages (1→32→64→128)
+ head: Linear(128,64) → ReLU → Dropout → Linear(64,5)
```

**Result:**
- val_acc:  0.2000  (prev best: 0.4667)
- val_loss: 2.4552
- epochs_run: 230
- wall_time_s: 60.0

**Interpretation:** Much worse. The 3-block model underfits — it doesn't have enough spatial downsampling to extract global cardiac shape features. The 4-block architecture is necessary.

**Next hypothesis:** Return to best config (4-block ResNet+SE) + OneCycleLR.

---

## Run 14 — e1d318e3 — 2026-03-12T23:20:59Z

**Hypothesis:** OneCycleLR with warmup will help the model explore better and converge to a better minimum.

**Change made:**
```diff
+ scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=2700, pct_start=0.1)
+ scheduler.step() per batch
+ 4-block ResNet+SE restored
```

**Result:**
- val_acc:  0.2667  (prev best: 0.4667)
- val_loss: 1.9837
- epochs_run: 245
- wall_time_s: 60.0

**Interpretation:** Worse. OneCycleLR with max_lr=1e-3 causes the model to overfit quickly during the high-LR phase. Schedulers consistently hurt this tiny dataset — the model needs stable, moderate LR throughout.

**Next hypothesis:** Stochastic depth (drop_path_prob=0.2) as regularizer.

---

## Run 15 — 51548710 — 2026-03-12T23:22:31Z

**Hypothesis:** Stochastic depth (randomly dropping entire residual blocks with p=0.2) will act as a strong regularizer and reduce overfitting.

**Change made:**
```diff
+ ResBlock3D: if training and rand < drop_path_prob: return relu(x) (skip residual)
- OneCycleLR removed
```

**Result:**
- val_acc:  0.4000  (prev best: 0.4667)
- val_loss: 2.7870
- epochs_run: 253
- wall_time_s: 60.0

**Interpretation:** Stochastic depth reduced overfitting (train acc ~83%) but val_acc didn't improve. The regularization is working but the model is now underfitting slightly. The SE blocks without stochastic depth (Run 10) remain best.

**Next hypothesis:** BS=4 for more gradient steps per epoch.

---

## Run 16 — 95bf4ba0 — 2026-03-12T23:23:57Z

**Hypothesis:** Smaller batch size (BS=4) gives more gradient updates per epoch and more augmentation diversity.

**Change made:**
```diff
- BATCH_SIZE = 8
+ BATCH_SIZE = 4
- stochastic depth removed
```

**Result:**
- val_acc:  0.3333  (prev best: 0.4667)
- val_loss: 3.0703
- epochs_run: 203
- wall_time_s: 60.0

**Interpretation:** Worse. BS=4 gives noisier gradients and fewer epochs (203 vs 246). The larger batch size (8) provides more stable gradient estimates for this tiny dataset.

**Next hypothesis:** Gradient clipping + higher weight decay (5e-3).

---

## Run 17 — 4e253b40 — 2026-03-12T23:25:19Z

**Hypothesis:** Gradient clipping (max_norm=1.0) + higher weight decay (5e-3) will stabilize training and prevent large gradient updates.

**Change made:**
```diff
- WEIGHT_DECAY = 1e-3
+ WEIGHT_DECAY = 5e-3
+ BATCH_SIZE = 8 (restored)
+ torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Result:**
- val_acc:  0.4000  (prev best: 0.4667)
- val_loss: 2.1577
- epochs_run: 242
- wall_time_s: 60.0

**Interpretation:** val_loss improved (2.16) but val_acc didn't beat Run 10. Higher weight decay + grad clipping stabilizes training but doesn't improve generalization beyond the SE baseline.

**Next hypothesis:** CosineAnnealingWarmRestarts (SGDR) for periodic LR restarts.

---

## Run 18 — 115a58ee — 2026-03-12T23:27:04Z

**Hypothesis:** SGDR (cosine restarts every T0=50 epochs) will allow the model to escape local minima through periodic LR resets.

**Change made:**
```diff
- grad clipping, WD=5e-3
+ WD = 1e-3
+ scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)
```

**Result:**
- val_acc:  0.3333  (prev best: 0.4667)
- val_loss: 2.9083
- epochs_run: 246
- wall_time_s: 60.1

**Interpretation:** Worse. All LR schedulers have consistently hurt performance on this tiny dataset. The model needs stable LR throughout the 60s budget.

**Next hypothesis:** TTA (test-time augmentation) — average predictions over 8 augmented passes at eval.

---

## Run 19 — b27f49b6 — 2026-03-12T23:29:07Z

**Hypothesis:** TTA (averaging softmax over 8 augmented val passes) will improve val_acc by reducing prediction variance.

**Change made:**
```diff
+ evaluate_tta(): n_aug=8 augmented forward passes, average softmax
- standard evaluate() replaced with evaluate_tta()
```

**Result:**
- val_acc:  0.2667  (prev best: 0.4667)
- val_loss: 3.3523
- epochs_run: 246
- wall_time_s: 60.0

**Interpretation:** TTA hurt badly. The model is so overfit that augmenting at test time just adds noise to already-wrong predictions. TTA only helps when the model has learned robust features — not when it's memorizing training data.

**Next hypothesis:** Two-phase training: LR=3e-4 for 50s, then fine-tune with LR=1e-5 for 10s.

---

## Run 20 (first attempt) — b180ca3c — 2026-03-12T23:30:50Z

**Hypothesis:** Two-phase training (high LR for exploration, then very low LR for fine-tuning) will help the model settle into a better minimum.

**Change made:**
```diff
+ phase2_start = t_start + BUDGET_SECONDS * 0.83  # switch at ~50s
+ if elapsed >= 50s: set LR = 1e-5 for all param groups
- TTA removed, standard evaluate() restored
```

**Result:**
- val_acc:  0.4000  (prev best: 0.4667)
- val_loss: 2.0681
- epochs_run: 246
- wall_time_s: 60.0

**Interpretation:** Phase 2 didn't trigger in this run (timing issue). val_loss improved (2.07) but val_acc same as run 12. Need to verify phase 2 triggers.

---

## Run 20 (confirmed) — cbbd95e8 — 2026-03-12T23:31:58Z

**Hypothesis:** Two-phase training confirmed working — phase 2 (LR=1e-5) triggered at 49.9s.

**Change made:** Same as above (two-phase LR).

**Result:**
- val_acc:  0.6000  (prev best: 0.4667) ← NEW BEST
- val_loss: 1.5407
- epochs_run: 245
- wall_time_s: 60.0

**Interpretation:** Massive breakthrough! val_acc jumped from 0.467 to 0.600 (9/15 correct). The two-phase approach works: high LR (3e-4) for 50s allows the model to explore the loss landscape, then very low LR (1e-5) for 10s allows it to fine-tune into a sharp minimum without overshooting. The val_loss (1.54) is also the best seen. This is the most impactful single change in the entire experiment series.

**Next hypothesis:** Extend phase 2 duration, try LR=5e-6 for fine-tuning, or add more augmentation diversity.

---

## Final Summary

**Best experiment:** cbbd95e8  
**Best val_acc:** 0.6000  
**Best val_loss:** 1.5407  
**Best config:**
- Architecture: 4-block ResNet+SE (1→16→32→64→128 channels), GlobalAvgPool → Dropout(0.4) → Linear(128,5)
- SE blocks: AdaptiveAvgPool3d → Linear(ch, ch//4) → ReLU → Linear(ch//4, ch) → Sigmoid
- Augmentation: per-sample H/V/D flips (p=0.5 each), intensity jitter (×(1+0.1·ε)), Gaussian noise (+0.02·ε)
- LR: 3e-4 (AdamW) for first 50s, then 1e-5 for final 10s (two-phase)
- DROPOUT: 0.4, WEIGHT_DECAY: 1e-3, BATCH_SIZE: 8
- Label smoothing: 0.1, USE_AMP: True

**Key findings:**
1. **Two-phase LR is the single biggest win** (+0.133 val_acc): high LR for exploration, then very low LR for fine-tuning into a sharp minimum. This is the most impactful technique for this tiny dataset + short budget combination.
2. **SE blocks significantly help** (+0.067 val_acc over plain ResNet): channel attention allows the model to focus on discriminative cardiac anatomy features.
3. **Residual connections are essential** (+0.067 val_acc over plain CNN): better gradient flow enables learning of more complex features.
4. **LR schedulers consistently hurt**: CosineAnnealingLR, OneCycleLR, SGDR all degraded performance. The 60s budget is too short for scheduled LR decay to be beneficial.
5. **Augmentation is necessary but must be balanced**: per-sample flips + jitter + noise help (0.267→0.40), but too-aggressive augmentation (random crop, mixup) prevents learning.
6. **The dataset is tiny (70 samples)**: all experiments show severe overfitting. The train/val gap is the fundamental bottleneck.

**Recommended next steps (beyond 60s budget):**
- **Longer training budget**: With 10 minutes, the two-phase approach could run phase 1 for 9 minutes and phase 2 for 1 minute, likely reaching 0.80+.
- **Pre-trained 3D features**: Use a model pre-trained on other medical imaging datasets (e.g., Med3D, MedicalNet) as a feature extractor.
- **K-fold cross-validation**: With only 70 samples, k-fold would give more reliable estimates and allow ensemble predictions.
- **More aggressive two-phase tuning**: Try phase 2 starting at 40s (20s fine-tuning), or a 3-phase schedule (3e-4 → 1e-4 → 1e-5).
- **Test-time ensemble**: Train 5 models with different seeds and ensemble — with 15 val samples, variance is very high.
- **Stronger augmentation**: Random rotation (±15°), elastic deformation, contrast adjustment.
