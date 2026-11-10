# Results

All offline numbers are on held-out episodes (every 5th recording, copies grouped) of
`demos/cube_grid_all_pi05` (100 demos, 92 parse) unless stated. "Live" means on the real arm.

## 1. Live, 2026-11-05

| run | what happened |
|---|---|
| `weak/001210`, `piper_1.json` | looked at the ceiling: wrong config (cube ckpts need `piper_1_banana.json`: 30 Hz, gripper 0..1); preflight now refuses it |
| `2ep/004033`, 1500 steps | gripper closed then opened 8 steps later; JEV `missed_place` |
| `004033`, 3000 steps ×2 | safety abort: wrist (J6) asked to jump 0.55 / 0.53 rad in one step → `clamp_jumps` |
| `004033` + clamp (dry) | lined up and **hovered** until timeout |
| `004033` + `--recover servo-grasp` | LR bucket stuck at +0.62 for 20 iterations; UD swung −0.23 → −1.07 → +2.12 → −1.5; arm driven into the table, CAN send failures, controller in teach mode, **arm dropped** → workspace guard |
| `004033` + `--recover descend` | hover → descend 0.171 → 0.152 m → **grasped**; pi0.5 carried it over the bowl; never released; JEV-9B said placed (0.82) |

## 2. JEV-lite vs JEV-9B (same 240 held-out frames, full resolution)

| | JEV-lite (`model_dinos_g4_aug`) | JEV-9B (direction adapter) |
|---|---|---|
| held over the bowl, said placed | **0/60** | 35/60 |
| placed AUROC / acc@0.5 | **0.991 / 98.3%** | 0.959 / 81.7% |
| holding AUROC / acc@0.5 | **0.993 / 96.7%** | 0.854 / 74.6% |
| latency | **25.9 ms** both questions | 167.6 ms per question |

All held-out frames (4,564): phase accuracy 91.3%, holding AUROC 0.995, placed AUROC 0.961. Placed recall
by time since release: 68% (0.5-1 s), 93% (1-2 s), 95% (2-4 s), 82% (4-8 s), 30% (arm back home).

Ablations (DINOv2-small): mean-pooled only 88.4% phase acc; + 4×4 grid 90.9%; + augmentation same clean,
+2 points under simulated drift (shifted copy: 88.1 → 90.0%, placed AUROC 0.923 → 0.935; 3 seeds,
std ≤ 0.005). Wrist only 88.1%, side only 82.1% (mean-pooled). Adding gripper width as input: no gain.

Live first frames (74 runs, never trained on): 67 idle, 2 correctly "holding" (the attempt started
with the cube in the gripper), misses with the cube at the image edge.

Thresholds re-checked: holding ≥ 0.35 → TPR 96.4%, FPR 0.3%; placed ≥ 0.6 → 0% on approach/idle,
1.9% on held-over-bowl. Kept.

## 3. Monitor rules (replay of all 24 live trajectories from 2026-11-05)

Longest "gripper open and joints < 0.02 rad/s": demos ≤ 0.93 s (p95 0.60 s); normal live runs ≤ 1.2 s;
live hovers 7-80 s. With a 3 s trigger: **7/7 hovers flagged at 21-26 s, 0/12 closed runs flagged.**

## 4. Grasp offset (3,062 train / 846 held-out frames within 15 cm of the grasp)

| model | radial | tangential | height |
|---|---|---|---|
| predict the mean (MAE) | 2.07 cm | 0.68 cm | 2.41 cm |
| single regressor (MAE) | 0.75 cm | 0.57 cm | 0.85 cm |

| distance to grasp | moved closer | xy error after one correction |
|---|---|---|
| 3-6 cm (median 2.5) | 93% | 1.1 cm |
| 6-15 cm (median 6.1) | 99% | 1.3 cm |
| runtime path, full-res frames, 3-15 cm | 94% | 4.1 → 1.3 cm |

Ensemble of 5: median xy error 0.92 cm; from ≥ 3 cm, closer 100%. Spread ≤ 0.25 cm on 90% of demo frames;
on a home-pose frame (out of distribution) still 0.31 cm, so spread catches noise, not novelty.

3-D buckets (JEV-lite features, 7 per axis):

| axis | bucket acc | within one | direction when off-centre |
|---|---|---|---|
| radial | 72% | 98.9% | 98% |
| tangential | 76% | 98.2% | 74% |
| height | 70% | 99.1% | 100% |

One step: 5.4 → 1.7 cm (median xy, ≥ 3 cm away), closer 100%. Held-out bucket coverage: radial never
negative (no overshoot in demos), tangential almost never beyond ±3 cm, height never "up".

## 5. JEV-9B on the 3-D buckets (`jev_lora_3d.py`, 600 held-out questions)

| | radial | tangential | height |
|---|---|---|---|
| zero-shot bucket acc | 28% | 61% | 22% |
| zero-shot direction | 73% | 68% | 35% |
| step 3,000 / 9,186 bucket acc | 73% | 70% | 69% |
| step 3,000 direction | **99%** | 38% | **100%** |
| step 3,000 within one bucket | 99.5% | 98.4% | 100% |

Final numbers: `outputs/jev_lora_3d/train_report.json` when training ends.

## 6. Transfer

| test | cube-only JEV-lite | trained on cube + October |
|---|---|---|
| cube held-out, phase acc | 91% | 91% |
| October session (`rubiks_bowl_v5`, other wrist mount) held-out, phase acc | 70% | **87%** |
| October held-out, placed recall | 51% | **90%** |

Side camera only on October: 60% (worse). Cross-object: not measurable (see LESSONS.md, data).

## Withdrawn

- "JEV-lite 91.5% on new objects (`general_v1_eval`)": those parsed episodes were cube recordings under
  other instructions.
- "General model 97.6% on cube held-out": 15 held-out cube demos were in its training set as copies.
