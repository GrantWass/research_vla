# JEV + pi0.5: verifier, retry, and trained visual servoing

pi0.5 does the moving. JEV ([autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B), vision, served
FP8 on the 3090 next to pi0.5) does two jobs:

1. **Verifier.** Calibrated yes/no on the camera frames at the moments a pick-and-place fails: after
   the grasp ("is the cube between the fingers?", rechecked every 2 s while carrying) and after the
   release ("is the cube in the bowl?"). A "yes in the bowl" ends the trial early.
2. **Corrector.** On a missed grasp, JEV steers the arm in joint space from the wrist camera:
   "where is the cube relative to the fingers: far left / slightly left / centred / slightly
   right / far right", plus the same for up/down. J1 steps toward it, reach (J2/J3/J5 via an
   end-pose Jacobian measured on the arm) steps near/far, and each step halves when the answer
   flips side. Then pi0.5 grasps from the centred hover (`--recover servo`), or the arm descends
   and closes itself (`--recover servo-grasp`). Plain retry from home is `--recover home`.

```
cameras + joints -> runner -> pi0.5 -> arm
      |
      +-> JevMonitor (thread) : grasp? -> no -> [servo: JEV buckets -> J1 / reach steps] -> pi0.5 again
                                carry? -> dropped -> fold home, retry
                                placed? -> yes: done / no: fold home, retry      (up to --attempts)
```

## What was tried overnight (2026-11-04/05) and what stuck

| Step | Result |
|---|---|
| JEV picks among motor commands | Not tried: JEV's own arm demo gets 0/10 that way. |
| Zero-shot verifier (196 labelled demo frames) | holding 99% / in-bowl 96% accuracy, AUROC 1.00 both, ~170 ms |
| Zero-shot direction buckets (shifted wrist frames, exact labels) | left/right 66% (biased: a centred cube reads "slightly left"), up/down 69% |
| + bias calibration | 84% / 81% |
| Zero-shot buckets → real demo joint deltas (ridge fit, held-out episodes) | J1 sign 84% (abs(dJ1) > 3°), reach sign 98% (abs(dr) > 3 cm) |
| Closed-loop servo sim, zero-shot JEV | 2/20 → 5/20 (calibration, rescaled sim); mirror-averaging made it worse (3/20) |
| Binary questions (as in JEV's demo) instead of buckets | same signal (ρ 0.80 vs 0.76): not the bottleneck |
| **Fine-tune JEV's own System 1 LoRA on our wrist frames** (QLoRA, 1,224 ex, 25 min) | held-out episodes: direction 57% → **96%** (left/right 93%, up/down 100%), bucket 20% → 71% |
| Closed-loop servo sim, fine-tuned JEV | **14/20**, median final error **0.45° J1 / 0.76 cm reach** (from ~6° / 3 cm) |
| Verifier after fine-tune | unchanged: 99% / 96%, AUROC 1.00, so one adapter serves both jobs |
| Second fine-tune (2× data, frames 12/25/40 before grasp; `outputs/jev_lora_dir2`) | held-out direction 99%, but closed-loop sim only 9/20 (stops early at 2–3° J1). Run 1 stays the default |

The servo sim (`real_robot/jev_servo_sim.py`) is a fake arm whose wrist view is a real near-grasp
frame shifted in proportion to its joint error, with random unknown signs. It tests JEV
perception + sign discovery + step halving together. It is *not* real kinematics: reach is a
pure vertical image shift (no change in cube size), so live reach behaviour may be better or worse.
All 6 sim misses are on reach (JEV reading a 2–4 cm offset as "centred").

## Dumbed-down pi0.5 (the 20/20 model is too good to show anything)

Same recipe as the 20/20 run (`cube_grid_rel_lora32_2ep_20261031-1538`, LoRA 32, relative actions,
same 6 held-out demos), stopped early. Offline joint error on the held-out demos (deg; lower is better):

| checkpoint | epochs | joints | J1 (where the gripper lands) |
|---|---|---|---|
| hold still (no model) | – | 2.54 | 0.71 |
| `cube_grid_rel_lora32_weak_20261104-2139/checkpoints/000400` | 0.05 | 2.59 | 2.64 |
| `.../000800` | 0.09 | 2.13 | 2.03 |
| `.../001210` | 0.14 | 2.22 | 1.98 |
| `cube_grid_rel_lora32_2ep_20261031-1538/checkpoints/004033` | 0.5 | 1.98 | 1.73 |
| `.../008066` | 1.0 | 1.98 | 1.30 |
| `.../016131` (went 20/20 live) | 2.0 | 1.84 | 1.01 |

Try `weak .../001210/pretrained_model_merged` first: J1 error is twice the 20/20 model's, which
should give missed grasps, exactly what the servo is for. If it still succeeds a lot, use `000800`;
`000400` is about as good as holding still.

## Live, 2026-11-05: what happened

- Cube checkpoints need `CONFIG=real_robot/configs/piper_1_banana.json` (30 Hz, gripper 0..1). With
  `piper_1.json` they look at the ceiling; preflight now refuses that pairing.
- `weak/001210` is about as good as holding still: useless live. `2ep/004033` lines up over the cube and
  then **hovers without closing**. That is the failure to fix, not missed grasps. It also asks for
  >0.5 rad wrist jumps; `real_robot/configs/piper_1_clamp.json` rate-limits them instead of aborting.
- The JEV-9B direction servo failed live: left/right never converged (bucket stuck at +0.62 while J1
  moved), and the up/down search drove the arm into the table, the controller faulted and the arm
  dropped. Servo moves are now bounded (z floor `--grasp-z - --z-margin`, J1 ±20°, reach ±8 cm) and
  stop on teach mode or motors off.
- `--recover descend` (straight down to `--grasp-z` in closed loop on the measured end pose, close,
  lift, check) grasped on its first live try; pi0.5 then carried the cube over the bowl.
- JEV-9B then scored the cube **held over the bowl, never released** as placed (P 0.82).

## JEV-lite (2026-11-10): a 25M-parameter judge that runs every step

`real_robot/jevlite.py`. Frozen DINOv2-small encodes the wrist and base frames (CLS, mean patch, and a
4×4 grid of patch features so position survives), and a small MLP predicts the task phase:
idle / approach / grasp / holding / release / placed. Labels come free from the demos' gripper traces
(`real_robot/jevlite_data.py`, 92 of 100 episodes, 20,386 frame pairs); the last third of every
carry is labelled "holding", never "placed". Trained with two augmented copies (colour jitter, crop,
shift); held out by episode (every 5th).

Same held-out frames, full resolution (`real_robot/jevlite_compare.py`, 60 frames per case):

| | JEV-lite | JEV-9B (direction LoRA) |
|---|---|---|
| held over the bowl, said "placed" | **0 / 60** | 35 / 60 |
| placed: AUROC / acc@0.5 | **0.991 / 98.3%** | 0.959 / 81.7% |
| holding: AUROC / acc@0.5 | **0.993 / 96.7%** | 0.854 / 74.6% |
| latency | **26 ms** for both questions (15 ms on GPU-resident frames) | 168 ms per question |

Over all held-out frames: phase accuracy 91%, holding AUROC 0.995. Placed recall is 93–95% 1–4 s after
release (when the place check asks) and drops only once the arm is back home. Augmentation keeps
clean accuracy and adds ~2 points under simulated lighting/framing drift. On 74 first frames from
real live runs (never seen): 67 idle, plus 2 correct "holding" (the attempt started with the cube in
the gripper); the misses are frames with the cube at the image edge.

Caveat: JEV-lite knows this rig and this task only; JEV-9B is general. Use `--judge both` to require
both to agree before calling a placement.

## Grasp-offset model and `--recover align` (2026-11-10)

`real_robot/jevlite_offset.py`. For every approach frame of a demo, the end-effector position when the
gripper starts closing is known, so the offset to it is an exact, real label (no synthetic image
shifts), expressed in the arm's frame: radial (reach), tangential (what J1 changes), dz. An MLP on the
same JEV-lite features regresses it (frames within 15 cm of the grasp; 3,062 train / 846 held out).

| held-out, by distance to the grasp | correction brings the gripper closer | xy error after one correction |
|---|---|---|
| 3–6 cm (median 2.5 cm) | 93% | 1.1 cm |
| 6–15 cm (median 6.1 cm) | 99% | 1.3 cm |
| < 3 cm | — (already aligned; deadband) | 0.6 cm |

MAE radial 0.75 / tangential 0.57 / height 0.85 cm (predicting the mean: 2.07 / 0.68 / 2.41). Through the
runtime path on full-resolution held-out frames: 4.1 cm → 1.3 cm median, closer in 94%.

`--recover align` (on `hover` or `missed_grasp`): average 3 predictions; if the sideways offset is over
`--align-deadband` (1.5 cm), shift there (≤ `--align-max` 6 cm, J1 included, closed loop on the end pose,
two rounds); then descend to the predicted grasp height (clamped to `--grasp-z` −1/+3 cm), close, lift,
check. All moves go through the servo's workspace guard. Fake-arm tests cover the shift and the
oracle case; **not yet run live**. Tangential is the weak axis: demos approach nearly on target sideways,
so there is little sideways variety to learn from; real hover logs (`judged/` frames) are the fix.

## Monitor rules (no model needed)

- **Hover**: gripper open and J1–J6 slower than 0.02 rad/s for 3 s (`--hover-s`, 0 = off) ends the
  attempt with verdict `hover`, which triggers `--recover`. Demos never stay open-and-still longer than
  0.93 s; replaying all 24 live runs from 2026-11-05 flags all 7 hovers at 21–26 s (they used to run
  to the 40–100 s step limit) and none of the 12 runs that closed.
- **Never released** at the end of the step budget is `not_released`, never `success`, whatever a
  model says. Never closed is `hover`/`timeout`.
- Every judgement saves its frames to `<log>/attemptN/judged/` with question, P and gripper phase:
  the next training set (real failures).

## Live runbook

1. Servers: `make policy-server CKPT_HF=outputs/cube_grid_rel_lora32_2ep_20261031-1538/checkpoints/004033/pretrained_model_merged`;
   JEV-9B only for `--judge jev|both` or the direction servo: `ADAPTER=$PWD/outputs/jev_lora_dir/adapter_vllm make jev-server`.
2. Home the arm before every trial (`python -m real_robot.arm home --arms 1`).
3. A/B, 10 trials each, same cube/bowl positions:
   ```bash
   make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 JEV=off
   make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 ATTEMPTS=2 JEV_FLAGS="--recover descend"
   make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 ATTEMPTS=2 JEV_FLAGS="--recover align"
   ```
   (`--judge lite` is the default; add `--judge both` to cross-check with JEV-9B.)

Knobs: `--grasp-thr 0.35`, `--place-thr 0.6`, `--hover-s 3`, `--grasp-z 0.151` (median grasp height of
the demos), `--z-margin 0.015`, `--servo-lift 0.04`, `--lite-model outputs/jevlite/model_dinos_g4_aug`.

Rebuild JEV-lite:
```bash
python -m real_robot.jevlite_data --dataset demos/cube_grid_all_pi05 --out outputs/jevlite/frames
python -m real_robot.jevlite features --frames outputs/jevlite/frames --out outputs/jevlite/feats_dinos_g4.npz --grid 4
for s in 1 2; do python -m real_robot.jevlite features --frames outputs/jevlite/frames --out outputs/jevlite/feats_dinos_g4_aug$s.npz --grid 4 --aug-seed $s; done
python -m real_robot.jevlite train --feats outputs/jevlite/feats_dinos_g4.npz --out outputs/jevlite/model_dinos_g4_aug --grid 4 --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz
python -m real_robot.jevlite_offset train --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames --feats outputs/jevlite/feats_dinos_g4.npz --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz --out outputs/jevlite/offset_dinos_g4
python -m real_robot.jevlite_compare --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames --lite outputs/jevlite/model_dinos_g4_aug --out outputs/jevlite/compare.json
```
(lerobot venv; ~6 min for frames, ~1.5 min per feature pass, seconds to train.)

## Files

- `real_robot/jev.py`: client + verifier monitor. `real_robot/jev_guard.py`: CLI (all runner flags + JEV flags).
- `real_robot/jev_servo.py`: direction-bucket servo. `real_robot/jev_servo_sim.py`: closed-loop sim.
- `real_robot/jev_lora_train.py`: fine-tunes JEV's System 1 adapter (vLLM-compatible output).
- `real_robot/jev_frames.py`, `jev_judge_eval.py`, `jev_dirs.py`, `jev_dir_eval.py`, `jev_dir_fit.py`: datasets + evals.
- `real_robot/runner.py`: `run(observer=, stop_event=)`, `build_parser()`, `make_policy()`.
- `scripts/serve_jev.sh` (`make jev-server`): JEV-9B FP8 + adapter on the 3090 (~15 GB).
- `outputs/weak_20261104.sh`: the weak-checkpoint training job.
- `real_robot/jevlite.py` (encoder, train, `JevLite`, `LiteJudge`), `jevlite_data.py` (phase labels from gripper traces), `jevlite_compare.py` (vs JEV-9B), `jevlite_offset.py` (grasp-offset regressor, `GraspOffset`).
- Tests: `tests/test_real_robot_jev.py` (monitor phases, runner hooks, servo convergence with an oracle judge).

