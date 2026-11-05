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

## Tomorrow: live runbook

0. CAN up, arm homed, scene set. The pi0.5 server left running since 2026-11-03 serves the 20/20
   model. Stop it (`make policy-server-stop`) and start one on the weak checkpoint:
   ```bash
   make policy-server CKPT_HF=outputs/cube_grid_rel_lora32_weak_20261104-2139/checkpoints/001210/pretrained_model_merged
   ```
1. JEV, second terminal (~3 min to load; ready when `curl localhost:8000/v1/decide/info` answers):
   ```bash
   ADAPTER=$PWD/outputs/jev_lora_dir/adapter_vllm make jev-server
   ```
2. **Dry first (5 trials).** JEV judges and logs but never stops or retries. Compare each verdict
   with what you saw:
   ```bash
   make jev-live CKPT_HF=<same ckpt> INSTR="pick up the Rubik's cube and put it in the bowl" CONFIG=<your piper config> STEPS=1500 SPEED=1 JEV_FLAGS=--jev-dry
   ```
3. **A/B, 10 trials each, same cube/bowl positions:**
   - baseline: `make jev-live ... JEV=off`
   - verifier + retry: `make jev-live ... JEV_FLAGS="--recover home"`
   - verifier + JEV servo: `make jev-live ... JEV_FLAGS="--recover servo"` (hand on the e-stop the
     first time: it jogs J2/J3/J5 by 2° to measure the Jacobian, then steps J1 by ≤4° at a time)
4. Each run writes `real_robot_logs/<stamp>-jev_*/jev_report.json`: every JEV question, P(yes),
   latency, servo trace. Note the real outcome per trial in `real_robot_logs/jev_trials.csv`.

Knobs: `--grasp-thr 0.35`, `--place-thr 0.6`, `--attempts 3`, `--servo-lift 0.04`, `--grasp-z 0.151`
(servo-grasp descend height, median of the demos), `--empty-check`.

## Files

- `real_robot/jev.py`: client + verifier monitor. `real_robot/jev_guard.py`: CLI (all runner flags + JEV flags).
- `real_robot/jev_servo.py`: direction-bucket servo. `real_robot/jev_servo_sim.py`: closed-loop sim.
- `real_robot/jev_lora_train.py`: fine-tunes JEV's System 1 adapter (vLLM-compatible output).
- `real_robot/jev_frames.py`, `jev_judge_eval.py`, `jev_dirs.py`, `jev_dir_eval.py`, `jev_dir_fit.py`: datasets + evals.
- `real_robot/runner.py`: `run(observer=, stop_event=)`, `build_parser()`, `make_policy()`.
- `scripts/serve_jev.sh` (`make jev-server`): JEV-9B FP8 + adapter on the 3090 (~15 GB).
- `outputs/weak_20261104.sh`: the weak-checkpoint training job.
- Tests: `tests/test_real_robot_jev.py` (monitor phases, runner hooks, servo convergence with an oracle judge).

