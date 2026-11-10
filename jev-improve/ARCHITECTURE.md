# Architecture

```
            ┌──────────────── pi0.5 policy server (drives every motion) ───────────────┐
cameras ──► │ runner.run(observer=JevMonitor, stop_event=...)                          │
(wrist,     └──────────────────────────────────────────────────────────────────────────┘
 side)            │ every step: images + joints + gripper width
                  ▼
           JevMonitor (real_robot/jev.py)
             gripper events: opened → closed+stalled → released
             rules: hover (open + still 3 s), never-released ≠ success
             judge.yes("holding?") 1.2 s after close, rechecks in transit
             judge.yes("placed?")  1.0 s after release
             saves judged frames → attemptN/judged/
                  │ verdict: success | missed_grasp | dropped | missed_place | hover | not_released | timeout
                  ▼
           jev_guard (real_robot/jev_guard.py): attempts loop
             missed_grasp / hover → --recover
               home     open, fold home, retry
               descend  straight down to --grasp-z, close, lift, judge
               align    grasp-offset model: shift (J1 incl.), descend to predicted height, close, judge
               servo / servo-grasp  JEV-9B direction buckets (failed live; kept, guarded)
             grasped → next attempt starts holding (pi0.5 carries and places)
```

## Judges (`--judge`)

| judge | module | what | latency |
|---|---|---|---|
| `lite` (default) | `real_robot/jevlite.py` `LiteJudge` | JEV-lite phase probabilities: holding = P(holding), placed = P(placed)+P(release) | ~15 ms on GPU frames, ~26 ms from camera frames |
| `jev` | `real_robot/jev.py` `JevClient` | JEV-9B System 1 yes/no over both cameras, via the vLLM server | ~170 ms per question |
| `both` | `LiteJudge(mode="both")` | lite for holding; placed needs lite AND JEV-9B (min) | ~200 ms |

## JEV-lite (`real_robot/jevlite.py`)

- Encoder: frozen `facebook/dinov2-small` (fp16), per camera: CLS + mean patch + 4×4 grid of patch
  features (6,912-d). Both cameras concatenated.
- Phase head: LayerNorm → Linear 256 → GELU → Linear 6, phases idle / approach / grasp / holding /
  release / placed. Class-weighted CE, label smoothing, feature noise; trained on clean + 2 augmented
  copies (colour jitter, 85-100% crop, shift).
- Labels (`real_robot/jevlite_data.py`): from each demo's gripper trace (open, close start, stall,
  release, relative thresholds); frames within 3 of a boundary dropped; last third of the carry is
  "holding" (late_carry), never "placed".
- Split: `test_mask` = every 5th recording, copies of a recording grouped by signature (first state,
  last frame, frame count).
- Models: `outputs/jevlite/model_dinos_g4_aug` (default), `model_2sess_g4` (cube + October session).

## Grasp offset (`real_robot/jevlite_offset.py`)

- Label: offset from the gripper to the end-effector position where the demo started closing, in the
  arm's frame: radial (reach), tangential (sideways, J1), dz. Frames within 15 cm.
- Regression: MLP → 3 (cm), smooth-L1. Ensemble of 5 (`offset_ens5`, default): mean, and spread
  (std of members' xy) gates shifting.
- Buckets (`offset_buckets`): per axis 7-way (edges ±1, ±3, ±6 cm); the step is Σ p·(bucket's
  training-median offset). Answers logged with probabilities.
- Runtime `GraspOffset(model_dir)(images, xyz)` → base-frame delta; `.answer` = bucket answers.

## `--recover align` (`jev_guard.align`)

1. Average 3 predictions; if ensemble spread > `--align-max-spread` (1 cm) do not shift.
2. If xy offset > `--align-deadband` (1.5 cm): `JevServo.shift_xy` (≤ `--align-max` 6 cm, ≤ 1 cm per
   move, J1/J2/J3/J5 via a measured Jacobian, closed loop on the end pose); repeat once.
3. Descend (`JevServo.descend`, closed loop, stops when blocked) to the predicted grasp height,
   clamped to `--grasp-z` −1/+3 cm; close; lift `--servo-lift`; judge holding.

## Safety (`real_robot/jev_servo.py`)

Every servo move: before and after, `control_problems()` (teach mode) and `motors_on()`; after,
end-effector z ≥ `--grasp-z − --z-margin`, J1 within 20° and reach within 8 cm of where recovery
started; a violation undoes the move and raises `ServoStop` (arm held). Policy jumps over
`abort_delta` abort, or are rate-limited with `clamp_jumps` (`configs/piper_1_clamp.json`).

## Data flywheel

- Live: `JevMonitor(frame_dir=...)` saves the frames of every judgement →
  `jevlite_live template` (CSV per attempt) → you fill grasped/placed/released → `build` →
  `jevlite train --live` (oversampled ×5).
- Arm collection: `jevlite_collect` visits radial/tangential ±8 cm at 3/6/9 cm above a good grasp
  pose (75 poses), saves frames with exact offsets → `jevlite_offset train --collected`.

## JEV-9B adapters

- `outputs/jev_lora_dir/adapter_vllm`: direction buckets on shifted wrist crops (2-D, image space).
- `outputs/jev_lora_3d/`: 3-D grasp buckets (reach, sideways, height) from demo poses, both cameras
  (`real_robot/jev_lora_3d.py`, QLoRA, continues the adapter above).

## Files

| file | role |
|---|---|
| `real_robot/jev.py` | `JevClient`, `JevMonitor` (events, rules, hover, frame saving) |
| `real_robot/jev_guard.py` | CLI: runner flags + JEV flags, attempts, recoveries, `align` |
| `real_robot/jev_servo.py` | `JevServo`: Jacobian, guarded moves, `shift_xy`, `descend`, bucket servo |
| `real_robot/jevlite.py` | encoder, phase head training/eval, `JevLite`, `LiteJudge`, `test_mask` |
| `real_robot/jevlite_data.py` | phase-labelled frames from a LeRobot dataset |
| `real_robot/jevlite_compare.py` | JEV-lite vs JEV-9B on the same frames |
| `real_robot/jevlite_offset.py` | grasp-offset regression / ensemble / buckets, `GraspOffset` |
| `real_robot/jevlite_live.py` | live judged frames → labelled training features |
| `real_robot/jevlite_collect.py` | arm grid collection for the offset model |
| `real_robot/jev_lora_train.py`, `jev_lora_3d.py` | JEV-9B System 1 LoRA fine-tunes |
| `real_robot/configs/piper_1_clamp.json` | cube config with jump clamping |
| `tests/test_real_robot_jev.py` | monitor, hover, rules, judge routing, servo/align/collect on a fake arm |
