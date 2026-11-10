# Lessons

## On the robot

- **Config must match the checkpoint.** Cube checkpoints were converted with `piper_1_banana.json`
  (30 Hz, gripper 0..1). Running them with `piper_1.json` (25 Hz, metres) produced "looking at the
  ceiling" and wrist jumps that looked like a bad model. A preflight check now refuses the pairing.
- **Offline error does not rank checkpoints for live use.** `weak/001210` had twice the 20/20 model's J1
  error offline (the intended "weak"), but its joint error equalled holding still; live it did nothing.
  Check against the hold-still baseline before choosing.
- **A weak policy fails in its own way.** We expected missed grasps; `004033` lined up and hovered.
  Watch dry runs before building recovery for an imagined failure.
- **Servo searches need hard workspace limits.** The direction servo had joint limits only. A stale
  Jacobian plus an expanding sign search drove the arm into the table; the controller faulted into teach
  mode and the arm dropped. Bound z, J1 and reach from where recovery started, check control state each
  move, undo and stop on violation.
- **Units leak.** `GRIPPER_OPEN_M = 0.07` "opened" the 0..1 gripper to 7%. Read units from the config.
- **Rules beat models where the robot knows the answer.** "Placed" requires the gripper to have opened;
  "hover" is a motion fact. Models judge only what the robot cannot sense.
- **Simple recovery first.** Straight-down descend worked on the first try where the clever servo failed.

## On models

- **Small task-specific beats big general for verification here.** A frozen DINOv2-small + MLP trained on
  free gripper-trace labels beat JEV-9B on every check and runs every step. JEV-9B's place check
  confused "above the bowl" with "in the bowl" because nothing told it the difference.
- **Offline accuracy ≠ closed-loop.** The JEV direction adapter run 2 was 99% offline but 9/20 in the
  servo sim (run 1: 96% / 14/20); live, LR never converged. Select by closed-loop behaviour.
- **Real labels beat synthetic shifts.** Grasp offsets from demo end poses are exact and in the arm's
  frame; shifted image crops were neither, and didn't transfer.
- **Coverage limits what any model can learn.** Demos never overshoot or sidestep, so those offset
  buckets are empty; no architecture fixes that. Collect the data (`jevlite_collect`).
- **Ensemble spread catches noise, not novelty.** Members agreed on a home-pose frame unlike any training
  frame. Keep physical caps as the real safety.

## On data

- **Datasets can contain copies.** `general_v1` repeats each cube recording ~4× under paraphrased or
  decoy instructions; splitting by episode index leaked 15 held-out demos and produced a 97.6% that
  wasn't real. Group recordings by content (first state + length) before splitting.
- **Check what the data is before claiming transfer.** The "new objects" were 20-frame stubs; the long
  episodes with other object names were cube videos. The 91.5% "new object" result was withdrawn.
- **A changed camera mount is a real distribution shift.** 91% → 70% on the October session; ~50 demos
  from it brought it back to 87%.

## On working on a shared machine

- Another user's policy server and uncommitted edits were on the robot PC. Never stop their processes;
  commit only your own hunks (stage a constructed file, not `git add -p` blindly); verify the commit in a
  clean worktree.
- `pkill -f pattern` matches your own SSH command line; use bracket patterns (`[j]ev_guard`).
- Tests that monkeypatch `time.sleep` must restore it, or they break every later timing test.
