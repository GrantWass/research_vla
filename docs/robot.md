# Real robot: 2x AgileX Piper

Two independent single-arm Pipers (6 joints + gripper each) on one
workstation. One process drives one arm. A policy runs either in-process
(a LeRobot checkpoint) or on a websocket policy server (the same XPolicyLab
server used in sim).

```
cameras + joints ──> real_robot.runner ──> policy (LeRobot in-process, or ws server)
       ^                                         │ action chunk (T x 7)
       └── motors <── SafetyFilter <── units ────┘
```

## The rig

| | |
|---|---|
| Workstation | Alienware Area-51 R2, Ubuntu 22.04, RTX 3090 24 GB, driver 580; `ssh csce990@100.90.226.52` (Tailscale) |
| Arms | firmware S-V1.8-3. **Close** arm (from the operator seat) = `can_piper1` = `configs/piper_1.json`; **far** arm = `can_piper2` = `configs/piper_2.json` |
| CAN | one USB-CAN adapter per arm, named by adapter serial and brought up at 1 Mbit/s on boot by `real_robot/can_persist.sh` (installed; it also handles replugs) |
| Cameras | three Orbbec Astra Pro Plus, 640x480 at 30 fps over plain UVC. All report serial `SN0001`, so configs address them **by USB port** (`/dev/v4l/by-path/...-usb-0:<port>:1.0-video-index0`). Keep each in its port: **5.2** = stationary high view (`cam_high`, shared by both arms), **9.2** = on the close arm, **1.2** = on the far arm |
| Python | `~/research_vla/.venv` (`make robot-venv`, `requirements/robot.txt`): `piper_sdk`, OpenCV, `pyorbbecsdk2`, for the repo code and tests. `~/lerobot_robot_piper/.venv` (`requirements/lerobot.txt`, an exact freeze: LeRobot 0.4.4, torch 2.10, `transformers` pinned to a git commit for pi0.5): for `--lerobot` policies. `~/openpi`: upstream openpi (`uv`) |

Configs: `configs/piper.json` holds what both arms share (limits, rate
limits, gripper stroke, speed), and each arm file `extends` it with its CAN
name and cameras. `piper_1_banana.json` extends `piper_1.json` for one
checkpoint's conventions.

## Code (`real_robot/`)

| File | Role |
|---|---|
| `backends.py` | `RobotBackend` interface (state, images, command, hold) and `MockBackend` for tests |
| `piper.py` | `PiperBackend` (follower, piper_sdk V2) and `PiperLeader` (teaching-mode arm) |
| `cameras.py` | threaded OpenCV reader (always the newest frame), Orbbec SDK reader (color + depth), and a per-camera router |
| `safety.py` | `SafetyFilter`: NaN and limit checks, per-step rate limit, abort on any jump > `abort_delta` |
| `mapping.py` | robot units <-> policy units (per-joint sign/offset, linear gripper map) |
| `runner.py` | the control loop. **Dry-run by default** (policy runs, nothing is sent); `--live` moves; always `hold()`s on exit |
| `lerobot_policy.py` | wraps a LeRobot checkpoint in the runner's policy interface |
| `teleop.py` | far arm leads by hand, close arm follows through `SafetyFilter` |
| `recorder.py` | records teleop demos into a LeRobot dataset (`teleop --record`) |
| `park.py` | fold an arm and power it down gently (`make park ARM=1`) |
| `verify.py` | GPU and robot I/O checks: joints in limits, camera fps, snapshot PNGs, `--wiggle` (joint 6 +0.05 rad and back) |

## Commands

Run from `~/research_vla`. Steps that move an arm say so and ask for confirmation.

```bash
# I/O check (read-only; --wiggle moves joint 6 by 3 deg and asks first)
.venv/bin/python -m real_robot.verify --stage robot --config real_robot/configs/piper_1.json --out verify/piper_1

# teleop: far arm (leader) goes limp, so hold its forearm; close arm follows
.venv/bin/python -m real_robot.teleop --leader can_piper2 --follower-config real_robot/configs/piper_1.json --seconds 300
.venv/bin/python -m real_robot.teleop --leader can_piper2 --restore-leader   # only take the leader out of teaching mode

# run a LeRobot policy: dry run first, then add --live (asks for YES; hand on e-stop)
~/lerobot_robot_piper/.venv/bin/python -m real_robot.runner --backend piper \
  --config real_robot/configs/piper_1_banana.json --lerobot richardshkim/pi05_piper_banana_v2_full \
  --end-pose-state --instruction "Put banana into basket." --max-steps 300 --execute-steps 30 --speed 0.5

# or a websocket policy server
.venv/bin/python -m real_robot.runner --backend piper --config real_robot/configs/piper_1.json \
  --url ws://localhost:6000 --instruction "..." --live --speed 0.25 --execute-steps 10
```

Each run logs to `real_robot_logs/<time>-{dry,live}/`: `summary.json`
(status, inference latency, safety clips), `trajectory.npz` (`q_real`,
`target_real`, `cmd_real` per step) and `first_frame.npz`. After a dry run,
check that `safety.max_requested_delta` is small and that there was no abort.

### Recording demos

Teleop with `--record` writes a new LeRobot dataset (v3, 30 fps). Run it with
the LeRobot venv:

```bash
~/lerobot_robot_piper/.venv/bin/python -m real_robot.teleop --record demos/pick_cube_01 \
  --task "put the cube in the cup" --episodes 10 --episode-seconds 30 --reset-seconds 10
```

Each frame holds `observation.state` (follower joints in rad and gripper in m,
7-D), `observation.end_pose` (6-D), `action` (the leader's target clipped to
the limits, 7-D) and one video per config camera. The session alternates
"reset the scene" pauses with recording windows, and the log says which one
you're in. Each episode is saved in the background during the next pause.
Ctrl-C discards the episode in progress and keeps the saved ones. Use a new
directory per session; LeRobot's dataset tools can merge sessions.

## Hardware behavior to know

- **Joint limits vs rest pose.** The firmware accepts only J2 >= 0 and J3 <= 0,
  but the arms rest about 2° past both. Commanding the rest pose lifts the arm
  2°. `SafetyFilter.check_state` has a 0.05 rad margin, so starting at rest is
  allowed.
- **Disabling drops the arm.** At the lowest pose the firmware accepts, each
  arm sits about 1 inch above its base. Disabling the motors there drops it
  that inch, and the wrist falls to about 37°. `make park ARM=1|2` is meant to
  avoid that: it folds with J5 already at 37°, triggers the damped e-stop
  (`EmergencyStop(0x01)`), waits, disables, then clears the e-stop (`0x02`).
  **It hasn't run on the arms yet.** Run it the first time with someone watching.
- **Teaching mode** (`MasterSlaveConfig 0xFA`) applies immediately, with no
  power cycle. The arm goes limp and the wrist sags about 30°. It broadcasts
  joint targets (0x155-0x157, gripper 0x159) **only while being moved**;
  otherwise it reads as zeros. Teleop waits for a complete, steady broadcast,
  refuses to start if the arms differ by more than 0.8 rad, and restores
  normal mode (0xFC) on exit, Ctrl-C or SIGTERM.
- **Speed.** The configs cap firmware speed at 30%, which can't keep up with a
  hand, so teleop uses 60%.
- **Orbbec depth.** The depth sensors once enumerated in bootloader mode
  (PID 0x0400); a replug fixed it. The udev rules are installed. Depth through
  the SDK hasn't been tested on these cameras yet.
- The WeGo `lerobot_robot_piper` GUIs (`piper-setup`, `piper-teleop`) rename
  the CAN ports and assume their own layout. Don't use them here. Its venv is
  only used for LeRobot.

## Policies for the Piper

There is no public policy that drives a Piper well out of the box (survey as of 2026-10-06):

| Model | Fit | Use |
|---|---|---|
| `richardshkim/pi05_piper_banana_v2_full` (LeRobot pi0.5) | one Piper, 20 demos of "Put banana into basket.", base + wrist camera, 13-D state (joints, gripper 0..1, end pose) | runs here today; one task. Scene: banana ~30 cm in front of the close gripper, basket front-left (`img/banana_scene_compare.jpg`) |
| LAP-3B (`lihzha/LAP-3B`, openpi) | end-effector poses from side + wrist cameras; >50% zero-shot on unseen arms | best zero-shot bet; needs an end-effector-target client and a workspace box |
| GigaBrain-0.7 | pretrained on AgileX (Piper-based) data, ships a Piper fine-tune config | best base for fine-tuning |
| pi0.5 base, X-VLA, SmolVLA, GR00T | no Piper checkpoint | need a fine-tune |
| DreamZero (14B), RDT2 | too big for 24 GB / needs its own hardware | no |

The path to a useful policy: record teleop demos of a few tasks, then
fine-tune pi0.5 base, LAP-3B or GigaBrain-0.7.

## Next steps

1. First run of `make park` (above), with someone watching.
2. Banana test on the close arm (needs a banana and a basket).
3. Record demos of a few tasks (`teleop --record`, above), then fine-tune
   (pi0.5 base, LAP-3B or GigaBrain-0.7). Recording has been tested against
   LeRobot with a mock robot, not yet on the arms.
