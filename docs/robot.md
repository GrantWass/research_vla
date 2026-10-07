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
| `arm.py` | everyday arm commands: `status`, `snapshot`, `move`, `home`, `park` (see Commands) |
| `verify.py` | GPU and robot I/O checks: joints in limits, camera fps, snapshot PNGs, `--wiggle` (joint 6 +0.05 rad and back) |

## Commands

Run from `~/research_vla`. Steps that move an arm say so and ask for confirmation.
Arm 1 is the close arm, arm 2 the far arm. `ARMS` defaults to both.

| Command | What it does |
|---|---|
| `make status [ARMS=1]` | Read-only: CAN link, joints (deg), gripper, end pose, mode, errors, motors on/off, base angle vs home |
| `make snapshot` | Save every camera's current frame to `verify/snapshots/<time>/` |
| `make move ARM=1 TO=-93,40,-30,0,30,0 [GRIP=30]` | Ease one arm to joint angles in degrees (J1 base ... J6 wrist roll), at most 0.3 rad/s, then hold. `BY=0,10,0,0,0,0` moves relative. Refuses targets outside the limits |
| `make home [ARMS=1]` | Fold, then turn the base to face the table (`home_base` in `configs/piper.json`). Motors stay on |
| `make park [ARMS=1]` | `home`, then damped stop and power down |
| `make cans` | Are both CAN links up? |
| `make leader-restore` | Take the far arm out of teaching mode (if a teleop session died) |
| `make teleop [SECONDS=300]` | Far arm leads by hand, close arm follows |
| `make record DIR=demos/<name> INSTR="..." [EPISODES=10]` | Teleop and record a LeRobot dataset |
| `make policy [CKPT_HF=... INSTR="..." STEPS=300]` | Fast pi0.5 dry run on the close arm (flags below) |
| `make policy-live [...]` | The same, moving the arm (asks YES; hand on e-stop) |

The `make` targets wrap `python -m real_robot.arm` (`--help` lists its
options), `real_robot.teleop` and `real_robot.runner`. The policy and record
targets use the LeRobot venv (`LEROBOT_PY`).

```bash
# I/O check (read-only; --wiggle moves joint 6 by 3 deg and asks first)
.venv/bin/python -m real_robot.verify --stage robot --config real_robot/configs/piper_1.json --out verify/piper_1

# teleop: far arm (leader) goes limp, so hold its forearm; close arm follows
.venv/bin/python -m real_robot.teleop --leader can_piper2 --follower-config real_robot/configs/piper_1.json --seconds 300
.venv/bin/python -m real_robot.teleop --leader can_piper2 --restore-leader   # only take the leader out of teaching mode

# run a LeRobot policy: dry run first, then add --live (asks for YES; hand on e-stop)
~/lerobot_robot_piper/.venv/bin/python -m real_robot.runner --backend piper \
  --config real_robot/configs/piper_1_banana.json --lerobot richardshkim/pi05_piper_banana_v2_full \
  --end-pose-state --base-offset-deg 90 --instruction "Put banana into basket." \
  --max-steps 300 --execute-steps 4 --speed 1.0 \
  --policy-process --compile reduce-overhead --async --blend-steps 4

# or a websocket policy server
.venv/bin/python -m real_robot.runner --backend piper --config real_robot/configs/piper_1.json \
  --url ws://localhost:6000 --instruction "..." --live --speed 0.25 --execute-steps 10
```

Each run logs to `real_robot_logs/<time>-{dry,live}/`: `summary.json`
(status, inference latency, safety clips), `trajectory.npz` (`q_real`,
`target_real`, `cmd_real` per step) and `first_frame.npz`. After a dry run,
check that `safety.max_requested_delta` is small and that there was no abort.

Flags for the LeRobot policy command:

- `--base-offset-deg 90`: the close arm's base sits at about -93 deg to face
  our table, and the banana checkpoint was trained at J1 between -15 and +26
  deg. The policy sees J1 + 90 deg and an end pose rotated to match; without
  the offset it tries to swing the base back on its first step.
- `--policy-process`: runs the policy in a child process. In the robot
  process, piper_sdk's CAN reader thread competes with PyTorch for the GIL
  (eager inference 358 ms alone, 682 ms next to the arm).
- `--compile reduce-overhead`: torch.compile with CUDA graphs. Eager pi0.5
  launches about 13,700 kernels per call and is CPU-bound. Inference drops to
  about 110 ms on the RTX 3090, and the actions stay within 0.2 deg of eager.
  The first compile on a machine takes about 6 minutes. Kernels are cached in
  `~/.cache/torchinductor`, so later starts compile in about 40 s. The runner
  warms up on a live observation before the episode (`--warmup`, default 2).
  `max-autotune` was no faster (124 ms) and compiles for 10 minutes.
- `--async`: infers the next chunk while the current one executes and skips
  the actions that elapsed during inference, so the arm never pauses. A
  200-step dry run held 30 Hz (33.3 ms median step, 39.7 ms worst) and never
  ran out of actions.
- `--execute-steps 4 --blend-steps 4`: replan as soon as each inference
  lands (every ~5 steps, ~6 plans/s instead of 1) and blend each switch.
  Consecutive chunks disagree by ~5 deg at a switch (12 deg max, live runs);
  `--blend-steps N` carries that gap from the last action sent and shrinks
  it over N steps. A 300-step dry run: switches 0.6 deg median (1.5 max),
  30 Hz held (35 ms worst step), no starvation.
- `--speed` (up to 2) plays actions faster than the demos were recorded;
  `--speed-percent` raises the firmware speed cap (30) to keep up.
- `--denoise-steps 5` halves the action-expert work (237 vs 358 ms eager) and
  moves actions by 0.5 deg on average (2.4 deg max). Compiled, it's no faster
  (109 vs 111 ms): the compiled model is GPU-bound in the 2B prefix, 93 of
  96 ms. Further speed needs fewer prefix tokens, INT8, or a faster GPU.

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
- **Parking.** This is how we park, so the arm never slams down: `make park`
  folds with the base held (J2 0°, J3 0°, J5 36.8°), turns the base back to
  face the table (`home_base`), triggers the damped e-stop
  (`EmergencyStop(0x01)`), waits 3 s, disables the motors, then clears the
  e-stop (`0x02`). `make home` does the same fold and turn but leaves the
  motors on. Never disable the motors from any other pose.
- **Cameras.** The three color cameras share one USB 2.0 bus, so they stream
  MJPG (raw YUYV doesn't fit three at once). They start one after another:
  the third delivers its first frame ~2.7 s after opening, then all run at
  30 fps.
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

1. Record demos of a few tasks (`teleop --record`, above), then fine-tune
   (pi0.5 base, LAP-3B or GigaBrain-0.7). Recording has been tested against
   LeRobot with a mock robot, not yet on the arms.
