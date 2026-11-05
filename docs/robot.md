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
| Cameras | stationary high view (`cam_high`, shared by both arms) and far-arm wrist: Orbbec Astra Pro Plus, 640x480 at 30 fps over plain UVC, all reporting serial `SN0001`, so configs address cameras **by USB port** (`/dev/v4l/by-path/...-usb-0:<port>:<interface>-video-index0`): **5.2** = `cam_high`, **1.2** = on the far arm. Close-arm wrist (`cam_wrist`): **Intel RealSense D455** since 2026-10-13, color stream on USB interface 1.3 (YUYV only, no MJPG; depth not used yet); its path follows the port it's in (now **6**, still USB 2). Before that, a Sonix USB 2.0 camera on **9.2** |
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
| `make status [ARMS=1]` | Read-only: CAN link, joints (deg), gripper, end pose, mode, errors, motors on/off, motor and driver temperatures (C) and faults, base angle vs home |
| `make snapshot` | Save every camera's current frame to `verify/snapshots/<time>/` |
| `make move ARM=1 TO=-93,40,-30,0,30,0 [GRIP=30]` | Ease one arm to joint angles in degrees (J1 base ... J6 wrist roll), at most 0.3 rad/s, then hold. `BY=0,10,0,0,0,0` moves relative. Refuses targets outside the limits |
| `make home [ARMS=1]` | Fold, then turn the base to face the table (`home_base` in `configs/piper.json`). Motors stay on |
| `make park [ARMS=1]` | `home`, then damped stop and power down |
| `make cans` | Are both CAN links up? |
| `make leader-restore` | Take the far arm out of teaching mode (if a teleop session died) |
| `make teleop [SECONDS=300]` | Far arm leads by hand, close arm follows |
| `make record DIR=demos/<name> INSTR="..." [EPISODES=10]` | Teleop and record a LeRobot dataset |
| `make quest-teleop [ARM=1\|2]` | Meta Quest controller drives one arm (see "Meta Quest teleop") |
| `make policy-server` | Load and compile the pi0.5 policy once (~95 s) and keep it on the GPU; leave it running in its own terminal. `make policy-server-stop` ends it |
| `make policy [CKPT_HF=... INSTR="..." STEPS=300]` | pi0.5 dry run on the close arm (flags below). Uses the policy server if it's running (starts in ~3 s), else loads the model itself (~95 s) |
| `make policy-live [...]` | The same, moving the arm (asks YES; hand on e-stop) |
| `make policy-live PARK=1 [...]` | Press `d` when the task is done: the policy stops and the arm parks like `make park` (also parks when `STEPS` run out, never after an abort or Ctrl-C). Same flag on `make jev-live`, which also parks when JEV sees the task done |

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
  --max-steps 300 --execute-steps 8 --speed 0.5 \
  --policy-process --compile reduce-overhead --async --blend-steps 4 --noise-seed 0 \
  --smooth-hz 2

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
- `--policy-server`: connect to `make policy-server` instead of loading the
  model. Loading takes ~60 s and re-tracing the compiled model ~36 s in every
  new process, even with the kernel cache warm; the server pays that once.
  The runner refuses a server started with a different checkpoint, base
  offset, noise seed or camera map.
- `--execute-steps 4 --blend-steps 4`: replan as soon as each inference
  lands (every ~5 steps, ~6 plans/s instead of 1) and blend each switch.
  Consecutive chunks disagree by ~5 deg at a switch (12 deg max, live runs);
  `--blend-steps N` carries that gap from the last action sent and shrinks
  it over N steps. A 300-step dry run: switches 0.6 deg median (1.5 max),
  30 Hz held (35 ms worst step), no starvation.
- `--noise-seed 0`: the same sampling noise for every plan, so consecutive
  plans differ only where the observation does. With fresh noise, replanning
  ~6x/s made the arm wobble in small circles (live run); with a fixed seed the
  commanded path's wobble fell from 0.66 to 0.12 deg RMS (dry runs, 300 steps).
  Replanning less often (every 15-25 steps) was worse (0.55-0.61 deg): each
  chunk carries its own wiggle. `--ensemble-steps N` (average the plans of the
  last N steps) is available but didn't help on top of the fixed seed.
- `--smooth-hz 2`: a background thread sends commands at 100 Hz
  (`--stream-hz`), each following the latest target through a critically
  damped filter (`real_robot/smoothing.py`). Sent one per step (15 Hz), the
  firmware sped up and slowed down for every target and each new plan added
  a ~3 Hz pulse: the arm moved fast, stopped, moved fast. Adds ~0.16 s lag.
  Keep 2 Hz: 1.5 Hz (40% less jerk offline) made the cube model fail live
  (2026-11-05): the extra lag costs accuracy. The slight shake while moving
  came from the replans instead: each new chunk is anchored on the lagging
  measured joints and starts ~1 step behind the command, so blending from
  the last action paused the target at every switch (~6 Hz).
  `--blend-from plan` (`BLEND_FROM=plan`) starts the blend where the arm was
  heading instead; after `--blend-steps` it follows the new chunk as before.
- The runner refuses `--live` when the config's rate or gripper units differ
  from the dataset a local checkpoint was trained on (cube models need
  `piper_1_banana.json`).
- Start with the gripper closed (`make home` closes it). From an open
  gripper the banana checkpoint pushed the shoulder down and curled up.
- `--speed 0.5`: half the trained rate. At 1.0, the per-step safety limits
  clipped most steps on J2/J3 and the motion looked jolty.
- `--speed` (up to 2) plays actions faster than the demos were recorded;
  `--speed-percent` raises the firmware speed cap (30) to keep up.
- `--denoise-steps 5` halves the action-expert work (237 vs 358 ms eager) and
  moves actions by 0.5 deg on average (2.4 deg max). Compiled, it's no faster
  (109 vs 111 ms): the compiled model is GPU-bound in the 2B prefix, 93 of
  96 ms. Further speed needs fewer prefix tokens, INT8, or a faster GPU.

### Recording demos

Run it **in your own terminal** (it reads single keys). The far arm leads,
the close arm follows:

```bash
make record-check                                  # optional pre-flight: CAN, arms, cameras, dataset; nothing moves
make record DIR=demos/pick_cube_01 INSTR="put the cube in the cup"
```

Between episodes **both arms are parked, motors off** (since 2026-10-15:
holding the fold idle heated the close arm's J5 to 67 C, and the far arm's J5
tripped its overheat fault twice).

1. The session parks any arm whose motors are on, then waits.
2. **Space**: checks every motor (refuses above 60 C or with a fault; clears
   a latched overheat once the motor has cooled), powers both arms up and
   homes them. The far arm then goes limp: wiggle it and **hold it still at
   home**. The close arm eases onto it (it refuses if they differ by more
   than ~46 deg), and recording starts with the first follow step.
3. **Enter** ends and saves the episode, **r** discards it, **q** discards
   it and ends the session. Either way both arms park again (~15 s) while
   the episode saves in the background (~0.8 s per second of episode;
   **don't Ctrl-C while it saves**: q or Ctrl-C waits for it. Until
   2026-10-14 Ctrl-C killed LeRobot's encoder processes and lost the
   episode; it now encodes in-process). Reset the scene by hand, then Space.
   An episode reaching 180 s without Enter is discarded
   (`--max-episode-seconds`). `EPISODES=N` ends the session after N saved.
4. The session ends with both arms parked.

Parked means the motors are off (no torque, no current, no heat); the arm's
controller stays powered (green light) and reports joints and temperatures.
`make status` shows each motor's temperature and any fault.

Move the far arm at a moderate speed: the close arm follows at up to ~43 deg/s
on J1-J3, so faster moves record actions ahead of the state (one test lagged
333 ms; at moderate speed, 100 ms and within ~1 deg). The grippers open to
100 mm (close arm: stops at 99.4; far arm reads up to 103, so the action is
cut off at 100). Before 2026-10-14 the limit was the spec sheet's 70 mm.

Each frame holds `observation.state` (follower joints in rad and gripper in m,
7-D), `observation.end_pose` (6-D), `action` (the leader's target clipped to
the limits, 7-D), the task text and one video per config camera (LeRobot v3,
30 fps). Use a new directory per session; LeRobot's dataset tools can merge
sessions. `--episode-seconds` gives the old fixed-length episodes instead of
the keys.

**For fine-tuning,** convert to the pi0.5 Piper format of the checkpoint
`make policy` runs (13-D state: joints with J1 + 90 deg, gripper 0..1, end
pose rotated by 90 deg; 7-D action; cameras `base`/`wrist`), using the same
functions inference uses, so a model trained on it runs with the same flags:

```bash
make convert DIR=demos/pick_cube_01 OUT=demos/pick_cube_01_pi05
```

The gripper stays 0..1 of 70 mm as in the checkpoint, so wider openings go
above 1 (100 mm = 1.43). The converter trims idle time: each episode starts
0.5 s before the far arm first moves and ends 1 s after it last moves
(`active_span`; `--no-trim` keeps everything). The raw recording keeps every
frame.

### Meta Quest teleop

A Meta Quest's controller drives one arm through AgileX's
[QuestArmTeleop](https://github.com/agilexrobotics/QuestArmTeleop). The
headset app sends controller poses over adb, an IK node turns them into joint
targets, and AgileX's ROS 2 driver (`agx_arm_ros`, using the pyAgxArm SDK)
moves the arm. First run on the close arm: 2026-10-08. The controller-to-arm frame
(`ros_to_arm_rpy`) and the IK weights are still untuned.

| Command | What it does |
|---|---|
| `make quest-setup` | Installs everything, without sudo, at pinned commits (`real_robot/quest/setup.sh`): conda env `vt` with ROS 2 Humble (RoboStack), pinocchio and casadi; `~/QuestArmTeleop` workspace; pyAgxArm; adb in `~/opt/platform-tools` |
| `make quest-check [ARM=1\|2]` | Read-only: starts the driver with `auto_enable false` and prints its joints and TCP pose. The motors stay off |
| `make quest-teleop [ARM=1\|2] [QSPEED=30] [SMOOTH=2]` | Teleop; see "Start and stop" below. Afterwards, `make park` |

**Start and stop** (the right controller drives the arm):

1. Move the arm to a ready pose first, e.g.
   `.venv/bin/python -m real_robot.arm move --arm 1 --to=-93.3,30,-40,0,30,0`.
   At the folded rest pose the arm sits on its joint limits, a bad start for the IK.
2. `make quest-teleop ARM=1`. The driver enables the arm, which holds still.
3. **Press and release A** to engage. While A is held, the controller's
   reference keeps resetting, so nothing moves until you let go. After that
   the arm follows the controller's motion, relative to the arm's pose when
   you engaged.
4. **B** stops following, and the arm holds. Press A again to re-engage from
   wherever the arm and the controller are now.
   **Workspace limits** (2026-10-13, after the gripper touched the table):
   the fingertips, gripper and wrist stay above z = 0 of the arm's base
   frame (`QFLOOR=0` mm; the table is ~15 mm below that plane, so they stop
   ~15 mm above the table; at 40 mm they stopped ~55 mm up), and each joint
   just inside its hardware limit (J1 ±148°, J2 0…178°, J3 −168…0°, J4 ±98°,
   J5 ±67°, J6 ±118°; tighter until 2026-10-15). The arm can now swing
   anywhere the hardware allows, **including into the other arm**: the
   guard only knows the table. Since
   2026-10-15 the arm just **stops at a limit and keeps following in every
   other direction**, no pause and no button: at the floor it slides along
   the table and lifts off freely. While your hand pushes past a limit, the
   controller is re-anchored at the arm's pose, so moving back moves the arm
   back at once (the log says `holding:` and which limit). Set them in
   `ik_guard.py` (`FLOOR_Z`, `SOFT_*_DEG`).
   The guard still **pauses** on a real fault (the log says `PAUSED` and
   why): an IK solution jumping > 29 deg in a frame, or the arm > 57 deg from
   its command (blocked). Press **A** to follow again from where the arm is.
5. The **trigger** sets the gripper: released is closed, a full pull is 100 mm.
   So the gripper is commanded closed for the whole session unless you pull.
6. Ctrl-C ends the launch. The arm stays enabled and holding. The driver
   sometimes hangs on exit (upstream's publisher thread); kill it, since that
   sends nothing to the arm. Then `make park`.

**Quest setup** (done once, 2026-10-08): developer mode, `teleop-debug.apk`
installed (package `com.rail.oculus.teleop`), and a udev rule so adb can open
the headset (`/etc/udev/rules.d/51-android-quest.rules`, vendor 2833). Accept
"Allow USB debugging" in the headset. The headset only sends poses while its
display is on. To run it off-head (e.g. sitting on the table), send
`adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`, which
makes it act as if worn until a reboot (`...automation_disable` undoes it), and
wake it with `adb shell input keyevent KEYCODE_WAKEUP`.

Our launch file, `real_robot/quest/teleop_quest_piper.launch.py`, differs from
upstream's `teleop_single_piper.launch.py` in these ways:

- **Position mode.** It uses `fast_mode false` (`move_j`, with the firmware
  speed cap). Upstream uses `move_js`, which is MIT mode with no smoothing;
  pyAgxArm's own docstring rates it "EXTREMELY HIGH" risk.
- **CAN names.** It uses `can_piper1`/`can_piper2`. Upstream's CAN scripts
  rename the adapters (`can0`, `can_left`, `can_right`); don't run them.
- **IK settings.** It locks the URDF's `gripper` joint (otherwise the IK would
  solve for 7 joints) and sets `w_reg 0`. Upstream's regularization pulls the
  joints toward 0, and our bases rest at −93°, so in an offline test it
  dragged the base about 12° during a 3 cm move.
- **Guard node.** `ik_guard.py` sits between the IK and the driver. It streams
  commands at 100 Hz through the critically damped filter of
  `real_robot/smoothing.py` (`smooth_hz`, default 2 Hz, about 0.16 s of lag).
  Each command is rate-limited to `max_step_delta` per 25 Hz step, about 43°/s
  on J1–J3 and 72°/s on J4–J6. The first version sent each IK target as its
  own `move_j` at 25 Hz, and the arm shook as it moved: the firmware slows
  down as it reaches each target, just as in the policy runs. It pauses if an IK target jumps more than
  29° between frames, or if the arm is more than 57° behind the target. The
  arm holds; B then A re-anchors and resumes. (Until 2026-10-13 a stop needed
  a restart, and the IK's configuration flips tripped it within seconds.) This matters
  because the IK can switch arm configuration from one frame to the next. In
  offline tests with random hand paths, upstream's weights trailed the hand
  by up to about 11 cm. Raising `w_pos` to 20 brought tracking within about
  2 cm, but some frames jumped 18–25°.

**Latency.** `make quest-teleop TRACE=verify/quest_trace.npz` adds a passive
recorder (`real_robot/quest/trace.py`, which only subscribes). In the `vt` env,
`python real_robot/quest/trace_report.py verify/quest_trace.npz` then reports
each stage's delay: hand to IK, IK to command (the guard), and command to arm
(the firmware). It also reports the IK's position error and how often the arm
stalls while its command moves. Traced runs on 2026-10-08:

| Run | hand → IK | IK → command | command → arm | IK rate | arm stalled |
|---|---|---|---|---|---|
| 1: upstream IK node, 30% speed cap, 2 Hz filter, RViz on | 340 ms | 180 ms | 310 ms | 16 Hz, 163 gaps over 100 ms | 25% |
| 3: `ik_node.py`, 60% speed cap, 3 Hz filter, RViz off | 65 ms | 155 ms | 320 ms | 23 Hz, no gaps | 15% |

The firmware stage (position-mode planning) is now the largest: about 250 ms
even for slow moves. Next: replace the guard's fixed filter with a One Euro
filter (smooths strongly when the hand is still, lightly when it moves), add
optional lead compensation for the firmware delay, then run another trace.

**Single sender.** The driver keeps only the newest control message. Until
2026-10-13, `pub_delta_pose` sent the gripper there too, so the two senders
overwrote each other: the arm stalled 17% of the time while its command was
moving, and the gripper felt slow. Now the controller node publishes to
`/quest/gripper`, and `ik_guard` includes the gripper in every command (100 Hz
while driving the arm, 33 Hz gripper-only otherwise).

Offline tests of the guard or the IK must use their own ROS domain
(`ROS_DOMAIN_ID=77`). On the default domain, a running launch picks up fake
targets and moves the arm: this happened once, a 4 s base sweep of up to ±20°.

The driver doesn't disable the arm on exit. It holds the last target with the
motors on, so always finish with `make park`. When the arm is enabled while
someone holds A, the controller's motion is applied relative to the arm's
current TCP pose. The IK weights (`arm_ik_pose_node.piper.yaml` plus our
overrides) still need tuning with real controller data.

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
  The RealSense D455 (close-arm `cam_wrist` since 2026-10-13) offers no MJPG:
  its raw 640x480 YUYV takes ~18 MB/s, so on the USB 2 bus it plus two MJPG
  cameras is about the limit. A blue USB 3 port (bus 4, empty) avoids that,
  and allows depth or 1280x720 at 30 fps.
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
