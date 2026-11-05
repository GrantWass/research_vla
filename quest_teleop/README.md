# Quest teleop for the Piper arm

A Meta Quest's right controller drives one Piper arm in real time. This is
AgileX's [QuestArmTeleop](https://github.com/agilexrobotics/QuestArmTeleop)
with the changes below. Everything of ours is in this folder; nothing in it
needs the rest of the repo.

```
Quest app --adb--> pub_pose --> quest_delta --> ik_node --> ik_guard --> agx_arm_ros driver --> arm
(upstream)         (upstream)   (ours)          (ours)      (ours)       (upstream)
```

## Install

Once per machine, from this folder:

1. `make setup` installs the upstream stack at pinned commits, without sudo: a conda env `vt` (ROS 2 Humble, pinocchio, casadi), `~/QuestArmTeleop`, pyAgxArm and adb. Safe to rerun.
2. Name the CAN adapter for your arm (ours: `can_piper1`) and set it as `can_port` in `config.json`. Do not run upstream's CAN scripts; they rename adapters.
3. Put the Quest in developer mode and install upstream's `teleop-debug.apk` (package `com.rail.oculus.teleop`).
4. Let adb open the headset. Create `/etc/udev/rules.d/51-android-quest.rules`, reload udev, replug the cable, and accept "Allow USB debugging" in the headset:
   ```
   SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
   ```
5. `make check` starts the driver with the motors off and prints the joints and TCP pose. Nothing moves.

The headset only sends poses while its display is on.

## Use

Keep your hand clear of the table and of any other arm.

1. `make ready` moves the arm to `ready_deg` in `config.json` (asks for YES). Don't start from the folded rest pose; it sits on its joint limits and the IK starts badly there.
2. `make teleop`. The driver enables the arm, which holds still.
3. **Calibrate** (once each time you start `make teleop`; see "Calibration" below). It takes two short strokes of the right controller.
4. **Press and release A** to engage. While A is held nothing moves; after release the arm follows your hand relative to where it was.
5. **B** stops following and the arm holds. A re-engages from wherever the arm and hand are now.
6. The **trigger** sets the gripper: released is closed, full pull is 100 mm.
7. Ctrl-C ends the launch, but the arm stays enabled and holding. Run `make park`. The driver sometimes hangs on exit; kill it, which sends nothing to the arm.

### Controller buttons (right controller only)

| Button | Where | What it does |
| --- | --- | --- |
| Grip | Side button, squeezed with your middle finger | Held during each calibration stroke |
| A | Lower face button, under your thumb | Press and release to start following; also re-anchors and clears a pause |
| B | Upper face button, above A | Stop following; the arm holds |
| Trigger | Front, under your index finger | Gripper: released is closed, a full pull is 100 mm |

### Calibration

Calibration tells the software which way "forward" and "up" are for you, because the headset's own axes can point anywhere. Do it after the log says `CALIBRATE FIRST`, with the headset on your head.

1. Stand or sit facing the direction the arm reaches (toward the table).
2. **Stroke 1, forward.** Squeeze and hold the grip with your middle finger. With the grip still held, move the controller straight forward, away from you, about 15 cm. Let go of the grip. The log says `forward stroke N cm recorded`.
3. **Stroke 2, up.** Squeeze and hold the grip again. Move the controller straight up toward the ceiling about 15 cm. Let go. The log says `calibrated`.
4. Press and release A to start following.

Tips: keep each stroke straight and at least 8 cm; the two strokes must be clearly different directions (at least 45 degrees apart). If a stroke is too short or the two are nearly parallel, the log says so and you repeat. Don't press A until you see `calibrated`; before that it only prints `not calibrated`. Holding the grip also stops following, so the calibration strokes never move the arm.

Settings you may want to change. The defaults are what worked for us; pass any on the command line, e.g. `make teleop SMOOTH=2 SPEED=40`:

| Variable | Default | What it does |
| --- | --- | --- |
| `SMOOTH` | 3.0 | Smoothing filter in Hz. Lower is smoother but laggier (2 Hz is about 0.16 s of lag). |
| `SPEED` | 60 | The arm firmware's speed cap, percent. Lower it for a first run. |
| `SCALE` | 1.0 | Arm motion per hand motion in room mode. Above 1 reaches farther with less hand travel. |
| `FORWARD` | -93.3 | Base yaw in degrees that your "forward" maps to. -93.3 faces our table; set it for your layout. |
| `FLOOR` | 0 | Lowest the gripper and wrist may go, in mm above the arm's base plane (the table is about 15 mm below it). |
| `MODE` | room | `room` moves 1:1 with your hand in the room; `controller` is upstream's mapping. |
| `CAN` | from `config.json` | CAN interface of the arm. |
| `RVIZ` | false | Opens RViz windows, which add lag. |
| `TRACE` | off | Path to an `.npz` file to record latency; report with `trace_report.py`. |

Limits, step sizes and the ready and park poses are in `config.json`.

`make park` is the only way we power the arm down: it folds with the base held, turns the base to `home_base_deg`, triggers the damped emergency stop, waits 3 s, disables and clears the stop. Disabling from any other pose drops the arm onto its base.

## What differs from the default install

| File | Replaces or adds | Change |
| --- | --- | --- |
| `teleop.launch.py` | upstream `teleop_single_piper.launch.py` | Position mode, CAN port from `config.json`, IK overrides, our nodes |
| `quest_delta.py` | upstream `pub_delta_pose` | Room-frame 1:1 mapping, two-stroke calibration, A re-anchors and clears pauses, B stops, gripper on its own topic |
| `ik_node.py` | wraps upstream `arm_ik_pose_node` | Latest pose only, no feedback resync while engaged; IK delay 340 ms to 65 ms |
| `ik_guard.py`, `smoothing.py` | new, between IK and driver | 100 Hz filtered stream, rate limit, joint and table-floor limits, pause on faults; the only sender to the driver |
| `arm_tool.py` | new | `ready` and `park` through pyAgxArm |
| `config.json` | new | CAN port, joint limits, step limits, ready and park poses |
| `setup.sh`, `Makefile` | new | Pinned no-sudo install and the targets above |
| `trace.py`, `trace_report.py` | new, optional | Passive latency recorder and report |

Why each change exists, in plain terms:

- **Gentler arm mode.** Upstream drives the arm in a mode with no smoothing or planning, which the SDK itself calls "extremely high" risk. We use the normal position mode, which has a built-in speed cap.
- **Better IK settings** (the IK is the math that turns your hand position into joint angles). Upstream's settings pull every joint toward zero, but our arms rest with the base at -93 deg, so the base got dragged about 12 deg on a 3 cm hand move. We turn that pull off and make matching your hand's position a priority. Offline, the arm then followed a 20 cm sweep to within 5 mm instead of 5 cm. We also lock the gripper joint so the IK only solves for the six arm joints.
- **One sender to the arm.** The driver only keeps the newest command. Upstream sent the arm and the gripper from two programs, so each overwrote the other and the arm stalled about 17% of the time. Now the gripper goes through `ik_guard`, which sends both together.
- **A safety guard between the IK and the arm.** The IK sometimes jumps to a different arm pose between frames (the base swung about 60 deg for a 3 cm hand move). `ik_guard` smooths the commands, limits how fast joints move, stops the arm at its limits and at the table, and pauses if the IK jumps or the arm falls far behind. The log says `PAUSED`; press A to continue.

The two settings in the launch file that make the IK change (for reference):

```python
"locked_joints": ["gripper_base_joint", "gripper", "gripper_joint1", "gripper_joint2"],
"w_reg": 0.0, "w_pos": 20.0,
```

## Limitations

- **Motion is relative to the headset, so it must be worn, and preferably on the head.** Poses are reported relative to the headset. If the headset moves while you drive, the arm moves with it. On the neck it is hard to reach fully down to the maximum range of motion.
- **Joints can drift into an extreme pose and stay there.** Sometimes one joint (often the wrist, J4/J6) turns to an odd angle near the end of its range, and the other joints compensate for it, so the arm keeps that posture instead of coming back. The IK picks whichever solution is nearest, so it doesn't correct itself. Joints also stop at soft limits just inside the hardware's, and the guard may pause. B only holds the arm where it is, so it does not fix this. To reset: Ctrl-C the launch, run `make ready` (moves the arm to the middle-of-range pose), then `make teleop` and calibrate again. Pressing A re-anchors at the arm's current pose, so it keeps the odd posture too.
- **The arm's range is bigger than yours.** It can reach and rotate farther than your arm and wrist comfortably do, so at 1:1 you run out of reach first. Press B, bring your hand back, press A (the re-anchor lets you "ratchet"), or raise `SCALE`.
- **Both controllers must be on and tracked, even though only the right one drives the arm.** Upstream's `pub_pose` publishes nothing unless the headset reports both. With only the right controller on, the gripper trigger still works but the arm never moves, and the log shows no calibration or engage lines. Wake the left controller (pick it up, press a button) and keep it in view.
- **A dead controller battery looks like a software fault.** If the right controller's battery dies, the headset drops it and the symptom is the same as above: gripper may work, the arm never moves, no calibration lines in the log. Check the controller's battery first (`adb logcat -d | grep -i "RIGHT slot"` shows when the headset dropped it).
- **The guard only knows the table.** The arm can swing into another arm or object.
- **Latency is about half a second end to end** (hand to IK 65 ms, guard 155 ms, firmware 320 ms in our traces). The firmware's position-mode planning is the largest part.
- The controller-to-arm frame (`ros_to_arm_rpy`) and the IK weights are still being tuned.

## Future work

- **Keep the arm out of its end range.** Today the guard only clips a joint at its soft limit, and the IK can leave the arm in an extreme pose with other joints compensating (see Limitations). Ideas: add a cost to the IK that pulls joints toward a mid-range pose (upstream's regularizer pulls toward zero, which is wrong for our base angle, so it needs a different center); warn or pause when a joint is within about 10 degrees of its limit; and offer a recovery that steers back to the ready pose without restarting.
- **Scale the directions separately.** The arm's workspace is bigger than a person's, and our reachable range differs by direction (for example reaching far down is hard). Replace the single `SCALE` with a scale per axis (and for wrist rotation), so a comfortable hand motion covers the arm's useful range.
- Remove the dependence on the left controller by reading the right controller directly in `quest_delta`.

## Tests

`make test` checks the arm tool's ready, park and feedback-wait logic against a fake arm.

Run on the real close arm (2026-10-21, no headset): `make check` read the joints and TCP pose with the motors off; `make ready` reached -93.3, 30, -40, 0, 30, 0 deg; `make park` folded, turned the base, stopped, disabled and left the arm at the same rest pose as our own park, motors off (confirmed with our repo's separate `real_robot.arm status`, which this folder does not need); and the arm enabled again right after, twice. Not yet run: `make teleop`, which needs the headset (calibration, A/B, trigger, guard behavior).
