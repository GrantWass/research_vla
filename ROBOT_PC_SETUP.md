# Robot PC setup (2x AgileX Piper)

Machine: Alienware Area-51 R2 (`int-alienware-area-51-r2`), user `csce990`,
Ubuntu 22.04, RTX 3090 24 GB, driver 580.178. Reached over Tailscale
(`ssh csce990@100.90.226.52`, Tailscale SSH). Bring-up procedure and checks:
[REAL_ROBOT.md](REAL_ROBOT.md).

## Done (2026-10-06)

| item | state |
|---|---|
| `~/research_vla` | this repo, git checkout of `GrantWass/research_vla`; pushes over SSH with a write deploy key (`~/.ssh/research_vla_deploy`, host alias `github-research_vla` in `~/.ssh/config`) |
| `~/research_vla/.venv` | `piper_sdk` 0.6.2, `python-can`, `numpy`, `opencv-python`; 27 real-robot tests pass |
| `~/openpi` | upstream openpi, `uv sync`; `pi05_droid` checkpoint cached |
| `uv` | `~/.local/bin/uv` |
| apt | `can-utils`, `v4l-utils`, `ethtool` |
| CAN | `can_piper1` -> `real_robot/configs/piper_1.json`, `can_piper2` -> `piper_2.json` (names pinned to adapter serials, up at 1 Mbit/s on boot via `real_robot/can_persist.sh`, installed 2026-10-06), both arms firmware S-V1.8-3 |
| checks | `verify --stage gpu` PASS; `verify --stage policy` PASS on mock frames (7 -> 8 -> 7 dims, 224x224 inputs, ~166 ms warm, 6.4 GB peak) |

Cameras: three Orbbec Astra Pro Plus (color 2bc5:050f is plain UVC, depth
2bc5:060f needs the Orbbec SDK). All three color sensors report the same
serial (`SN0001`), so `/dev/v4l/by-id` shows only one; the configs address
them by USB port through `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:<port>:1.0-video-index0`.
Keep each camera in its port. All read 640x480 at 30 fps.

| port | camera | config |
|---|---|---|
| 5.2 | high stationary view | `cam_high` in both configs (only one runner at a time) |
| 9.2 | on top of the close arm (`can_piper1`) | `cam_wrist` in `piper_1.json` |
| 1.2 | on top of the far arm (`can_piper2`) | `cam_wrist` in `piper_2.json` |

"Close" and "far" are from the operator seat. Arm-to-CAN mapping confirmed by
unplugging the close arm's adapter (`can_piper1` disappeared). Port-to-camera
mapping per the operator, from snapshots.

## Done without sudo (2026-10-06)

| item | state |
|---|---|
| ffmpeg | already installed (4.4.2) |
| `~/pyorbbecsdk` | `orbbec/pyorbbecsdk` clone (branch `v2-main`), for its udev rules and examples |
| Orbbec SDK | `pyorbbecsdk2` 2.1.2 (SDK 2.9.3) in `~/research_vla/.venv`, installed `--no-deps` so numpy 2.2.6 / OpenCV 5 stay; it lists the camera but cannot open it until the udev rules are in |
| Orbbec reader | `OrbbecCameras` in `real_robot/backends.py`: set a camera to `"orbbec"` (or `"orbbec:<serial>"`) in a config's `cameras`; color goes to the policy, depth (mm) is kept in `.depth`. A `Cameras` router picks Orbbec or OpenCV per camera. Tested against a fake SDK only |
| CAN persistence script | `real_robot/can_persist.sh`, pins the two adapters by USB serial to `can_piper1` / `can_piper2` and brings them up at boot (needs sudo, below) |
| `~/lerobot_robot_piper` | `WeGo-Robotics/lerobot_robot_piper` clone with its own `.venv` (8 GB: LeRobot 0.4.4, torch 2.10, `piper_sdk`), separate from the repo venv. `piper-doctor` passes packages, plugin registration, both adapters, both arms broadcasting; it fails only on the missing `~/piper_config.json` (made by `piper-setup`) and warns about boot persistence. It looks for WeGo's own rules and service, so it keeps warning after `can_persist.sh` too |

## Details

### Done with sudo (2026-10-06)

- Orbbec udev rules (`/etc/udev/rules.d/99-obsensor-libusb.rules`): the
  device nodes are now `0666`, group `video`. The SDK now opens the device
  and reports it as **"Bootloader device, PID 0x0400"**, with no serial and
  no sensors. So the depth side really is in bootloader mode, and the
  SDK reader (`OrbbecCameras`) could not be used. A replug fixed it: all
  three depth sensors now enumerate as 2bc5:060f (normal). Depth through the
  SDK is still untested on the real cameras.
- `real_robot/can_persist.sh` (first version; rerun the fixed one, below):
  `can_piper1` (was `can0`, joints about
  [4, -2, 2, 9, 23, 1] deg) and `can_piper2` (was `can1`, joint 1 about
  -90 deg) are up at 1 Mbit/s; configs and tests updated.

### CAN replug fix (2026-10-06)

The first `can_persist.sh` started the bring-up service only on the udev
"add" event; a replug's rename also emits "move", so the adapter came back
DOWN. The fixed script (rerun 2026-10-06) matches both, and a replug of the
far arm's adapter then came back up at 1 Mbit/s on its own.

### Teleop (works, 2026-10-06)

`real_robot/teleop.py`: the far arm (`can_piper2`) is the leader, put in
teaching mode (`MasterSlaveConfig 0xFA`, goes limp, wrist sags ~30 deg); the
close arm follows through `SafetyFilter`. Hold the leader's forearm when it
starts.

```bash
cd ~/research_vla && .venv/bin/python -m real_robot.teleop --leader can_piper2 --follower-config real_robot/configs/piper_1.json --seconds 300
```

Learned on the hardware:
- A teaching-mode arm broadcasts its joint targets (0x155-0x157, gripper
  0x159) only while it is being moved, and is otherwise silent on the bus
  (reads as all zeros). The script waits for a complete, steady broadcast
  before using it, and refuses to start if the arms differ by > 0.8 rad.
- The mode switch takes effect immediately, no power cycle needed. The script
  restores the leader to normal mode (0xFC) on exit, Ctrl-C or SIGTERM;
  `--restore-leader` does only that.
- At the configs' 30% firmware speed the follower can't keep up with a hand;
  teleop uses `--speed-percent 60` by default.
- The WeGo GUIs (`piper-setup`, `piper-teleop`) were not used: they rename the
  CAN ports and assume a shared leader/follower layout.

## Next session

0. **Gentle shutdown (do first).** "Zero" (the lowest pose the firmware
   accepts: J2 >= 0, J3 <= 0) holds each arm ~1 inch above the base it rests
   on, and the wrist straight. Disabling the motors there drops the arm that
   inch and the wrist to ~37 deg (seen 2026-10-06). Write a `park` step:
   fold, lower J5 to ~37 deg, then try the software e-stop
   (`MotionCtrl_1(0x01, 0, 0)`; AgileX describes it as a damped stop, so the
   arm should descend slowly) before `DisablePiper`, and reset after
   (`MotionCtrl_1(0x02, 0, 0)`). Test it once with someone watching.
   Widening the firmware J2/J3 limits a few degrees would also work, but it
   changes stored arm settings: only if wanted.

1. **Banana test** (needs a banana and a basket). Scene as in their training
   data, `verify/banana_scene_compare.jpg`: banana ~30 cm in front of the
   close arm's gripper, basket front-left, stationary camera (port 5.2) high
   and behind-left so it sees arm + banana + basket. The close arm's base is
   at about -93 deg, facing the table. Dry run first, then add `--live`:

```bash
cd ~/research_vla && ~/lerobot_robot_piper/.venv/bin/python -m real_robot.runner --backend piper --config real_robot/configs/piper_1_banana.json --lerobot richardshkim/pi05_piper_banana_v2_full --end-pose-state --instruction "Put banana into basket." --max-steps 300 --execute-steps 30 --speed 0.5
```

2. **Record demos for fine-tuning.** Still to write: recording during
   teleop (`teleop()` has an `on_step(state, cmd, leader)` hook) that saves the
   follower's joints, gripper, end pose and both cameras at 30 fps in LeRobot
   format, with an instruction per episode. Then a batch of demos of a few
   tasks, and a fine-tune on the RTX PRO 6000 box (`pi05_base`, LAP-3B or
   GigaBrain-0.7, see REAL_ROBOT.md).

## Not needed here

openpi/JAX/CUDA, `piper_sdk`, OpenCV, the NVIDIA driver: done. Fine-tuning
runs on the RTX PRO 6000 box, not this machine.

## Open questions

- Joint readings at rest: can_piper1 about [4, -2, 2, 9, 23, 1] deg, can_piper2 joint 1
  about -90 deg. Confirm against the arms by eye. Joint 2 lower and joint 3
  upper limits were widened 0.1 rad because both arms rest just past them.
