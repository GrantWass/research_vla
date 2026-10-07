# Real-robot bring-up (pi0.5)

Drive a physical dual-arm robot with the same pi0.5 policy server used in sim.
The server is unchanged; `real_robot/` replaces the Isaac Sim env loop.

```
cameras + joints -> real_robot.runner -> ws -> pi0.5 server (linuxbox, port 6000)
       ^                                               |
       +--- motors <- SafetyFilter <- unit mapping <---+  action chunk (T x 14)
```

## Day 1 checklist (first access to the robots)

`real_robot/verify.py` writes `report.json` + PNGs to `--out`; every check
prints PASS / FAIL / INFO. Nothing is sent to a motor except `--wiggle`
(asks first).

**0. Setup (workstation).** `pip install piper_sdk`; CAN comes up
at boot as `can_piper1` / `can_piper2` (`sudo bash real_robot/can_persist.sh`
once, pins names to adapter serials; without it, `can_activate.sh can0 1000000`
from piper_sdk);
upstream openpi checkout + `uv sync` on the GPU box (RoboDojo's fork lacks
`pi05_droid`). Fill camera paths in `configs/piper_1.json` from
`ls /dev/v4l/by-id/`.

**1. GPU.** Driver, VRAM, JAX/torch see the card.

```bash
python3 -m real_robot.verify --stage gpu --out verify/gpu
```

**2. Robot I/O (read-only).** Joint angles in degrees (compare with the arm
by eye / piper_sdk UI), within limits, camera fps >= 25, raw PNGs + snapshot.

```bash
python3 -m real_robot.verify --stage robot --config real_robot/configs/piper_1.json --out verify/piper_1
```

Then the command path, units and sign (joint 6, +0.05 rad and back):

```bash
python3 -m real_robot.verify --stage robot --wiggle --config real_robot/configs/piper_1.json --out verify/piper_1
```

**3. pi0.5 plumbing.** Loads `pi05_droid` (single-arm, base + wrist camera:
closest official layout to ours), feeds it the snapshot, saves
`model_sees_*.png` (the exact 224x224 inputs; the third slot is masked
out), checks 7 -> 8 -> 7 dimension hops, latency, peak VRAM. Its actions are
Franka actions and are never sent.

```bash
python3 -m real_robot.verify --stage policy --config real_robot/configs/piper_1.json --snapshot verify/piper_1/snapshot.npz --out verify/piper_1
```

Look at `model_sees_base_0_rgb.png` / `model_sees_left_wrist_0_rgb.png`:
scene and gripper visible, not letterboxed into a sliver, colours not
swapped (blue objects should look blue). Repeat 2-3 for `piper_2.json`.

**4. Teleop (1 leader / 1 follower).** Uses
[WeGo-Robotics/lerobot_robot_piper](https://github.com/WeGo-Robotics/lerobot_robot_piper)
(records straight to LeRobot). Per its README (not yet run here):
`pip install lerobot_robot_wego_piper`, `piper-setup` (choose 1 Leader /
1 Follower, persistent CAN names), `piper-calibrate` (both arms in the same
pose), `piper-teleop` to practise, then `lerobot.record` with
`--robot.type=piper_follower --teleop.type=piper_leader`. Start slow; check
the follower tracks without lag or offset before recording.

## Our hardware: 2 separate single-arm AgileX Pipers

Two independent robots (6-DoF + gripper each), one workstation. Each robot
gets its own config (`real_robot/configs/piper_1.json`, `piper_2.json`:
state/action 7-D, cameras `cam_high` + `cam_wrist`) and its own runner
process; both can share one pi0.5 server (see MULTI_ROBOT_SERVING.md).

**No official or general-purpose pi0.5 checkpoint for Piper.** openpi ships
`pi05_base` (pretrain only) plus DROID/LIBERO/ALOHA fine-tunes; AgileX's
[openpi-agilex](https://github.com/agilexrobotics/openpi-agilex) has configs
but only the base checkpoint. Community single-task, single-arm fine-tunes
exist on Hugging Face (as of 2026-10-06): PiPER-X
`theroboticscompany/pi05-piperx-*` and `abdulrahimmirani/pi05-piperx-*`,
49-200 real teleop demos each. They are the closest public match (same arm
family, single arm) and a good end-to-end smoke test, but they were trained
on someone else's task, cameras and scene. The RoboDojo checkpoint is sim
dual ARX X5: wrong embodiment. Path: record demos -> fine-tune `pi05_base`
-> run with this loop.

1. CAN: one USB-CAN adapter per robot; `can_piper1`/`can_piper2` via
   `real_robot/can_persist.sh` (match `arms` in each config).
2. `pip install piper_sdk`; note `pip show piper_sdk` and arm firmware
   (`dh_is_offset` differs before V1.6-3).
3. Per config: verify limits/gripper stroke, set camera paths from
   `/dev/v4l/by-id/`, then `"verified": true`.
4. Teleop: Piper supports leader/follower linkage, so one Piper can lead the
   other (record the follower). Or hand-guide one arm in teach mode. The
   recorder is read-only either way:

```bash
python3 -m real_robot.record --backend piper --config real_robot/configs/piper_2.json --out demos/pick_cube --instruction "put the cube in the cup" --seconds 30 --episodes 10
```

5. Fine-tune `pi05_base` on the RTX PRO 6000 box (LeRobot conversion +
   openpi single-arm train config + a Piper `Model` for the server: still to
   write; the RoboDojo Pi_05 server Model hardcodes 3 ALOHA cameras).
6. Run each robot with `real_robot.runner --backend piper --config
   real_robot/configs/piper_N.json` (commands below).

The ARX X5 material below documents what the RoboDojo sim checkpoint expects.

## Generalist policies for the Piper (surveyed 2026-10-06)

No public policy drives a Piper well out of the box. What exists:

| model | open | Piper fit | verdict |
|---|---|---|---|
| `richardshkim/pi05_piper_banana_v2_full` (LeRobot pi0.5) | yes | trained on a single Piper: 20 demos of "Put banana into basket.", base + wrist camera, joints in rad, 13-D state (joints, gripper 0..1, end pose) | **runs here today** (`--lerobot`, `piper_1_banana.json`); one task only |
| [LAP-3B](https://lap-vla.github.io/) (`lihzha/LAP-3B`, openpi/JAX) | yes, Apache-2.0 | outputs gripper (end-effector) poses in the base frame from a side + wrist camera; >50% zero-shot on unseen arms (YAM, Kinova, Franka), never tried on a Piper | best zero-shot bet; needs a Piper client sending end-effector targets (firmware does the IK), plus a workspace box limit |
| [GigaBrain-0.7](https://huggingface.co/open-gigaai/GigaBrain-0.7-3.5B-Base) | yes | pretraining includes AgileX Cobot Magic (Piper-based) data; repo ships a Piper fine-tuning config | best base for **fine-tuning**; no live-control path yet |
| [RDT2](https://github.com/thu-ml/RDT2) | yes | zero-shot only with its own gripper + camera hardware, printed mounts, and Vive-tracker calibration | needs hardware purchase |
| [DreamZero](https://arxiv.org/abs/2602.15922) (NVIDIA, 14B) | yes | adapts to a new arm with ~30 min of play data | **too heavy for the RTX 3090**: 14B at 16-bit is ~28 GB of weights (> 24 GB), and its 7 Hz needs 2x GB200 with FP4 weights (Blackwell only); unoptimized it took 5.7 s per step even on GB200. Candidate for the RTX PRO 6000 box (96 GB, Blackwell) |
| X-VLA, pi0.5 base, SmolVLA, GR00T | yes | no Piper checkpoint | need a fine-tune |

Path to a useful policy: record teleop demos of a few tasks here, then
fine-tune (pi0.5 base, LAP-3B, or GigaBrain-0.7) on the RTX PRO 6000 box.

## What the policy expects (from the sim it was trained in)

| item | value | source |
|---|---|---|
| state / action | 14-D `[L arm 6, L grip, R arm 6, R grip]`, joint positions | `XPolicyLab/utils/process_data.py` `pack_robot_state`, `_robot_info.json` (`dual_x5`) |
| cameras | `cam_high` (head), `cam_left_wrist`, `cam_right_wrist`, RGB | `XPolicyLab/policy/Pi_05/model.py` `encode_obs` |
| resolution | 640x480 (pi0.5 resizes to 224 itself) | `env_cfg/camera/template.py` `GEMINI_345LG` |
| head camera pose | pos `[0, -0.41, 1.308]` m, pitched 30 deg | `env_cfg/camera/camera_config.yml` |
| arm bases | `[±0.3, -0.45, 0.765]` m | `env_cfg/robot/dual_x5.yml` |
| control rate | 25 Hz | `env_cfg/arx_x5.yml` `collect_freq` |
| checkpoint | `RoboDojo-sim-arx_x5-joint-0` (sim-only data) | `GPU_BOX_SETUP.md` |

Expect ~0% task success zero-shot: the checkpoint has never seen real
images. Today's bar is "moves sensibly toward the objects, no unsafe motion".

## Files

- `real_robot/backends.py` — `RobotBackend` interface, `MockBackend`,
  `OpenCVCameras`, `ArxX5Backend` (SDK calls are TODOs, see below).
- `real_robot/safety.py` — NaN/limit/rate checks; aborts on any single-step
  jump > `abort_delta` (catches wrong joint order/sign/units).
- `real_robot/mapping.py` — real <-> sim units (per-joint sign/offset,
  linear gripper map).
- `real_robot/runner.py` — the loop. **Dry-run by default**; `--live` moves.
  Always calls `hold()` on exit, error, timeout, or Ctrl-C.
- `real_robot/configs/arx_x5.json` — limits/mapping/cameras, `verified: false`.
- `tests/test_real_robot.py` — mock robot + fake policy (`make test`).

## Still to do with the robot in front of you

1. **SDK.** Fill the four `TODO(sdk)` methods in `ArxX5Backend` from the SDK
   installed on the robot PC. Test them alone first (read state, then
   command the current position, then a 0.05 rad move on one joint).
2. **Config.** In `configs/arx_x5.json`: real joint limits; real gripper
   range; sim gripper range (x5 `gripper_scale` in RoboDojo Assets on
   linuxbox); camera device ids (`v4l2-ctl --list-devices`). Compare a
   real joint reading at a known pose against the sim's home pose to set
   `joint_sign` / `joint_offset`. Then `"verified": true`.
3. **Cameras.** Mount the head camera as close to the sim pose as you can.
   Save a frame and eyeball it next to a sim head-camera mp4.

## Run order

On linuxbox, start the server (as in sim evals; `make lowvram` profile if
needed). The robot PC must reach it over Tailscale/LAN.

```bash
python3 -m real_robot.runner --backend arx_x5 --config real_robot/configs/arx_x5.json --preflight-only
```

```bash
python3 -m real_robot.runner --backend arx_x5 --config real_robot/configs/arx_x5.json --url ws://grant-pc-linux:6000 --instruction "stack the bowls" --max-steps 200
```

Dry-run: check `real_robot_logs/<ts>-dry/summary.json` — `safety.max_requested_delta`
should be small, no abort, `infer_ms_median` ~100-150 ms. Plot
`trajectory.npz` `target_real` vs `q_real` per joint for direction sanity.

```bash
python3 -m real_robot.runner --backend arx_x5 --config real_robot/configs/arx_x5.json --url ws://grant-pc-linux:6000 --instruction "stack the bowls" --live --speed 0.25 --execute-steps 10 --max-steps 200
```

Live: hand on e-stop, `--speed 0.25`, short `--execute-steps` so it
re-plans often. The arm pauses during each inference (no async execution yet).
