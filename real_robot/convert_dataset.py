"""Convert a recorded demo dataset (recorder.py) to the pi0.5 Piper training format.

Recordings keep raw robot units: joints rad (our bases face the table at
J1 ~ -93 deg), gripper m (0..0.07), end pose separate, cameras cam_high /
cam_wrist. The pi0.5 Piper checkpoint that runs on this rig
(richardshkim/pi05_piper_banana_v2_full, `make policy`) wants:

  observation.state (13): joints with J1 + base_offset, gripper 0..1 of the
      0.07 m stroke, end pose [x, y, z m, roll, pitch, yaw rad] rotated about
      the base z axis by base_offset
  action (7): joints with J1 + base_offset, gripper 0..1
  observation.images.base / .wrist: cam_high / cam_wrist

Every number goes through the same functions inference uses (mapping.real_to_sim
with the banana config's gripper range, the base offset, lerobot_policy.
rotate_end_pose), so a model fine-tuned on the output runs with the same
`make policy` flags (--base-offset-deg 90 --end-pose-state, camera map
cam_high:base,cam_wrist:wrist).

Idle time is trimmed from each episode (active_span): it starts LEAD_S before
the leader first moves and ends TAIL_S after it last moves, so the model
doesn't learn to sit still at the start. --no-trim keeps every frame.

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.convert_dataset \\
      demos/pick_cube demos/pick_cube_pi05
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .config import load_config
from .lerobot_policy import rotate_end_pose
from .mapping import real_to_sim

STATE_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper", "x", "y", "z", "roll", "pitch", "yaw"]
ACTION_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
CAMERA_MAP = {"cam_high": "base", "cam_wrist": "wrist"}
LEAD_S, TAIL_S = 0.5, 1.0
MOVING_DEG_S, MOVING_GRIP_M_S = 3.0, 0.005  # leader at rest reads ~0 on both


def active_span(actions, fps: float, lead_s: float = LEAD_S, tail_s: float = TAIL_S) -> tuple[int, int]:
    """[start, end) frames from lead_s before the action first moves to tail_s after it last moves."""
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 2:
        return 0, len(a)
    joint = np.degrees(np.abs(np.diff(a[:, :6], axis=0))).max(1) * fps
    grip = np.abs(np.diff(a[:, 6])) * fps
    moving = np.flatnonzero((joint > MOVING_DEG_S) | (grip > MOVING_GRIP_M_S))
    if len(moving) == 0:
        return 0, len(a)
    first, last = moving[0], moving[-1] + 1  # diff i spans frames i -> i + 1
    return max(0, first - round(lead_s * fps)), min(len(a), last + 1 + round(tail_s * fps))


def pi05_frame(state7, end_pose6, action7, cfg, base_offset: float):
    """(state 13, action 7) in the policy's units, exactly as inference builds them."""
    state = real_to_sim(state7, cfg)
    state[0] += base_offset
    pose = rotate_end_pose(np.asarray(end_pose6, dtype=np.float64), base_offset)
    action = real_to_sim(action7, cfg)
    action[0] += base_offset
    return (np.concatenate([state, pose]).astype(np.float32), action.astype(np.float32))


def convert(src: str, dst: str, config: str, base_offset_deg: float, trim: bool = True, log=print) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists: convert into a new directory")
    cfg = load_config(config)
    off = float(np.radians(base_offset_deg))
    source = LeRobotDataset(f"local/{os.path.basename(os.path.normpath(src))}", root=src)
    image_shape = source.features[f"observation.images.{next(iter(CAMERA_MAP))}"]["shape"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (13,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }
    for theirs in CAMERA_MAP.values():
        features[f"observation.images.{theirs}"] = {
            "dtype": "video", "shape": tuple(image_shape), "names": ["height", "width", "channels"]}
    out = LeRobotDataset.create(repo_id=f"local/{os.path.basename(os.path.normpath(dst))}",
                                fps=source.fps, features=features, root=dst, robot_type="piper",
                                use_videos=True, image_writer_threads=4)
    episodes = np.asarray(source.hf_dataset["episode_index"])
    actions = np.stack([np.asarray(a) for a in source.hf_dataset["action"]])
    keep = np.zeros(len(episodes), dtype=bool)
    for ep in np.unique(episodes):
        idx = np.flatnonzero(episodes == ep)
        start, end = active_span(actions[idx], source.fps) if trim else (0, len(idx))
        keep[idx[start:end]] = True
        log(f"episode {ep}: keeping frames {start}..{end - 1} of {len(idx)} "
            f"({start / source.fps:.1f} s idle start, {(len(idx) - end) / source.fps:.1f} s idle end trimmed)")
    episode, n = None, 0
    for i in np.flatnonzero(keep):
        item = source[int(i)]
        ep = int(item["episode_index"])
        if episode is not None and ep != episode:
            out.save_episode()
            log(f"episode {episode}: {n} frames")
            n = 0
        episode = ep
        state, action = pi05_frame(item["observation.state"].numpy(), item["observation.end_pose"].numpy(),
                                   item["action"].numpy(), cfg, off)
        frame = {"observation.state": state, "action": action, "task": item["task"]}
        for ours, theirs in CAMERA_MAP.items():
            chw = item[f"observation.images.{ours}"].numpy()  # float 0..1, (3, H, W)
            frame[f"observation.images.{theirs}"] = (np.transpose(chw, (1, 2, 0)) * 255).round().astype(np.uint8)
        out.add_frame(frame)
        n += 1
    if episode is not None:
        out.save_episode()
        log(f"episode {episode}: {n} frames")
    out.finalize()
    log(f"wrote {dst}: {source.num_episodes} episodes, {int(keep.sum())} of {source.num_frames} frames")
    return source.num_episodes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="dataset recorded by teleop --record")
    ap.add_argument("dst", help="new directory for the converted dataset")
    ap.add_argument("--config", default="real_robot/configs/piper_1_banana.json",
                    help="config whose gripper ranges map metres -> 0..1 (as `make policy`)")
    ap.add_argument("--base-offset-deg", type=float, default=90.0,
                    help="added to J1 (as `make policy`'s --base-offset-deg)")
    ap.add_argument("--no-trim", action="store_true", help="keep idle frames at episode start and end")
    args = ap.parse_args(argv)
    convert(args.src, args.dst, args.config, args.base_offset_deg, trim=not args.no_trim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
