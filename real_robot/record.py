"""Record demonstrations for a pi0.5 fine-tune. Never commands the motors.

Hand-guide the arms (Piper teach/drag mode) or drive them from leader arms;
this only reads joints + cameras at the control rate. Per episode it writes

  <out>/episode_NNNN/<camera>.mp4         one per config camera_names
  <out>/episode_NNNN/data.npz   state (T,14), action (T,14), t (T,)
  <out>/episode_NNNN/meta.json  instruction, fps, frames, config used

action[t] = state[t+1] (absolute joint target, the openpi ALOHA convention);
the last frame repeats its state. Convert to LeRobot with a later script.

  python3 -m real_robot.record --backend piper --config real_robot/configs/piper_1.json \\
      --out demos/stack_cups --instruction "stack the cups" --seconds 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .backends import MockBackend, PiperBackend, RobotBackend
from .config import RobotConfig, load_config


def next_episode_dir(out: str) -> str:
    os.makedirs(out, exist_ok=True)
    taken = [int(d.split("_")[1]) for d in os.listdir(out)
             if d.startswith("episode_") and d.split("_")[1].isdigit()]
    path = os.path.join(out, f"episode_{(max(taken) + 1) if taken else 0:04d}")
    os.makedirs(path)
    return path


class VideoSink:
    """mp4 per camera via OpenCV; falls back to a .npy stack if cv2 is absent."""

    def __init__(self, path: str, fps: float, names):
        self.path, self.fps, self.names = path, fps, list(names)
        self.frames: dict[str, list] = {c: [] for c in self.names}
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
        self.writers: dict = {}

    def add(self, images: dict[str, np.ndarray]) -> None:
        for c in self.names:
            img = images[c]
            if self.cv2 is None:
                self.frames[c].append(img)
                continue
            if c not in self.writers:
                h, w = img.shape[:2]
                self.writers[c] = self.cv2.VideoWriter(
                    os.path.join(self.path, f"{c}.mp4"),
                    self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.writers[c].write(self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        for w in self.writers.values():
            w.release()
        if self.cv2 is None:
            for c, frames in self.frames.items():
                if frames:
                    np.save(os.path.join(self.path, f"{c}.npy"), np.stack(frames))


def record_episode(robot: RobotBackend, cfg: RobotConfig, out: str, instruction: str,
                   seconds: float, sleep=time.sleep) -> str:
    ep = next_episode_dir(out)
    period = 1.0 / cfg.control_hz
    n = int(round(seconds * cfg.control_hz))
    video = VideoSink(ep, cfg.control_hz, cfg.camera_names)
    states, ts = [], []
    late = 0
    try:
        for _ in range(n):
            tick = time.perf_counter()
            q = robot.get_joint_state()
            if not np.all(np.isfinite(q)):
                raise RuntimeError(f"non-finite joint state {q}")
            video.add(robot.get_images())
            states.append(q)
            ts.append(time.time())
            spare = period - (time.perf_counter() - tick)
            if spare < 0:
                late += 1
            else:
                sleep(spare)
    except KeyboardInterrupt:
        pass  # keep what was recorded
    finally:
        video.close()

    state = np.asarray(states, dtype=np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0) if len(state) else state
    np.savez_compressed(os.path.join(ep, "data.npz"), state=state, action=action,
                        t=np.asarray(ts))
    meta = {"instruction": instruction, "fps": cfg.control_hz, "frames": len(state),
            "late_frames": late, "arms": list(cfg.arms), "cameras": cfg.camera_names,
            "state_layout": "per arm: 6 joints rad, gripper m",
            "action": "next-frame absolute joint state", "config_verified": cfg.verified}
    with open(os.path.join(ep, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "piper"])
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    robot = MockBackend(cfg) if args.backend == "mock" else PiperBackend(cfg, passive=True)
    robot.connect()
    try:
        for i in range(args.episodes):
            input(f"[{i + 1}/{args.episodes}] reset the scene, press Enter to record "
                  f"{args.seconds:.0f}s (Ctrl-C ends early)...")
            ep = record_episode(robot, cfg, args.out, args.instruction, args.seconds)
            meta = json.load(open(os.path.join(ep, "meta.json")))
            print(f"saved {ep}: {meta['frames']} frames, {meta['late_frames']} late")
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
