"""Record teleop demonstrations into a LeRobot dataset (teleop's on_step hook).

Each frame: the follower's joints + gripper (observation.state, rad / m), its
end pose (observation.end_pose, m / rad), the leader's target clipped to the
limits (action, same units as the state) and every config camera as video.
Episodes alternate: record `episode_s`, then save in the background while the
operator resets the scene for `reset_s`. A partial episode (Ctrl-C, abort) is
discarded.

Needs the LeRobot venv (requirements/lerobot.txt); see docs/robot.md.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .config import ARM_DOF, IMAGE_HW, RobotConfig

JOINT_NAMES = [f"joint_{i}" for i in range(1, ARM_DOF + 1)] + ["gripper"]
POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]


def dataset_features(cfg: RobotConfig) -> dict:
    """LeRobot feature spec for one Piper arm and the config's cameras."""
    if len(cfg.arms) != 1:
        raise ValueError("demo recording expects a single-arm config")
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
        "observation.end_pose": {"dtype": "float32", "shape": (len(POSE_NAMES),), "names": POSE_NAMES},
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    }
    for cam in cfg.camera_names:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channels"]}
    return feats


def create_dataset(root: str, cfg: RobotConfig, fps: int):
    """A new LeRobot dataset at root (must not exist yet)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: LeRobot venv only

    if os.path.exists(root):
        raise SystemExit(f"{root} exists: record each session into a new directory")
    return LeRobotDataset.create(repo_id=f"local/{os.path.basename(os.path.normpath(root))}",
                                 fps=fps, features=dataset_features(cfg), root=root,
                                 robot_type="piper", use_videos=True, image_writer_threads=4)


class DemoRecorder:
    """Callable for teleop(on_step=...): returns True once all episodes are saved."""

    def __init__(self, dataset, robot, cfg: RobotConfig, task: str, *, fps: int,
                 episodes: int, episode_s: float, reset_s: float,
                 clock=time.monotonic, log=print):
        self.ds, self.robot, self.cfg, self.task = dataset, robot, cfg, task
        self.period, self.episodes = 1.0 / fps, episodes
        self.episode_s, self.reset_s = episode_s, reset_s
        self.clock, self.log = clock, log
        self.saved = 0
        self.recording = False
        self.next_frame = self.phase_end = clock() + reset_s  # first: get ready
        self.saver: threading.Thread | None = None
        self.save_error: Exception | None = None
        log(f"recording {episodes} x {episode_s:.0f} s of '{task}'; first episode in {reset_s:.0f} s")

    def __call__(self, state, cmd, lead) -> bool:
        if self.save_error is not None:
            raise RuntimeError(f"saving an episode failed: {self.save_error}")
        now = self.clock()
        if not self.recording:
            if self.saved >= self.episodes:
                return not self._saving()
            if now >= self.phase_end and not self._saving():
                self.recording, self.phase_end, self.next_frame = True, now + self.episode_s, now
                self.log(f"episode {self.saved + 1}/{self.episodes}: RECORDING")
            return False
        if now >= self.phase_end:
            self.recording = False
            self.phase_end = now + self.reset_s
            self._save_async()
            if self.saved + 1 < self.episodes:
                self.log(f"episode {self.saved + 1} done; reset the scene ({self.reset_s:.0f} s)")
            return False
        if now >= self.next_frame:
            self.next_frame += self.period
            target = np.clip(lead, self.cfg.joint_lower, self.cfg.joint_upper)
            frame = {"observation.state": np.asarray(state, dtype=np.float32),
                     "observation.end_pose": np.asarray(self.robot.get_end_pose(), dtype=np.float32),
                     "action": np.asarray(target, dtype=np.float32),
                     "task": self.task}
            for cam, img in self.robot.get_images().items():
                frame[f"observation.images.{cam}"] = img
            self.ds.add_frame(frame)
        return False

    def _saving(self) -> bool:
        return self.saver is not None and self.saver.is_alive()

    def _save_async(self) -> None:
        def save():
            try:
                self.ds.save_episode()
                self.saved += 1
                self.log(f"episode {self.saved} saved")
            except Exception as e:  # noqa: BLE001  surfaced on the next step
                self.save_error = e
        self.saver = threading.Thread(target=save, daemon=False)
        self.saver.start()

    def close(self) -> int:
        """Discard a partial episode, wait for saving, finalize. Returns episodes saved."""
        if self.recording:
            self.ds.clear_episode_buffer()
            self.log("partial episode discarded")
        if self.saver is not None:
            self.saver.join()
        self.ds.finalize()
        return self.saved
