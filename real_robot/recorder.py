"""Record teleop demonstrations into a LeRobot dataset (teleop's on_step hook).

Each frame: the follower's joints + gripper (observation.state, rad / m), its
end pose (observation.end_pose, m / rad), the leader's target clipped to the
limits (action, same units as the state) and every config camera as video.
Two ways to run episodes:
  KeyedRecorder (default for `make record`, driven by teleop.keyed_session):
    the arms stay parked, motors off, between episodes. Space powers them
    up and records from the first follow step; Enter ends and saves the
    episode, r discards it, q ends the session; each parks the arms. An
    episode reaching max_episode_s is discarded.
  DemoRecorder (--episode-seconds): fixed-length episodes alternating with
    `reset_s` pauses.
Episodes are saved in the background (in this process, so Ctrl-C can't kill
the video encoding); the session waits for a save before it ends. A partial
episode (q, Ctrl-C, abort) is discarded.

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


# Frames the streaming encoder may queue per camera: more than a whole episode
# (KeyedRecorder max_episode_s 180 s x 30 fps), so it never drops one.
STREAM_QUEUE_FRAMES = 6000


def create_dataset(root: str, cfg: RobotConfig, fps: int):
    """A new LeRobot dataset at root (must not exist yet).

    Videos are AV1-encoded while recording (LeRobot's streaming encoder, one
    thread per camera) instead of from PNGs after the episode ends, so saving
    no longer waits ~0.8 s per second of episode. Same encoder (libsvtav1),
    settings and frames as the PNG path; StreamingGuard keeps the image stats
    and the no-dropped-frames guarantee identical (2026-10-16: two live episodes
    recorded through both paths gave byte-identical videos and identical data).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: LeRobot venv only

    if os.path.exists(root):
        raise SystemExit(f"{root} exists: record each session into a new directory")
    ds = LeRobotDataset.create(repo_id=f"local/{os.path.basename(os.path.normpath(root))}",
                               fps=fps, features=dataset_features(cfg), root=root,
                               robot_type="piper", use_videos=True, image_writer_threads=4,
                               streaming_encoding=True, encoder_queue_maxsize=STREAM_QUEUE_FRAMES)
    StreamingGuard(ds)
    return ds


class StreamingGuard:
    """Makes a streaming-encoded dataset match the PNG path exactly.

    - Image stats: LeRobot's streaming encoder computes them over every frame;
      the PNG path (compute_episode_stats) over sample_indices() frames,
      downsampled. Here a downsampled copy of each frame is kept (~60 kB) and
      the stats are computed the PNG path's way from the same samples.
    - Dropped frames: when its queue is full LeRobot drops a frame and only
      warns, which would leave the video shorter than the joint data. The
      queue holds a whole episode; if a frame was dropped anyway, saving fails.
    - Stats order: the streaming path adds the camera stats after the others,
      which reorders the per-episode stats columns in meta/episodes; they are
      put back in the PNG path's order (the episode buffer's).
    """

    def __init__(self, ds):
        from lerobot.datasets import compute_stats

        self.ds, self.cs = ds, compute_stats
        self.keys = list(ds.meta.video_keys)
        self.small: dict[str, list] = {k: [] for k in self.keys}
        enc = ds._streaming_encoder
        self._add, self._clear, self._finish = ds.add_frame, ds.clear_episode_buffer, enc.finish_episode
        ds.add_frame, ds.clear_episode_buffer = self.add_frame, self.clear_episode_buffer
        enc.finish_episode = self.finish_episode
        self._save_meta = ds.meta.save_episode
        ds.meta.save_episode = self.save_episode_meta

    def save_episode_meta(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
        order = [k for k in self.ds.episode_buffer if k in episode_stats]
        order += [k for k in episode_stats if k not in order]
        return self._save_meta(episode_index, episode_length, episode_tasks,
                               {k: episode_stats[k] for k in order}, episode_metadata)

    def _reset(self) -> None:
        self.small = {k: [] for k in self.keys}

    def add_frame(self, frame: dict) -> None:
        if self.ds.episode_buffer is None or self.ds.episode_buffer["size"] == 0:
            self._reset()  # first frame of an episode
        for k in self.keys:
            chw = np.asarray(frame[k], dtype=np.uint8).transpose(2, 0, 1)
            self.small[k].append(np.ascontiguousarray(self.cs.auto_downsample_height_width(chw)))
        self._add(frame)

    def clear_episode_buffer(self, *args, **kwargs):
        self._reset()
        return self._clear(*args, **kwargs)

    def finish_episode(self) -> dict:
        dropped = {k: n for k, n in self.ds._streaming_encoder._dropped_frames.items() if n}
        results = self._finish()
        if dropped:
            raise RuntimeError(f"video encoder dropped frames {dropped}: episode not saved")
        out = {}
        for k, (path, _) in results.items():
            frames = self.small[k]
            imgs = np.stack([frames[i] for i in self.cs.sample_indices(len(frames))])
            out[k] = (path, self.cs.get_feature_stats(imgs, axis=(0, 2, 3), keepdims=True,
                                                      quantile_list=self.cs.DEFAULT_QUANTILES))
        self._reset()
        return out


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
            raise RuntimeError(f"saving an episode failed: {self.save_error!r}")
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
            self._add_frame(state, lead)
        return False

    def _add_frame(self, state, lead) -> None:
        target = np.clip(lead, self.cfg.joint_lower, self.cfg.joint_upper)
        frame = {"observation.state": np.asarray(state, dtype=np.float32),
                 "observation.end_pose": np.asarray(self.robot.get_end_pose(), dtype=np.float32),
                 "action": np.asarray(target, dtype=np.float32),
                 "task": self.task}
        for cam, img in self.robot.get_images().items():
            frame[f"observation.images.{cam}"] = img
        self.ds.add_frame(frame)
        self.frames_in_episode = getattr(self, "frames_in_episode", 0) + 1

    def _saving(self) -> bool:
        return self.saver is not None and self.saver.is_alive()

    def _save_async(self) -> None:
        def save():
            try:
                # Encode in this process. LeRobot's parallel encoding uses child
                # processes, which Ctrl-C (sent to the whole process group) kills
                # mid-video: the episode was lost (2026-10-14). In-process it
                # takes ~0.8 s per second of episode and leaves the 50 Hz
                # teleop loop on time (measured: one 100 ms step in 40 s).
                self.ds.save_episode(parallel_encoding=False)
                self.saved += 1
                self.log(f"episode {self.saved} saved")
            except BaseException as e:  # noqa: BLE001  surfaced on the next step and in close()
                self.save_error = e
                self.log(f"SAVING EPISODE {self.saved + 1} FAILED: {e!r}")
        self.saver = threading.Thread(target=save, daemon=False)
        self.saver.start()

    def _wait_for_save(self) -> None:
        """Wait for a background save, through Ctrl-C (an interrupted Thread.join
        can return early on Python 3.10, and the exiting process then kills the save)."""
        told = False
        while self._saving():
            try:
                if not told:
                    self.log("waiting for the episode to finish saving (Ctrl-C won't stop it)")
                    told = True
                time.sleep(0.2)
            except KeyboardInterrupt:
                told = False

    def close(self) -> int:
        """Discard a partial episode, wait for saving, finalize. Returns episodes saved."""
        if self.recording:
            self.ds.clear_episode_buffer()
            self.log("partial episode discarded")
        self._wait_for_save()
        if self.save_error is not None:
            self.log(f"an episode was NOT saved ({self.save_error!r}); its frames stay in the "
                     f"dataset's images/ folder")
        self.ds.finalize()
        return self.saved


KEY_HELP = ("Space = power up both arms and record an episode | Enter = end + save | "
            "r = discard | q = end the session")


class KeyedRecorder(DemoRecorder):
    """One operator-driven episode per teleop() call; keys from key_source()
    (e.g. KeyReader.get).

    begin() starts recording (call it right before teleop(): the first frame
    is the first follow step). Enter ends and saves the episode, r discards
    it, q discards it and ends the session; an episode reaching
    max_episode_s is discarded (likely a missed key). Any of these ends the
    episode: the hook returns True and teleop() returns, so the caller can
    park the arms. Space (power up, next episode) is the caller's: see
    teleop.keyed_session. episodes > 0: done() once that many are saved.
    """

    def __init__(self, dataset, robot, cfg: RobotConfig, task: str, *, fps: int,
                 key_source, episodes: int = 0, max_episode_s: float = 180.0,
                 clock=time.monotonic, log=print):
        self.ds, self.robot, self.cfg, self.task = dataset, robot, cfg, task
        self.period, self.episodes, self.max_episode_s = 1.0 / fps, episodes, max_episode_s
        self.key_source, self.clock, self.log = key_source, clock, log
        self.saved = self.discarded = self.queued = 0
        self.recording = self.quit = False
        self.started = self.next_frame = 0.0
        self.frames_in_episode = 0
        self.saver: threading.Thread | None = None
        self.save_error: Exception | None = None

    def begin(self) -> None:
        if self._saving():
            raise RuntimeError("begin() while the last episode is still saving")
        while self.key_source() is not None:  # keys pressed before following
            pass
        now = self.clock()
        self.recording, self.started, self.next_frame = True, now, now
        self.frames_in_episode = 0
        self.log(f"episode {self.queued + 1}: RECORDING (Enter = save, r = discard, q = quit)")

    def done(self) -> bool:
        return self.quit or bool(self.episodes and self.queued >= self.episodes)

    def abandon(self, why: str) -> None:
        """teleop() stopped on its own (error, Ctrl-C): drop the episode."""
        if self.recording:
            self._discard(why)

    def _discard(self, why: str) -> None:
        self.ds.clear_episode_buffer()
        self.recording = False
        self.discarded += 1
        self.log(f"episode discarded ({why})")

    def _end(self, now) -> None:
        self.recording = False
        if self.frames_in_episode == 0:
            self.discarded += 1
            self.log("no frames recorded; nothing to save")
            return
        took = now - self.started
        self.queued += 1
        self.log(f"episode {self.queued}: saving ({self.frames_in_episode} frames, {took:.1f} s); "
                 f"takes ~{0.8 * took + 3:.0f} s, while the arms park")
        self._save_async()

    def __call__(self, state, cmd, lead) -> bool:
        if self.save_error is not None:
            raise RuntimeError(f"saving an episode failed: {self.save_error!r}")
        if not self.recording:
            return True
        now = self.clock()
        while self.recording:  # stop reading at the episode's end: later keys are the session's
            key = self.key_source()
            if key is None:
                break
            if key in ("\n", "\r"):
                self._end(now)
            elif key in ("r", "R"):
                self._discard("r pressed")
            elif key in ("q", "Q"):
                self._discard("q pressed during the episode")
                self.quit = True
        if not self.recording:
            return True
        if now - self.started >= self.max_episode_s:
            self._discard(f"reached {self.max_episode_s:.0f} s without Enter")
            return True
        if now >= self.next_frame:
            self.next_frame += self.period
            self._add_frame(state, lead)
        return False

    def wait_for_save(self) -> None:
        self._wait_for_save()
        if self.save_error is not None:
            raise RuntimeError(f"saving an episode failed: {self.save_error!r}")

    def close(self) -> int:
        if self.recording:
            self.recording = False
            self.ds.clear_episode_buffer()
            self.log("episode in progress discarded")
        saved = super().close()
        self.log(f"{saved} episode(s) saved, {self.discarded} discarded")
        return saved
