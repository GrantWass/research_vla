"""Hardware backends. All values here are in REAL robot units."""
from __future__ import annotations

import abc

import numpy as np

from .config import IMAGE_HW, RobotConfig


class RobotBackend(abc.ABC):
    """Minimal interface the runner needs. Implement one per robot."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """(7 * n_arms,) [arm 6, grip] per arm in cfg.arms order, real units."""

    @abc.abstractmethod
    def get_images(self) -> dict[str, np.ndarray]:
        """cfg.camera_names -> HxWx3 uint8 RGB."""

    @abc.abstractmethod
    def command_joints(self, q: np.ndarray) -> None:
        """Send one position target (already safety-filtered)."""

    @abc.abstractmethod
    def hold(self) -> None:
        """Freeze at the current position. Must be safe to call any time."""

    def close(self) -> None:
        pass


class MockBackend(RobotBackend):
    """Software robot: tracks commands with first-order lag, random images."""

    def __init__(self, cfg: RobotConfig, seed: int = 0, lag: float = 0.5):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.lag = lag
        self.q = np.zeros(cfg.state_dim)
        self.commands: list[np.ndarray] = []
        self.held = 0

    def connect(self) -> None:
        pass

    def get_joint_state(self) -> np.ndarray:
        return self.q.copy()

    def get_images(self) -> dict[str, np.ndarray]:
        return {c: self.rng.integers(0, 256, (*IMAGE_HW, 3), dtype=np.uint8)
                for c in self.cfg.camera_names}

    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        self.commands.append(q.copy())
        self.q = self.q + self.lag * (q - self.q)

    def hold(self) -> None:
        self.held += 1


def make_backend(name: str, cfg: RobotConfig) -> RobotBackend:
    if name == "mock":
        return MockBackend(cfg)
    if name == "piper":
        from .piper import PiperBackend  # lazy: piper_sdk lives on the robot PC

        return PiperBackend(cfg)
    raise ValueError(f"unknown backend {name!r}")
