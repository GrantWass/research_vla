"""Last line of defence between policy output and motor commands (real units)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RobotConfig


class SafetyAbort(RuntimeError):
    """Raised when a command is unsafe to send at all; caller must hold."""


@dataclass
class SafetyStats:
    steps: int = 0
    limit_clips: int = 0
    rate_clips: int = 0
    jump_clamps: int = 0
    max_requested_delta: float = 0.0
    events: list[str] = field(default_factory=list)


class SafetyFilter:
    def __init__(self, cfg: RobotConfig):
        self.lo = np.asarray(cfg.joint_lower, dtype=np.float64)
        self.hi = np.asarray(cfg.joint_upper, dtype=np.float64)
        self.max_delta = np.asarray(cfg.max_step_delta, dtype=np.float64)
        self.abort_delta = float(cfg.abort_delta)
        self.clamp_jumps = bool(getattr(cfg, "clamp_jumps", False))
        self.max_tracking_error = float(cfg.max_tracking_error)
        self.joints = np.setdiff1d(np.arange(cfg.state_dim), cfg.gripper_idx)
        self.dim = cfg.state_dim
        self.stats = SafetyStats()

    def check_state(self, q) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.dim,) or not np.all(np.isfinite(q)):
            raise SafetyAbort(f"bad robot state {q!r}")
        margin = 0.05
        out = np.where((q < self.lo - margin) | (q > self.hi + margin))[0]
        if out.size:
            raise SafetyAbort(f"robot already outside limits on joints {out.tolist()}")

    def filter(self, target, current, prev_cmd=None) -> np.ndarray:
        """Return the command to send this step, or raise SafetyAbort.

        Without prev_cmd, the jump check and rate limit start from the measured
        joints (current), so the command never leads the arm by more than
        max_step_delta. That caps a policy run at ~18 deg/s (2026-10-28). With
        prev_cmd (the last command sent), they start from it, so commands can
        move at max_step_delta per step; the tracking check then stops the run
        if the command gets max_tracking_error ahead of the measured joints."""
        target = np.asarray(target, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        if target.shape != (self.dim,) or not np.all(np.isfinite(target)):
            raise SafetyAbort(f"bad policy action {target!r}")
        self.stats.steps += 1
        base = current if prev_cmd is None else np.asarray(prev_cmd, dtype=np.float64)

        delta = target - base
        worst = float(np.max(np.abs(delta)))
        self.stats.max_requested_delta = max(self.stats.max_requested_delta, worst)
        if worst > self.abort_delta:
            j = int(np.argmax(np.abs(delta)))
            if self.clamp_jumps:
                self.stats.jump_clamps += 1
                if self.stats.jump_clamps <= 20:
                    self.stats.events.append(
                        f"step {self.stats.steps}: clamped joint {j} jump {delta[j]:+.3f}")
            else:
                raise SafetyAbort(
                    f"policy asked joint {j} to jump {delta[j]:+.3f} in one step "
                    f"(abort_delta={self.abort_delta}); check joint order/units/sign")

        clipped = np.clip(target, self.lo, self.hi)
        if not np.array_equal(clipped, target):
            self.stats.limit_clips += 1

        step = np.clip(clipped - base, -self.max_delta, self.max_delta)
        if not np.allclose(step, clipped - base):
            self.stats.rate_clips += 1
        cmd = base + step
        if prev_cmd is not None:
            lead = np.abs(cmd - current)[self.joints]
            if lead.max() > self.max_tracking_error:
                j = int(self.joints[np.argmax(lead)])
                raise SafetyAbort(
                    f"joint {j} is {lead.max():.3f} rad behind its command "
                    f"(max_tracking_error={self.max_tracking_error}); blocked or not following?")
        return cmd
