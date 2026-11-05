"""Smooth command streaming between the policy and the arm.

The runner produces one joint target per control step (15 Hz at --speed 0.5).
Sent as is, the Piper firmware moves to each target and slows down as it
arrives, so the arm stutters at the command rate, and each new plan adds a
pulse (~3 Hz on the live banana runs, 2026-10-07). SmoothStreamer instead
sends commands at rate_hz from a background thread, each following the latest
target through a critically damped second-order filter: position and velocity
stay continuous, and changes faster than cutoff_hz are damped. It adds
~1 / (pi * cutoff_hz) s of lag (0.16 s at 2 Hz).
"""
from __future__ import annotations

import threading
import time

import numpy as np


class SmoothStreamer:
    def __init__(self, send, lower, upper, cutoff_hz: float = 2.0, rate_hz: float = 100.0,
                 sleep=time.sleep, clock=time.monotonic):
        if cutoff_hz <= 0 or rate_hz <= 0:
            raise ValueError("cutoff_hz and rate_hz must be positive")
        self.send = send
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.w = 2 * np.pi * cutoff_hz
        self.dt = 1.0 / rate_hz
        self.sleep, self.clock = sleep, clock
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error: BaseException | None = None
        self.sent = 0

    def reset(self, q0) -> None:
        """Start from q0 at rest; the target is q0 until set_target()."""
        q0 = np.asarray(q0, dtype=np.float64)
        with self.lock:
            self.y = q0.copy()
            self.v = np.zeros_like(q0)
            self.x = q0.copy()

    def set_target(self, x) -> None:
        with self.lock:
            self.x = np.asarray(x, dtype=np.float64).copy()

    def step(self) -> np.ndarray:
        """Advance the filter by one dt; returns the command to send."""
        with self.lock:
            a = self.w ** 2 * (self.x - self.y) - 2 * self.w * self.v
            self.v = self.v + a * self.dt  # semi-implicit Euler: stable for w * dt << 1
            y = self.y + self.v * self.dt
            clipped = np.clip(y, self.lower, self.upper)
            self.v[clipped != y] = 0.0
            self.y = clipped
            return clipped.copy()

    def start(self, q0) -> None:
        self.reset(q0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        next_tick = self.clock()
        try:
            while not self.stop_event.is_set():
                self.send(self.step())
                self.sent += 1
                next_tick += self.dt
                self.sleep(max(0.0, next_tick - self.clock()))
        except BaseException as e:  # noqa: BLE001  surfaced by check()
            self.error = e

    def check(self) -> None:
        """Re-raise a failure from the streaming thread (e.g. an SDK error)."""
        if self.error is not None:
            raise RuntimeError(f"command streamer failed: {self.error!r}") from self.error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class OneEuroFilter:
    """Speed-adaptive low-pass (Casiez et al., CHI 2012), per element of a vector.

    The cutoff rises with the filtered speed: cutoff = min_cutoff + beta * |v|.
    Held still, a target is smoothed at min_cutoff (removes jitter); moving
    fast, the cutoff goes up and the lag goes down. Samples may arrive at an
    irregular rate (dt from their timestamps). Used on the Quest IK targets
    (real_robot/quest/ik_guard.py), ahead of SmoothStreamer.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be positive, beta >= 0")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or t <= self.t:
            if self.x is None:
                self.x, self.dx = x.copy(), np.zeros_like(x)
            self.t = t if self.t is None else self.t
            return self.x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x.copy()
