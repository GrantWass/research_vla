"""SmoothStreamer: critically damped command streaming (no hardware)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np

    from real_robot.config import RobotConfig
    from real_robot.backends import MockBackend
    from real_robot.runner import run
    from real_robot.smoothing import SmoothStreamer
except ImportError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class TestSmoothStreamer(unittest.TestCase):
    def make(self, cutoff=2.0, lo=-10.0, hi=10.0):
        s = SmoothStreamer(lambda y: None, [lo] * 7, [hi] * 7, cutoff_hz=cutoff, rate_hz=100.0)
        s.reset(np.zeros(7))
        return s

    def test_step_response_is_smooth_without_overshoot(self):
        s = self.make()
        s.set_target(np.full(7, 1.0))
        ys = np.array([s.step()[0] for _ in range(300)])  # 3 s
        self.assertLess(ys.max(), 1.0 + 1e-9)          # critically damped: no overshoot
        self.assertGreater(ys[-1], 0.99)                # and it gets there
        self.assertTrue(np.all(np.diff(ys) >= -1e-12))  # monotone
        v = np.diff(ys)
        w, dt = 2 * np.pi * 2.0, 0.01
        # velocity changes by at most w^2 dt^2 per unit of step: no velocity jumps
        self.assertLessEqual(np.abs(np.diff(v)).max(), (w * dt) ** 2 + 1e-12)

    def test_damps_plan_to_plan_pulsing(self):
        s = self.make()
        swing = []
        for i in range(400):  # target alternates +-0.1 at 3 Hz (plan switches)
            s.set_target(np.full(7, 0.1 if (i // 17) % 2 else -0.1))
            swing.append(s.step()[0])
        swing = np.array(swing[200:])
        self.assertLess(swing.max() - swing.min(), 0.2 * 0.5)  # under half the commanded swing

    def test_clips_to_limits(self):
        s = self.make(lo=-0.5, hi=0.5)
        s.set_target(np.full(7, 3.0))
        ys = np.array([s.step() for _ in range(300)])
        self.assertLessEqual(ys.max(), 0.5)

    def test_runner_streams_and_holds(self):
        cfg = RobotConfig(verified=True, control_hz=100.0)
        robot = MockBackend(cfg, lag=1.0)
        drift = lambda st, n: [st + 0.002 * (i + 1) for i in range(n)]  # noqa: E731

        class P:
            obs = None

            def call(self, func_name=None, obs=None):
                if func_name == "update_obs":
                    self.obs = obs
                if func_name == "get_action":
                    return drift(np.asarray(self.obs["state"], dtype=np.float64), 10)

        s = run(robot, P(), cfg, instruction="x", live=True, speed=1.0, max_steps=40,
                execute_steps=5, smooth_hz=2.0, stream_hz=200.0)
        self.assertEqual(s["status"], "ok")
        self.assertGreater(s["streamed_commands"], 40)  # streamed faster than the 40 steps
        self.assertEqual(robot.held, 1)
        sent = np.asarray(robot.commands)[:, 0]
        self.assertLess(np.abs(np.diff(sent, 2)).max(), 1e-3)  # smooth: tiny accel per command


if __name__ == "__main__":
    unittest.main()
