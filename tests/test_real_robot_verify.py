"""verify.py robot stage against a mock robot."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REPO = os.path.join(os.path.dirname(__file__), "..")

if np is not None:
    from real_robot.backends import MockBackend
    from real_robot.config import load_config
    from real_robot.verify import Report, stage_robot


@unittest.skipIf(np is None, "numpy not installed")
class TestVerify(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(os.path.join(REPO, "real_robot/configs/piper_1.json"))

    def test_robot_stage(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Report(d)
            robot = MockBackend(self.cfg)
            snap = stage_robot(rep, robot, self.cfg, frames=3)
            self.assertFalse(rep.failed("robot"))
            self.assertTrue(os.path.exists(os.path.join(d, "snapshot.npz")))
            self.assertEqual(snap["state"].shape, (7,))
            self.assertIn("raw_cam_wrist.png", os.listdir(d))
            rep.save()
            self.assertTrue(os.path.exists(os.path.join(d, "report.json")))
            self.assertEqual(robot.commands, [])

    def test_wiggle_declined_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            robot = MockBackend(self.cfg)
            stage_robot(Report(d), robot, self.cfg, wiggle=True, frames=1,
                        confirm=lambda _: "no")
            self.assertEqual(robot.commands, [])

    def test_out_of_limits_fails(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Report(d)
            robot = MockBackend(self.cfg)
            robot.q[0] = 3.0  # beyond J1 limit 2.618
            stage_robot(rep, robot, self.cfg, frames=1)
            self.assertTrue(rep.failed("robot"))


if __name__ == "__main__":
    unittest.main()
