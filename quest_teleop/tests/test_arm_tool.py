import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import arm_tool  # noqa: E402


class FakeArm:
    def __init__(self, q):
        self.q, self.calls = np.asarray(q, dtype=float), []

    def get_joint_angles(self):
        return type("M", (), {"msg": self.q.tolist()})()

    def enable(self):
        self.calls.append("enable")
        return True

    def move_j(self, q):
        self.q = np.asarray(q)
        self.calls.append("move_j")

    def electronic_emergency_stop(self):
        self.calls.append("estop")

    def disable(self):
        self.calls.append("disable")

    def reset(self):
        self.calls.append("reset")


class ArmToolTest(unittest.TestCase):
    def setUp(self):
        self.cfg = arm_tool.load()
        self.nosleep = lambda s: None

    def test_config_has_every_key_the_nodes_read(self):
        for k in ("can_port", "joint_lower", "joint_upper", "max_step_delta", "control_hz",
                  "ready_deg", "home_base_deg", "park_fold_deg", "tool_speed_percent"):
            self.assertIn(k, self.cfg)
        self.assertEqual(len(self.cfg["joint_lower"]), 7)
        self.assertEqual(len(self.cfg["park_fold_deg"]), 5)

    def test_ready_pose_is_inside_the_limits(self):
        arm_tool.check_target(self.cfg, np.radians(self.cfg["ready_deg"]))

    def test_park_folds_with_base_held_then_turns_then_stops_gently(self):
        arm = FakeArm(np.radians([20, 60, -80, 10, 40, 5]))
        arm_tool.park(arm, self.cfg, sleep=self.nosleep, log=lambda *a: None)
        calls = arm.calls
        self.assertEqual(calls[0], "enable")
        self.assertEqual(calls[-3:], ["estop", "disable", "reset"])
        folded, home = arm_tool.fold_targets(self.cfg, np.radians([20, 60, -80, 10, 40, 5]))
        self.assertAlmostEqual(np.degrees(folded[0]), 20.0)  # base kept while folding
        self.assertAlmostEqual(np.degrees(home[0]), self.cfg["home_base_deg"])
        np.testing.assert_allclose(arm.q, home, atol=1e-9)
        np.testing.assert_allclose(np.degrees(home[1:]), self.cfg["park_fold_deg"])

    def test_waits_for_live_feedback_and_gives_up_without_it(self):
        class Live:
            n = 0

            def get_joint_angles(self):
                Live.n += 1
                return type("M", (), {"timestamp": Live.n * 0.01, "msg": [0] * 6})()

        arm_tool.wait_for_feedback(Live(), sleep=self.nosleep)

        class Frozen:
            def get_joint_angles(self):
                return type("M", (), {"timestamp": 1.0, "msg": [0] * 6})()

        t = iter(range(100))
        with self.assertRaises(RuntimeError):
            arm_tool.wait_for_feedback(Frozen(), timeout_s=3, sleep=self.nosleep, clock=lambda: next(t))

    def test_target_outside_limits_is_refused(self):
        with self.assertRaises(ValueError):
            arm_tool.check_target(self.cfg, np.radians([0, -10, 0, 0, 0, 0]))  # J2 < 0


if __name__ == "__main__":
    unittest.main()
