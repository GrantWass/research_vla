"""convert_dataset.pi05_frame: recorded raw units -> the pi0.5 Piper policy's units (no hardware)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np

    from real_robot.config import load_config
    from real_robot.convert_dataset import active_span, pi05_frame
    from real_robot.lerobot_policy import rotate_end_pose
    from real_robot.mapping import real_to_sim, sim_to_real
except ImportError:  # pragma: no cover
    np = None

CFG = os.path.join(os.path.dirname(__file__), "..", "real_robot", "configs", "piper_1_banana.json")


@unittest.skipIf(np is None, "numpy not installed")
class TestPi05Frame(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CFG)
        self.off = np.pi / 2  # make policy: --base-offset-deg 90
        self.q = np.r_[np.radians([-93.3, 30, -40, 5, 30, 10]), 0.035]
        self.pose = np.array([-0.013, -0.224, 0.326, 3.1, 1.3, 1.5])
        self.act = np.r_[np.radians([-90.0, 35, -45, 0, 25, 0]), 0.07]

    def test_state_matches_inference(self):
        state, _ = pi05_frame(self.q, self.pose, self.act, self.cfg, self.off)
        # what LerobotPolicy._batch builds from the runner's real_to_sim state
        want = real_to_sim(self.q, self.cfg)
        want[0] += self.off
        want = np.concatenate([want, rotate_end_pose(self.pose, self.off)])
        np.testing.assert_allclose(state, want, rtol=1e-6, atol=1e-6)
        self.assertEqual(state.shape, (13,))
        self.assertAlmostEqual(np.degrees(state[0]), -3.3, places=3)  # base as the training rig saw it
        self.assertAlmostEqual(state[6], 0.5, places=6)                 # 35 mm of the 70 mm stroke

    def test_action_maps_back_to_the_leader_command(self):
        _, action = pi05_frame(self.q, self.pose, self.act, self.cfg, self.off)
        self.assertEqual(action.shape, (7,))
        self.assertAlmostEqual(action[6], 1.0, places=6)
        executed = action.astype(np.float64).copy()
        executed[0] -= self.off  # LerobotPolicy.get_action, then the runner's sim_to_real
        np.testing.assert_allclose(sim_to_real(executed, self.cfg), self.act, atol=1e-6)


@unittest.skipIf(np is None, "numpy not installed")
class TestActiveSpan(unittest.TestCase):
    def episode(self, idle_start, move, idle_end, fps=30):
        """Leader still, then J1 sweeping 10 deg/s, then still again."""
        a = np.zeros((idle_start + move + idle_end, 7))
        a[idle_start:idle_start + move, 0] = np.radians(10.0) * np.arange(1, move + 1) / fps
        a[idle_start + move:, 0] = a[idle_start + move - 1, 0]
        return a

    def test_trims_idle_start_and_end_with_margins(self):
        a = self.episode(150, 300, 90)  # 5 s idle, 10 s moving, 3 s idle
        start, end = active_span(a, 30)
        self.assertEqual(start, 150 - 1 - 15)  # 0.5 s before the first moving step
        self.assertEqual(end, 450 + 30)        # 1 s after the last one

    def test_gripper_only_motion_counts(self):
        a = np.zeros((300, 7))
        a[100:130, 6] = np.linspace(0, 0.05, 30)
        a[130:, 6] = 0.05
        start, end = active_span(a, 30)
        self.assertLess(start, 100)
        self.assertGreater(end, 130)

    def test_no_motion_keeps_everything(self):
        self.assertEqual(active_span(np.zeros((60, 7)), 30), (0, 60))

    def test_short_episode_is_clamped(self):
        start, end = active_span(self.episode(2, 20, 2), 30)
        self.assertEqual((start, end), (0, 24))


if __name__ == "__main__":
    unittest.main()
