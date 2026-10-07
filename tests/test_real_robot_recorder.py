"""Demo recording during teleop, against a fake LeRobot dataset and clock."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from real_robot.backends import MockBackend
    from real_robot.config import RobotConfig
    from real_robot.recorder import DemoRecorder, dataset_features


class FakeDataset:
    def __init__(self):
        self.buffer, self.episodes, self.finalized, self.cleared = [], [], False, 0

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_episode(self):
        self.episodes.append(self.buffer)
        self.buffer = []

    def clear_episode_buffer(self):
        self.buffer = []
        self.cleared += 1

    def finalize(self):
        self.finalized = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@unittest.skipIf(np is None, "numpy not installed")
class TestDemoRecorder(unittest.TestCase):
    def setUp(self):
        self.cfg = RobotConfig(verified=True, joint_upper=[1.0] * 7)
        self.robot = MockBackend(self.cfg)
        self.ds, self.clock = FakeDataset(), Clock()

    def recorder(self, **kw):
        args = dict(fps=10, episodes=2, episode_s=1.0, reset_s=0.5)
        args.update(kw)
        return DemoRecorder(self.ds, self.robot, self.cfg, "stack", clock=self.clock,
                            log=lambda *a: None, **args)

    def drive(self, rec, seconds, hz=50):
        """Call the hook like teleop does; returns True if it asked to stop."""
        lead = np.full(7, 2.0)  # beyond joint_upper: the action must be clipped
        for _ in range(int(seconds * hz)):
            self.clock.t += 1.0 / hz
            if rec(np.zeros(7), np.zeros(7), lead):
                return True
            if rec.saver is not None:
                rec.saver.join()  # make background saves deterministic
        return False

    def test_records_episodes_then_stops(self):
        rec = self.recorder()
        self.assertTrue(self.drive(rec, 10))
        self.assertEqual(rec.close(), 2)
        self.assertEqual(len(self.ds.episodes), 2)
        for ep in self.ds.episodes:
            self.assertAlmostEqual(len(ep), 10, delta=1)  # 1 s at 10 fps
        f = self.ds.episodes[0][0]
        self.assertEqual(f["task"], "stack")
        self.assertEqual(f["observation.state"].shape, (7,))
        self.assertEqual(f["observation.end_pose"].shape, (6,))
        np.testing.assert_allclose(f["action"], 1.0)  # leader target clipped to the limit
        self.assertEqual(f["observation.images.cam_wrist"].shape, (480, 640, 3))
        self.assertTrue(self.ds.finalized)

    def test_nothing_recorded_during_reset(self):
        rec = self.recorder(reset_s=5.0)
        self.assertFalse(self.drive(rec, 4.9))
        self.assertEqual(self.ds.buffer, [])

    def test_partial_episode_discarded(self):
        rec = self.recorder()
        self.drive(rec, 1.0)  # reset (0.5 s) then half an episode
        self.assertTrue(rec.recording)
        self.assertEqual(rec.close(), 0)
        self.assertEqual(self.ds.cleared, 1)
        self.assertEqual(self.ds.episodes, [])

    def test_features_match_frames(self):
        feats = dataset_features(self.cfg)
        self.assertEqual(feats["action"]["shape"], (7,))
        self.assertIn("observation.images.cam_high", feats)
        with self.assertRaises(ValueError):
            dataset_features(RobotConfig(arms={"a": "x", "b": "y"}))


if __name__ == "__main__":
    unittest.main()
