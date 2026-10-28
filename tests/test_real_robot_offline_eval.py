"""offline_eval: frame sampling and chunk metrics (no GPU, model or dataset)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np

    from real_robot.offline_eval import FRACS, chunk_metrics, sample_indices
except ImportError:  # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class TestOfflineEval(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n, t = 6, 50
        self.state = np.zeros((n, 13))
        ramp = np.linspace(0, np.radians(20), t)                       # every joint moves +20 deg
        self.gt = np.zeros((n, t, 7))
        self.gt[..., :6] = ramp[None, :, None]
        self.gt[..., 6] = 0.5
        self.pad = np.zeros((n, t), bool)
        self.rng = rng

    def test_sample_indices_stay_in_episode(self):
        ep_from, ep_to = np.array([0, 100, 350]), np.array([100, 350, 351])
        idx = sample_indices(ep_from, ep_to)
        self.assertEqual(len(idx), 3 * len(FRACS))
        for e, f, i in idx:
            self.assertTrue(ep_from[e] <= i < ep_to[e], (e, f, i))
        self.assertEqual(idx[-1][2], 350)                              # 1-frame episode

    def test_perfect_prediction(self):
        m = chunk_metrics(self.gt, self.gt, self.pad, self.state)
        self.assertEqual(m["joints_deg"], 0.0)
        self.assertEqual(m["gripper"], 0.0)
        self.assertEqual(m["direction"], 1.0)

    def test_constant_offset_and_direction(self):
        pred = self.gt.copy()
        pred[..., :6] += np.radians(2.0)
        pred[..., 6] += 0.1
        m = chunk_metrics(pred, self.gt, self.pad, self.state)
        self.assertAlmostEqual(m["joints_deg"], 2.0, places=6)
        self.assertAlmostEqual(m["gripper"], 0.1, places=6)
        self.assertAlmostEqual(m["by_horizon_deg"]["t+1"], 2.0, places=6)
        self.assertEqual(m["direction"], 1.0)
        m = chunk_metrics(-self.gt, self.gt, self.pad, self.state)   # moves the wrong way
        self.assertEqual(m["direction"], 0.0)

    def test_padded_steps_ignored(self):
        pred = self.gt.copy()
        pred[:, 40:] += 1.0                                            # garbage past the episode end
        self.pad[:, 40:] = True
        m = chunk_metrics(pred, self.gt, self.pad, self.state)
        self.assertAlmostEqual(m["joints_deg"], 0.0, places=9)
        self.assertNotEqual(m["direction"], m["direction"])            # no frame has a full chunk: nan


if __name__ == "__main__":
    unittest.main()
