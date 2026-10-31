"""relative_actions: chunk-relative joint actions (no GPU or model)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np

    from real_robot.relative_actions import (chunk_stats, relative_chunks, relative_joints,
                                             to_absolute, to_relative)
    from real_robot.train_relative import _arg, with_action_stats
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


@unittest.skipIf(np is None, "numpy not installed")
class TestRelativeActions(unittest.TestCase):
    def test_round_trip_keeps_gripper_absolute(self):
        rng = np.random.default_rng(0)
        state = rng.normal(size=(4, 13))
        chunk = rng.normal(size=(4, 50, 7))
        rel = to_relative(chunk, state)
        np.testing.assert_allclose(rel[..., :6], chunk[..., :6] - state[:, None, :6])
        np.testing.assert_array_equal(rel[..., 6], chunk[..., 6])
        np.testing.assert_allclose(to_absolute(rel, state), chunk)
        self.assertFalse(np.shares_memory(rel, chunk))

    def test_unbatched_state(self):
        state, chunk = np.arange(13.0), np.ones((50, 7))
        np.testing.assert_allclose(to_relative(chunk, state)[0, :6], 1 - np.arange(6.0))

    def test_chunks_pad_with_last_action(self):
        a = np.arange(5.0)[:, None].repeat(7, 1)       # joint value = frame index
        s = np.zeros((5, 13))
        s[:, :6] = np.arange(5.0)[:, None]
        c = relative_chunks(a, s, chunk=3)
        self.assertEqual(c.shape, (5, 3, 7))
        np.testing.assert_array_equal(c[0, :, 0], [0, 1, 2])
        np.testing.assert_array_equal(c[4, :, 0], [0, 0, 0])   # last frame: padding repeats it
        np.testing.assert_array_equal(c[3, :, 6], [3, 4, 4])   # gripper stays absolute

    def test_holding_still_is_zero(self):
        s = np.tile(np.r_[np.linspace(-1, 1, 6), 0.3, np.zeros(6)], (10, 1))
        a = s[:, :7].copy()
        self.assertEqual(np.abs(relative_chunks(a, s)[..., :6]).max(), 0.0)

    def test_relative_joints_of_pipeline(self):
        class Step:
            _registry_name = "piper_relative_actions"
            joints = 6

        class Other:
            pass

        class Pipe:
            steps = [Other(), Step()]

        self.assertEqual(relative_joints(Pipe()), 6)
        Pipe.steps = [Other()]
        self.assertEqual(relative_joints(Pipe()), 0)

    def test_with_action_stats_keeps_others(self):
        stats = {"action": {"q01": np.zeros(7), "foo": np.ones(7)}, "observation.state": {"q01": 1}}
        out = with_action_stats(stats, {"q01": np.full(7, -1.0)})
        self.assertEqual(out["action"]["q01"][0], -1.0)
        self.assertEqual(out["action"]["foo"][0], 1.0)
        self.assertIs(out["observation.state"], stats["observation.state"])
        self.assertEqual(stats["action"]["q01"][0], 0.0)                # input untouched

    def test_arg(self):
        argv = ["--dataset.root=demos/x_pi05", "--dataset.episodes=[1,2]"]
        self.assertEqual(_arg(argv, "dataset.root"), "demos/x_pi05")
        self.assertIsNone(_arg(argv, "steps"))

    @unittest.skipIf(pd is None, "pandas not installed")
    def test_chunk_stats_from_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/data/chunk-000")
            rows = []
            for ep in range(2):
                for t in range(20):
                    st = np.zeros(13, np.float32)
                    st[:6] = t * 0.01
                    act = np.r_[np.full(6, (t + 1) * 0.01), 0.5].astype(np.float32)
                    rows.append({"episode_index": ep, "action": act, "observation.state": st})
            try:
                pd.DataFrame(rows).to_parquet(f"{d}/data/chunk-000/file-000.parquet")
            except ImportError:
                self.skipTest("no parquet engine")
            s = chunk_stats(d, episodes=[1], chunk=5)
            self.assertEqual(int(s["count"][0]), 20 * 5)
            self.assertAlmostEqual(float(s["max"][0]), 0.05, places=5)  # 5 steps ahead, 0.01 each
            self.assertAlmostEqual(float(s["min"][0]), 0.01, places=5)  # last frame: padding
            self.assertAlmostEqual(float(s["mean"][6]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
