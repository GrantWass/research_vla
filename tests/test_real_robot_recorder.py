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
    from real_robot.recorder import DemoRecorder, KeyedRecorder, dataset_features


class FakeDataset:
    def __init__(self):
        self.buffer, self.episodes, self.finalized, self.cleared = [], [], False, 0
        self.fail = False

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_episode(self, parallel_encoding=True):
        if self.fail:
            raise KeyboardInterrupt  # what a killed encoder process surfaces as
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


@unittest.skipIf(np is None, "numpy not installed")
class TestKeyedRecorder(unittest.TestCase):
    """One episode per teleop() call: begin(), then Enter save / r discard / q quit."""

    def setUp(self):
        self.cfg = RobotConfig(verified=True, joint_upper=[1.0] * 7)
        self.robot = MockBackend(self.cfg)
        self.ds, self.clock, self.pending, self.logs = FakeDataset(), Clock(), [], []

    def recorder(self, **kw):
        return KeyedRecorder(self.ds, self.robot, self.cfg, "stack", fps=10, clock=self.clock,
                             key_source=lambda: self.pending.pop(0) if self.pending else None,
                             log=self.logs.append, **kw)

    def drive(self, rec, seconds, keys="", hz=50):
        """Call the hook like teleop does; True once it ends the episode."""
        self.pending.extend(keys)
        for _ in range(int(seconds * hz)):
            self.clock.t += 1.0 / hz
            if rec(np.zeros(7), np.zeros(7), np.full(7, 0.5)):
                return True
            if rec.saver is not None:
                rec.saver.join()
        return False

    def test_nothing_before_begin(self):
        rec = self.recorder()
        self.assertTrue(self.drive(rec, 1.0))  # not recording: teleop would stop at once
        self.assertEqual(self.ds.buffer, [])

    def test_begin_records_until_enter_saves(self):
        rec = self.recorder()
        rec.begin()
        self.assertFalse(self.drive(rec, 1.0))
        self.assertTrue(self.drive(rec, 0.1, "\n"))
        rec.saver.join()
        self.assertEqual(len(self.ds.episodes), 1)
        self.assertAlmostEqual(len(self.ds.episodes[0]), 10, delta=1)
        np.testing.assert_allclose(self.ds.episodes[0][0]["action"], 0.5)
        self.assertFalse(rec.done())

    def test_keys_pressed_before_begin_are_dropped(self):
        rec = self.recorder()
        self.pending.extend("\nr")  # e.g. pressed while wiggling the leader
        rec.begin()
        self.assertFalse(self.drive(rec, 0.5))
        self.assertTrue(rec.recording)

    def test_r_discards_then_next_episode_saves(self):
        rec = self.recorder()
        rec.begin()
        self.drive(rec, 0.5)
        self.assertTrue(self.drive(rec, 0.1, "r"))
        self.assertEqual((self.ds.cleared, self.ds.episodes, rec.discarded), (1, [], 1))
        rec.begin()
        self.drive(rec, 0.5)
        self.assertTrue(self.drive(rec, 0.1, "\r"))
        rec.saver.join()
        self.assertEqual(len(self.ds.episodes), 1)

    def test_q_discards_and_ends_session(self):
        rec = self.recorder()
        rec.begin()
        self.drive(rec, 0.5)
        self.assertTrue(self.drive(rec, 0.1, "q"))
        self.assertTrue(rec.done())
        self.assertEqual(rec.close(), 0)
        self.assertEqual(self.ds.cleared, 1)
        self.assertTrue(self.ds.finalized)

    def test_enter_before_any_frame_saves_nothing(self):
        rec = self.recorder()
        rec.begin()
        self.pending.append("\n")
        self.assertTrue(rec(np.zeros(7), np.zeros(7), np.full(7, 0.5)))
        self.assertIsNone(rec.saver)
        self.assertEqual(rec.discarded, 1)

    def test_max_length_discards(self):
        rec = self.recorder(max_episode_s=2.0)
        rec.begin()
        self.assertTrue(self.drive(rec, 3.0))
        self.assertEqual((self.ds.cleared, self.ds.episodes), (1, []))

    def test_done_after_target_episodes(self):
        rec = self.recorder(episodes=2)
        for _ in range(2):
            self.assertFalse(rec.done())
            rec.begin()
            self.drive(rec, 0.3)
            self.drive(rec, 0.1, "\n")
        self.assertTrue(rec.done())
        self.assertEqual(rec.close(), 2)

    def test_abandon_drops_the_episode(self):
        rec = self.recorder()
        rec.begin()
        self.drive(rec, 0.5)
        rec.abandon("leader glitch")
        self.assertEqual((self.ds.cleared, rec.recording), (1, False))

    def test_failed_save_is_reported_and_session_still_finalizes(self):
        self.ds.fail = True  # e.g. Ctrl-C killed an encoder: a BaseException, not an Exception
        rec = self.recorder()
        rec.begin()
        self.drive(rec, 0.5)
        self.drive(rec, 0.1, "\n")
        rec.saver.join()
        with self.assertRaises(RuntimeError):
            rec.wait_for_save()
        with self.assertRaises(RuntimeError):  # and the hook refuses to go on
            rec(np.zeros(7), np.zeros(7), np.zeros(7))
        self.assertEqual(rec.close(), 0)
        self.assertTrue(self.ds.finalized)
        self.assertTrue(any("NOT saved" in m for m in self.logs))

    def test_close_waits_for_a_slow_save(self):
        import threading
        gate, real_save = threading.Event(), self.ds.save_episode

        def slow_save(parallel_encoding=True):
            self.assertFalse(parallel_encoding)  # child encoders die on Ctrl-C
            gate.wait(5)
            real_save()
        self.ds.save_episode = slow_save
        rec = self.recorder()
        rec.begin()
        for _ in range(20):
            self.clock.t += 0.02
            rec(np.zeros(7), np.zeros(7), np.full(7, 0.5))
        self.pending.append("\n")
        self.assertTrue(rec(np.zeros(7), np.zeros(7), np.full(7, 0.5)))
        self.assertTrue(rec._saving())
        with self.assertRaises(RuntimeError):  # no new episode while saving
            rec.begin()
        threading.Timer(0.3, gate.set).start()
        self.assertEqual(rec.close(), 1)
        self.assertTrue(self.ds.finalized)

if __name__ == "__main__":
    unittest.main()
