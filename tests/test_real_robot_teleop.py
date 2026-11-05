"""Teleop safety: wait for the leader, refuse big start gaps, rate-limit, stop on glitches."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from real_robot.backends import MockBackend
    from real_robot.config import RobotConfig
    from real_robot.piper import PiperLeader
    from real_robot.piper import MotorsTooHot
    from real_robot.recorder import KeyedRecorder
    from real_robot.teleop import keyed_session, teleop


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLeaderArm:
    """Broadcast starts only after `start_after` polls, like a teaching arm at rest."""

    def __init__(self, q_deg, start_after=3, partial_polls=0):
        self.q_mdeg = [int(v * 1000) for v in q_deg]
        self.partial_polls = partial_polls
        self.polls, self.start_after, self.grip = 0, start_after, 0

    def GetArmJointCtrl(self):
        self.polls += 1
        live = self.polls > self.start_after
        joints = list(self.q_mdeg) if live else [0] * 6
        if live and self.polls <= self.start_after + self.partial_polls:
            joints[2:] = [0, 0, 0, 0]  # only the 0x155 frame (joints 1-2) has arrived
        return _NS(time_stamp=self.polls if live else 0,
                   joint_ctrl=_NS(**{f"joint_{i + 1}": v for i, v in enumerate(joints)}))

    def GetArmGripperCtrl(self):
        return _NS(gripper_ctrl=_NS(grippers_angle=self.grip))


@unittest.skipIf(np is None, "numpy not installed")
class TestTeleop(unittest.TestCase):
    def setUp(self):
        self.cfg = RobotConfig(arms={"arm": "x"}, control_hz=50)
        self.robot = MockBackend(self.cfg, lag=1.0)

    def leader(self, q_deg, **kw):
        return PiperLeader("x", sdk_factory=lambda can: FakeLeaderArm(q_deg, **kw))

    def run_teleop(self, leader, seconds=0.05):
        return teleop(leader, self.robot, self.cfg, seconds, self.cfg.control_hz,
                      sleep=lambda s: None, log=lambda *a: None)

    def test_waits_for_broadcast_then_follows(self):
        self.robot.q[:] = np.radians([5, 10, -10, 0, 20, 0, 0])
        lead = self.leader([8, 12, -12, 2, 22, 1])
        s = self.run_teleop(lead, seconds=0.2)
        self.assertEqual(s["status"], "ok")
        np.testing.assert_allclose(np.degrees(self.robot.q[:6]), [8, 12, -12, 2, 22, 1], atol=0.5)
        self.assertGreater(lead.arm.polls, 3)  # never targeted the pre-broadcast zeros

    def test_partial_first_broadcast_is_not_a_target(self):
        # Follower at zeros; leader's real wrist is at 20 deg but its first
        # frames only carry joints 1-2, so joints 3-6 read 0 for a while.
        self.robot.q[:] = np.zeros(7)
        lead = self.leader([0, 0, 0, 0, 20, 0], partial_polls=5)
        s = self.run_teleop(lead, seconds=0.2)
        self.assertEqual(s["status"], "ok")
        np.testing.assert_allclose(np.degrees(self.robot.q[4]), 20, atol=0.5)

    def test_refuses_large_start_gap(self):
        self.robot.q[:] = np.radians([-90, 0, 0, 0, 20, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "differ"):
            self.run_teleop(self.leader([0, 0, 0, 0, 20, 0]))
        self.assertEqual(len(self.robot.commands), 0)

    def test_rate_limited_per_step(self):
        self.robot.q[:] = np.zeros(7)
        self.run_teleop(self.leader([20, 0, 0, 0, 0, 0]), seconds=0.2)
        steps = np.diff(np.asarray(self.robot.commands)[:, 0])
        self.assertLessEqual(np.max(np.abs(steps)), 0.05 + 1e-9)  # default max_step_delta

    def test_on_step_true_ends_session(self):
        self.robot.q[:] = np.zeros(7)
        calls = []
        s = teleop(self.leader([1, 0, 0, 0, 0, 0]), self.robot, self.cfg, 10, self.cfg.control_hz,
                   on_step=lambda *a: calls.append(a) or len(calls) >= 3,
                   sleep=lambda s: None, log=lambda *a: None)
        self.assertEqual((s["status"], s["steps"], len(calls)), ("ok", 3, 3))
        self.assertGreater(self.robot.held, 0)

    def test_stops_on_leader_glitch(self):
        self.robot.q[:] = np.zeros(7)
        lead = self.leader([1, 0, 0, 0, 0, 0])
        base = lead.arm.GetArmJointCtrl
        t_glitch = time.monotonic() + 0.4  # after startup has settled

        def glitch():
            m = base()
            if time.monotonic() > t_glitch:
                m.joint_ctrl.joint_1 = 90000
            return m
        lead.arm.GetArmJointCtrl = glitch
        s = self.run_teleop(lead, seconds=10)
        self.assertIn("jumped", s["status"])
        self.assertGreater(self.robot.held, 0)


class SessionArm(MockBackend if np is not None else object):
    """MockBackend with the power controls keyed_session uses; logs them."""

    def __init__(self, cfg, name, calls, hot=False):
        super().__init__(cfg, lag=1.0)
        self.name, self.calls, self.hot, self.on = name, calls, hot, True

    def motors_on(self):
        return self.on

    def check_motors(self, log=print):
        self.calls.append((self.name, "check"))
        if self.hot:
            raise MotorsTooHot(f"{self.name} J5 at 67 C")

    def enable(self):
        self.calls.append((self.name, "enable"))
        self.on = True

    def home(self, log=print):
        self.calls.append((self.name, "home"))

    def park(self, log=print):
        self.calls.append((self.name, "park"))
        self.on = False


class TeachingArm(FakeLeaderArm):
    def __init__(self, q_deg, calls):
        super().__init__(q_deg)
        self.calls = calls

    def MasterSlaveConfig(self, mode, *a):
        self.calls.append(("leader", hex(mode)))


class ScriptedKeys:
    """get() returns the script in order (None = no key on that poll), then None."""

    def __init__(self, script):
        self.script = list(script)

    def get(self):
        return self.script.pop(0) if self.script else None


class SessionDataset:
    def __init__(self):
        self.buffer, self.episodes = [], []

    def add_frame(self, f):
        self.buffer.append(f)

    def save_episode(self, parallel_encoding=True):
        self.episodes.append(self.buffer)
        self.buffer = []

    def clear_episode_buffer(self):
        self.buffer = []

    def finalize(self):
        pass


@unittest.skipIf(np is None, "numpy not installed")
class TestKeyedSession(unittest.TestCase):
    """make record: arms parked between episodes; Space powers up and records."""

    def setUp(self):
        self.cfg = RobotConfig(arms={"arm": "x"}, control_hz=50)
        self.calls = []
        self.ds = SessionDataset()

    def run_session(self, script, hot=False):
        robot = SessionArm(self.cfg, "close", self.calls, hot=hot)
        far = SessionArm(self.cfg, "far", self.calls)
        leader = PiperLeader("x", sdk_factory=lambda can: TeachingArm([0] * 6, self.calls))
        keys = ScriptedKeys(script)
        rec = KeyedRecorder(self.ds, robot, self.cfg, "stack", fps=50, key_source=keys.get,
                            log=lambda *a: None)
        keyed_session(robot, far, leader, self.cfg, rec, keys, hz=200.0, log=lambda *a: None)
        rec.close()
        return rec

    def test_space_powers_up_records_then_parks(self):
        rec = self.run_session([" ", None] + [None] * 10 + ["\n", "q"])
        self.assertEqual(len(self.ds.episodes), 1)
        self.assertGreater(len(self.ds.episodes[0]), 5)
        self.assertEqual(rec.saved, 1)
        c = self.calls
        self.assertEqual(c[:2], [("close", "park"), ("far", "park")])  # parked at start
        i_enable, i_teach = c.index(("close", "enable")), c.index(("leader", "0xfa"))
        self.assertLess(c.index(("close", "check")), i_enable)  # motors checked first
        self.assertLess(c.index(("far", "home")), i_teach)      # homed before going limp
        tail = c[c.index(("leader", "0xfc")):]
        self.assertIn(("close", "park"), tail)  # parked after the episode
        self.assertIn(("far", "park"), tail)

    def test_hot_motor_refuses_to_start(self):
        rec = self.run_session([" ", "q"], hot=True)
        self.assertNotIn(("close", "enable"), self.calls)
        self.assertNotIn(("leader", "0xfa"), self.calls)
        self.assertEqual(rec.saved, 0)

    def test_r_discards_and_parks_ready_for_space(self):
        rec = self.run_session([" ", None] + [None] * 10 + ["r", " ", None] + [None] * 10 + ["\n", "q"])
        self.assertEqual((rec.saved, rec.discarded), (1, 1))
        self.assertEqual(self.calls.count(("leader", "0xfa")), 2)


if __name__ == "__main__":
    unittest.main()
