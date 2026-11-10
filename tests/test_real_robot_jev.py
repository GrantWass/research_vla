"""JevMonitor phase logic and runner hooks, with a fake JEV (no server, no GPU)."""
import time

import numpy as np

from real_robot.jev import JevMonitor, questions


class FakeJev:
    def __init__(self, answers):
        self.answers, self.asked = dict(answers), []

    def yes(self, images, question, task, gripper_m=None):
        name = next(k for k, v in questions("cube", "bowl").items() if v == question)
        self.asked.append(name)
        a = self.answers[name]
        return (a.pop(0) if isinstance(a, list) else a), 0.01


def drive(mon, widths, dt=0.05):
    q = np.zeros(7)
    for i, g in enumerate(widths):
        q[6] = g
        mon({}, q, i)
        time.sleep(dt)
        if mon.stop_event.is_set():
            break
    time.sleep(0.2)  # let the worker answer


def trace(hold=0.062, after=0.090):
    return [0.0] * 3 + [0.095] * 4 + [0.08, 0.07, hold] + [hold] * 40 + [after] * 40


def mon_for(fake, **kw):
    kw = dict(lift_s=0.1, settle_s=0.1, recheck_s=0.5, log=lambda *_: None) | kw
    return JevMonitor(fake, "put the cube in the bowl", "cube", "bowl", **kw)


def test_success_stops_after_place_check():
    fake = FakeJev({"holding": 0.9, "placed": 0.8})
    m = mon_for(fake)
    drive(m, trace())
    assert m.verdict == "success" and m.stop_event.is_set()
    assert fake.asked[0] == "holding" and fake.asked[-1] == "placed"


def test_missed_grasp():
    m = mon_for(FakeJev({"holding": 0.1, "placed": 0.0}))
    drive(m, trace())
    assert m.verdict == "missed_grasp"


def test_drop_in_transit():
    m = mon_for(FakeJev({"holding": [0.9, 0.05], "placed": 0.0}), recheck_s=0.2)
    drive(m, trace(after=0.062))  # never releases
    assert m.verdict == "dropped"


def test_missed_place():
    m = mon_for(FakeJev({"holding": 0.9, "placed": 0.2}))
    drive(m, trace())
    assert m.verdict == "missed_place"


def test_jev_errors_never_stop():
    class Down:
        def yes(self, *a, **k):
            raise ConnectionError("down")
    m = mon_for(Down())
    drive(m, trace())
    assert m.verdict is None and not m.stop_event.is_set()
    assert any(e.kind == "jev_error" for e in m.events)


def test_empty_check_without_jev_call():
    fake = FakeJev({"holding": 0.9, "placed": 0.9})
    m = mon_for(fake, empty_check=True)
    drive(m, trace(hold=0.005))
    assert m.verdict == "missed_grasp" and fake.asked == []


def test_runner_observer_and_stop():
    import threading

    from real_robot.backends import MockBackend
    from real_robot.config import load_config
    from real_robot.runner import run

    cfg = load_config("real_robot/configs/piper_1.json")
    seen, stop = [], threading.Event()

    class Pol:
        def call(self, func_name, **kw):
            if func_name == "get_action":
                return np.zeros((10, cfg.state_dim))

    def obs(images, q, step):
        seen.append(step)
        if step >= 20:
            stop.set()
    s = run(MockBackend(cfg), Pol(), cfg, instruction="x", live=False, max_steps=100,
            observer=obs, stop_event=stop)
    assert s["status"] == "monitor_stop" and seen[:3] == [0, 10, 20] and s["steps"] == 20


def test_servo_converges_with_oracle_judge():
    """JevServo on the sim arm with a judge that reads the true image offset."""
    import random

    from real_robot import backends, jev_servo
    from real_robot.config import load_config
    from real_robot.jev_servo import JevServo
    from real_robot.jev_servo_sim import PX_PER_DEG, PX_PER_M, SimArm

    def oracle(cli, img, obj, which):
        e_deg, e_m = arm.err()
        off = arm.sx * PX_PER_DEG * e_deg if which == "lr" else arm.sy * PX_PER_M * e_m
        # cube appears opposite to the crop shift
        b = int(np.clip(np.round(-off / 0.04), -2, 2)) + 2
        p = [0.02] * 5
        p[b] = 0.92
        return p

    sleep0 = jev_servo.time.sleep
    SimArm.get_images = lambda self: {"cam_wrist": None}
    jev_servo.ask, jev_servo.time.sleep, backends.time.sleep = oracle, (lambda s: None), (lambda s: None)
    try:
        cfg = load_config("real_robot/configs/piper_1.json")
        for seed in range(12):
            arm = SimArm(None, random.Random(seed), 10.0, 0.06)
            JevServo(None, arm, cfg, log=lambda *_: None).recover(lift_m=0.0)
            e = arm.err()
            assert abs(e[0]) < 3.0 and abs(e[1]) < 0.03, (seed, e)
    finally:
        jev_servo.time.sleep = backends.time.sleep = sleep0


import unittest  # noqa: E402  (make test runs unittest discover)


def _fake_clock_drive(mon, qs, hz=30.0):
    """Feed joint vectors at hz with the monitor's clock faked (no sleeping)."""
    import real_robot.jev as J

    real, t = J.time.time, [1000.0]
    J.time.time = lambda: t[0]
    try:
        for i, q in enumerate(qs):
            t[0] = 1000.0 + i / hz
            mon({}, np.asarray(q, float), i)
            if mon.stop_event.is_set():
                break
    finally:
        J.time.time = real


def test_hover_open_and_still_stops():
    m = mon_for(FakeJev({"holding": 0.9, "placed": 0.9}))
    q = np.zeros(7)
    q[6] = 0.095
    _fake_clock_drive(m, [q] * (30 * 6))  # open and motionless for 6 s
    assert m.verdict == "hover"
    hover_at = next(e.step for e in m.events if e.kind == "verdict")
    assert 30 * 3 <= hover_at <= 30 * 4.5


def test_no_hover_while_moving_or_disabled():
    m = mon_for(FakeJev({"holding": 0.9, "placed": 0.9}))
    qs = []
    for i in range(30 * 8):  # open, J1 sweeping at 0.05 rad/s (a slow approach)
        q = np.zeros(7)
        q[0], q[6] = 0.05 * i / 30, 0.095
        qs.append(q)
    _fake_clock_drive(m, qs)
    assert m.verdict is None
    m2 = mon_for(FakeJev({"holding": 0.9, "placed": 0.9}), hover_s=0)
    q = np.zeros(7)
    q[6] = 0.095
    _fake_clock_drive(m2, [q] * (30 * 6))
    assert m2.verdict is None


def test_final_check_rules_before_jev():
    # held over the bowl, never released: JEV says placed (the 2026-11-05 false positive)
    fake = FakeJev({"holding": 0.9, "placed": 0.95})
    m = mon_for(fake)
    drive(m, trace()[:30])  # closes and holds, no release
    assert m.final_check({}, np.r_[np.zeros(6), 0.062], 99) == "not_released"
    assert "placed" not in fake.asked
    m = mon_for(FakeJev({"holding": 0.9, "placed": 0.95}), hover_s=0)
    drive(m, [0.0] * 3 + [0.095] * 3)  # opened, never closed
    assert m.final_check({}, np.r_[np.zeros(6), 0.095], 9) == "hover"


def test_lite_judge_question_routing():
    from real_robot.jev import questions as Q
    from real_robot.jevlite import LiteJudge

    q = Q("Rubik's cube", "bowl")
    assert LiteJudge.kind(q["holding"]) == "holding"
    assert LiteJudge.kind(q["placed"]) == "placed"
    assert LiteJudge.kind("Is the Rubik's cube held between the robot gripper's fingers right now "
                          "(lifted off the table, not lying on the table)?") == "holding"


def test_judged_frames_saved(tmp=None):
    import importlib.util
    import json, os, tempfile

    if importlib.util.find_spec("PIL") is None:
        return  # robot venv: no Pillow (live runs use the lerobot venv, which has it)
    d = tempfile.mkdtemp()
    m = mon_for(FakeJev({"holding": 0.9, "placed": 0.8}), frame_dir=d)
    q = np.zeros(7)
    widths = trace()
    for i, g in enumerate(widths):
        q[6] = g
        m({"cam_wrist": np.zeros((8, 8, 3), np.uint8)}, q, i)
        time.sleep(0.05)
        if m.stop_event.is_set():
            break
    time.sleep(0.2)
    recs = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
    assert [r["question"] for r in recs][0] == "holding" and os.path.exists(recs[0]["cam_wrist"])


class _ArmXYZ:
    """Fake arm: J1 rotates the reach plane; J2/J3/J5 move the gripper linearly."""
    A = np.array([[0.30, 0.10, 0.05], [0.0, 0.0, 0.0], [0.25, -0.20, 0.08]])

    def __init__(self):
        self.q = np.array([0.0, 1.0, -1.0, 0, 0.6, 0, 0.5])
        self.q0 = self.q.copy()

    def _p(self, q):
        local = np.array([0.30, 0.0, 0.20]) + self.A @ (q - self.q0)[[1, 2, 4]]
        c, s = np.cos(q[0]), np.sin(q[0])
        return np.array([c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]])

    def get_joint_state(self): return self.q.copy()
    def command_joints(self, q): self.q = np.asarray(q, float)
    def get_end_pose(self): return np.r_[self._p(self.q), 0, 0, 0]
    def get_images(self): return {"cam_wrist": np.zeros((4, 4, 3), np.uint8)}
    def hold(self): pass


def _servo(arm):
    import real_robot.backends as B
    import real_robot.jev_servo as S
    from real_robot.config import load_config
    from real_robot.safety import SafetyFilter

    cfg = load_config("real_robot/configs/piper_1_clamp.json")
    s = S.JevServo.__new__(S.JevServo)
    s.robot, s.cfg, s.hz, s.log, s.trace = arm, cfg, 500.0, lambda *a: None, []
    s.safety = SafetyFilter(cfg)
    return s


def _no_sleep(fn):
    """Run fn with time.sleep stubbed (servo moves), then restore it for the other tests."""
    real = time.sleep
    time.sleep = lambda t: None
    try:
        return fn()
    finally:
        time.sleep = real


def test_shift_xy_moves_sideways_with_j1():
    _no_sleep(_shift_xy)


def _shift_xy():
    arm = _ArmXYZ()
    s = _servo(arm)
    s.guard_from_here(0.10)
    p0 = arm.get_end_pose()[:3]
    out = s.shift_xy(s.jacobian(joints=(0, 1, 2, 4)), [0.01, 0.03])  # mostly tangential
    p1 = arm.get_end_pose()[:3]
    assert out["reached"], out
    assert np.allclose(p1[:2] - p0[:2], [0.01, 0.03], atol=0.004) and abs(p1[2] - p0[2]) < 0.004


def test_align_shifts_to_predicted_grasp_then_reports_height():
    _no_sleep(_align)


def _align():
    import types

    import real_robot.jev_guard as G

    arm = _ArmXYZ()
    s = _servo(arm)
    s.guard_from_here(0.10)
    goal = arm.get_end_pose()[:3] + np.array([0.0, 0.04, -0.05])  # 4 cm sideways, 5 cm down

    class Oracle:
        def __call__(self, images, p): return goal - np.asarray(p)

    G._OFFSET["oracle"] = Oracle()
    args = types.SimpleNamespace(offset_model="oracle", align_deadband=0.015, align_max=0.06, grasp_z=0.151)
    out = G.align(arm, s, args, log=lambda *a: None)
    p = arm.get_end_pose()[:3]
    assert np.linalg.norm(p[:2] - goal[:2]) < 0.006, (p, goal)
    assert abs(out["grasp_z"] - float(np.clip(goal[2], 0.141, 0.181))) < 0.003


class TestJev(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(TestJev, _name, lambda self, fn=_fn: fn())

