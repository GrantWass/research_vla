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


class TestJev(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(TestJev, _name, lambda self, fn=_fn: fn())

