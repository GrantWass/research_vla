#!/usr/bin/env python3
"""Move the arm to the teleop start pose, or power it down, with pyAgxArm only.

  python arm_tool.py ready [--can can0] [--yes]   # to config.json's ready_deg, motors stay on
  python arm_tool.py park  [--can can0] [--yes]   # fold, face home_base_deg, damped stop, disable

Run in the conda env "vt" (make ready / make park do). Both ask for YES first
and move at most --vel rad/s.

park is the procedure we use on our rig, so the arm never slams down:
disabling at the fold drops it ~1 inch onto its base, so after folding (base
held) and turning the base, trigger the firmware's damped emergency stop (the
joints sink slowly), wait, disable, then clear the stop so the next enable works.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "config.json")) as f:
        return json.load(f)


def connect(can_port: str):
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    arm = AgxArmFactory.create_arm(create_agx_arm_config(robot="piper", comm="can", channel=can_port))
    arm.connect()
    wait_for_feedback(arm)
    return arm


def wait_for_feedback(arm, timeout_s: float = 3.0, sleep=time.sleep, clock=time.monotonic) -> None:
    """Block until joint feedback is arriving: right after connect the readings
    are stale (a read 0.3 s after connect once showed an enabled arm at the wrong pose)."""
    deadline = clock() + timeout_s
    stamps = set()
    while clock() < deadline:
        ja = arm.get_joint_angles()
        if ja is not None:
            stamps.add(ja.timestamp)
            if len(stamps) >= 3:  # three distinct frames: live
                sleep(0.1)
                return
        sleep(0.02)
    raise RuntimeError("no live joint feedback: is the CAN adapter up and the arm powered?")


def joints(arm) -> np.ndarray:
    ja = arm.get_joint_angles()
    if ja is None:
        raise RuntimeError("no joint feedback: is the CAN adapter up and the arm powered?")
    return np.asarray(ja.msg, dtype=np.float64)


def enable(arm, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not arm.enable():
        if time.monotonic() > deadline:
            raise RuntimeError("the arm did not enable")
        time.sleep(0.01)


def check_target(cfg: dict, target) -> None:
    lo, hi = np.asarray(cfg["joint_lower"][:6]), np.asarray(cfg["joint_upper"][:6])
    bad = [f"J{i + 1}" for i in range(6) if not lo[i] - 1e-6 <= target[i] <= hi[i] + 1e-6]
    if bad:
        raise ValueError(f"{', '.join(bad)} outside config.json's joint limits")


def ease_to(arm, cfg: dict, target, vel: float = 0.2, hz: float = 50.0,
            sleep=time.sleep) -> None:
    """Straight line in joint space at the slowest joint's `vel` rad/s."""
    target = np.asarray(target, dtype=np.float64)
    check_target(cfg, target)
    q0 = joints(arm)
    steps = max(1, int(np.ceil(np.max(np.abs(target - q0)) / vel * hz)))
    for k in range(1, steps + 1):
        arm.move_j((q0 + (target - q0) * k / steps).tolist())
        sleep(1.0 / hz)
    sleep(0.5)


def ready(arm, cfg: dict, log=print) -> None:
    enable(arm)
    target = np.radians(cfg["ready_deg"])
    log(f"to ready pose, deg {np.degrees(target).round(1)}")
    ease_to(arm, cfg, target)


def fold_targets(cfg: dict, q) -> tuple[np.ndarray, np.ndarray]:
    """(folded with the base kept, folded with the base turned to home_base_deg)."""
    folded = np.asarray(q, dtype=np.float64)[:6].copy()
    folded[1:] = np.radians(cfg["park_fold_deg"])
    home = folded.copy()
    home[0] = np.radians(cfg["home_base_deg"])
    return folded, home


def park(arm, cfg: dict, settle_s: float = 3.0, sleep=time.sleep, log=print) -> None:
    enable(arm)
    folded, home = fold_targets(cfg, joints(arm))
    log(f"folding to deg {np.degrees(folded).round(0)}")
    ease_to(arm, cfg, folded, sleep=sleep)
    log(f"turning base to deg {np.degrees(home[0]):.1f}")
    ease_to(arm, cfg, home, sleep=sleep)
    log("damped stop, then disable")
    arm.electronic_emergency_stop()
    sleep(settle_s)
    arm.disable()
    arm.reset()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ready", "park"])
    ap.add_argument("--can", help="CAN interface (default: config.json's can_port)")
    ap.add_argument("--config", help="default: config.json next to this file")
    ap.add_argument("--yes", action="store_true", help="don't ask first")
    args = ap.parse_args(argv)
    cfg = load(args.config)
    can = args.can or cfg["can_port"]
    what = {"ready": "Move to the ready pose (motors stay on)",
            "park": "Fold, face the base home and power down"}[args.command]
    if not args.yes and input(f"{what} on {can}? type YES: ") != "YES":
        return 1
    arm = connect(can)
    arm.set_speed_percent(int(cfg["tool_speed_percent"]))
    {"ready": ready, "park": park}[args.command](arm, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
