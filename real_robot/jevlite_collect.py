"""Collect grasp-offset training frames on the real arm (fills the empty buckets).

The demos only approach from short of the cube and nearly on-line sideways, so
the offset model (real_robot/jevlite_offset.py) has never seen "too far",
"far left/right" or most of the 3-D space around a hover. This script makes
that data with exact labels:

  1. put the arm at a good grasp pose over the cube (teach it there, or let
     `--recover align` finish a grasp and open) and start the script;
  2. it records that end pose as the grasp point, then visits a grid of offsets
     around it (radial and tangential ±R, heights +H above, never below), saving
     both camera frames with the offset (radial, tangential, dz) to the grasp;
  3. it returns to the start pose.

Every move goes through JevServo's workspace guard (z floor at the grasp
height minus 1 cm, J1 and reach excursion) and stops if the arm leaves our
control. Hand on the e-stop.

  python -m real_robot.jevlite_collect --config real_robot/configs/piper_1_clamp.json --out outputs/jevlite/collect_<stamp> --yes
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .jevlite_offset import to_arm_frame


def grid(radius_m: float = 0.08, step_m: float = 0.04, heights=(0.03, 0.06, 0.09)) -> list[np.ndarray]:
    """(radial, tangential, +height) offsets FROM the grasp point, nearest first."""
    r = np.arange(-radius_m, radius_m + 1e-9, step_m)
    pts = [np.array([a, b, h]) for h in heights for a in r for b in r]
    return sorted(pts, key=lambda v: (v[2], np.hypot(v[0], v[1])))


def collect(robot, cfg, servo, out: str, points, settle_s: float = 0.4, log=print) -> int:
    """Visit points (arm-frame offsets from the grasp), save frames + labels. Returns frames saved."""
    from PIL import Image

    from .jev_servo import ServoStop
    from .jevlite_offset import from_arm_frame

    os.makedirs(out, exist_ok=True)
    grasp = np.asarray(robot.get_end_pose()[:3], float)
    q_start = robot.get_joint_state()
    servo.guard_from_here(z_floor=grasp[2] - 0.01)
    servo.max_radial_m = max(servo.max_radial_m, 0.12)
    J = servo.jacobian(joints=(0, 1, 2, 4))
    n = 0
    fh = open(os.path.join(out, "labels.jsonl"), "a")
    try:
        for k, off in enumerate(points):
            target = grasp + from_arm_frame(grasp, off)
            p = np.asarray(robot.get_end_pose()[:3], float)
            # rise first, then go sideways at the target height: never sweep low over the cube
            if target[2] > p[2] + 0.002:
                servo.shift_xy(J, [0.0, 0.0])  # settle
                _vertical(servo, J, target[2] - p[2])
            servo.shift_xy(J, (target - np.asarray(robot.get_end_pose()[:3], float))[:2])
            p = np.asarray(robot.get_end_pose()[:3], float)
            if p[2] > target[2] + 0.004:
                _vertical(servo, J, target[2] - p[2])
            time.sleep(settle_s)
            p = np.asarray(robot.get_end_pose()[:3], float)
            ims = robot.get_images()
            rec = {"k": k, "grasp": grasp.tolist(), "pose": p.tolist(),
                   "offset_rtz": to_arm_frame(p, grasp - p).tolist()}
            for cam in ("cam_wrist", "cam_high"):
                if cam in ims:
                    path = os.path.join(out, f"k{k:03d}_{cam}.jpg")
                    Image.fromarray(ims[cam]).save(path, quality=90)
                    rec[cam] = path
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            log(f"[collect] {k + 1}/{len(points)} offset to grasp (cm) "
                + " ".join(f"{100 * x:+.1f}" for x in rec["offset_rtz"]))
    except ServoStop as e:
        log(f"[collect] stopped: {e}")
    finally:
        fh.close()
        try:  # back up, over, and down to where we started
            p = np.asarray(robot.get_end_pose()[:3], float)
            if p[2] < grasp[2] + 0.03:
                _vertical(servo, J, grasp[2] + 0.03 - p[2])
            servo.shift_xy(J, (grasp - np.asarray(robot.get_end_pose()[:3], float))[:2])
            from .backends import ease_to
            from .safety import SafetyFilter

            ease_to(robot, SafetyFilter(cfg), q_start, cfg.control_hz, vel=0.15)
        except Exception as e:  # leave it holding where it is
            robot.hold()
            log(f"[collect] return failed, holding: {e}")
    return n


def _vertical(servo, J, dz: float, step_m: float = 0.01) -> None:
    """Straight up/down by dz in <= step_m moves (closed loop on z, x/y held)."""
    p0 = np.asarray(servo.robot.get_end_pose()[:3], float)
    goal = p0[2] + dz
    for _ in range(int(abs(dz) / step_m) + 6):
        p = np.asarray(servo.robot.get_end_pose()[:3], float)
        e = goal - p[2]
        if abs(e) < 0.003:
            return
        want = np.array([p0[0] - p[0], p0[1] - p[1], float(np.clip(e, -step_m, step_m))])
        sol, *_ = np.linalg.lstsq(J, want, rcond=None)
        dq = np.zeros(len(servo.robot.get_joint_state()))
        dq[[0, 1, 2, 4]] = sol
        servo._move(dq)


def main(argv=None) -> int:
    from .backends import make_backend
    from .config import load_config
    from .jev_servo import JevServo

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.08)
    ap.add_argument("--step", type=float, default=0.04)
    ap.add_argument("--heights", default="0.03,0.06,0.09")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    pts = grid(a.radius, a.step, tuple(float(h) for h in a.heights.split(",")))
    if not a.yes and input(f"LIVE: {len(pts)} poses around the current grasp pose. Hand on e-stop? "
                           "type YES: ") != "YES":
        return 1
    cfg = load_config(a.config)
    robot = make_backend("piper", cfg)
    robot.connect()
    servo = JevServo(None, robot, cfg, log=print)
    n = collect(robot, cfg, servo, a.out, pts)
    print(f"{n} frames -> {a.out}/labels.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
