"""Closed-loop test of real_robot.jev_servo against the real JEV server, no arm.

A fake arm whose wrist view is a real near-grasp wrist frame (cube centred
between the fingers) shifted in proportion to its joint error: J1 error moves
the cube left/right, reach error moves it up/down, each with a random sign
the servo has to discover. Start errors are random. Success = the arm ends
within --tol of the grasp pose on both axes.

  python -m real_robot.jev_servo_sim --dirs outputs/jev_dirs --episodes 20 --out outputs/jev_servo_sim
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from .jev import JevClient
from .jev_dir_eval import shifted

PX_PER_DEG = 0.035  # image width fraction per degree of J1 error (~0.1 at 2-3 deg near the cube, demo frames)
PX_PER_M = 2.5      # image height fraction per metre of reach error
MAX_SHIFT = 0.20


class SimArm:
    def __init__(self, img, rng: random.Random, err_deg: float, err_m: float):
        self.img = img
        self.q = np.zeros(7)
        self.q[1], self.q[2], self.q[4] = 1.2, -1.0, 0.5
        self.goal_j1 = np.radians(rng.uniform(-err_deg, err_deg))
        self.goal_r = self.radius() + rng.uniform(-err_m, err_m)
        self.sx, self.sy = rng.choice([-1, 1]), rng.choice([-1, 1])

    def radius(self) -> float:  # toy planar arm: reach from J2, J3
        return 0.25 + 0.20 * np.sin(self.q[1]) + 0.15 * np.sin(self.q[1] + self.q[2])

    def height(self) -> float:
        return 0.30 - 0.20 * np.cos(self.q[1]) - 0.15 * np.cos(self.q[1] + self.q[2]) - 0.02 * self.q[4]

    def err(self) -> tuple[float, float]:
        return float(np.degrees(self.q[0] - self.goal_j1)), float(self.radius() - self.goal_r)

    def get_joint_state(self):
        return self.q.copy()

    def get_end_pose(self):
        r = self.radius()
        return np.array([r * np.cos(self.q[0]), r * np.sin(self.q[0]), self.height(), 0, 0, 0])

    def command_joints(self, q):
        self.q = np.asarray(q, float).copy()

    def get_images(self):
        e_deg, e_m = self.err()
        dx = float(np.clip(self.sx * PX_PER_DEG * e_deg, -MAX_SHIFT, MAX_SHIFT))
        dy = float(np.clip(self.sy * PX_PER_M * e_m, -MAX_SHIFT, MAX_SHIFT))
        return {"cam_wrist": shifted(self.img, dx, dy)}


def main(argv=None) -> int:
    from PIL import Image

    from . import jev_servo
    from .jev_servo import JevServo

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--err-deg", type=float, default=10.0)
    ap.add_argument("--err-m", type=float, default=0.08)
    ap.add_argument("--tol-deg", type=float, default=2.0)
    ap.add_argument("--tol-m", type=float, default=0.02)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--calib", default=None, help="servo_calib.json from jev_dir_fit")
    ap.add_argument("--config", default="real_robot/configs/piper_1.json")
    ap.add_argument("--url", default=os.environ.get("JEV_URL", "http://localhost:8000"))
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(os.path.join(a.dirs, "labels.jsonl"))]
    base = [r for r in rows if r["before_grasp"] == 12]
    rng = random.Random(0)
    jev_servo.time.sleep = lambda s: None  # no settling needed in sim
    from . import backends
    backends.time.sleep = lambda s: None
    cli = JevClient(a.url)
    from .config import load_config
    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    res = []
    for ep in range(a.episodes):
        r = base[ep % len(base)]
        arm = SimArm(np.asarray(Image.open(r["wrist"]).convert("RGB")), rng, a.err_deg, a.err_m)
        e0 = arm.err()
        s = JevServo(cli, arm, cfg, calib=a.calib, mirror=not a.no_mirror, log=lambda *_: None)
        out = s.recover(lift_m=0.0)
        e1 = arm.err()
        ok = abs(e1[0]) <= a.tol_deg and abs(e1[1]) <= a.tol_m
        calls = len(s.trace)
        res.append({"ep": ep, "src_episode": r["episode"], "start_err": e0, "end_err": e1,
                    "signs": [arm.sx, arm.sy], "success": ok, "servo_says_centred": out["centred"],
                    "jev_calls": calls, "trace": out["trace"]})
        print(f"ep {ep}: start J1 {e0[0]:+.1f} deg reach {e0[1] * 100:+.1f} cm -> "
              f"end {e1[0]:+.1f} deg {e1[1] * 100:+.1f} cm  {'OK' if ok else 'MISS'} ({calls} steps)", flush=True)
    sr = np.mean([x["success"] for x in res])
    end = np.array([x["end_err"] for x in res])
    summary = {"episodes": a.episodes, "success": float(sr),
               "median_end_err_deg": float(np.median(abs(end[:, 0]))),
               "median_end_err_cm": float(np.median(abs(end[:, 1])) * 100)}
    json.dump({"summary": summary, "episodes": res}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
