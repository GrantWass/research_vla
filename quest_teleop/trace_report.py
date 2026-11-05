#!/usr/bin/env python3
"""Latency report for a Quest teleop trace (quest_teleop/trace.py).

For each stage it estimates the delay (the time shift that best lines up
the two signals, 5 ms resolution) and the tracking error:
  hand -> IK      FK of the IK solution vs the /delta_pose target (mm)
  IK -> command   ik_guard: filter + rate limit
  command -> arm  firmware (speed cap, planning)
plus how jerky the arm's motion is (share of time the arm is stopped while
its command moves). Run in the conda env "vt" (needs pinocchio):

  python quest_teleop/trace_report.py quest_trace.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

GRID_HZ = 1000.0


def _resample(rows: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, rows[:, 0], rows[:, i]) for i in range(1, rows.shape[1])], axis=1)


def delay_ms(a: np.ndarray, b: np.ndarray, max_ms: float = 1000.0) -> float:
    """Lag of b behind a (both (t, dims) rows): the shift that best lines b up with a."""
    t0, t1 = max(a[0, 0], b[0, 0]) + max_ms / 1000, min(a[-1, 0], b[-1, 0])
    t = np.arange(t0, t1, 1 / GRID_HZ)
    pb = _resample(b, t)
    lags = np.arange(0, int(max_ms * GRID_HZ / 1000), 5)
    rms = [np.mean((_resample(a, t - k / GRID_HZ) - pb) ** 2) for k in lags]
    return float(lags[int(np.argmin(rms))] * 1000 / GRID_HZ)


def fk_positions(ik_rows: np.ndarray) -> np.ndarray:
    """(t, x, y, z) of the IK's end-effector frame, same model as the IK node."""
    import pinocchio as pin
    from ament_index_python.packages import get_package_share_directory
    sys.path.insert(0, os.path.expanduser("~/QuestArmTeleop/install/oculus_reader/lib/oculus_reader"))
    from arm_ik_pose_node import ArmIK
    pkg = get_package_share_directory("agx_arm_description")
    ik = ArmIK(os.path.join(pkg, "agx_arm_urdf/piper/urdf/piper_with_gripper_description.urdf"), [pkg],
               ["gripper_base_joint", "gripper", "gripper_joint1", "gripper_joint2"], "joint6", "ee",
               [0, 0, 0], [0, 0, 0.13], [], 1, 1, 0, 0, 1, 1e-4, False, False, "", "", 0.1, 1)
    m = ik.reduced_robot.model
    d = m.createData()
    out = []
    for row in ik_rows:
        pin.framesForwardKinematics(m, d, row[1:7])
        out.append([row[0], *d.oMf[ik.ee_id].translation])
    return np.asarray(out)


def main(path: str) -> None:
    tr = np.load(path)
    ik, cmd, fb, dp = tr["ik"], tr["cmd"], tr["feedback"], tr["delta_pose"]
    print(f"{path}: delta_pose {len(dp)}, ik {len(ik)}, cmd {len(cmd)}, feedback {len(fb)}")
    for name, rows in (("delta_pose", dp), ("ik", ik), ("cmd", cmd)):
        if len(rows) > 1:
            dt = np.diff(rows[:, 0])
            print(f"  {name:10s} rate {1 / np.median(dt):5.0f} Hz, gaps > 100 ms: {int(np.sum(dt > 0.1))}")
    if len(ik) < 10 or len(cmd) < 10:
        print("not enough teleop data (engage with A and move the controller)")
        return

    ee = fk_positions(ik)
    target = _resample(dp[:, :4], ee[:, 0])
    err = np.linalg.norm(ee[:, 1:] - target, axis=1) * 1000
    print(f"hand -> IK:      delay {delay_ms(dp[:, :4], ee):4.0f} ms, IK position error "
          f"median {np.median(err):.0f} mm, p95 {np.percentile(err, 95):.0f} mm")
    print(f"IK -> command:   delay {delay_ms(ik, cmd):4.0f} ms (ik_guard filter + rate limit)")
    t = np.arange(max(cmd[0, 0], fb[0, 0]), min(cmd[-1, 0], fb[-1, 0]), 1 / GRID_HZ)
    gap = np.degrees(np.abs(_resample(cmd, t) - _resample(fb, t))).max(axis=1)
    print(f"command -> arm:  delay {delay_ms(cmd, fb):4.0f} ms (firmware), arm behind command "
          f"median {np.median(gap):.1f} deg, p95 {np.percentile(gap, 95):.1f} deg")

    # Jerkiness: while the command moves, how often is the arm (nearly) stopped?
    t = np.arange(cmd[0, 0], cmd[-1, 0], 0.01)
    vc = np.degrees(np.abs(np.diff(_resample(cmd, t), axis=0))).max(axis=1) * 100
    vf = np.degrees(np.abs(np.diff(_resample(fb, t), axis=0))).max(axis=1) * 100
    moving = vc > 5
    if moving.any():
        stalls = np.mean(vf[moving] < 1)
        print(f"jerkiness:       arm stopped (<1 deg/s) during {stalls * 100:.0f}% of the time its command "
              f"moves (>5 deg/s); arm speed p95 {np.percentile(vf, 95):.0f} deg/s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quest_trace.npz")
