"""JEV visual servoing over the wrist camera: recover from a missed grasp.

JEV answers direction buckets about the wrist image ("where is the cube
relative to the point between the fingers: far left .. far right", and the
same up/down); the arm steps in joint space toward it and halves the step
whenever the answer flips side (the halving rule from JEV's own arm demo).

No hand-written geometry:
  - image direction -> joint direction is probed online: jog a little, ask
    again, keep the sign that moved the cube toward the centre
  - "reach farther / closer" -> joint deltas uses a Jacobian of the end
    pose (x, y, z) over J2, J3, J5 measured by small jogs at the start
  - J1 handles left/right directly

  bucket value v = sum(p_i * [-2, -1, 0, 1, 2]); |v| < CENTRE: aligned on that axis
"""
from __future__ import annotations

import time

import numpy as np

from .jev_dir_eval import LR, UD, VALUES, ask

CENTRE = {"lr": 0.3, "ud": 0.2, "reach": 0.3}  # |bucket value| below this: aligned on that axis
REACH_UNIT = 0.03  # m of predicted reach per unit of value (CENTRE -> ~1 cm)


class ServoStop(RuntimeError):
    """The servo stopped itself: arm out of bounds or not under control."""


def bucket_value(p) -> float:
    return float(np.dot(np.asarray(p, float), VALUES))


class JevServo:
    def __init__(self, client, robot, cfg, obj: str = "Rubik's cube", step_deg: float = 4.0,
                 reach_step_m: float = 0.02, min_step_deg: float = 0.5, max_iters: int = 20,
                 hz: float = 25.0, calib: dict | str | None = None, dir_map: dict | str | None = None,
                 mirror: bool = False, log=print):
        from .safety import SafetyFilter

        self.cli, self.robot, self.cfg, self.obj, self.log = client, robot, cfg, obj, log
        self.step0, self.reach0, self.min_step, self.max_iters = step_deg, reach_step_m, min_step_deg, max_iters
        self.hz = hz
        self.safety = SafetyFilter(cfg)
        self.trace: list[dict] = []
        # JEV reads a centred cube as slightly left / up: subtract that bias,
        # fitted by real_robot.jev_dir_fit on shifted demo frames (servo_calib.json)
        if isinstance(calib, str):
            import json
            calib = json.load(open(calib))
        self.bias = {k: v["bias"] for k, v in (calib or {}).items()}
        # mirror: also ask about the left-right mirrored image and average
        # v - v_mirrored; cancels JEV's left bias and halves its noise.
        self.mirror = mirror
        # dir_map (real_robot.jev_dir_fit, fitted on demo approach frames):
        # both questions' probabilities -> reach still to go (m). Used for the
        # near/far axis instead of the up/down bucket alone (98% sign accuracy
        # on held-out demos vs ~70%).
        if isinstance(dir_map, str):
            import json
            dir_map = json.load(open(dir_map))
        self.w_reach = np.asarray(dir_map["dr_m"]) if dir_map else None

    # -- primitives --------------------------------------------------------------------
    def _see(self, which: str) -> tuple[float, list]:
        img = self.robot.get_images()["cam_wrist"]
        if which == "reach":  # learned: metres to go, in units of REACH_UNIT
            p_lr, p_ud = ask(self.cli, img, self.obj, "lr"), ask(self.cli, img, self.obj, "ud")
            x = np.r_[p_lr, p_ud, bucket_value(p_lr), bucket_value(p_ud), 1.0]
            return float(x @ self.w_reach) / REACH_UNIT, p_lr + p_ud
        p = ask(self.cli, img, self.obj, which)
        if which == "lr" and self.mirror:
            pm = ask(self.cli, np.ascontiguousarray(img[:, ::-1]), self.obj, "lr")
            return 0.5 * (bucket_value(p) - bucket_value(pm)), p
        return bucket_value(p) - self.bias.get(which, 0.0), p

    # -- workspace guard (2026-11-05: an oscillating reach search drove the
    # arm into the table; the controller faulted and the arm dropped) ----------
    z_floor = None          # m: never command below this end-effector height
    max_j1_deg = 20.0       # from where the guard was armed
    max_radial_m = 0.08
    _home = None
    _hold_z = None

    def guard_from_here(self, z_floor: float) -> None:
        q = self.robot.get_joint_state()
        p = np.asarray(self.robot.get_end_pose()[:3], float)
        self._home = (q.copy(), p.copy())
        self.z_floor = z_floor

    def _arm_ok(self) -> None:
        probs = getattr(self.robot, "control_problems", lambda: [])()
        motors = getattr(self.robot, "motors_on", lambda: True)()
        if probs or not motors:
            raise ServoStop("arm not under control: " + ("; ".join(probs) or "motors off"))

    def _check_pose(self) -> str | None:
        if self._home is None:
            return None
        q0, p0 = self._home
        q = self.robot.get_joint_state()
        p = np.asarray(self.robot.get_end_pose()[:3], float)
        if self.z_floor is not None and p[2] < self.z_floor:
            return f"z {p[2]:.3f} below floor {self.z_floor:.3f}"
        if abs(np.degrees(q[0] - q0[0])) > self.max_j1_deg:
            return f"J1 {np.degrees(q[0] - q0[0]):+.1f} deg from start (max {self.max_j1_deg})"
        dr = abs(np.linalg.norm(p[:2]) - np.linalg.norm(p0[:2]))
        if dr > self.max_radial_m:
            return f"reach {dr:.3f} m from start (max {self.max_radial_m})"
        return None

    def _move(self, dq: np.ndarray) -> None:
        self._arm_ok()
        self._raw_move(dq)
        self._arm_ok()
        bad = self._check_pose()
        if bad:
            self._raw_move(-dq)  # undo the step that crossed the line
            self.robot.hold()
            raise ServoStop(bad)

    def _raw_move(self, dq: np.ndarray) -> None:
        from .backends import ease_to

        q = self.robot.get_joint_state()
        target = np.clip(q + dq, self.cfg.joint_lower, self.cfg.joint_upper)
        ease_to(self.robot, self.safety, target, self.hz, vel=0.15)
        time.sleep(0.3)  # let the image settle

    def jacobian(self, joints=(1, 2, 4), jog_deg: float = 2.0) -> np.ndarray:
        """d(x, y, z) / d(q_j) for the given joint indices, by +jog / back."""
        J = np.zeros((3, len(joints)))
        for k, j in enumerate(joints):
            p0 = np.asarray(self.robot.get_end_pose()[:3], float)
            dq = np.zeros(len(self.robot.get_joint_state()))
            dq[j] = np.radians(jog_deg)
            self._move(dq)
            p1 = np.asarray(self.robot.get_end_pose()[:3], float)
            self._move(-dq)
            J[:, k] = (p1 - p0) / dq[j]
        return J

    def reach_dq(self, J: np.ndarray, d_radial: float, joints=(1, 2, 4)) -> np.ndarray:
        """Joint deltas moving the gripper d_radial (m) away from the base, height kept."""
        p = np.asarray(self.robot.get_end_pose()[:3], float)
        u = p[:2] / max(np.linalg.norm(p[:2]), 1e-6)
        dz = 0.0 if self._hold_z is None else float(np.clip(self._hold_z - p[2], -0.01, 0.01))
        want = np.array([u[0] * d_radial, u[1] * d_radial, dz])
        sol, *_ = np.linalg.lstsq(J, want, rcond=None)
        dq = np.zeros(len(self.robot.get_joint_state()))
        dq[list(joints)] = sol
        return dq

    # -- one axis ------------------------------------------------------------------
    def servo_axis(self, which: str, make_dq, known_sign: bool = False) -> bool:
        """make_dq(signed step) -> joint delta. Returns True when centred.
        known_sign: v > 0 already means "step +" (the learned reach map)."""
        step, sign, last = 1.0, 1.0, None
        locked, flat, limit = known_sign, 0, 2  # sign search: see below
        v, p = self._see(which)
        for it in range(self.max_iters):
            self.trace.append({"axis": which, "iter": it, "v": round(v, 3), "p": [round(x, 3) for x in p],
                               "step": step, "sign": sign})
            self.log(f"[servo] {which} iter {it}: bucket value {v:+.2f}")
            if abs(v) < CENTRE[which]:  # confirm with a second look: one reading is noisy
                v, p = self._see(which)
                if abs(v) < CENTRE[which]:
                    return True
                continue
            if last is not None and np.sign(v) != last:
                step *= 0.5  # overshot: halve
                if step * self.step0 < self.min_step:
                    return abs(v) < 2 * CENTRE[which]
            last = np.sign(v)
            mag = 1.0 if abs(v) > 1.2 else 0.5  # "far" buckets take full steps
            self._move(make_dq(sign * np.sign(v) * step * mag))
            v2, p = self._see(which)
            # Which joint direction moves the cube toward the centre is not
            # known up front (wrist roll, camera mount). A visible change
            # settles it: better or crossed = keep, worse = flip. While the view
            # is saturated ("far" either way) nothing changes, so keep going
            # `limit` moves, then flip with a doubled range (expanding search).
            if not locked:
                if np.sign(v2) != np.sign(v) or abs(v2) < abs(v) - 0.25:
                    locked = True
                elif abs(v2) > abs(v) + 0.25:
                    sign, locked = -sign, True
                    self.log(f"[servo] {which}: moving the wrong way, flipping")
                else:
                    flat += 1
                    if flat >= limit:
                        sign, flat, limit = -sign, 0, 2 * limit
                        self.log(f"[servo] {which}: no change, searching the other way")
            v = v2
        return False

    def descend(self, J: np.ndarray, z_target: float, step_m: float = 0.01,
                max_iters: int = 30, joints=(1, 2, 4)) -> dict:
        """Straight down to z_target in <= step_m moves, re-reading the end pose
        each move and correcting x/y drift. Stops early when the arm stops
        going down (fingers on the cube or the table)."""
        p0 = np.asarray(self.robot.get_end_pose()[:3], float)
        stuck = 0
        for it in range(max_iters):
            p = np.asarray(self.robot.get_end_pose()[:3], float)
            if p[2] <= z_target + 0.003:
                return {"reached": True, "z": float(p[2]), "iters": it}
            dz = max(z_target - p[2], -step_m)
            want = np.array([p0[0] - p[0], p0[1] - p[1], dz])
            sol, *_ = np.linalg.lstsq(J, want, rcond=None)
            dq = np.zeros(len(self.robot.get_joint_state()))
            dq[list(joints)] = sol
            self._move(dq)
            z1 = float(self.robot.get_end_pose()[2])
            stuck = stuck + 1 if p[2] - z1 < 0.25 * abs(dz) else 0
            self.log(f"[servo] descend {it}: z {p[2]:.3f} -> {z1:.3f} (target {z_target:.3f})")
            if stuck >= 2:
                return {"reached": False, "blocked": True, "z": z1, "iters": it + 1}
        return {"reached": False, "z": float(self.robot.get_end_pose()[2]), "iters": max_iters}

    def recover(self, lift_m: float = 0.04) -> dict:
        """Lift, centre left/right with J1, then near/far along the reach. The
        caller decides what comes next (hand back to pi0.5, or descend + grasp)."""
        t0 = time.time()
        n = len(self.robot.get_joint_state())
        J = self.jacobian()
        lift, *_ = np.linalg.lstsq(J, np.array([0, 0, lift_m]), rcond=None)
        dq = np.zeros(n)
        dq[[1, 2, 4]] = lift
        self._move(dq)
        self._hold_z = float(self.robot.get_end_pose()[2])

        def j1(s):
            d = np.zeros(n)
            d[0] = np.radians(self.step0 * s)
            return d

        def reach(s):
            return self.reach_dq(J, self.reach0 * s)

        ok_lr = self.servo_axis("lr", j1)
        ok_ud = (self.servo_axis("reach", reach, known_sign=True) if self.w_reach is not None
                 else self.servo_axis("ud", reach))
        ok_lr2 = self.servo_axis("lr", j1) if ok_ud else ok_lr  # reach moves can shift left/right
        return {"centred": bool(ok_lr2 and ok_ud), "lr": ok_lr2, "ud": ok_ud,
                "seconds": round(time.time() - t0, 1), "trace": self.trace}
