"""Leader/follower teleop between two Pipers on separate CAN adapters.

The leader is put in teaching mode (MasterSlaveConfig 0xFA): it can be moved by
hand and broadcasts its joint targets. Each control step the follower is sent
those targets through SafetyFilter (limits + per-step rate), so a fast hand
motion makes the follower lag instead of jump. Startup first eases the follower
onto the leader's pose.

  python3 -m real_robot.teleop --leader can_piper2 \\
      --follower-config real_robot/configs/piper_1.json --seconds 120

The leader is switched back to normal mode (0xFC) when the session ends;
--restore-leader does only that.
"""
from __future__ import annotations

import argparse
import signal
import time

import numpy as np

from .backends import _MDEG_PER_RAD, _UM_PER_M, PiperBackend, _default_piper_factory
from .config import load_config
from .safety import SafetyFilter

START_MISMATCH_RAD = 0.8  # refuse to start if follower and leader differ more (~46 deg; eased at 0.3 rad/s)
LEADER_GLITCH_RAD = 0.5  # leader joint jump between two samples that we treat as bad data
FOLLOW_LAG_ABORT = 1.0   # follower this far behind the leader on any joint: stop (stuck/blocked)


class PiperLeader:
    def __init__(self, can: str, sdk_factory=None):
        self.arm = (sdk_factory or _default_piper_factory)(can)

    def connect(self) -> None:
        self.arm.ConnectPort()
        self.arm.MasterSlaveConfig(0xFA, 0, 0, 0)  # teaching input arm
        time.sleep(0.3)

    def wait_for_broadcast(self, timeout_s: float = 30.0, settle_s: float = 0.2,
                           tol_rad: float = 0.02, sleep=time.sleep) -> None:
        """Block until the leader's broadcast is complete and steady.

        A teaching arm broadcasts only while it is being moved, and its six
        joints arrive in three CAN frames (0x155-0x157): until each has been
        seen the SDK reports 0 for those joints. So wait for the first frame,
        then for readings that stay within tol_rad for settle_s.
        """
        deadline = time.monotonic() + timeout_s
        ts0 = self.arm.GetArmJointCtrl().time_stamp
        while self.arm.GetArmJointCtrl().time_stamp == ts0:
            if time.monotonic() > deadline:
                raise RuntimeError("leader sent nothing: move it a little by hand "
                                   "(is it in teaching mode and powered?)")
            sleep(0.02)
        ref, steady_since = self.read(), time.monotonic()
        while time.monotonic() - steady_since < settle_s:
            if time.monotonic() > deadline:
                raise RuntimeError("leader readings never settled; hold it still for a moment")
            sleep(0.02)
            q = self.read()
            if np.max(np.abs(q[:6] - ref[:6])) > tol_rad:
                ref, steady_since = q, time.monotonic()

    def restore_follower_mode(self) -> None:
        self.arm.MasterSlaveConfig(0xFC, 0, 0, 0)

    def read(self) -> np.ndarray:
        """(7,) joint targets the leader broadcasts: 6 joints rad + gripper m."""
        j = self.arm.GetArmJointCtrl().joint_ctrl
        g = self.arm.GetArmGripperCtrl().gripper_ctrl
        return np.asarray([getattr(j, f"joint_{i}") / _MDEG_PER_RAD for i in range(1, 7)]
                          + [abs(g.grippers_angle) / _UM_PER_M], dtype=np.float64)

    def read_feedback(self) -> np.ndarray:
        """(7,) measured joints (rad) + gripper (m), for comparison."""
        j = self.arm.GetArmJointMsgs().joint_state
        g = self.arm.GetArmGripperMsgs().gripper_state
        return np.asarray([getattr(j, f"joint_{i}") / _MDEG_PER_RAD for i in range(1, 7)]
                          + [g.grippers_angle / _UM_PER_M], dtype=np.float64)


def ease_to(robot: PiperBackend, safety: SafetyFilter, target: np.ndarray, hz: float,
            vel: float = 0.3, sleep=time.sleep) -> None:
    """Move the follower to target with a cosine profile at <= vel rad/s."""
    start = robot.get_joint_state()
    n = max(int(hz), int(np.max(np.abs(target[:6] - start[:6])) / vel * hz))
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        robot.command_joints(safety.filter(start + s * (target - start), robot.get_joint_state()))
        sleep(1.0 / hz)


def teleop(leader: PiperLeader, robot: PiperBackend, cfg, seconds: float, hz: float,
           on_step=None, sleep=time.sleep, log=print, wait_s: float = 120.0) -> dict:
    safety = SafetyFilter(cfg)
    log("waiting for the leader: move it a little by hand, then hold it still")
    leader.wait_for_broadcast(timeout_s=wait_s, sleep=sleep)
    lead = np.clip(leader.read(), cfg.joint_lower, cfg.joint_upper)
    gap = np.abs(lead[:6] - robot.get_joint_state()[:6])
    if np.max(gap) > START_MISMATCH_RAD:
        robot.hold()
        raise RuntimeError(f"leader and follower differ by {np.degrees(gap).round(0)} deg; "
                           f"bring them closer (e.g. both folded at rest) before starting")
    log(f"leader at deg {np.degrees(lead[:6]).round(1)}, grip {lead[6] * 1000:.1f} mm; "
        f"easing follower onto it")
    ease_to(robot, safety, lead, hz, sleep=sleep)
    log("FOLLOWING: move the leader arm now")
    status, steps, prev, t0 = "ok", 0, lead, time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            tick = time.monotonic()
            lead = leader.read()
            if not np.all(np.isfinite(lead)) or np.max(np.abs(lead[:6] - prev[:6])) > LEADER_GLITCH_RAD:
                raise RuntimeError(f"leader reading jumped: {np.degrees(lead[:6]).round(1)}")
            prev = lead
            cur = robot.get_joint_state()
            target = np.clip(lead, cfg.joint_lower, cfg.joint_upper)
            gaps = np.abs(target[:6] - cur[:6])
            lag = float(np.max(gaps))
            if lag > FOLLOW_LAG_ABORT:
                raise RuntimeError(f"follower joint {int(np.argmax(gaps)) + 1} is {lag:.2f} rad "
                                   f"behind the leader")
            cmd = cur + np.clip(target - cur, -safety.max_delta, safety.max_delta)
            robot.command_joints(cmd)
            if on_step is not None:
                on_step(cur, cmd, lead)
            steps += 1
            if steps % int(hz * 5) == 0:
                log(f"t={time.monotonic() - t0:5.1f}s leader deg {np.degrees(lead[:6]).round(0)} "
                    f"lag {np.degrees(lag):.1f} deg")
            sleep(max(0.0, 1.0 / hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # noqa: BLE001  any failure: stop and hold
        status = f"stopped: {e}"
    finally:
        robot.hold()
    return {"status": status, "steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leader", default="can_piper2", help="leader CAN interface")
    ap.add_argument("--follower-config", default="real_robot/configs/piper_1.json")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--speed-percent", type=int, default=60,
                    help="follower firmware speed cap for teleop (configs use 30 for policies; "
                         "at 30 a hand-guided leader outruns the follower)")
    ap.add_argument("--restore-leader", action="store_true",
                    help="only switch the leader arm back to follower mode, then exit")
    args = ap.parse_args(argv)

    # A SIGTERM (e.g. from `timeout` or kill) must still run the cleanup below:
    # hold the follower and take the leader out of teaching mode.
    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)

    leader = PiperLeader(args.leader)
    if args.restore_leader:
        leader.arm.ConnectPort()
        leader.restore_follower_mode()
        print(f"{args.leader}: MasterSlaveConfig -> follower (0xFC)")
        return 0
    cfg = load_config(args.follower_config)
    cfg.arm_speed_percent = args.speed_percent
    robot = PiperBackend(cfg, cameras=_NoCameras())
    leader.connect()
    robot.connect()
    try:
        summary = teleop(leader, robot, cfg, args.seconds, args.hz)
    except RuntimeError as e:  # e.g. leader never moved, or start gap too large
        summary = {"status": f"stopped: {e}", "steps": 0}
    finally:
        robot.close()
        # Leave the leader in normal mode (in teaching mode it stays silent on
        # the bus and ignores commands). It stays where it is, motors holding.
        leader.restore_follower_mode()
    print(summary)
    return 0 if summary["status"] in ("ok", "interrupted") else 2


class _NoCameras:
    def read(self):
        return {}

    def close(self):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
