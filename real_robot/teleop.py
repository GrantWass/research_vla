"""Leader/follower teleop between two Pipers on separate CAN adapters.

The leader is put in teaching mode (MasterSlaveConfig 0xFA): it can be moved by
hand and broadcasts its joint targets. Each control step the follower is sent
those targets through SafetyFilter (limits + per-step rate), so a fast hand
motion makes the follower lag instead of jump. Startup first eases the follower
onto the leader's pose.

  python3 -m real_robot.teleop --leader can_piper2 \\
      --follower-config real_robot/configs/piper_1.json --seconds 120

With --record DIR it also records demos into a new LeRobot dataset (run it
with the LeRobot venv; see real_robot/recorder.py):

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.teleop --record demos/pick_cube \\
      --task "put the cube in the cup" --episodes 10 --episode-seconds 30

The leader is switched back to normal mode (0xFC) when the session ends;
--restore-leader does only that.
"""
from __future__ import annotations

import argparse
import signal
import time

import numpy as np

from .config import load_config
from .piper import PiperBackend, PiperLeader
from .safety import SafetyFilter

START_MISMATCH_RAD = 0.8  # refuse to start if follower and leader differ more (~46 deg; eased at 0.3 rad/s)
LEADER_GLITCH_RAD = 0.5  # leader joint jump between two samples that we treat as bad data
FOLLOW_LAG_ABORT = 1.0   # follower this far behind the leader on any joint: stop (stuck/blocked)


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
    """Follow the leader for `seconds`. on_step(state, cmd, lead) runs after each
    command; returning True ends the session early (status "ok")."""
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
            steps += 1
            if on_step is not None and on_step(cur, cmd, lead):
                break
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
    rec = ap.add_argument_group("demo recording (LeRobot venv)")
    rec.add_argument("--record", metavar="DIR", help="new LeRobot dataset directory")
    rec.add_argument("--task", help="instruction stored with every frame")
    rec.add_argument("--episodes", type=int, default=10)
    rec.add_argument("--episode-seconds", type=float, default=30.0)
    rec.add_argument("--reset-seconds", type=float, default=10.0,
                     help="pause between episodes (also before the first) to reset the scene")
    rec.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)
    if args.record and not args.task:
        ap.error("--record needs --task")

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
    recorder = None
    if args.record:
        from .recorder import DemoRecorder, create_dataset

        dataset = create_dataset(args.record, cfg, args.fps)
    robot = PiperBackend(cfg, cameras=None if args.record else _NoCameras())
    leader.connect()
    robot.connect()
    if args.record:
        recorder = DemoRecorder(dataset, robot, cfg, args.task, fps=args.fps,
                                episodes=args.episodes, episode_s=args.episode_seconds,
                                reset_s=args.reset_seconds)
    seconds = float("inf") if recorder else args.seconds
    try:
        summary = teleop(leader, robot, cfg, seconds, args.hz, on_step=recorder)
    except RuntimeError as e:  # e.g. leader never moved, or start gap too large
        summary = {"status": f"stopped: {e}", "steps": 0}
    finally:
        if recorder is not None:
            summary_saved = recorder.close()
        robot.close()
        # Leave the leader in normal mode (in teaching mode it stays silent on
        # the bus and ignores commands). It stays where it is, motors holding.
        leader.restore_follower_mode()
    if recorder is not None:
        summary["episodes_saved"] = summary_saved
        summary["dataset"] = args.record
    print(summary)
    return 0 if summary["status"] in ("ok", "interrupted") else 2


class _NoCameras:
    def read(self):
        return {}

    def close(self):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
