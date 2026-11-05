"""Leader/follower teleop between two Pipers on separate CAN adapters.

The leader is put in teaching mode (MasterSlaveConfig 0xFA): it can be moved by
hand and broadcasts its joint targets. Each control step the follower is sent
those targets through SafetyFilter (limits + per-step rate), so a fast hand
motion makes the follower lag instead of jump. Startup first eases the follower
onto the leader's pose.

  python3 -m real_robot.teleop --leader can_piper2 \\
      --follower-config real_robot/configs/piper_1.json --seconds 120

With --record DIR it also records demos into a new LeRobot dataset (run it
with the LeRobot venv, in your own terminal; see real_robot/recorder.py).
Between episodes both arms are parked, motors off. Space checks the motors
(temperature, faults), powers both arms up and homes them; wiggle the
leader, hold it still at home, and recording starts once the follower
follows. Enter saves the episode, r discards it, q ends the session; each
parks both arms again (keyed_session):

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.teleop --record demos/pick_cube \\
      --task "put the cube in the cup"

The leader is switched back to normal mode (0xFC) when the session ends;
--restore-leader does only that.
"""
from __future__ import annotations

import argparse
import signal
import time

import numpy as np

from .backends import ease_to
from .cameras import NoCameras
from .config import load_config
from .piper import PiperBackend, PiperLeader
from .recorder import KEY_HELP
from .safety import SafetyFilter

START_MISMATCH_RAD = 0.8  # refuse to start if follower and leader differ more (~46 deg; eased at 0.3 rad/s)
LEADER_GLITCH_RAD = 0.5  # leader joint jump between two samples that we treat as bad data
FOLLOW_LAG_ABORT = 1.0   # follower this far behind the leader on any joint: stop (stuck/blocked)


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
                    f"grip {lead[6] * 1000:.0f} mm (follower {cur[6] * 1000:.0f}) lag {np.degrees(lag):.1f} deg")
            sleep(max(0.0, 1.0 / hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # noqa: BLE001  any failure: stop and hold
        status = f"stopped: {e}"
    finally:
        robot.hold()
    return {"status": status, "steps": steps}


def wait_key(keys, wanted: str, sleep=time.sleep) -> str:
    while True:
        k = keys.get()
        if k is not None and k in wanted:
            return k
        sleep(0.05)


def park_if_on(arm, name: str, log=print) -> None:
    """Fold, damped stop, motors off; skipped if the motors are already off."""
    try:
        if arm.motors_on():
            log(f"parking the {name} (motors off)")
            arm.park(log=lambda *a: None)
    except Exception as e:  # noqa: BLE001  report; the arm holds or rests where it is
        log(f"parking the {name} failed ({e}); run `make park`")


def keyed_session(robot, leader_arm, leader, cfg, recorder, keys, hz: float,
                  wait_s: float = 120.0, log=print) -> None:
    """Record episodes with both arms parked, motors off, between them.

    Space: motor check (temperature, faults), power up, home both, far arm
    into teaching mode, follow and record (recorder.begin). The episode
    ends on Enter / r / q (or an error); then the far arm leaves teaching
    mode and both park while the episode saves in the background. Idle
    arms holding the fold heated the close arm's J5 to 67 C (2026-10-15).
    """
    from .piper import MotorsTooHot

    park_if_on(robot, "close arm", log)
    park_if_on(leader_arm, "far arm", log)
    while not recorder.done():
        log(f"\nREADY: arms parked. {KEY_HELP}")
        if wait_key(keys, " qQ") in "qQ":
            break
        try:
            robot.check_motors(log=log)
            leader_arm.check_motors(log=log)
        except MotorsTooHot as e:
            log(f"NOT STARTING, motors need to cool: {e}. Check `make status`; Space to try again")
            continue
        try:
            log("powering up and homing both arms")
            robot.enable()
            leader_arm.enable()
            leader_arm.home(log=lambda *a: None)
            robot.home(log=lambda *a: None)
            leader.teach()
            recorder.wait_for_save()
            recorder.begin()
            summary = teleop(leader, robot, cfg, float("inf"), hz, on_step=recorder,
                             wait_s=wait_s, log=log)
            if summary["status"] not in ("ok",):
                recorder.abandon(summary["status"])
                log(f"teleop {summary['status']}")
                if summary["status"] == "interrupted":
                    raise KeyboardInterrupt
        except RuntimeError as e:  # leader never moved, start gap, enable failed
            recorder.abandon(str(e))
            if recorder.save_error is not None:
                raise  # a lost episode ends the session
            log(f"episode not recorded: {e}")
        finally:
            leader.restore_follower_mode()
            try:
                leader_arm.enable()  # after teaching mode: hold, so park can fold it
            except RuntimeError as e:
                log(f"far arm: {e}")
            park_if_on(robot, "close arm", log)
            park_if_on(leader_arm, "far arm", log)


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
    rec.add_argument("--episodes", type=int, default=0,
                     help="end the session after this many saved episodes (0: until q)")
    rec.add_argument("--episode-seconds", type=float, default=None,
                     help="fixed-length episodes instead of the keys (Space/Enter/r/q)")
    rec.add_argument("--reset-seconds", type=float, default=10.0,
                     help="with --episode-seconds: pause between episodes to reset the scene")
    rec.add_argument("--max-episode-seconds", type=float, default=180.0,
                     help="with the keys: discard an episode this long (missed Enter)")
    rec.add_argument("--fps", type=int, default=30)
    rec.add_argument("--leader-config", default="real_robot/configs/piper_2.json",
                     help="leader arm config, for homing it")
    rec.add_argument("--no-home", action="store_true",
                     help="don't move both arms home before and after the session")
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
    recorder = keys = leader_arm = None
    home = bool(args.record) and not args.no_home
    if args.record:
        from .recorder import DemoRecorder, create_dataset

        dataset = create_dataset(args.record, cfg, args.fps)
        if args.episode_seconds is None:
            from .keys import KeyReader

            keys = KeyReader()  # fails early without a terminal
    if keys is not None:
        return keyed_main(args, cfg, leader, dataset, keys)
    robot = PiperBackend(cfg, cameras=None if args.record else NoCameras())
    robot.connect()
    if home:  # both arms start folded, facing the table: the start check passes
        print("moving both arms home")
        leader_arm = PiperBackend(load_config(args.leader_config), cameras=NoCameras())
        leader_arm.connect()
        leader_arm.home()
        robot.home()
    leader.connect()
    if args.record:
        recorder = DemoRecorder(dataset, robot, cfg, args.task, fps=args.fps,
                                episodes=args.episodes or 10, episode_s=args.episode_seconds,
                                reset_s=args.reset_seconds)
    seconds = float("inf") if recorder else args.seconds
    try:
        summary = teleop(leader, robot, cfg, seconds, args.hz, on_step=recorder)
    except RuntimeError as e:  # e.g. leader never moved, or start gap too large
        summary = {"status": f"stopped: {e}", "steps": 0}
    finally:
        if recorder is not None:
            summary_saved = recorder.close()
        # Leave the leader in normal mode (in teaching mode it stays silent on
        # the bus and ignores commands). It stays where it is, motors holding.
        leader.restore_follower_mode()
        if home:
            print("moving both arms home")
            try:
                robot.home()
                leader_arm.home()
            except Exception as e:  # noqa: BLE001  report; the arms hold where they are
                print(f"homing failed ({e}); the arms hold where they are")
        robot.close()
    if recorder is not None:
        summary["episodes_saved"] = summary_saved
        summary["dataset"] = args.record
    print(summary)
    return 0 if summary["status"] in ("ok", "interrupted") else 2


def keyed_main(args, cfg, leader, dataset, keys) -> int:
    """make record: keyed_session, then park both arms, motors off."""
    from .recorder import KeyedRecorder

    robot = PiperBackend(cfg)
    leader_arm = PiperBackend(load_config(args.leader_config), cameras=NoCameras())
    recorder = None
    status = "ok"
    try:
        robot.connect(enable=False)
        leader_arm.connect(enable=False)
        leader.arm.ConnectPort()
        recorder = KeyedRecorder(dataset, robot, cfg, args.task, fps=args.fps,
                                 key_source=keys.get, episodes=args.episodes,
                                 max_episode_s=args.max_episode_seconds)
        print(f"task '{args.task}', dataset {args.record}")
        keyed_session(robot, leader_arm, leader, cfg, recorder, keys, args.hz)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # noqa: BLE001  stop, park, keep what is saved
        status = f"stopped: {e}"
        print(status)
    finally:
        keys.close()  # give the terminal back
        saved = recorder.close() if recorder is not None else 0
        leader.restore_follower_mode()
        park_if_on(robot, "close arm")
        park_if_on(leader_arm, "far arm")
        robot.close()
    print({"status": status, "episodes_saved": saved, "dataset": args.record})
    return 0 if status in ("ok", "interrupted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
