"""pi0.5 + JEV: run the policy, let JEV judge grasp and place, retry on failure.

Takes every real_robot.runner flag, plus the --jev-* flags below. Each
attempt is one runner episode watched by real_robot.jev.JevMonitor. When JEV
says the grasp missed, the cube fell, or the place missed, the arm opens its
gripper, folds home and pi0.5 starts again from the new scene, up to
--attempts times. JEV saying "in the bowl" ends the trial early.

  # JEV server first (scripts/serve_jev.sh), policy server optional (make policy-server)
  python -m real_robot.jev_guard --backend piper --config real_robot/configs/piper_1.json \\
      --lerobot outputs/<run>/checkpoints/004033/pretrained_model_merged ... --live \\
      --jev-object "Rubik's cube" --jev-container bowl --attempts 3

--jev off runs the same loop with JEV disabled (one attempt, no early stop):
the A/B baseline. --jev-dry asks JEV and logs its verdicts but never stops
or retries: to score JEV's judgement against what you see.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from .backends import ease_to, make_backend
from .config import load_config
from .jev import JevClient, JevMonitor
from .runner import DoneKey, build_parser, make_policy, preflight, run
from .safety import SafetyFilter

GRIPPER_OPEN_M = 0.07


def reset_arm(robot, cfg, log=print) -> None:
    """Open the gripper where it is (drop anything half-held), then fold home."""
    q = robot.get_joint_state()
    opened = q.copy()
    for k in cfg.gripper_idx:
        opened[k] = gripper_open(cfg)
    ease_to(robot, SafetyFilter(cfg), np.clip(opened, cfg.joint_lower, cfg.joint_upper),
            cfg.control_hz)
    time.sleep(0.5)
    if hasattr(robot, "home"):
        robot.home(log=log)


def gripper_open(cfg) -> float:
    """Fully open in the config's gripper units (m, or 0..1 for the banana/cube
    configs: 0.07 there would barely open it)."""
    rng = getattr(cfg, "gripper_sim_range", None)
    return float(rng[1]) if rng else GRIPPER_OPEN_M


def servo_recover(robot, cfg, client, args, log=print) -> dict:
    """After a missed grasp or a hover: open, then
    --recover servo        JEV centres the gripper over the object, pi0.5 takes over
    --recover servo-grasp  JEV centres, then descend to --grasp-z, close, check
    --recover descend      no centring (pi0.5 already lined up): descend, close, check
    Every servo move is bounded (z floor, J1 and reach excursion) and stops
    if the arm leaves our control."""
    from .jev_servo import JevServo, ServoStop

    q = robot.get_joint_state()
    opened = q.copy()
    for k in cfg.gripper_idx:
        opened[k] = gripper_open(cfg)
    ease_to(robot, SafetyFilter(cfg), np.clip(opened, cfg.joint_lower, cfg.joint_upper), cfg.control_hz)
    s = JevServo(client, robot, cfg, obj=args.jev_object, calib=args.servo_calib,
                 dir_map=args.servo_map, mirror=False, log=log)
    s.guard_from_here(z_floor=args.grasp_z - args.z_margin)
    try:
        if args.recover == "descend":
            out = {"centred": True, "result": "skipped centring"}
        else:
            out = s.recover(lift_m=args.servo_lift)
            out["result"] = "centred" if out["centred"] else "not_centred"
        if args.recover == "servo" or not out["centred"]:
            return out
        J = s.jacobian()
        out["descend"] = s.descend(J, args.grasp_z)
        closed = robot.get_joint_state()
        for k in cfg.gripper_idx:
            closed[k] = 0.0
        ease_to(robot, SafetyFilter(cfg), closed, cfg.control_hz)
        time.sleep(0.5)
        lift, *_ = np.linalg.lstsq(J, np.array([0.0, 0.0, args.servo_lift]), rcond=None)
        dq = np.zeros(len(q))
        dq[[1, 2, 4]] = lift
        s._move(dq)
    except ServoStop as e:
        robot.hold()
        log(f"[servo] STOPPED: {e}")
        return {"result": f"stopped: {e}", "centred": False, "trace": s.trace}
    except Exception as e:  # never leave the arm mid-servo on an error
        robot.hold()
        return {"result": f"error: {type(e).__name__}: {e}", "centred": False, "trace": s.trace}
    p, _ = client.yes(robot.get_images(), (
        f"Is the {args.jev_object} held between the robot gripper's fingers right now "
        f"(lifted off the table, not lying on the table)?"), args.instruction,
        float(robot.get_joint_state()[cfg.gripper_idx[0]]))
    out.update(result="grasped" if p >= args.grasp_thr else "grasp_failed", p_holding=p)
    return out


def add_jev_args(ap) -> None:
    g = ap.add_argument_group("JEV")
    g.add_argument("--jev", choices=["on", "off"], default="on")
    g.add_argument("--jev-url", default=os.environ.get("JEV_URL", "http://localhost:8000"))
    g.add_argument("--jev-object", default="Rubik's cube")
    g.add_argument("--jev-container", default="bowl")
    g.add_argument("--jev-dry", action="store_true", help="judge and log, never stop or retry")
    g.add_argument("--attempts", type=int, default=3)
    g.add_argument("--grasp-thr", type=float, default=0.35, help="P(holding) below this = missed")
    g.add_argument("--place-thr", type=float, default=0.6, help="P(in container) at or above = success")
    g.add_argument("--no-gripper-text", action="store_true",
                   help="do not tell JEV the gripper width (images only)")
    g.add_argument("--recover", choices=["home", "servo", "servo-grasp", "descend"], default="home",
                   help="after a missed grasp: fold home and retry (home); JEV-servo over the object, "
                        "then pi0.5 retries from there (servo); or servo, descend, close (servo-grasp)")
    g.add_argument("--servo-calib", default=None,
                   help="per-axis bias of zero-shot JEV's direction buckets (outputs/jev_dir_eval/"
                        "servo_calib.json); not needed with the fine-tuned adapter")
    g.add_argument("--servo-map", default=None,
                   help="zero-shot JEV probabilities -> reach map (outputs/jev_dir_eval/dir_map.json); "
                        "default: the up/down bucket, which the fine-tuned adapter answers directly")
    g.add_argument("--servo-lift", type=float, default=0.04, help="lift before servoing, m")
    g.add_argument("--z-margin", type=float, default=0.015,
                   help="servo never goes below --grasp-z minus this (m)")
    g.add_argument("--grasp-z", type=float, default=0.151,
                   help="end-pose z of a grasp, m (median of the cube_grid demos)")
    g.add_argument("--empty-check", action="store_true",
                   help="also call a grasp missed when the gripper closes on nothing (no JEV)")


class _Either:
    """Stop when any of these stop events is set (each may be None)."""

    def __init__(self, *events):
        self.events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self.events)


def main(argv=None) -> int:
    ap = build_parser()
    ap.description = __doc__
    add_jev_args(ap)
    args = ap.parse_args(argv)
    if args.park_when_done and not args.live:
        ap.error("--park-when-done needs --live")
    cfg = load_config(args.config)
    if args.speed_percent is not None:
        cfg.arm_speed_percent = args.speed_percent
    robot = make_backend(args.backend, cfg)
    robot.connect()
    out = args.log_dir or os.path.join(
        "real_robot_logs", time.strftime("%Y%m%d-%H%M%S") + f"-jev_{args.jev}" +
        ("-live" if args.live else "-dry"))
    os.makedirs(out, exist_ok=True)
    try:
        problems = preflight(robot, cfg, SafetyFilter(cfg), args.lerobot)
        for p in problems:
            print(f"PREFLIGHT: {p}")
        if args.live:
            if problems:
                print("refusing --live with preflight problems")
                return 1
            if not args.yes and input(
                    f"LIVE at speed {args.speed}, up to {args.attempts} attempts. "
                    f"Hand on e-stop? type YES: ") != "YES":
                return 1
        client = JevClient(args.jev_url) if args.jev == "on" else None
        policy = make_policy(args, robot, cfg)
        attempts, result, held_width = [], "failed", None
        n = args.attempts if args.jev == "on" and not args.jev_dry else 1
        done = None
        if args.park_when_done:
            from .keys import KeyReader

            done = DoneKey(KeyReader())
            print("press d when the task is done: the policy stops and the arm parks"
                  " (it also parks when JEV sees the task done)")
        try:
            for a in range(1, n + 1):
                print(f"=== attempt {a}/{n}")
                mon = None
                if a > 1 and attempts[-1].get("servo", {}).get("result") == "grasped":
                    held_width = float(robot.get_joint_state()[cfg.gripper_idx[0]])
                else:
                    held_width = None
                if client is not None:
                    mon = JevMonitor(client, args.instruction, args.jev_object, args.jev_container,
                                     grasp_thr=args.grasp_thr, place_thr=args.place_thr,
                                     gripper_text=not args.no_gripper_text,
                                     empty_check=args.empty_check, gripper_idx=cfg.gripper_idx[0],
                                     holding_width=held_width)
                summary = run(robot, policy, cfg, instruction=args.instruction, live=args.live,
                              speed=args.speed, max_steps=args.max_steps,
                              execute_steps=args.execute_steps,
                              log_dir=os.path.join(out, f"attempt{a}"),
                              async_infer=args.async_infer, blend_steps=args.blend_steps,
                              blend_from=args.blend_from,
                              ensemble_steps=args.ensemble_steps, smooth_hz=args.smooth_hz,
                              stream_hz=args.stream_hz, observer=mon,
                              stop_event=_Either(done, None if (mon is None or args.jev_dry)
                                                 else mon.stop_event))
                verdict = None
                if mon is not None:
                    if mon.verdict is None and summary["status"] == "ok":
                        try:
                            mon.final_check(robot.get_images(), robot.get_joint_state(), summary["steps"])
                        except Exception as e:
                            print(f"[jev] final check failed: {e}")
                    verdict = mon.verdict
                    if verdict == "timeout" and not any(e.kind == "closed" for e in mon.events):
                        verdict = "missed_grasp"  # hovered and never closed: same fix
                        print("[jev] timed out without ever closing: treating as missed_grasp (hover)")
                    summary["jev_events"] = [dict(t=e.t, step=e.step, kind=e.kind, **e.detail)
                                             for e in mon.events]
                summary["jev_verdict"] = verdict
                attempts.append(summary)
                print(f"attempt {a}: status {summary['status']}, JEV verdict {verdict}")
                if done is not None and done.done:
                    result = "done (operator)"
                    break
                if summary["status"] not in ("ok", "monitor_stop"):
                    result = "aborted"
                    break
                if verdict == "success":
                    result = "success"
                    break
                if a < n:
                    if args.live and args.recover != "home" and verdict == "missed_grasp":
                        summary["servo"] = servo_recover(robot, cfg, client, args)
                        print(f"JEV servo: {summary['servo'].get('result')}")
                        if summary["servo"].get("result") == "grasped":
                            continue  # pi0.5 carries and places from here
                        if summary["servo"].get("centred") and args.recover == "servo":
                            continue  # pi0.5 grasps from the centred hover
                    if args.live:
                        print("JEV: failed attempt; opening the gripper and folding home")
                        reset_arm(robot, cfg)
                    time.sleep(1.0)
        finally:
            policy.close()
            if done is not None:
                done.reader.close()
        if done is not None:
            if result != "aborted":
                print("parking: fold, face the table, damped stop, motors off")
                robot.park()
            else:  # leave it holding where it stopped
                print("not parking after an abort; arm holding (make park when clear)")
        report = {"result": result if client else "unjudged", "attempts": len(attempts),
                  "jev": args.jev, "jev_dry": args.jev_dry, "lerobot": args.lerobot,
                  "instruction": args.instruction, "per_attempt": attempts}
        with open(os.path.join(out, "jev_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"result: {report['result']} after {len(attempts)} attempt(s); logs: {out}")
        print("Score it yourself too: record the real outcome in "
              "real_robot_logs/jev_trials.csv (see docs/jev.md).")
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
