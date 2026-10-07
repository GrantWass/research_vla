"""Closed loop: robot -> pi0.5 server -> safety filter -> robot.

Dry-run is the default: the policy runs on live camera/joint data, but no
command reaches the motors; the would-be commands are logged instead.

  python3 -m real_robot.runner --backend mock --url ws://localhost:6000 \\
      --instruction "stack the bowls"            # dry-run
  python3 -m real_robot.runner ... --live --speed 0.25   # moves the robot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .backends import RobotBackend, make_backend
from .config import RobotConfig, load_config
from .mapping import real_to_sim, sim_to_real
from .safety import SafetyAbort, SafetyFilter

_ARM_KEYS = (("left_arm_joint_state", "left_ee_joint_state"),
             ("right_arm_joint_state", "right_ee_joint_state"))


def build_obs(images: dict[str, np.ndarray], q_sim: np.ndarray, instruction: str,
              camera_names=None) -> dict:
    """Pre-packed obs, the branch Pi_05 Model.encode_obs takes first."""
    camera_names = list(camera_names or images)
    for c in camera_names:
        img = images[c]
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{c}: want HxWx3 uint8, got {img.dtype} {img.shape}")
    return {"images": {c: images[c] for c in camera_names},
            "state": np.asarray(q_sim, dtype=np.float32),
            "instruction": instruction}


def chunk_to_array(actions: Any, state_dim: int) -> np.ndarray:
    """Server returns (T,D) when env_cfg_type is unset, else list[dict]."""
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        rows = []
        for a in actions:
            row = []
            for arm_k, ee_k in _ARM_KEYS:
                row.extend(np.ravel(a[arm_k]).tolist())
                row.extend(np.ravel(a[ee_k]).tolist())
            rows.append(row)
        actions = rows
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] > state_dim:
        arr = arr[:, :state_dim]  # openpi pads actions (e.g. to 32); keep ours
    if arr.ndim != 2 or arr.shape[1] != state_dim:
        raise ValueError(f"expected action chunk (T,{state_dim}), got {arr.shape}")
    return arr


def connect_policy(url: str, timeout_s: float):
    here = os.path.dirname(os.path.abspath(__file__))
    xpl = os.path.join(here, "..", "RoboDojo", "XPolicyLab")
    sys.path.insert(0, os.path.abspath(os.path.join(xpl, "..")))
    sys.path.insert(0, os.path.abspath(xpl))
    from client_server.ws.model_client import WsModelClient  # lazy: box only

    return WsModelClient(url=url, evaluation_id="real_robot",
                         trial_id=f"real-{int(time.time())}",
                         request_timeout_s=timeout_s)


def preflight(robot: RobotBackend, cfg: RobotConfig, safety: SafetyFilter) -> list[str]:
    """Checks that need no policy. Returns problems (empty = OK)."""
    problems = []
    if not cfg.verified:
        problems.append("config.verified is false: joint limits/mapping are placeholders")
    q = robot.get_joint_state()
    try:
        safety.check_state(q)
    except SafetyAbort as e:
        problems.append(str(e))
    imgs = robot.get_images()
    for c in cfg.camera_names:
        if c not in imgs:
            problems.append(f"camera {c} missing")
        elif imgs[c].mean() < 5:
            problems.append(f"camera {c} is (nearly) black")
    return problems


def run(robot: RobotBackend, policy, cfg: RobotConfig, *, instruction: str,
        live: bool, speed: float = 1.0, max_steps: int = 500,
        execute_steps: int | None = None, log_dir: str | None = None) -> dict:
    """One episode. Always leaves the robot holding position."""
    safety = SafetyFilter(cfg)
    period = 1.0 / (cfg.control_hz * speed)
    log: dict[str, list] = {"t": [], "q_real": [], "target_real": [], "cmd_real": [],
                            "chunk_idx": []}
    infer_ms: list[float] = []
    status = "ok"
    first_images = None

    q = robot.get_joint_state()
    safety.check_state(q)
    virtual_q = q.copy()  # dry-run: pretend commands are tracked exactly
    policy.call(func_name="reset")
    step = chunk_idx = 0
    try:
        while step < max_steps:
            q_now = robot.get_joint_state() if live else virtual_q
            safety.check_state(q_now)
            images = robot.get_images()
            if first_images is None:
                first_images = images
            obs = build_obs(images, real_to_sim(q_now, cfg), instruction, cfg.camera_names)
            t0 = time.perf_counter()
            policy.call(func_name="update_obs", obs=obs)
            chunk = chunk_to_array(policy.call(func_name="get_action"), cfg.state_dim)
            infer_ms.append((time.perf_counter() - t0) * 1e3)

            n = len(chunk) if execute_steps is None else min(execute_steps, len(chunk))
            for a_sim in chunk[:n]:
                tick = time.perf_counter()
                current = robot.get_joint_state() if live else virtual_q
                target = sim_to_real(a_sim, cfg)
                cmd = safety.filter(target, current)
                if live:
                    robot.command_joints(cmd)
                else:
                    virtual_q = cmd
                for k, v in (("t", time.time()), ("q_real", current),
                             ("target_real", target), ("cmd_real", cmd),
                             ("chunk_idx", chunk_idx)):
                    log[k].append(v)
                step += 1
                if step >= max_steps:
                    break
                if live:
                    time.sleep(max(0.0, period - (time.perf_counter() - tick)))
            chunk_idx += 1
    except SafetyAbort as e:
        status = f"safety_abort: {e}"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as e:  # server down, camera died, SDK error...
        status = f"error: {type(e).__name__}: {e}"
    finally:
        if live:
            robot.hold()

    summary = {
        "status": status, "live": live, "steps": step, "chunks": chunk_idx,
        "instruction": instruction, "speed": speed,
        "infer_ms_median": float(np.median(infer_ms)) if infer_ms else None,
        "infer_ms_max": float(np.max(infer_ms)) if infer_ms else None,
        "safety": asdict(safety.stats),
    }
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        np.savez_compressed(os.path.join(log_dir, "trajectory.npz"),
                            **{k: np.asarray(v) for k, v in log.items()})
        if first_images is not None:
            np.savez_compressed(os.path.join(log_dir, "first_frame.npz"), **first_images)
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "piper"])
    ap.add_argument("--config", help="robot JSON config (real_robot/configs/*.json)")
    ap.add_argument("--url", default="ws://localhost:6000")
    ap.add_argument("--lerobot", metavar="PATH",
                    help="run this LeRobot policy (Hub id or dir) in-process instead of --url")
    ap.add_argument("--camera-map", default="cam_high:base,cam_wrist:wrist",
                    help="--lerobot: our_camera:policy_image_key,...")
    ap.add_argument("--end-pose-state", action="store_true",
                    help="--lerobot: append the arm end pose (6-D) to the state")
    ap.add_argument("--base-offset-deg", type=float, default=0.0,
                    help="--lerobot: joint 1 as the policy sees it = ours + this "
                         "(base turned vs the training rig; 90 for our close arm)")
    ap.add_argument("--instruction", default="stack the bowls")
    ap.add_argument("--live", action="store_true", help="actually move the robot")
    ap.add_argument("--speed", type=float, default=0.25,
                    help="fraction of sim control rate (live only)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--execute-steps", type=int, default=None,
                    help="steps of each chunk to run before re-planning (default: all)")
    ap.add_argument("--timeout", type=float, default=10.0, help="policy call timeout, s")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the live confirmation prompt")
    args = ap.parse_args(argv)

    if not 0 < args.speed <= 1.0:
        ap.error("--speed must be in (0, 1]")
    cfg = load_config(args.config)
    robot = make_backend(args.backend, cfg)
    robot.connect()
    try:
        problems = preflight(robot, cfg, SafetyFilter(cfg))
        for p in problems:
            print(f"PREFLIGHT: {p}")
        print("PREFLIGHT:", "OK" if not problems else f"{len(problems)} problem(s)")
        if args.preflight_only:
            return 0 if not problems else 1
        if args.live:
            if problems:
                print("refusing --live with preflight problems")
                return 1
            if not args.yes and input(
                    f"LIVE at speed {args.speed}. Hand on e-stop? type YES: ") != "YES":
                return 1
        log_dir = args.log_dir or os.path.join(
            "real_robot_logs", time.strftime("%Y%m%d-%H%M%S") + ("-live" if args.live else "-dry"))
        if args.lerobot:
            from .lerobot_policy import LerobotPolicy

            cmap = dict(kv.split(":") for kv in args.camera_map.split(","))
            policy = LerobotPolicy(args.lerobot, cmap,
                                   extra_state=robot.get_end_pose if args.end_pose_state else None,
                                   base_offset=np.radians(args.base_offset_deg))
        else:
            policy = connect_policy(args.url, args.timeout)
        try:
            summary = run(robot, policy, cfg, instruction=args.instruction,
                          live=args.live, speed=args.speed, max_steps=args.max_steps,
                          execute_steps=args.execute_steps, log_dir=log_dir)
        finally:
            policy.close()
        print(json.dumps(summary, indent=2))
        print(f"logs: {log_dir}")
        return 0 if summary["status"] == "ok" else 2
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
