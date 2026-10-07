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

MAX_SPEED = 2.0  # --speed cap: at most twice the policy's trained control rate

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
    """Client for an XPolicyLab policy server (the one RoboDojo evals use)."""
    here = os.path.dirname(os.path.abspath(__file__))
    xpl = os.path.abspath(os.path.join(here, "..", "RoboDojo", "XPolicyLab"))
    if not os.path.isdir(os.path.join(xpl, "client_server")):
        raise SystemExit(f"--url needs the XPolicyLab client in {xpl} (run `bash setup.sh`), "
                         f"or use --lerobot for an in-process policy")
    sys.path.insert(0, os.path.dirname(xpl))
    sys.path.insert(0, xpl)
    from client_server.ws.model_client import WsModelClient

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
        execute_steps: int | None = None, log_dir: str | None = None,
        async_infer: bool = False, blend_steps: int = 0, ensemble_steps: int = 0,
        smooth_hz: float = 0.0, stream_hz: float = 100.0) -> dict:
    """One episode. Always leaves the robot holding position.

    async_infer: compute the next chunk in a background thread while the
    current one executes (see _async_loop), instead of pausing for each.
    blend_steps, ensemble_steps (async only): smooth between chunks.
    smooth_hz > 0: stream commands at stream_hz through a critically damped
    filter with this cutoff (real_robot/smoothing.py) instead of sending one
    per step. Dry-run streams too, into the log only.
    """
    safety = SafetyFilter(cfg)
    period = 1.0 / (cfg.control_hz * speed)
    log: dict[str, list] = {"t": [], "q_real": [], "target_real": [], "cmd_real": [],
                            "chunk_idx": []}
    infer_ms: list[float] = []
    extra: dict[str, int] = {}
    status = "ok"
    first_images: list = []

    q = robot.get_joint_state()
    safety.check_state(q)
    virtual_q = q.copy()  # dry-run: pretend commands are tracked exactly
    stream_log: list = []
    streamer = None
    if smooth_hz > 0:
        from .smoothing import SmoothStreamer

        def send(y):
            stream_log.append((time.time(), y))
            if live:
                robot.command_joints(y)
        streamer = SmoothStreamer(send, cfg.joint_lower, cfg.joint_upper,
                                  cutoff_hz=smooth_hz, rate_hz=stream_hz)
    policy.call(func_name="reset")
    if streamer is not None:
        streamer.start(np.clip(q, cfg.joint_lower, cfg.joint_upper))
    step = chunk_idx = 0

    def observe():
        q_now = robot.get_joint_state() if live else virtual_q
        safety.check_state(q_now)
        images = robot.get_images()
        if not first_images:
            first_images.append(images)
        return build_obs(images, real_to_sim(q_now, cfg), instruction, cfg.camera_names)

    def infer(obs):
        t0 = time.perf_counter()
        policy.call(func_name="update_obs", obs=obs)
        chunk = chunk_to_array(policy.call(func_name="get_action"), cfg.state_dim)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        return chunk

    def _bump():
        nonlocal chunk_idx
        chunk_idx += 1

    def act(a_sim):
        """Send one action through the safety filter (or log it, dry-run)."""
        nonlocal step, virtual_q
        current = robot.get_joint_state() if live else virtual_q
        target = sim_to_real(a_sim, cfg)
        cmd = safety.filter(target, current)
        if streamer is not None:
            streamer.check()
            streamer.set_target(cmd)
        elif live:
            robot.command_joints(cmd)
        if not live:
            virtual_q = cmd
        for k, v in (("t", time.time()), ("q_real", current),
                     ("target_real", target), ("cmd_real", cmd),
                     ("chunk_idx", chunk_idx)):
            log[k].append(v)
        step += 1

    try:
        if async_infer:
            _async_loop(observe, infer, act, lambda: step, max_steps, execute_steps,
                        period, extra, bump_chunk=lambda: _bump(), blend_steps=blend_steps,
                        ensemble_steps=ensemble_steps)
        else:
            while step < max_steps:
                chunk = infer(observe())
                n = len(chunk) if execute_steps is None else min(execute_steps, len(chunk))
                for a_sim in chunk[:n]:
                    tick = time.perf_counter()
                    act(a_sim)
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
        if streamer is not None:
            streamer.stop()
        if live:
            robot.hold()
    first_images = first_images[0] if first_images else None

    summary = {
        "status": status, "live": live, "steps": step, "chunks": chunk_idx,
        "instruction": instruction, "speed": speed,
        "infer_ms_median": float(np.median(infer_ms)) if infer_ms else None,
        "infer_ms_max": float(np.max(infer_ms)) if infer_ms else None,
        "async": async_infer, **extra,
        "smooth_hz": smooth_hz, "streamed_commands": len(stream_log),
        "safety": asdict(safety.stats),
    }
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        stream = ({"stream_t": np.asarray([t for t, _ in stream_log]),
                   "stream_cmd": np.asarray([y for _, y in stream_log])} if stream_log else {})
        np.savez_compressed(os.path.join(log_dir, "trajectory.npz"),
                            **{k: np.asarray(v) for k, v in log.items()}, **stream)
        if first_images is not None:
            np.savez_compressed(os.path.join(log_dir, "first_frame.npz"), **first_images)
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def _async_loop(observe, infer, act, get_step, max_steps, execute_steps, period,
                extra, bump_chunk, blend_steps: int = 0, ensemble_steps: int = 0) -> None:
    """Execute chunks at `period` while the next is inferred in a thread.

    Inference for the next chunk starts when the steps left before
    execute_steps fall to the measured latency (+2), so it lands about when
    it is needed. A chunk's action 0 belongs to the moment its observation was
    taken, so on switching, the actions that elapsed during inference are
    skipped. If inference is late the current chunk runs on past
    execute_steps; if it runs out, the arm holds until the new chunk arrives.
    Runs in real time in dry-run too.

    blend_steps: on a switch, carry the gap between the last action sent and
    the new chunk, and shrink it linearly to 0 over this many steps, instead
    of jumping. Consecutive pi0.5 chunks disagree by ~5 deg at a switch
    (12 deg max, live runs 2026-10-07) against ~0.5 deg per step within a
    chunk. Relative to the last action sent, so it stays continuous when the
    next switch comes before the blend ends (frequent replanning).

    ensemble_steps: each step, average what every chunk observed within the
    last ensemble_steps steps says for that step (temporal ensembling, as in
    ACT), instead of following only the newest. One chunk's quirk then moves
    the arm by a fraction of it.
    """
    import concurrent.futures as cf

    sys.setswitchinterval(0.001)  # keep the control loop on time next to the inference thread
    pool = cf.ThreadPoolExecutor(max_workers=1)
    try:
        t0 = time.perf_counter()
        chunk, chunk_step = infer(observe()), get_step()
        lat_steps = (time.perf_counter() - t0) / period
        n_exec = len(chunk) if execute_steps is None else min(execute_steps, len(chunk))
        pending = last = offset = switch_step = None
        history = [(chunk, chunk_step)]  # chunks for ensembling, oldest first
        extra.update(skipped_actions=0, starved_steps=0)
        next_tick = time.perf_counter()
        while get_step() < max_steps:
            i = get_step() - chunk_step
            if pending is None and i >= n_exec - lat_steps - 2:
                obs_step, t_sub = get_step(), time.perf_counter()
                pending = pool.submit(infer, observe())
            if pending is not None and (pending.done() or i >= len(chunk)):
                if not pending.done():
                    extra["starved_steps"] += 1  # chunk ran out: hold and wait
                new = pending.result()  # re-raises inference errors
                switch_step = get_step()
                lat_steps = 0.7 * lat_steps + 0.3 * (time.perf_counter() - t_sub) / period
                chunk, chunk_step, pending = new, obs_step, None
                n_exec = len(chunk) if execute_steps is None else min(execute_steps, len(chunk))
                bump_chunk()
                i = min(get_step() - chunk_step, len(chunk) - 1)
                chunk_step = get_step() - i
                extra["skipped_actions"] += i
                next_tick = time.perf_counter()
                history.append((chunk, chunk_step))
            a = chunk[i]
            if ensemble_steps > 0:
                s_now = get_step()
                history = [(c, cs) for c, cs in history
                           if c is chunk or (s_now - cs < ensemble_steps and s_now - cs < len(c))]
                a = np.mean([c[s_now - cs] for c, cs in history], axis=0)
            if blend_steps > 0 and last is not None and switch_step is not None:
                k = get_step() - switch_step
                if k == 0:
                    offset = last - a
                if k < blend_steps:
                    a = a + offset * (1 - (k + 1) / (blend_steps + 1))
            act(a)
            last = a
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


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
                    help="multiple of the policy's control rate (cfg.control_hz). Up to "
                         "%.0f; above 1 the arm moves faster than the demos it learned "
                         "from, and the per-step safety limits allow proportionally "
                         "faster joints" % MAX_SPEED)
    ap.add_argument("--speed-percent", type=int, default=None,
                    help="override the arm firmware's speed cap (config: "
                         "arm_speed_percent, 30); at 30 the arm lags fast policies")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--execute-steps", type=int, default=None,
                    help="steps of each chunk to run before re-planning (default: all)")
    ap.add_argument("--async", dest="async_infer", action="store_true",
                    help="infer the next chunk while the current one executes")
    ap.add_argument("--ensemble-steps", type=int, default=0,
                    help="--async: average every chunk observed in the last N steps "
                         "for each step (temporal ensembling; try 20)")
    ap.add_argument("--smooth-hz", type=float, default=0.0,
                    help="stream commands at --stream-hz through a critically damped "
                         "filter with this cutoff (try 2); 0 = one command per step")
    ap.add_argument("--stream-hz", type=float, default=100.0)
    ap.add_argument("--blend-steps", type=int, default=0,
                    help="--async: crossfade from the old chunk to the new one over "
                         "this many steps instead of jumping (try 8)")
    ap.add_argument("--compile", default=None, metavar="MODE",
                    help="--lerobot: torch.compile mode, e.g. max-autotune (default: eager)")
    ap.add_argument("--policy-server", nargs="?", const="default", metavar="SOCKET",
                    help="--lerobot: use a running policy server (make policy-server) "
                         "instead of loading the model; it must match --lerobot, "
                         "--base-offset-deg, --noise-seed and --camera-map")
    ap.add_argument("--policy-process", action="store_true",
                    help="--lerobot: run the policy in a child process (own GIL; "
                         "faster next to the arm's CAN thread)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="--lerobot: inferences on a live obs before the episode "
                         "(absorbs torch.compile and CUDA-graph capture)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="--lerobot: same sampling noise for every chunk (e.g. 0), so "
                         "replanning doesn't wobble between random plans")
    ap.add_argument("--denoise-steps", type=int, default=None,
                    help="--lerobot: flow-matching steps (default: the checkpoint's)")
    ap.add_argument("--timeout", type=float, default=10.0, help="policy call timeout, s")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the live confirmation prompt")
    args = ap.parse_args(argv)

    if not 0 < args.speed <= MAX_SPEED:
        ap.error(f"--speed must be in (0, {MAX_SPEED:g}]")
    for name in ("blend_steps", "ensemble_steps"):
        n = getattr(args, name)
        if n < 0 or (n and not args.async_infer):
            ap.error(f"--{name.replace('_', '-')} needs --async and must be >= 0")
    if args.speed_percent is not None and not 1 <= args.speed_percent <= 100:
        ap.error("--speed-percent must be in [1, 100]")
    cfg = load_config(args.config)
    if args.speed_percent is not None:
        cfg.arm_speed_percent = args.speed_percent
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
            from .lerobot_policy import DEFAULT_SOCKET, LerobotPolicy, PolicyClient, PolicyProcess

            kw = dict(path=args.lerobot,
                      camera_map=dict(kv.split(":") for kv in args.camera_map.split(",")),
                      base_offset=float(np.radians(args.base_offset_deg)),
                      compile_mode=args.compile, num_steps=args.denoise_steps,
                      noise_seed=args.noise_seed)
            extra = robot.get_end_pose if args.end_pose_state else None
            if args.policy_server:
                addr = DEFAULT_SOCKET if args.policy_server == "default" else args.policy_server
                policy = PolicyClient(addr, extra_state=extra, expect={
                    k: kw[k] for k in ("path", "camera_map", "base_offset")}
                    | {"noise_seed": args.noise_seed})
                print(f"policy server: {policy.info}")
            elif args.policy_process:
                policy = PolicyProcess(extra_state=extra, **kw)
            else:
                policy = LerobotPolicy(extra_state=extra, **kw)
            for i in range(args.warmup):
                t0 = time.perf_counter()
                policy.call(func_name="update_obs", obs=build_obs(
                    robot.get_images(), real_to_sim(robot.get_joint_state(), cfg),
                    args.instruction, cfg.camera_names))
                policy.call(func_name="get_action")
                print(f"warmup {i + 1}/{args.warmup}: {time.perf_counter() - t0:.1f} s")
        else:
            policy = connect_policy(args.url, args.timeout)
        try:
            summary = run(robot, policy, cfg, instruction=args.instruction,
                          live=args.live, speed=args.speed, max_steps=args.max_steps,
                          execute_steps=args.execute_steps, log_dir=log_dir,
                          async_infer=args.async_infer, blend_steps=args.blend_steps,
                          ensemble_steps=args.ensemble_steps,
                          smooth_hz=args.smooth_hz, stream_hz=args.stream_hz)
        finally:
            policy.close()
        print(json.dumps(summary, indent=2))
        print(f"logs: {log_dir}")
        return 0 if summary["status"] == "ok" else 2
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
