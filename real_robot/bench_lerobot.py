"""Time a LeRobot policy's inference on a logged frame (no robot needed).

  python -m real_robot.bench_lerobot --lerobot richardshkim/pi05_piper_banana_v2_full \\
      --frame real_robot_logs/<run>/first_frame.npz --compile default --steps 5

Reports load time, the first call (includes torch.compile), and the median of
the timed calls. --out saves the chunks (fixed seed per call) so two settings
can be compared with --compare.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

# Rest pose of the close arm (2026-10-07), policy units: joints rad, gripper 0..1,
# then end pose [x, y, z m, roll, pitch, yaw rad].
REST_STATE = [-1.630, -0.038, -0.017, 0.005, 0.642, 0.0, 0.004]
REST_END_POSE = [-0.002, -0.041, 0.165, 3.129, 1.072, 1.501]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot", required=True)
    ap.add_argument("--frame", required=True, help="first_frame.npz from a runner log")
    ap.add_argument("--camera-map", default="cam_high:base,cam_wrist:wrist")
    ap.add_argument("--instruction", default="Put banana into basket.")
    ap.add_argument("--compile", default=None, help="torch.compile mode (default: eager)")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps")
    ap.add_argument("--base-offset-deg", type=float, default=90.0)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--gap", type=float, default=0.0,
                    help="idle seconds between calls (the runner idles ~1 s per chunk)")
    ap.add_argument("--robot-config",
                    help="also run this config's cameras and a read-only (passive) arm "
                         "connection, as the runner does; nothing is commanded")
    ap.add_argument("--out", help="save chunks to this .npz")
    ap.add_argument("--compare", nargs=2, metavar="NPZ", help="only compare two --out files")
    args = ap.parse_args(argv)

    if args.compare:
        a, b = (np.load(p)["chunks"] for p in args.compare)
        d = np.abs(a - b)
        print(json.dumps({"max_abs_diff_joints_deg": float(np.degrees(d[..., :6]).max()),
                          "mean_abs_diff_joints_deg": float(np.degrees(d[..., :6]).mean()),
                          "max_abs_diff_gripper": float(d[..., 6].max())}, indent=2))
        return 0

    import torch

    from .lerobot_policy import LerobotPolicy

    t0 = time.perf_counter()
    cmap = dict(kv.split(":") for kv in args.camera_map.split(","))
    policy = LerobotPolicy(args.lerobot, cmap, extra_state=lambda: REST_END_POSE,
                           base_offset=np.radians(args.base_offset_deg),
                           compile_mode=args.compile, num_steps=args.steps)
    load_s = time.perf_counter() - t0
    images = dict(np.load(args.frame))
    robot = None
    if args.robot_config:
        from .backends import PiperBackend
        from .config import load_config

        robot = PiperBackend(load_config(args.robot_config), passive=True)
        robot.connect()
        robot.get_images()
    obs = {"state": np.asarray(REST_STATE), "images": images, "instruction": args.instruction}
    policy.call("update_obs", obs=obs)

    def timed():
        torch.manual_seed(0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = policy.call("get_action")
        torch.cuda.synchronize()
        return time.perf_counter() - t, out

    first_s, _ = timed()
    times, chunks = [], []
    for _ in range(args.calls):
        if args.gap:
            time.sleep(args.gap)
        if robot is not None:
            robot.get_joint_state()
            robot.get_images()
        t, out = timed()
        times.append(t)
        chunks.append(out)
    if robot is not None:
        robot.cameras.close()
    if args.out:
        np.savez(args.out, chunks=np.stack(chunks))
    print(json.dumps({"compile": args.compile, "steps": args.steps or "checkpoint",
                      "load_s": round(load_s, 1), "first_call_s": round(first_s, 1),
                      "median_ms": round(1000 * float(np.median(times)), 1),
                      "p90_ms": round(1000 * float(np.percentile(times, 90)), 1),
                      "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
