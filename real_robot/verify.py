"""Hardware check: GPU and robot I/O. Writes report.json + PNGs to --out.

Stages (run separately or `all`):

  gpu     driver/VRAM, JAX + torch see the GPU (no robot, no model)
  robot   read joints (read-only), check limits, grab camera frames, save
          snapshot.npz + raw PNGs. `--wiggle` additionally commands joint 6
          +0.05 rad and back (asks first) to verify command units/sign.

  python3 -m real_robot.verify --stage robot --config real_robot/configs/piper_1.json --out verify/piper_1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import ARM_DOF, RobotConfig, load_config
from .safety import SafetyAbort, SafetyFilter


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, out: str):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.path = os.path.join(out, "report.json")
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)

    def check(self, stage: str, name: str, ok: bool | None, detail="") -> bool:
        tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
        print(f"[{stage}] {tag:4} {name}" + (f": {detail}" if detail != "" else ""))
        self.data.setdefault(stage, {})[name] = {"ok": ok, "detail": detail}
        return bool(ok) or ok is None

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def failed(self, stage: str) -> bool:
        return any(v["ok"] is False for v in self.data.get(stage, {}).values())


def save_png(path: str, rgb: np.ndarray) -> None:
    """RGB uint8 HxWx3 -> PNG. Stdlib only, so it runs on any box."""
    import struct
    import zlib

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- gpu
def stage_gpu(rep: Report) -> None:
    S = "gpu"
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True)
        gpus = [g.strip() for g in q.stdout.strip().splitlines() if g.strip()]
        rep.check(S, "nvidia-smi", bool(gpus), gpus)
    else:
        rep.check(S, "nvidia-smi", False, "not found")
    frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    rep.check(S, "XLA_PYTHON_CLIENT_MEM_FRACTION", None, frac or "unset (JAX grabs 75%)")
    try:
        import jax
        devs = [str(d) for d in jax.devices()]
        rep.check(S, "jax devices", any("cuda" in d.lower() or "gpu" in d.lower() for d in devs), devs)
    except Exception as e:  # noqa: BLE001
        rep.check(S, "jax devices", False, f"{type(e).__name__}: {e}")
    try:
        import torch
        rep.check(S, "torch cuda", torch.cuda.is_available(),
                  [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:
        rep.check(S, "torch cuda", None, "torch not installed (fine for JAX openpi)")


# ---------------------------------------------------------------- robot
def stage_robot(rep: Report, robot, cfg: RobotConfig, wiggle: bool = False,
                frames: int = 25, confirm=input) -> dict:
    S = "robot"
    q = np.asarray(robot.get_joint_state(), dtype=np.float64)
    rep.check(S, "state dim", q.shape == (cfg.state_dim,), f"{q.shape} vs ({cfg.state_dim},)")
    rep.check(S, "state finite", bool(np.all(np.isfinite(q))))
    deg = [round(float(np.degrees(v)), 2) for v in q]
    for k, arm in enumerate(cfg.arms):
        base = 7 * k
        rep.check(S, f"{arm} joints deg", None, deg[base:base + ARM_DOF])
        rep.check(S, f"{arm} gripper m", None, round(float(q[base + ARM_DOF]), 4))
    try:
        SafetyFilter(cfg).check_state(q)
        rep.check(S, "within joint limits", True)
    except SafetyAbort as e:
        rep.check(S, "within joint limits", False, str(e))

    for _ in range(5):  # cameras deliver the first frames slowly after opening
        robot.get_images()
    counts = getattr(getattr(robot, "cameras", None), "frame_counts", None)
    t0 = time.perf_counter()
    imgs = None
    if counts:  # threaded reader: read() is instant, so count frames delivered
        c0 = dict(counts)
        time.sleep(1.0)
        imgs = robot.get_images()
        c1 = robot.cameras.frame_counts
        fps = min(c1[c] - c0[c] for c in c0) / (time.perf_counter() - t0)
    else:
        for _ in range(frames):
            imgs = robot.get_images()
        fps = frames / max(time.perf_counter() - t0, 1e-9)
    rep.check(S, "camera fps", fps >= cfg.control_hz, f"{fps:.1f} (need >= {cfg.control_hz})")
    for c in cfg.camera_names:
        img = imgs.get(c)
        if img is None:
            rep.check(S, f"camera {c}", False, "missing")
            continue
        ok = img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        rep.check(S, f"camera {c}", ok and img.mean() > 5,
                  f"{img.shape} {img.dtype} mean={img.mean():.0f}")
        save_png(os.path.join(rep.out, f"raw_{c}.png"), img)
    np.savez_compressed(os.path.join(rep.out, "snapshot.npz"), state=q,
                        **{f"img_{c}": imgs[c] for c in cfg.camera_names if c in imgs})
    rep.check(S, "snapshot", None, os.path.join(rep.out, "snapshot.npz"))

    if wiggle:
        j = ARM_DOF - 1  # wrist roll of the first arm: smallest swept volume
        if confirm(f"Command {list(cfg.arms)[0]} joint {j + 1} +0.05 rad and back? type YES: ") != "YES":
            rep.check(S, "wiggle", None, "skipped")
        else:
            # Home = start pose clipped to the limits: Pipers rest ~2 deg past
            # J2/J3 limits and the firmware clamps any command back inside.
            home = np.clip(q, cfg.joint_lower, cfg.joint_upper)
            target = home.copy()
            target[j] += 0.05
            for cmd in (target, home):
                for _ in range(int(cfg.control_hz)):  # ~1 s each way
                    robot.command_joints(cmd)
                    time.sleep(1.0 / cfg.control_hz)
            moved = robot.get_joint_state()
            err = float(np.max(np.abs(moved - home)))
            rep.check(S, "wiggle returns home", err < 0.01, f"max err {err:.4f} rad")
            rep.check(S, "wiggle direction", None,
                      "watch the arm: joint 6 should rotate +0.05 rad (~3 deg) and back")
    return {"state": q, "images": imgs}


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gpu", "robot", "all"], default="all")
    ap.add_argument("--backend", choices=["mock", "piper"], default="piper")
    ap.add_argument("--config", help="robot config JSON (piper_1.json, piper_2.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiggle", action="store_true", help="robot stage: tiny commanded move")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rep = Report(args.out)
    robot = None
    try:
        if args.stage in ("gpu", "all"):
            stage_gpu(rep)
        if args.stage in ("robot", "all"):
            if args.backend == "mock":
                from .backends import MockBackend
                robot = MockBackend(cfg)
            else:
                from .piper import PiperBackend
                robot = PiperBackend(cfg, passive=not args.wiggle)
            robot.connect()
            stage_robot(rep, robot, cfg, wiggle=args.wiggle)
    finally:
        rep.save()
        if robot is not None:
            robot.hold()
            robot.close()
    bad = [s for s in ("gpu", "robot") if rep.failed(s)]
    print(f"report: {rep.path}  ->  " + ("ALL PASS" if not bad else f"FAILED: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
