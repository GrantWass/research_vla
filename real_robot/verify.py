"""First-contact verification: GPU, robot I/O, and pi0.5 plumbing.

Stages (run separately or `all`); everything lands in --out with a report.json:

  gpu     driver/VRAM, JAX + torch see the GPU (no robot, no model)
  robot   read joints (read-only), check limits, grab camera frames, save
          snapshot.npz + raw PNGs. `--wiggle` additionally commands joint 6
          +0.05 rad and back (asks first) to verify command units/sign.
  policy  load a pi0.5 checkpoint, feed it the snapshot (or live robot),
          save the exact 224x224 images the model receives, check every
          dimension hop (ours 7 -> model -> ours 7), latency and VRAM.

Policy stage uses upstream openpi (`pi05_droid` is not in RoboDojo's fork).
pi05_droid is a Franka policy: this checks plumbing, not behaviour; its
actions are NOT meaningful positions for a Piper and are never sent.

  python3 -m real_robot.verify --stage robot  --config real_robot/configs/piper_1.json --out verify/piper_1
  python3 -m real_robot.verify --stage policy --snapshot verify/piper_1/snapshot.npz --out verify/piper_1
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


# ---------------------------------------------------------------- policy
def to_droid_obs(state: np.ndarray, images: dict, cfg: RobotConfig, prompt: str) -> dict:
    """Single-arm ours (6 joints + grip, base + wrist cam) -> openpi DROID keys.

    DROID is a 7-joint Franka: our 6 joints are zero-padded to 7. Only the
    first arm is used.
    """
    joints = np.zeros(7, dtype=np.float64)
    joints[:ARM_DOF] = state[:ARM_DOF]
    lo, hi = cfg.gripper_real_range
    grip = np.clip((state[ARM_DOF] - lo) / (hi - lo), 0, 1)
    names = cfg.camera_names
    return {
        "observation/exterior_image_1_left": images[names[0]],
        "observation/wrist_image_left": images[names[1] if len(names) > 1 else names[0]],
        "observation/joint_position": joints,
        "observation/gripper_position": np.asarray([grip]),
        "prompt": prompt,
    }


def from_droid_actions(actions: np.ndarray, cfg: RobotConfig) -> np.ndarray:
    """(T, 8) DROID [7 joints, grip] -> (T, 7) ours [6 joints, grip]."""
    a = np.asarray(actions)
    return np.concatenate([a[:, :ARM_DOF], a[:, 7:8]], axis=1)


def load_openpi_policy(config_name: str, checkpoint: str):
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    try:
        tc = _config.get_config(config_name)
    except ValueError as e:
        raise SystemExit(f"{e}\nUse an upstream openpi checkout (RoboDojo's fork lacks "
                         f"{config_name}).") from e
    return policy_config.create_trained_policy(tc, download.maybe_download(checkpoint))


def stage_policy(rep: Report, policy, cfg: RobotConfig, state: np.ndarray, images: dict,
                 prompt: str, n_infer: int = 5) -> None:
    S = "policy"
    obs = to_droid_obs(state, images, cfg, prompt)
    rep.check(S, "our state -> model state", None,
              f"{state.shape} -> joint_position {obs['observation/joint_position'].shape} "
              f"+ gripper {obs['observation/gripper_position'].shape}")

    # Exactly what the network receives, after openpi's input transforms.
    import copy
    seen = policy._input_transform(copy.deepcopy(obs))
    for name, img in seen["image"].items():
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8) if img.min() < 0 \
                else (img * 255).clip(0, 255).astype(np.uint8)
        masked = not bool(np.asarray(seen["image_mask"][name]))
        save_png(os.path.join(rep.out, f"model_sees_{name}.png"), img)
        rep.check(S, f"model image {name}", None,
                  f"{img.shape}{' (masked out: unused slot)' if masked else ''}")
    st = np.asarray(seen["state"])
    rep.check(S, "normalized state", bool(np.all(np.abs(st) < 10)),
              f"dim {st.shape}, max |x| {np.max(np.abs(st)):.2f} (huge = norm-stat mismatch)")

    times = []
    out = None
    for _ in range(n_infer):
        t0 = time.perf_counter()
        out = policy.infer(copy.deepcopy(obs))
        times.append((time.perf_counter() - t0) * 1e3)
    acts = np.asarray(out["actions"])
    rep.check(S, "model action chunk", acts.ndim == 2 and acts.shape[1] >= 8, str(acts.shape))
    ours = from_droid_actions(acts, cfg)
    rep.check(S, "model -> our action", ours.shape[1] == 7, f"{acts.shape} -> {ours.shape}")
    rep.check(S, "actions finite", bool(np.all(np.isfinite(ours))))
    rep.check(S, "latency ms (first, then warm median)", None,
              f"{times[0]:.0f}, {np.median(times[1:]) if len(times) > 1 else float('nan'):.0f}")
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        rep.check(S, "peak VRAM GiB", None, round(peak / 2**30, 2) if peak else "n/a")
    except Exception:  # noqa: BLE001
        pass
    rep.check(S, "actions sent to robot", None, "none (verification only)")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gpu", "robot", "policy", "all"], default="all")
    ap.add_argument("--backend", choices=["mock", "piper"], default="piper")
    ap.add_argument("--config", help="robot config JSON (piper_1.json, piper_2.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiggle", action="store_true", help="robot stage: tiny commanded move")
    ap.add_argument("--snapshot", help="policy stage: use this snapshot.npz, not a live robot")
    ap.add_argument("--policy-config", default="pi05_droid")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_droid")
    ap.add_argument("--prompt", default="pick up the cube")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rep = Report(args.out)
    robot = None
    try:
        if args.stage in ("gpu", "all"):
            stage_gpu(rep)
        snap = None
        if args.stage in ("robot", "all") or (args.stage == "policy" and not args.snapshot):
            from .backends import MockBackend, PiperBackend
            robot = (MockBackend(cfg) if args.backend == "mock"
                     else PiperBackend(cfg, passive=not args.wiggle))
            robot.connect()
            snap = stage_robot(rep, robot, cfg, wiggle=args.wiggle)
        if args.stage in ("policy", "all"):
            if args.snapshot:
                z = np.load(args.snapshot)
                snap = {"state": z["state"],
                        "images": {k[4:]: z[k] for k in z.files if k.startswith("img_")}}
            policy = load_openpi_policy(args.policy_config, args.checkpoint)
            stage_policy(rep, policy, cfg, snap["state"], snap["images"], args.prompt)
    finally:
        rep.save()
        if robot is not None:
            robot.hold()
            robot.close()
    bad = [s for s in ("gpu", "robot", "policy") if rep.failed(s)]
    print(f"report: {rep.path}  ->  " + ("ALL PASS" if not bad else f"FAILED: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
