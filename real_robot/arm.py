"""Everyday arm commands for the Piper rig. Arms are named 1 (close) and 2 (far);
each maps to real_robot/configs/piper_<n>.json.

  python -m real_robot.arm status   [--arms 1 2]       # read-only
  python -m real_robot.arm snapshot [--arms 1 2] [--out DIR]
  python -m real_robot.arm move     --arm 1 --to -93,40,-30,0,30,0 [--grip 30]
  python -m real_robot.arm move     --arm 1 --by 0,10,0,0,0,0     # relative, deg
  python -m real_robot.arm home     [--arms 1 2]       # fold, face the table, motors on
  python -m real_robot.arm park     [--arms 1 2]       # home, then power down gently

Joint angles are in degrees (J1 base ... J6 wrist roll), the gripper in mm.
Commands that move an arm ask for YES first (--yes skips it); they move at
most --vel rad/s and leave the arm holding, except park, which powers it off.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .config import load_config

ARM_NAMES = {1: "close", 2: "far"}
HERE = os.path.dirname(os.path.abspath(__file__))


def config_path(n: int) -> str:
    return os.path.join(HERE, "configs", f"piper_{n}.json")


def _connect(n: int, passive: bool, cameras=None):
    from .cameras import NoCameras
    from .piper import PiperBackend

    robot = PiperBackend(load_config(config_path(n)), cameras=cameras or NoCameras(),
                         passive=passive)
    robot.connect()
    time.sleep(0.3)  # let the first feedback frames arrive
    return robot


def _deg(q) -> str:
    return "[" + ", ".join(f"{v:7.1f}" for v in np.degrees(q[:6])) + "]"


def _confirm(args, what: str) -> bool:
    return args.yes or input(f"{what}? type YES: ") == "YES"


def status(args) -> int:
    for n in args.arms:
        cfg = load_config(config_path(n))
        can = cfg.arms["arm"]
        try:
            with open(f"/sys/class/net/{can}/operstate") as f:
                link = f.read().strip()
        except OSError:
            link = "missing"
        print(f"arm {n} ({ARM_NAMES.get(n, '?')}) on {can}: link {link}")
        if link == "missing":
            continue
        robot = _connect(n, passive=True)
        arm = robot.arms["arm"]
        q, pose = robot.get_joint_state(), robot.get_end_pose()
        st = arm.GetArmStatus().arm_status
        enabled = arm.GetArmEnableStatus()
        print(f"  joints deg  {_deg(q)}   gripper {q[6] * 1000:.1f} mm")
        print(f"  end pose    xyz [{', '.join(f'{v * 1000:.0f}' for v in pose[:3])}] mm, "
              f"rpy [{', '.join(f'{v:.0f}' for v in np.degrees(pose[3:]))}] deg")
        print(f"  mode {st.ctrl_mode}, status {st.arm_status}, error {st.err_code}, "
              f"motors {'on' if all(enabled) else 'off' if not any(enabled) else enabled}")
        if cfg.home_base is not None:
            off = np.degrees(q[0] - cfg.home_base[0])
            print(f"  base {off:+.1f} deg from home (facing the table)"
                  + ("" if abs(off) < 2 else "  <- run `home` or `park` to turn it back"))
    return 0


def snapshot(args) -> int:
    import cv2

    from .cameras import Cameras

    out = args.out or os.path.join("verify", "snapshots", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    devices: dict[str, str] = {}  # file name -> device, each camera once
    for n in args.arms:
        cfg = load_config(config_path(n))
        for name, dev in cfg.cameras.items():
            if dev not in devices.values():
                devices[name if name == "cam_high" else f"{name}_arm{n}"] = dev
    cams = Cameras(devices, list(devices))
    try:
        for _ in range(5):  # let auto-exposure settle
            images = cams.read()
    finally:
        cams.close()
    for name, rgb in images.items():
        path = os.path.join(out, f"{name}.jpg")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(path)
    return 0


def _parse_deg(text: str) -> np.ndarray:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 6:
        raise SystemExit(f"need 6 joint angles (deg), got {len(vals)}: {text}")
    return np.radians(vals)


def move(args) -> int:
    from .backends import ease_to
    from .safety import SafetyFilter

    cfg = load_config(config_path(args.arm))
    if not _confirm(args, f"Move arm {args.arm} ({ARM_NAMES.get(args.arm, '?')})"):
        return 1
    robot = _connect(args.arm, passive=False)
    q = robot.get_joint_state()
    target = q.copy()
    target[:6] = _parse_deg(args.to) if args.to else q[:6] + _parse_deg(args.by)
    if args.grip is not None:
        target[6] = args.grip / 1000
    lo, hi = np.asarray(cfg.joint_lower), np.asarray(cfg.joint_upper)
    bad = [f"J{i + 1}" if i < 6 else "gripper" for i in range(7)
           if not lo[i] - 1e-6 <= target[i] <= hi[i] + 1e-6]
    if bad:
        robot.hold()
        print(f"refusing: {', '.join(bad)} outside the limits "
              f"(deg lower {_deg(lo)}, upper {_deg(hi)}; gripper 0..{hi[6] * 1000:.0f} mm)")
        return 2
    print(f"from deg {_deg(q)}\nto   deg {_deg(target)}")
    ease_to(robot, SafetyFilter(cfg), target, 50, vel=args.vel)
    time.sleep(0.5)
    q = robot.get_joint_state()
    print(f"now  deg {_deg(q)}   gripper {q[6] * 1000:.1f} mm  (holding, motors on)")
    robot.hold()
    return 0


def home(args) -> int:
    if not _confirm(args, f"Fold arm(s) {args.arms} and turn to face the table"):
        return 1
    for n in args.arms:
        robot = _connect(n, passive=False)
        robot.home()
        print(f"arm {n}: deg {_deg(robot.get_joint_state())} (holding, motors on)")
        robot.hold()
    return 0


def park(args) -> int:
    if not _confirm(args, f"Fold arm(s) {args.arms}, face the table and power down"):
        return 1
    for n in args.arms:
        robot = _connect(n, passive=False)
        robot.park()
        time.sleep(0.5)
        print(f"arm {n}: parked at deg {_deg(robot.get_joint_state())}, motors off")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def arms_arg(p):
        p.add_argument("--arms", type=int, nargs="+", choices=[1, 2], default=[1, 2],
                       help="1 = close arm, 2 = far arm (default: both)")

    p = sub.add_parser("status", help="read-only: joints, end pose, mode, errors, motors")
    arms_arg(p)
    p = sub.add_parser("snapshot", help="save each camera's current frame as JPEG")
    arms_arg(p)
    p.add_argument("--out", help="directory (default verify/snapshots/<time>)")
    for name, helptext in (("home", "fold, turn the base to face the table; motors stay on"),
                           ("park", "home, then damped stop and power down")):
        p = sub.add_parser(name, help=helptext)
        arms_arg(p)
        p.add_argument("--yes", action="store_true", help="skip the confirmation")
    p = sub.add_parser("move", help="ease one arm to joint angles (deg); stays holding")
    p.add_argument("--arm", type=int, choices=[1, 2], required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--to", help="absolute J1..J6 in deg, comma-separated")
    g.add_argument("--by", help="relative J1..J6 in deg, comma-separated")
    p.add_argument("--grip", type=float, help="gripper opening, mm (default: unchanged)")
    p.add_argument("--vel", type=float, default=0.3, help="max joint speed, rad/s")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    # "--to -93,40,..." would read as an unknown option (argparse only accepts
    # a leading "-" for plain numbers), and base angles here are negative.
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv[:-1]):
        if a in ("--to", "--by"):
            argv[i:i + 2] = [f"{a}={argv[i + 1]}", ""]
    args = ap.parse_args([a for a in argv if a != ""])
    return {"status": status, "snapshot": snapshot, "move": move,
            "home": home, "park": park}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
