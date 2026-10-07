"""Fold an arm and power it down gently (PiperBackend.park). Moves the arm.

  python3 -m real_robot.park --config real_robot/configs/piper_1.json

Untested on the arms: run it the first time with someone watching, hand near
the power switch.
"""
from __future__ import annotations

import argparse

from .cameras import NoCameras
from .config import load_config
from .piper import PiperBackend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if not args.yes and input(f"Fold and power down {list(cfg.arms.values())}? type YES: ") != "YES":
        return 1
    robot = PiperBackend(cfg, cameras=NoCameras())
    robot.connect()
    try:
        robot.park()
    finally:
        robot.close()
    print("parked: motors disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
