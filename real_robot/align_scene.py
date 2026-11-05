"""Place objects to match a recorded episode: the live base camera blended with
that episode's first frame, in a window.

A policy trained on a few dozen demos reproduces the layouts it saw but not new
ones (2026-10-29: on held-out layouts it scored worse than holding still), so
set the scene up like a training episode before a run:

  make align EP=20                          # then move the cube/bowl until they overlap
  make align EP=20 DATA=demos/x_pi05

Keys: n / p next / previous episode, s save the blend, q quit.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .config import IMAGE_HW, load_config


def first_frame(ds, ep: int, key: str) -> np.ndarray:
    """BGR uint8 first frame of episode ep (camera key, e.g. observation.images.base)."""
    i = int(ds.meta.episodes["dataset_from_index"][ep])
    rgb = (ds[i][key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="converted dataset (demos/<name>_pi05)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--config", default="real_robot/configs/piper_1.json")
    ap.add_argument("--camera", default="cam_high", help="live camera name in the config")
    ap.add_argument("--key", default="observation.images.base", help="dataset image key it was recorded as")
    args = ap.parse_args(argv)

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{os.path.basename(args.dataset.rstrip('/'))}", root=args.dataset)
    n = ds.meta.total_episodes
    cap = cv2.VideoCapture(load_config(args.config).cameras[args.camera])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_HW[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HW[0])
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    ep = args.episode % n
    ref = first_frame(ds, ep, args.key)
    print(f"{n} episodes; n/p next/previous, s save, q quit")
    while True:
        ok, live = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
        view = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
        cv2.putText(view, f"episode {ep}/{n - 1}  (n/p, s, q)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("align: live + recorded first frame", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("n"), ord("p")):
            ep = (ep + (1 if k == ord("n") else -1)) % n
            ref = first_frame(ds, ep, args.key)
            print(f"episode {ep}")
        if k == ord("s"):
            path = os.path.join("verify", f"align_ep{ep}_{time.strftime('%Y%m%d-%H%M%S')}.jpg")
            os.makedirs("verify", exist_ok=True)
            cv2.imwrite(path, view)
            print(f"saved {path}")
    print(f"last episode shown: {ep}")
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
