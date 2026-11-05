"""Direction-labelled frames: where must the arm go from here to grasp?

For each demo, the grasp pose is the state where the gripper stalls on the cube
(real_robot.jev_frames.phases). For approach frames before it, the label is
the joint/end-pose delta from the current state to that grasp pose:
  dj1_deg   base yaw still to turn (J1, deg)
  dr_m      horizontal reach still to go (end-pose radius from the base)
  dz_m      height still to drop
These are the targets for JEV's direction questions (real_robot.jev_dir_eval).

  python -m real_robot.jev_dirs --dataset demos/cube_grid_all_pi05 --out outputs/jev_dirs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .jev_frames import phases

OFFSETS = (240, 180, 120, 90, 60, 40, 25, 12)  # frames before the grasp (30 fps)


def grasp_frame(g: np.ndarray) -> int | None:
    ph = phases(g)
    if ph is None:
        return None
    i = int(np.flatnonzero(g > 1.2)[0])
    while g[i] > 1.2:
        i += 1
    while abs(g[i + 10] - g[i]) > 0.02:
        i += 1
    return i


def main(argv=None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args(argv)
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    eps = ds.meta.episodes
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for e in range(min(a.episodes, ds.meta.total_episodes)):
        lo, hi = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        st = np.stack(ds.hf_dataset.select(range(lo, hi))["observation.state"]).astype(np.float64)
        j = grasp_frame(st[:, 6])
        if j is None:
            continue
        sg = st[j]
        for off in OFFSETS:
            f = j - off
            if f < 0:
                continue
            s = st[f]
            item = ds[lo + f]
            paths = {}
            for cam in ("base", "wrist"):
                im = (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                paths[cam] = os.path.join(a.out, f"e{e:03d}_f{f:04d}_{cam}.jpg")
                Image.fromarray(im).save(paths[cam], quality=90)
            rows.append({"episode": e, "frame": f, "before_grasp": off, "task": item["task"],
                         "gripper": float(s[6]),
                         "dj1_deg": float(np.degrees(sg[0] - s[0])),
                         "dr_m": float(np.hypot(sg[7], sg[8]) - np.hypot(s[7], s[8])),
                         "dz_m": float(sg[9] - s[9]),
                         "q": s[:7].tolist(), "q_grasp": sg[:7].tolist(),
                         "pose": s[7:13].tolist(), "pose_grasp": sg[7:13].tolist(), **paths})
        print(f"episode {e}: grasp at {j}")
    with open(os.path.join(a.out, "labels.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    d = np.array([[r["dj1_deg"], r["dr_m"], r["dz_m"]] for r in rows])
    print(f"{len(rows)} frames; |dJ1| deg p50/p90 {np.percentile(abs(d[:, 0]), [50, 90]).round(1)}, "
          f"dr p50/p90 {np.percentile(abs(d[:, 1]), [50, 90]).round(3)}, dz p50 {np.median(d[:, 2]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
