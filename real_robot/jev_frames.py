"""Labelled frames from a recorded dataset, for scoring JEV as a verifier.

Labels come from each demo's gripper trace (sim units, 1.0 = 70 mm): closed
at the start, opens wide, closes onto the cube and stalls (holding), opens to
release, closes. Per episode:
  start    first frames: not holding, cube not in the bowl
  holding  middle of the stall: holding, cube not in the bowl
  end      last frames, after release: not holding, cube in the bowl

  python -m real_robot.jev_frames --dataset demos/cube_grid_all_pi05 --out outputs/jev_frames
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def phases(g: np.ndarray, open_thr: float = 1.2) -> dict[str, list[int]] | None:
    """Frame indices per phase from one episode's gripper trace, or None."""
    n = len(g)
    opened = np.flatnonzero(g > open_thr)
    if not len(opened):
        return None
    i = opened[0]
    while i < n and g[i] > open_thr:  # until it starts closing
        i += 1
    j = i
    while j + 10 < n and abs(g[j + 10] - g[j]) > 0.02:  # until closing stalls
        j += 1
    if j + 10 >= n or g[j] < 0.4:  # closed on nothing
        return None
    k = j
    while k < n and g[k] < g[j] + 0.05:  # until it opens to release
        k += 1
    if k >= n or k - j < 40:
        return None
    mid = (j + k) // 2
    return {"start": [2, 10], "holding": [mid - 10, mid, mid + 10], "end": [n - 12, n - 3]}


def main(argv=None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    a = ap.parse_args(argv)

    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    eps = ds.meta.episodes
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for e in range(min(a.episodes, ds.meta.total_episodes)):
        lo, hi = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        g = np.array([float(x[6]) for x in ds.hf_dataset.select(range(lo, hi))["observation.state"]])
        ph = phases(g)
        if ph is None:
            print(f"episode {e}: gripper trace not parsed, skipped")
            continue
        for name, idx in ph.items():
            for f in idx:
                item = ds[lo + f]
                paths = {}
                for cam in ("base", "wrist"):
                    im = (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    paths[cam] = os.path.join(a.out, f"e{e:03d}_f{f:04d}_{cam}.jpg")
                    Image.fromarray(im).save(paths[cam], quality=90)
                rows.append({"episode": e, "frame": int(f), "phase": name, "task": item["task"],
                             "gripper": float(g[f]), "holding": name == "holding",
                             "in_bowl": name == "end", **paths})
        print(f"episode {e}: {len(g)} frames, phases {ph}")
    with open(os.path.join(a.out, "labels.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} labelled frames -> {a.out}/labels.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
