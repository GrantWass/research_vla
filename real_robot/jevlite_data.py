"""Phase-labelled frames from every demo, for training JEV-lite (real_robot/jevlite.py).

Each cube demo's gripper trace (sim units, 1.0 = 70 mm) gives exact labels:
  idle      before the gripper first opens wide
  approach  open, moving to the cube
  grasp     the last GRASP_LEAD frames before closing starts, and the closing itself
  holding   closed and stalled on the cube (lift + carry); the last third is
            "over the bowl, not released" (late_carry=True), the case JEV-9B
            scored as placed on 2026-11-05
  release   opening over the bowl
  placed    released, cube in the bowl
Frames within MARGIN of a boundary are dropped (labels are ambiguous there).

  python -m real_robot.jevlite_data --dataset demos/cube_grid_all_pi05 --out outputs/jevlite/frames
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PHASES = ["idle", "approach", "grasp", "holding", "release", "placed"]
GRASP_LEAD = 15  # frames (0.5 s at 30 fps) before closing that count as "ready to grasp"
RELEASE_LEN = 15
MARGIN = 3


def segments(g: np.ndarray) -> tuple[int, int, int, int] | None:
    """(open, close_start, stall, release) frame indices, or None if unparsed.
    Thresholds are relative to the episode (operators open 1.17..1.42 wide)."""
    n = len(g)
    open_thr = 0.85 * float(g.max())
    if open_thr < 0.6:
        return None
    o = int(np.flatnonzero(g > open_thr)[0])
    i = o
    while i < n and g[i] > open_thr:
        i += 1
    j = i
    while j + 10 < n and abs(g[j + 10] - g[j]) > 0.02:
        j += 1
    if j + 10 >= n or g[j] < 0.4:  # closed on nothing
        return None
    hold = float(np.median(g[j + 20:j + 60]))  # fingers settle on the cube over ~0.5 s
    k = j + 20
    while k < n and abs(g[k] - hold) < 0.06:  # release: opens, or the operator closes after
        k += 1
    if k >= n or k - j < 40:
        return None
    return o, i, j, k


def label(n: int, seg: tuple[int, int, int, int]) -> list[tuple[str, bool] | None]:
    """Per-frame (phase, late_carry) or None near a boundary."""
    o, i, j, k = seg
    bounds = [0, o, max(o, i - GRASP_LEAD), j, k, min(n, k + RELEASE_LEN), n]
    out: list[tuple[str, bool] | None] = []
    for f in range(n):
        p = int(np.searchsorted(bounds, f, side="right")) - 1
        p = min(max(p, 0), len(PHASES) - 1)
        near = min(abs(f - b) for b in bounds[1:-1]) < MARGIN
        out.append(None if near else (PHASES[p], PHASES[p] == "holding" and f > j + 0.67 * (k - j)))
    return out


def main(argv=None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args(argv)

    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    eps = ds.meta.episodes
    os.makedirs(a.out, exist_ok=True)
    rows, skipped = [], []
    for e in range(ds.meta.total_episodes):
        lo, hi = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        st = np.asarray(ds.hf_dataset.select(range(lo, hi))["observation.state"], dtype=np.float32)
        seg = segments(st[:, 6])
        if seg is None:
            skipped.append(e)
            continue
        labs = label(len(st), seg)
        for f in range(0, len(st), a.stride):
            if labs[f] is None:
                continue
            item = ds[lo + f]
            paths = {}
            for cam in ("base", "wrist"):
                im = (item[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                paths[cam] = os.path.join(a.out, f"e{e:03d}_f{f:04d}_{cam}.jpg")
                Image.fromarray(im).resize((a.size, a.size), Image.BILINEAR).save(paths[cam], quality=90)
            phase, late = labs[f]
            rows.append({"episode": e, "frame": f, "n": len(st), "phase": phase, "late_carry": late,
                         "state": [round(float(x), 5) for x in st[f]], "seg": list(seg), **paths})
        print(f"episode {e}: {len(st)} frames, seg {seg}", flush=True)
    with open(os.path.join(a.out, "labels.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    counts = {p: sum(r["phase"] == p for r in rows) for p in PHASES}
    print(f"{len(rows)} frames, skipped episodes {skipped}, per phase {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
