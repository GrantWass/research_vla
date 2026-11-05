"""Merge recorded demo datasets (same features) into one, e.g. the batches of a long session.

LeRobot keeps a session's data parquet open, and buffers episode metadata
(10 episodes), until finalize(): a hard crash (killed process, power cut)
mid-session can lose that session's joint data. So record long sessions in
batches (make record EPISODES=10, a new DIR each) and merge them afterwards:

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.merge_datasets \\
      demos/pick_v5 demos/pick_v5_a demos/pick_v5_b ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def merge(dst: str, srcs: list[str], log=print) -> int:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists: merge into a new directory")
    name = lambda p: f"local/{Path(p).name}"  # noqa: E731
    aggregate_datasets(repo_ids=[name(s) for s in srcs], aggr_repo_id=name(dst),
                       roots=[Path(s) for s in srcs], aggr_root=Path(dst))
    out = LeRobotDataset(name(dst), root=dst)
    log(f"wrote {dst}: {out.num_episodes} episodes, {out.num_frames} frames from {len(srcs)} datasets")
    return out.num_episodes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dst", help="new directory for the merged dataset")
    ap.add_argument("srcs", nargs="+", help="datasets to merge, in order")
    args = ap.parse_args(argv)
    merge(args.dst, args.srcs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
