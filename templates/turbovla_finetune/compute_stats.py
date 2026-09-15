"""Normalization stats for a RoboDojo-trained TurboVLA checkpoint.

The eval adapter (`adapters/turbovla_robodojo/model.py`) normalizes proprio
with mean/std and denormalizes arm actions with min/max. Training-time stats
come from the dataloader; for inference this script computes the same
quantities over the training frames and writes the JSON the adapter's
`stats_path` points at.

GPU-box usage (needs the `lerobot` package from the training env):
  python templates/turbovla_finetune/compute_stats.py \
    --data-root /data/RoboDojo_ee_lerobot_v30_video \
    --out /runs/turbovla_robodojo_arx_x5_55k/robodojo_stats.json

Only `summarize_frames` (pure numpy) is imported by the repo test-suite, so
this module must stay importable without torch/lerobot/GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_frames(states: np.ndarray, actions: np.ndarray) -> dict:
    """Per-dim proprio mean/std + action min/max over [N, D] frame arrays."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"Expected 2-D frame arrays, got {states.shape} / {actions.shape}"
        )
    if states.shape[0] != actions.shape[0] or states.shape[0] == 0:
        raise ValueError("states/actions must be non-empty with equal frame counts.")
    return {
        "proprio": {
            "mean": states.mean(axis=0).astype(np.float32).tolist(),
            "std": states.std(axis=0).astype(np.float32).tolist(),
        },
        "action": {
            "min": actions.min(axis=0).astype(np.float32).tolist(),
            "max": actions.max(axis=0).astype(np.float32).tolist(),
        },
    }


def discover_tasks(data_root: Path) -> list:
    return sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").is_file()
    )


def load_frames(data_root: Path, tasks: list) -> tuple:
    """Stack observation.state / action frames across tasks (float64)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "compute_stats needs the `lerobot` package (training env on the GPU box)."
        ) from exc
    state_parts, action_parts = [], []
    for task in tasks:
        ds = LeRobotDataset(repo_id=task, root=data_root / task)
        for episode in ds.episodes():
            frame = ds[episode["episode_index"]]
            state_parts.append(
                np.asarray(frame["observation.state"], dtype=np.float64).reshape(-1)
            )
            action_parts.append(
                np.asarray(frame["action"], dtype=np.float64).reshape(-1)
            )
    if not state_parts:
        raise ValueError(f"No frames found under {data_root} for tasks={tasks}.")
    return np.stack(state_parts), np.stack(action_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TurboVLA inference stats over RoboDojo LeRobot data."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dir of per-task LeRobot datasets (lerobot_v3.0_ee).",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task dirs (default: auto-discover).",
    )
    parser.add_argument(
        "--out", default="robodojo_stats.json", help="Output JSON path."
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or discover_tasks(
        data_root
    )
    if not tasks:
        raise SystemExit(f"No LeRobot datasets under {data_root}.")
    print(
        f"[stats] {len(tasks)} tasks: {', '.join(tasks[:5])}"
        f"{' ...' if len(tasks) > 5 else ''}"
    )

    states, actions = load_frames(data_root, tasks)
    print(
        f"[stats] {states.shape[0]} frames, state_dim={states.shape[1]}, "
        f"action_dim={actions.shape[1]}"
    )
    payload = summarize_frames(states, actions)
    payload["metadata"] = {
        "data_root": str(data_root),
        "tasks": tasks,
        "frames": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[stats] wrote {out}")


if __name__ == "__main__":
    main()
