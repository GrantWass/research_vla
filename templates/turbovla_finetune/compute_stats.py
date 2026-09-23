"""Normalization stats for a RoboDojo-trained TurboVLA checkpoint.

The eval adapter (`adapters/turbovla_robodojo/model.py`) normalizes proprio
with mean/std and denormalizes arm actions with min/max. Training-time stats
come from the dataloader; for inference this script computes the same
quantities over the training frames and writes the JSON the adapter's
`stats_path` points at.

GPU-box usage (needs the `lerobot` package from the training env):
  python templates/turbovla_finetune/compute_stats.py \
    --data-root data/robodojo_tasks_joint \
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
    """Stack observation.state / action frames across tasks (float64).

    Reads the LeRobot v3 parquet shards directly (pyarrow), so this runs in any
    env — no lerobot install, no video decoding. Only the episodes listed in
    each task's meta/episodes/ are counted, which is what make_task_dataset.py
    writes for a single-task view of the combined RoboDojo download.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "compute_stats needs pyarrow (conda activate RoboDojo)."
        ) from exc

    state_parts, action_parts = [], []
    for task in tasks:
        root = data_root / task
        info = json.loads((root / "meta" / "info.json").read_text())
        episodes = [
            pq.read_table(f).to_pydict()
            for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))
        ]
        wanted, shards = set(), set()
        for cols in episodes:
            wanted.update(cols["episode_index"])
            shards.update(zip(cols["data/chunk_index"], cols["data/file_index"]))
        for chunk_index, file_index in sorted(shards):
            rel = info["data_path"].format(
                chunk_index=chunk_index, file_index=file_index
            )
            table = pq.read_table(
                root / rel,
                columns=["episode_index", "observation.state", "action"],
                memory_map=True,
            ).to_pydict()
            for ep, state, action in zip(
                table["episode_index"], table["observation.state"], table["action"]
            ):
                if ep in wanted:
                    state_parts.append(np.asarray(state, dtype=np.float64))
                    action_parts.append(np.asarray(action, dtype=np.float64))
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
    parser.add_argument(
        "--key",
        default="robodojo_arx_x5",
        help="Top-level stats key the adapter resolves (deploy.yml stats_key; "
        "null there auto-resolves when the file has exactly one).",
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
    payload = {args.key: summarize_frames(states, actions)}
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
