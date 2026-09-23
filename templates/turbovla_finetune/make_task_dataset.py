#!/usr/bin/env python3
"""Carve a single-task LeRobot dataset out of RoboDojo's combined download.

`download_data.sh ... lerobot_v3.0_ee` produces ONE dataset holding every task
(3500 episodes / 1.86 M frames / 120 GB). The TurboVLA training recipe
(`data_registry/data_config.py`, `train.sh`) wants a directory of per-task
LeRobot datasets, one `meta/info.json` each. This script writes that view for
the tasks you name, without copying frame data:

    python templates/turbovla_finetune/make_task_dataset.py \
      --source RoboDojo/.cache/robodojo_assets_repo/data/RoboDojo_ee_lerobot_v30_video \
      --out data/robodojo_tasks --task stack_bowls

Output: <out>/<task>/{meta,data,videos} where `data/` and `videos/` are
symlinks to the source files and `meta/` is rewritten to list only that task's
episodes. Row indices inside the parquet files are left untouched (the episode
metadata addresses frames by absolute index), so only metadata is filtered.

Tasks are matched against RoboDojo task names (`meta/tasks.parquet`), either by
the benchmark task id (`stack_bowls`, matched on its words) or by a substring
of the language string. `--list` prints what is available.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - stdlib-only CI
    pq = None

# GR00T-style modality map the TurboVLA loader needs (meta/modality.json).
# RoboDojo packs dual_x5 as [arm_0(6), ee_0(1), arm_1(6), ee_1(1)], which is the
# same layout as the upstream RoboTwin recipe's modality.json.
DUAL_ARM_14D_MODALITY = {
    "action": {
        "left_joints": {"start": 0, "end": 6, "original_key": "action"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
        "right_joints": {"start": 7, "end": 13, "original_key": "action"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}

VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _episode_rows(source: Path) -> dict:
    files = sorted((source / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        sys.exit(
            f"[make_task_dataset] no episode metadata under {source}/meta/episodes"
        )
    tables = [pq.read_table(f) for f in files]
    import pyarrow as pa

    return pa.concat_tables(tables)


def _task_of(row_tasks) -> str:
    # `tasks` is a list column (one language string per episode in RoboDojo).
    if isinstance(row_tasks, (list, tuple)):
        return row_tasks[0] if row_tasks else ""
    return str(row_tasks)


def _matches(task_text: str, wanted: str) -> bool:
    text = task_text.lower()
    if wanted.lower() in text:
        return True
    # `stack_bowls` -> every word must appear ("Stack the three bowls together.")
    words = [w for w in wanted.lower().replace("-", "_").split("_") if w]
    return bool(words) and all(w.rstrip("s") in text for w in words)


def _state_dim(source: Path, info: dict, cols: dict, episode: int) -> int:
    """Width of observation.state in the source parquet (14 = joints, 16 = ee poses)."""
    rel = info["data_path"].format(
        chunk_index=cols["data/chunk_index"][episode],
        file_index=cols["data/file_index"][episode],
    )
    table = pq.read_table(source / rel, columns=["observation.state"], memory_map=True)
    return len(table.column("observation.state")[0].as_py())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", required=True, help="combined lerobot_v3.0_ee dataset dir"
    )
    ap.add_argument("--out", required=True, help="output dir (one subdir per task)")
    ap.add_argument(
        "--task", action="append", default=[], help="task id or text (repeatable)"
    )
    ap.add_argument(
        "--list", action="store_true", help="list tasks with episode counts and exit"
    )
    ap.add_argument(
        "--copy", action="store_true", help="copy files instead of symlinking"
    )
    args = ap.parse_args()

    if pq is None:
        sys.exit("[make_task_dataset] needs pyarrow (conda activate RoboDojo)")
    source = Path(args.source).resolve()
    episodes = _episode_rows(source)
    cols = episodes.to_pydict()
    task_texts = [_task_of(t) for t in cols["tasks"]]

    if args.list:
        counts: dict[str, int] = {}
        for text in task_texts:
            counts[text] = counts.get(text, 0) + 1
        for text, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{n:5d}  {text}")
        return
    if not args.task:
        sys.exit("[make_task_dataset] pass --task (or --list)")

    info = json.loads((source / "meta" / "info.json").read_text())
    out_root = Path(args.out).resolve()

    for wanted in args.task:
        keep = [i for i, text in enumerate(task_texts) if _matches(text, wanted)]
        if not keep:
            sys.exit(f"[make_task_dataset] no episodes match {wanted!r} (try --list)")
        texts = {task_texts[i] for i in keep}
        if len(texts) > 1:
            sys.exit(
                f"[make_task_dataset] {wanted!r} matches {len(texts)} tasks: {sorted(texts)}"
            )

        dest = out_root / wanted
        (dest / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        kept = episodes.take(keep)

        # Link only the parquet/video files this task's episodes reference.
        needed = {
            Path(
                info["data_path"].format(
                    chunk_index=cols["data/chunk_index"][i],
                    file_index=cols["data/file_index"][i],
                )
            )
            for i in keep
        }
        for key in VIDEO_KEYS:
            needed |= {
                Path(
                    info["video_path"].format(
                        video_key=key,
                        chunk_index=cols[f"videos/{key}/chunk_index"][i],
                        file_index=cols[f"videos/{key}/file_index"][i],
                    )
                )
                for i in keep
            }

        missing = [p for p in sorted(needed) if not (source / p).exists()]
        if missing:
            sys.exit(
                f"[make_task_dataset] {len(missing)} source files missing, e.g. {missing[0]}\n"
                "  pull them first (git lfs pull --include=...) — see GPU_BOX_SETUP.md"
            )
        for rel in sorted(needed):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                continue
            if args.copy:
                shutil.copy2(source / rel, target)
            else:
                os.symlink(source / rel, target)

        pq.write_table(
            kept, dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        shutil.copy2(source / "meta" / "tasks.parquet", dest / "meta" / "tasks.parquet")
        if (source / "meta" / "stats.json").exists():
            shutil.copy2(source / "meta" / "stats.json", dest / "meta" / "stats.json")
        task_info = dict(info)
        task_info["total_episodes"] = len(keep)
        task_info["total_frames"] = int(sum(cols["length"][i] for i in keep))
        task_info["total_tasks"] = 1
        (dest / "meta" / "info.json").write_text(json.dumps(task_info, indent=2))
        # Fail loudly rather than write a modality map that does not match the data.
        state_dim = _state_dim(source, info, cols, keep[0])
        if state_dim != 14:
            sys.exit(
                f"[make_task_dataset] observation.state is {state_dim}-D, expected 14 "
                "(dual_x5 joints+grippers). lerobot_v3.0_ee is 16-D end-effector poses: "
                "use the joint variant lerobot_v3.0."
            )
        (dest / "meta" / "modality.json").write_text(
            json.dumps(DUAL_ARM_14D_MODALITY, indent=2)
        )

        print(
            f"[make_task_dataset] {wanted}: {len(keep)} episodes, "
            f"{task_info['total_frames']} frames, {len(needed)} source files -> {dest}"
        )


if __name__ == "__main__":
    main()
