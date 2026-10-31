"""lerobot-train with chunk-relative joint actions (real_robot/relative_actions.py).

Takes lerobot-train's arguments unchanged. Before training it computes the
action stats on relative chunks of --dataset.root (only --dataset.episodes if
given), and it puts RelativeActionsStep ahead of the normalizer in the
preprocessor, so the saved checkpoints carry it and LerobotPolicy turns their
chunks back into joint angles.

  RELATIVE=1 make finetune ...      # scripts/finetune_pi05.sh runs this
"""
from __future__ import annotations

import sys

import numpy as np

from .relative_actions import chunk_stats, register


def _arg(argv, name):
    for a in argv:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def with_action_stats(stats: dict, action_stats: dict) -> dict:
    """Copy of dataset stats with "action" swapped for the relative ones, keeping
    any absolute stat the relative set lacks."""
    out = dict(stats)
    out["action"] = {**stats.get("action", {}), **{k: np.asarray(v, np.float32)
                                                   for k, v in action_stats.items()}}
    return out


def insert_before_normalizer(pipeline, step) -> None:
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    i = next((i for i, s in enumerate(pipeline.steps) if isinstance(s, NormalizerProcessorStep)), None)
    if i is None:
        raise RuntimeError("preprocessor has no normalizer step")
    pipeline.steps.insert(i, step)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = _arg(argv, "dataset.root")
    if root is None:
        raise SystemExit("train_relative needs --dataset.root=<converted dataset>")
    eps = _arg(argv, "dataset.episodes")
    eps = [int(e) for e in eps.strip("[]").split(",") if e.strip()] if eps else None
    rel = chunk_stats(root, eps)
    print(f">> relative action stats over {int(rel['count'][0])} chunk steps: "
          f"q01 {np.round(np.degrees(rel['q01'][:6]), 1)} q99 {np.round(np.degrees(rel['q99'][:6]), 1)} deg",
          flush=True)
    step_cls = register()

    import lerobot.scripts.lerobot_train as lt

    make = lt.make_pre_post_processors

    def make_relative(*a, **kw):
        if "dataset_stats" in kw:
            kw["dataset_stats"] = with_action_stats(kw["dataset_stats"], rel)
        for key, name in (("preprocessor_overrides", "normalizer_processor"),
                          ("postprocessor_overrides", "unnormalizer_processor")):
            o = kw.get(key, {}).get(name)
            if o is not None and o.get("stats") is not None:
                o["stats"] = with_action_stats(o["stats"], rel)
        pre, post = make(*a, **kw)
        if not any(isinstance(s, step_cls) for s in pre.steps):
            insert_before_normalizer(pre, step_cls())
        return pre, post

    lt.make_pre_post_processors = make_relative
    sys.argv = [sys.argv[0], *argv]
    lt.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
