"""lerobot-train with chunk-relative joint actions (real_robot/relative_actions.py).

Takes lerobot-train's arguments unchanged. Before training it computes the
action stats on relative chunks of --dataset.root (only --dataset.episodes if
given), and it puts RelativeActionsStep ahead of the normalizer in the
preprocessor, so the saved checkpoints carry it and LerobotPolicy turns their
chunks back into joint angles.

  RELATIVE=1 make finetune ...      # scripts/finetune_pi05.sh runs this

Two optional env settings target joints that rarely move (the wrist roll J4/J6
moves in ~15% of cube_grid frames and plain training learns ~10% of it):
  WRIST_OVERSAMPLE=3                  # frames whose next chunk turns J4 or J6 > 3 deg drawn 3x as often
  JOINT_LOSS_WEIGHTS=1,1,1,2,1.5,2,1  # loss weight per action dim (rescaled to mean 1)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .relative_actions import chunk_stats, register, relative_chunks

WRIST = (3, 5)  # J4, J6: wrist roll


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


def rotation_weights(episode, action, state, factor: float, joints=WRIST, deg: float = 3.0):
    """Sampling weight per frame: factor where the chunk starting there moves any
    of joints more than deg from the observed state, else 1."""
    w = np.ones(len(episode))
    for e in np.unique(episode):
        i = np.flatnonzero(episode == e)
        rel = relative_chunks(action[i], state[i])[..., list(joints)]
        w[i[np.abs(rel).max((1, 2)) > np.radians(deg)]] = factor
    return w


def loss_weights(spec: str) -> np.ndarray:
    w = np.array([float(x) for x in spec.split(",")])
    return w / w.mean()


def patch_loss(weights: np.ndarray) -> None:
    """Scale pi0.5's per-dimension flow-matching loss (B, T, max_action_dim)."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    forward = PI05Pytorch.forward

    def weighted(self, *a, **kw):
        losses = forward(self, *a, **kw)
        w = torch.ones(losses.shape[-1], device=losses.device, dtype=losses.dtype)
        w[:len(weights)] = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype)
        return losses * w

    PI05Pytorch.forward = weighted


def patch_sampler(lt, factor: float) -> None:
    """Swap the training DataLoader's shuffle for a WeightedRandomSampler over rotation_weights."""
    import torch

    make_dataset, loader = lt.make_dataset, torch.utils.data.DataLoader
    seen = {}

    def capture(*a, **kw):
        seen["ds"] = make_dataset(*a, **kw)
        return seen["ds"]

    def weighted_loader(dataset, *a, **kw):
        if dataset is seen.get("ds") and kw.get("sampler") is None:
            hf = dataset.hf_dataset
            w = rotation_weights(np.asarray(hf["episode_index"]), np.stack(hf["action"]),
                                 np.stack(hf["observation.state"]), factor)
            print(f">> WRIST_OVERSAMPLE={factor}: {np.mean(w > 1):.0%} of frames turn J4/J6, now "
                  f"{w[w > 1].sum() / w.sum():.0%} of draws", flush=True)
            kw["shuffle"] = False
            kw["sampler"] = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
            torch.utils.data.DataLoader = loader  # accelerate isinstance-checks the real class
        return loader(dataset, *a, **kw)

    lt.make_dataset = capture
    torch.utils.data.DataLoader = weighted_loader


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
    if os.environ.get("WRIST_OVERSAMPLE"):
        patch_sampler(lt, float(os.environ["WRIST_OVERSAMPLE"]))
    if os.environ.get("JOINT_LOSS_WEIGHTS"):
        w = loss_weights(os.environ["JOINT_LOSS_WEIGHTS"])
        print(f">> JOINT_LOSS_WEIGHTS: {np.round(w, 2)}", flush=True)
        patch_loss(w)
    sys.argv = [sys.argv[0], *argv]
    lt.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
