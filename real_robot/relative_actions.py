"""Chunk-relative joint actions for pi0.5 (openpi's DeltaActions).

The model predicts each 50-step chunk as joint angles minus the joints at the
observation, so "stay where you are" is all zeros and a chunk can't start
offset from the arm. The gripper stays absolute (0..1), as in openpi's
ALOHA/DROID configs.

Training (real_robot/train_relative.py, `RELATIVE=1 make finetune`) inserts
RelativeActionsStep into the saved preprocessor ahead of the normalizer, with
the action normalization stats computed on relative chunks. That step in a
checkpoint's policy_preprocessor.json is what marks it relative: LerobotPolicy
finds it and adds the observed joints back to every chunk it returns.

Everything but register() works without LeRobot.
"""
from __future__ import annotations

import glob
import os

import numpy as np

JOINTS = 6
CHUNK = 50
REGISTRY_NAME = "piper_relative_actions"


def to_relative(action, state, joints: int = JOINTS):
    """(..., T, A) chunk minus the (..., S) state on the first `joints` dims."""
    out = action.clone() if hasattr(action, "clone") else np.array(action, copy=True)
    out[..., :joints] -= state[..., None, :joints]
    return out


def to_absolute(action, state, joints: int = JOINTS):
    """Inverse of to_relative."""
    out = action.clone() if hasattr(action, "clone") else np.array(action, copy=True)
    out[..., :joints] += state[..., None, :joints]
    return out


def relative_chunks(actions: np.ndarray, states: np.ndarray, chunk: int = CHUNK,
                    joints: int = JOINTS) -> np.ndarray:
    """Every training chunk of one episode, relative: (L, chunk, A). Steps past
    the end repeat the last action, as LeRobot pads them."""
    n = len(actions)
    idx = np.minimum(np.arange(n)[:, None] + np.arange(chunk)[None, :], n - 1)
    return to_relative(actions[idx], states, joints)


def chunk_stats(root: str, episodes=None, chunk: int = CHUNK, joints: int = JOINTS) -> dict:
    """Action normalization stats (q01/q99/mean/std/min/max, per dim) over the
    relative chunks of a converted dataset; the gripper's come out unchanged."""
    import pandas as pd

    files = sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}/data")
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
                   for f in files)
    if episodes is not None:
        df = df[df.episode_index.isin(list(episodes))]
    parts = []
    for _, g in df.groupby("episode_index", sort=True):
        a = np.stack(g.action.values).astype(np.float32)
        s = np.stack(g["observation.state"].values).astype(np.float32)
        parts.append(relative_chunks(a, s, chunk, joints).reshape(-1, a.shape[1]))
    x = np.concatenate(parts)
    return {"q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0),
            "q10": np.quantile(x, 0.10, axis=0), "q90": np.quantile(x, 0.90, axis=0),
            "mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0),
            "count": np.array([len(x)])}


def relative_joints(pipeline) -> int:
    """Joints a loaded preprocessor makes relative (0: absolute checkpoint)."""
    for step in getattr(pipeline, "steps", []):
        if getattr(type(step), "_registry_name", None) == REGISTRY_NAME:
            return step.joints
    return 0


def register():
    """Register RelativeActionsStep with LeRobot (idempotent) and return it."""
    from dataclasses import dataclass

    from lerobot.processor.core import TransitionKey
    from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
    from lerobot.utils.constants import OBS_STATE

    if REGISTRY_NAME in ProcessorStepRegistry._registry:
        return ProcessorStepRegistry._registry[REGISTRY_NAME]

    @ProcessorStepRegistry.register(name=REGISTRY_NAME)
    @dataclass
    class RelativeActionsStep(ProcessorStep):
        """Action chunk joints minus observed joints, before normalization.
        Passes through when there is no action (inference)."""

        joints: int = JOINTS

        def __call__(self, transition):
            action = transition.get(TransitionKey.ACTION)
            if action is None:
                return transition
            state = transition[TransitionKey.OBSERVATION][OBS_STATE]
            transition = transition.copy()
            transition[TransitionKey.ACTION] = to_relative(action, state.to(action.dtype), self.joints)
            return transition

        def get_config(self):
            return {"joints": self.joints}

        def transform_features(self, features):
            return features

    return RelativeActionsStep
