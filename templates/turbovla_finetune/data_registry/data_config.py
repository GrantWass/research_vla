"""RoboDojo data configuration for TurboVLA training.

Installed (copied) to <turbovla>/experiments/robodojo/data_registry/ by
`bash scripts/install_turbovla_training.sh`. The StarVLA registry
auto-discovers `experiments/*/data_registry/data_config.py`, which is how the
`robodojo_arx_x5` mix and robot type become visible to the training recipe --
no upstream code changes needed.

RoboDojo LeRobot data (`lerobot_v3.0_ee`) already uses the exact feature
schema TurboVLA's RoboTwin path expects:
  video  observation.images.{cam_high,cam_left_wrist,cam_right_wrist}
  state  observation.state  [left_arm(6), left_ee(1), right_arm(6), right_ee(1)]
  action action             (same 14-D layout, absolute joint targets)
  language task_index -> task string (standard LeRobot task metadata)

Task list: `ROBODOJO_TASKS` env (comma-separated dataset dir names, written by
train.sh after discovering <ROBODOJO_DATA_ROOT>/*/meta/info.json), else every
such subdir of `ROBODOJO_DATA_ROOT` is used.
"""

import os
from pathlib import Path
from typing import ClassVar

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboDojoArxX5DataConfig:
    """Three-camera, dual-arm (dual_x5) RoboDojo data, 50-step action horizon."""

    video_keys: ClassVar[list] = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys: ClassVar[list] = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys: ClassVar[list] = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]
    language_keys: ClassVar[list] = ["annotation.human.action.task_description"]
    observation_indices: ClassVar[list] = [0]
    action_indices: ClassVar[list] = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_modes = {
            "state.left_joints": "min_max",
            "state.right_joints": "min_max",
            "state.left_gripper": "binary",
            "state.right_gripper": "binary",
        }
        action_modes = {
            "action.left_joints": "min_max",
            "action.right_joints": "min_max",
            "action.left_gripper": "binary",
            "action.right_gripper": "binary",
        }
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    binary_threshold=0.49,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    binary_threshold=0.49,
                    normalization_modes=action_modes,
                ),
            ]
        )


def _discover_tasks() -> list:
    explicit = os.environ.get("ROBODOJO_TASKS", "").strip()
    if explicit:
        return [t.strip() for t in explicit.split(",") if t.strip()]
    root = os.environ.get("ROBODOJO_DATA_ROOT", "").strip()
    if root:
        found = sorted(
            p.name
            for p in Path(root).iterdir()
            if p.is_dir() and (p / "meta" / "info.json").is_file()
        )
        if found:
            return found
    raise RuntimeError(
        "No RoboDojo tasks: set ROBODOJO_TASKS='task_a,task_b' or point "
        "ROBODOJO_DATA_ROOT at a dir of LeRobot datasets (each with "
        "meta/info.json). See templates/turbovla_finetune/README.md."
    )


ROBOT_TYPE = "robodojo_arx_x5"
MIX_NAME = "robodojo_arx_x5"

ROBOT_TYPE_CONFIG_MAP = {ROBOT_TYPE: RoboDojoArxX5DataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    MIX_NAME: [(task, 1.0, ROBOT_TYPE) for task in _discover_tasks()],
}
