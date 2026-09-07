"""Zero-action policy adapter template.

Copy this directory to RoboDojo/XPolicyLab/policy/<YourPolicy>/ and replace
get_action() with real inference. Imports of XPolicyLab resolve once the file
lives inside the XPolicyLab tree (not from this templates/ folder).
"""
import numpy as np

try:
    from XPolicyLab.model_template import ModelTemplate
    from XPolicyLab.utils.process_data import (
        get_action_dim,
        get_batch_size,
        get_robot_action_dim_info,
    )
except ImportError:  # allow syntax/import checks outside the XPolicyLab tree
    ModelTemplate = object

    def _missing(*args, **kwargs):
        raise ImportError("Run this adapter from inside the XPolicyLab tree.")

    get_action_dim = get_batch_size = get_robot_action_dim_info = _missing


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]

        self.action_dim = get_action_dim(self.env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)

        assert len(self.robot_action_dim_info["arm_dim"]) == len(
            self.robot_action_dim_info["ee_dim"]
        ), "Arm and EE action dimensions must match"

        # TODO: load model weights here, e.g.
        #   ckpt = os.path.join("checkpoints", model_cfg["ckpt_name"])
        print(f"[Model] initialized, action_type={self.action_type}")

    def update_obs(self, obs):
        # TODO: preprocess/store single observation (images, proprio, instruction).
        pass

    def update_obs_batch(self, obs_list):
        # TODO: batched variant; required only if deploy.yml sets eval_batch: true.
        for obs in obs_list:
            self.update_obs(obs)

    def get_action(self):
        # TODO: replace zero actions with model inference. Return a *list* of
        # per-step action dicts (action chunk); the eval loop steps through it.
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms == 1:
            arm_keys = ["arm_joint_state"] if self.action_type == "joint" else ["ee_pose"]
            ee_keys = ["ee_joint_state"]
        elif num_arms == 2:
            arm_keys = (
                ["left_arm_joint_state", "right_arm_joint_state"]
                if self.action_type == "joint"
                else ["left_ee_pose", "right_ee_pose"]
            )
            ee_keys = ["left_ee_joint_state", "right_ee_joint_state"]
        else:
            raise NotImplementedError(f"Unsupported number of arms: {num_arms}")

        action_list = []
        for _ in range(2):  # TODO: chunk length = your policy's action horizon
            action_dict = {}
            for i, (arm_key, ee_key) in enumerate(zip(arm_keys, ee_keys)):
                if self.action_type == "joint":
                    action_dict[arm_key] = np.zeros(
                        self.robot_action_dim_info["arm_dim"][i], dtype=np.float32
                    )
                else:
                    # 7-D EE pose [x, y, z, qw, qx, qy, qz]
                    action_dict[arm_key] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )
                action_dict[ee_key] = np.zeros(
                    self.robot_action_dim_info["ee_dim"][i], dtype=np.float32
                )
            action_list.append(action_dict)
        return action_list

    def get_action_batch(self, env_idx_list=None):
        batch_size = len(env_idx_list) if env_idx_list is not None else self.batch_size
        return [self.get_action() for _ in range(batch_size)]

    def reset(self):
        # TODO: clear recurrent/hidden state at episode start.
        pass
