"""Convert between real-robot units and the sim units the policy was trained in."""
from __future__ import annotations

import numpy as np

from .config import RobotConfig


def _lerp(x, src, dst):
    return dst[0] + (np.asarray(x) - src[0]) * (dst[1] - dst[0]) / (src[1] - src[0])


def real_to_sim(q_real, cfg: RobotConfig) -> np.ndarray:
    q = np.asarray(q_real, dtype=np.float64).reshape(cfg.state_dim)
    out = np.asarray(cfg.joint_sign) * q + np.asarray(cfg.joint_offset)
    for i in cfg.gripper_idx:
        out[i] = _lerp(q[i], cfg.gripper_real_range, cfg.gripper_sim_range)
    return out


def sim_to_real(q_sim, cfg: RobotConfig) -> np.ndarray:
    q = np.asarray(q_sim, dtype=np.float64).reshape(cfg.state_dim)
    out = (q - np.asarray(cfg.joint_offset)) / np.asarray(cfg.joint_sign)
    for i in cfg.gripper_idx:
        out[i] = _lerp(q[i], cfg.gripper_sim_range, cfg.gripper_real_range)
    return out
