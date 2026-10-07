"""Robot/safety config. Loaded from JSON so the Mac test suite needs no PyYAML."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# Per arm: 6 joints then 1 gripper; arms are concatenated in `arms` order.
# For the RoboDojo sim checkpoint (dual ARX X5, XPolicyLab pack_robot_state):
#   [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
ARM_DOF = 6
PER_ARM = ARM_DOF + 1
STATE_DIM = 2 * PER_ARM  # sim default (dual arm)

# Sim cameras (RoboDojo/env_cfg/camera/template.py GEMINI_345LG) and the names
# the Pi_05 Model.encode_obs expects. pi0.5 resizes to 224 internally.
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SIM_IMAGE_HW = (480, 640)
# RoboDojo/env_cfg/arx_x5.yml observation.collect_freq
SIM_CONTROL_HZ = 25.0

_PER_JOINT = ("joint_lower", "joint_upper", "max_step_delta", "joint_sign", "joint_offset")


@dataclass
class RobotConfig:
    """One robot: one or more arms driven by one runner/recorder process."""

    # Arm name -> CAN interface (Piper: names from piper_sdk can_activate.sh).
    # The state/action vector concatenates arms in this order, PER_ARM each.
    arms: dict[str, str] = field(default_factory=lambda: {"left": "can_left", "right": "can_right"})
    # Per-joint lists, len == state_dim. None -> loose placeholder defaults;
    # fill from the vendor spec before any live run.
    joint_lower: list[float] | None = None
    joint_upper: list[float] | None = None
    # Max change in commanded position per control step (rad, or gripper units).
    # Larger requested jumps are rate-limited toward the target.
    max_step_delta: list[float] | None = None
    # A requested jump this large means the policy and robot disagree badly
    # (wrong joint order, units, sign): stop instead of rate-limiting.
    abort_delta: float = 0.6
    # Linear map between policy gripper units and real gripper units.
    gripper_sim_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    gripper_real_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    # Per-joint sign/offset if the real zero or direction differs from the
    # policy's: policy = sign * real + offset (gripper uses the range map).
    joint_sign: list[float] | None = None
    joint_offset: list[float] | None = None
    control_hz: float = SIM_CONTROL_HZ
    # Camera names the policy sees, in order, and the device for each
    # (OpenCV index or /dev/v4l/by-id path). Devices only needed on hardware.
    camera_names: list[str] = field(default_factory=lambda: list(CAMERAS))
    cameras: dict[str, str | int] = field(default_factory=dict)
    # Piper firmware-side speed cap (MotionCtrl_2 speed, percent).
    arm_speed_percent: int = 30
    verified: bool = False  # set true once limits/mapping are checked on hardware

    def __post_init__(self) -> None:
        n = self.state_dim
        for name, default in (("joint_lower", -3.14), ("joint_upper", 3.14),
                              ("max_step_delta", 0.05), ("joint_sign", 1.0),
                              ("joint_offset", 0.0)):
            if getattr(self, name) is None:
                setattr(self, name, [default] * n)

    @property
    def state_dim(self) -> int:
        return PER_ARM * len(self.arms)

    @property
    def gripper_idx(self) -> list[int]:
        return [PER_ARM * k + ARM_DOF for k in range(len(self.arms))]

    def validate(self) -> None:
        if not self.arms:
            raise ValueError("arms must name at least one arm")
        for name in _PER_JOINT:
            if len(getattr(self, name)) != self.state_dim:
                raise ValueError(f"{name} must have {self.state_dim} entries "
                                 f"({len(self.arms)} arm(s) x {PER_ARM})")
        if any(lo >= hi for lo, hi in zip(self.joint_lower, self.joint_upper)):
            raise ValueError("joint_lower must be < joint_upper everywhere")
        if any(d <= 0 for d in self.max_step_delta):
            raise ValueError("max_step_delta must be positive")
        for name in ("gripper_sim_range", "gripper_real_range"):
            r = getattr(self, name)
            if len(r) != 2 or r[0] == r[1]:
                raise ValueError(f"{name} must be [lo, hi] with lo != hi")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")


def _read_json(path: str) -> dict:
    """A config file; "extends": "<file>" (relative to it) loads that file first
    and overrides its keys. Keys starting with "_" are comments."""
    with open(path) as f:
        raw = json.load(f)
    base = raw.pop("extends", None)
    if base is not None:
        raw = {**_read_json(os.path.join(os.path.dirname(path), base)), **raw}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_config(path: str | None) -> RobotConfig:
    cfg = RobotConfig() if path is None else RobotConfig(**_read_json(path))
    cfg.validate()
    return cfg
