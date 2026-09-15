"""TurboVLA policy adapter for XPolicyLab / RoboDojo eval.

Upstream model: https://github.com/H-EmbodVis/TurboVLA
  forward(instructions: [str], samples: [B,V,3,H,W], state: [B,S]) -> [B,H,A]
Released ckpts: LIBERO single-arm (.pth, 2 views, S=8, A=7) and RoboTwin
bimanual (.safetensors, 3 views, A=14). There is no RoboDojo-finetuned ckpt
yet — see README.md; dims that do not line up fail fast with a pointer.

Conventions mirror RoboDojo/XPolicyLab/policy/SmolVLA/model.py:
  - update_obs accepts the server-decoded "images/state" payload or a raw env
    obs with a "vision" dict (same camera fallback lists).
  - get_action returns a list of per-step action dicts built with
    XPolicyLab.utils.process_data.unpack_robot_state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
# Installed location is RoboDojo/XPolicyLab/policy/TurboVLA, so the workspace
# root (which holds the `turbovla/` checkout from setup.sh) is 3 levels up.
_WORKSPACE_ROOT = _CUR_DIR.parents[3]

for _path in (str(_WORKSPACE_ROOT),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots  # noqa: E402
from XPolicyLab.utils.process_data import (  # noqa: E402
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

# Fallback normalization = TurboVLA LIBERO suite stats
# (turbovla/evaluation/policy.py in the upstream repo). Used only when the
# adapter is pointed at a LIBERO ckpt without an explicit stats file.
_LIBERO_PROPRIO_MEAN = np.array(
    [-0.04190646, 0.03539438, 0.82570666, 2.90831566,
     -0.55621588, -0.16649103, 0.02831535, -0.02856156], dtype=np.float32)
_LIBERO_PROPRIO_STD = np.array(
    [0.10743438, 0.14424760, 0.25723374, 0.34413809,
     1.23443019, 0.35798806, 0.01330879, 0.01317459], dtype=np.float32)
_LIBERO_ACTION_MIN = np.array(
    [-0.9375, -0.9375, -0.9375, -0.23642857, -0.30535713, -0.3675, -1.0],
    dtype=np.float32)
_LIBERO_ACTION_MAX = np.array(
    [0.9375, 0.9375, 0.9375, 0.30000001, 0.29357144, 0.375, 1.0],
    dtype=np.float32)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PRIMARY_CANDIDATES = ("cam_high", "cam_head", "head_camera", "top_camera")
_WRIST_CANDIDATES = ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left")


def _extract_image(observation: dict, candidates: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision", {})
    for name in candidates:
        if name not in vision:
            continue
        image = vision[name]
        if isinstance(image, dict):
            for key in ("color", "rgb"):
                if key in image:
                    return np.asarray(image[key])
        else:
            return np.asarray(image)
    raise KeyError(f"Could not find any camera image for candidates: {candidates}")


def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[-1] in (1, 3):
        hwc = image
    elif image.shape[0] in (1, 3):
        hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    return np.ascontiguousarray(hwc)


def _normalize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif ((isinstance(value, np.ndarray) and value.ndim == 0)
            or isinstance(value, np.generic)):
        value = value.item()
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_prompt(item)
            if normalized is not None:
                return normalized
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _resolve_prompt(observation: dict, default_prompt: str) -> str:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt(observation.get(key))
        if prompt is not None:
            return prompt
    fallback = _normalize_prompt(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or model config.")
    return fallback


def _load_stats(stats_path: str | None, stats_key: str | None):
    """Return (proprio_mean, proprio_std, action_min, action_max)."""
    if not stats_path:
        return (_LIBERO_PROPRIO_MEAN, _LIBERO_PROPRIO_STD,
                _LIBERO_ACTION_MIN, _LIBERO_ACTION_MAX)
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    if stats_key:
        if stats_key not in payload:
            raise KeyError(f"stats_key={stats_key!r} not found in {stats_path}")
        stats = payload[stats_key]
    else:
        keys = [k for k in payload if k != "metadata"]
        if len(keys) != 1:
            raise KeyError(
                f"stats_key is required for {stats_path}; available keys: {keys}")
        stats = payload[keys[0]]
    state_section = "proprio" if "proprio" in stats else "state"
    return (
        np.asarray(stats[state_section]["mean"], dtype=np.float32),
        np.asarray(stats[state_section]["std"], dtype=np.float32),
        np.asarray(stats["action"]["min"], dtype=np.float32),
        np.asarray(stats["action"]["max"], dtype=np.float32),
    )


def _preprocessor_stats(dinov3_path: str | None):
    """(mean, std) from the DINOv3 preprocessor config, else ImageNet."""
    if dinov3_path:
        cfg_path = Path(dinov3_path) / "preprocessor_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return (
                np.asarray(cfg.get("image_mean", [0.485, 0.456, 0.406]),
                           dtype=np.float32),
                np.asarray(cfg.get("image_std", [0.229, 0.224, 0.225]),
                           dtype=np.float32),
            )
    return _IMAGENET_MEAN, _IMAGENET_STD


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        import torch  # GPU-box dependency; repo tests never import this module

        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError(
                "TurboVLA adapter currently supports only action_type='joint' "
                f"(got {self.action_type!r}).")

        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(env_cfg) if env_cfg is not None else None)
        if self.robot_action_dim_info is None:
            raise ValueError("env_cfg_type is required for the TurboVLA adapter.")
        self.packed_dim = int(sum(self.robot_action_dim_info["arm_dim"])
                              + sum(self.robot_action_dim_info["ee_dim"]))
        self.num_arms = len(self.robot_action_dim_info["arm_dim"])

        self.default_prompt = self.model_cfg.get("prompt") or self.task_name
        self.device = self._resolve_device(self.model_cfg.get("device", "cuda"))
        self.precision = str(self.model_cfg.get("precision", "bf16")).lower()
        self.num_views = int(self.model_cfg.get("num_views", 2))
        self.image_size = int(self.model_cfg.get("image_size", 256))
        self.chunk_size = int(self.model_cfg.get("chunk_size", 12))
        self.num_open_loop_steps = int(
            self.model_cfg.get("num_open_loop_steps", self.chunk_size))
        self.state_dim = int(self.model_cfg.get("state_dim", 8))
        self.action_dim = int(self.model_cfg.get("action_dim", 7))
        self.dual_arm_mode = str(self.model_cfg.get("dual_arm_mode", "error"))

        self.dinov3_path = (self.model_cfg.get("dinov3_path")
                            or os.environ.get("DINOV3_PATH"))
        self.bert_path = (self.model_cfg.get("bert_path")
                          or os.environ.get("BERT_PATH"))
        if not self.dinov3_path or not self.bert_path:
            raise ValueError(
                "dinov3_path and bert_path are required (deploy.yml or "
                "$DINOV3_PATH / $BERT_PATH). See adapters/turbovla_robodojo/README.md.")
        self.img_mean, self.img_std = _preprocessor_stats(self.dinov3_path)

        stats_path = (self.model_cfg.get("stats_path")
                      or os.environ.get("TURBOVLA_STATS"))
        self.proprio_mean, self.proprio_std, self.action_min, self.action_max = \
            _load_stats(stats_path, self.model_cfg.get("stats_key"))
        if self.proprio_mean.shape[0] != self.state_dim:
            raise ValueError(
                f"Proprio stats dim {self.proprio_mean.shape[0]} != "
                f"state_dim {self.state_dim}; point stats_path at the ckpt's "
                "training stats.")

        self.torch = torch
        self.policy = self._load_policy()
        self.model = self.policy

        self._obs: dict | None = None
        self._chunk: list[dict] | None = None
        self._warned_single_view = False
        print(f"[TurboVLA] ready: views={self.num_views} S={self.state_dim} "
              f"A={self.action_dim} chunk={self.chunk_size} "
              f"open_loop={self.num_open_loop_steps}")

    # ---- setup helpers ----

    def _resolve_device(self, name: str):
        import torch

        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("[TurboVLA] CUDA unavailable, falling back to CPU.")
            return torch.device("cpu")
        return dev

    def _turbovla_repo(self) -> Path:
        override = (self.model_cfg.get("turbovla_repo")
                    or os.environ.get("TURBOVLA_REPO"))
        if override:
            repo = Path(override).expanduser()
            if not repo.is_dir():
                raise FileNotFoundError(f"turbovla_repo not found: {repo}")
            return repo
        repo = _WORKSPACE_ROOT / "turbovla"
        if not (repo / "turbovla" / "models" / "turbovla.py").is_file():
            raise FileNotFoundError(
                f"turbovla checkout not found at {repo} (run setup.sh; "
                "or set turbovla_repo / $TURBOVLA_REPO).")
        return repo

    def _resolve_checkpoint(self) -> Path:
        candidates = candidate_checkpoint_roots(
            self.model_cfg, _CHECKPOINTS_DIR, policy_dir=_CUR_DIR,
            explicit_keys=("checkpoint_path", "ckpt_path", "pretrained_path",
                           "model_path"))
        if not candidates:
            raise ValueError(
                "ckpt_name, checkpoint_path, ckpt_path, or pretrained_path is "
                "required for TurboVLA.")
        files: list[Path] = []
        for root in candidates:
            root = Path(root)
            if root.is_file() and root.suffix in (
                    ".pth", ".safetensors", ".pt", ".bin"):
                files.append(root)
            elif root.is_dir():
                files.extend(sorted(root.glob("*.pth")))
                files.extend(sorted(root.glob("*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No TurboVLA checkpoint (*.pth/*.safetensors) under: {candidates}")
        # Prefer EMA weights (upstream LIBERO/RoboTwin releases ship EMA).
        ema = [f for f in files if "ema" in f.name.lower()]
        pool = ema or files
        return max(pool, key=lambda f: f.stat().st_mtime)

    def _load_policy(self):
        torch = self.torch
        repo = self._turbovla_repo()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from turbovla.models.configuration import (
                ActionHeadConfig, InteractionConfig, TextEncoderConfig,
                TurboVLAConfig, VisionEncoderConfig)
            from turbovla.models.turbovla import TurboVLA
        except ImportError as exc:
            raise ImportError(
                f"Could not import turbovla from {repo}: {exc}. "
                "Did install.sh finish (pip install -e <turbovla>)?") from exc

        ckpt_path = self._resolve_checkpoint()
        print(f"[TurboVLA] loading checkpoint: {ckpt_path}")
        config = TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=self.bert_path, frozen=True,
                local_files_only=True),
            vision=VisionEncoderConfig(
                model_name_or_path=self.dinov3_path,
                image_size=self.image_size, num_views=self.num_views,
                frozen=True, local_files_only=True),
            interaction=InteractionConfig(),
            action=ActionHeadConfig(
                action_dim=self.action_dim, state_dim=self.state_dim,
                horizon=self.chunk_size),
        )
        policy = TurboVLA(config)
        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(blob, dict):
            for key in ("ema_model_state_dict", "model_state_dict", "state_dict"):
                if isinstance(blob.get(key), dict):
                    blob = blob[key]
                    break
        if not isinstance(blob, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(blob)}")
        cleaned = {k[len("module."):] if k.startswith("module.") else k: v
                   for k, v in blob.items()}
        missing, unexpected = policy.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} has {len(unexpected)} unexpected keys "
                f"(wrong ckpt family?); e.g. {sorted(unexpected)[:3]}")
        if missing:
            print(f"[TurboVLA] WARNING: {len(missing)} missing keys "
                  f"(e.g. {sorted(missing)[:3]}).")
        policy.to(self.device)
        policy.eval()
        if self.precision == "bf16" and self.device.type == "cuda":
            policy = policy.to(dtype=torch.bfloat16)
        return policy

    # ---- observation handling ----

    def _encode_obs(self, observation: dict) -> dict:
        if "images" in observation and "state" in observation:
            primary = _ensure_hwc_uint8(observation["images"]["cam_high"])
            wrist_key = next(
                (k for k in ("cam_left_wrist", "cam_right_wrist", "cam_wrist")
                 if k in observation["images"]), None)
            wrist = (_ensure_hwc_uint8(observation["images"][wrist_key])
                     if wrist_key else None)
            state = np.asarray(observation["state"], dtype=np.float32)
        else:
            primary = _ensure_hwc_uint8(
                _extract_image(observation, _PRIMARY_CANDIDATES))
            try:
                wrist = _ensure_hwc_uint8(
                    _extract_image(observation, _WRIST_CANDIDATES))
            except KeyError:
                wrist = None
            state = pack_robot_state(
                observation, self.action_type, self.robot_action_dim_info,
                source_type="obs").astype(np.float32)
        if wrist is None:
            if not self._warned_single_view:
                print("[TurboVLA] WARNING: no wrist camera; duplicating primary "
                      "view. Prefer a 2-camera env_cfg.")
                self._warned_single_view = True
            wrist = primary
        prompt = _resolve_prompt(observation, self.default_prompt)
        return {"primary": primary, "wrist": wrist, "state": state, "prompt": prompt}

    def update_obs(self, obs):
        self._obs = self._encode_obs(obs)

    def update_obs_batch(self, obs_list):
        # Single-policy server: keep the latest; batch fan-out is not supported.
        self._obs = self._encode_obs(obs_list[-1])

    # ---- inference ----

    def _preprocess_view(self, hwc: np.ndarray):
        from PIL import Image  # Pillow ships with the policy env

        torch = self.torch
        img = Image.fromarray(hwc).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.from_numpy(self.img_mean).view(3, 1, 1).to(ten.dtype)
        std = torch.from_numpy(self.img_std).view(3, 1, 1).to(ten.dtype)
        return (ten - mean) / std

    @staticmethod
    def _gripper_from_norm(value: float) -> float:
        return 1.0 if float(value) >= 0.0 else -1.0

    def _denormalize_row(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.action_dim:
            raise ValueError(
                f"Model action dim {row.shape[0]}, expected {self.action_dim}.")
        arm = 0.5 * (row[:6] + 1.0) * (self.action_max[:6] - self.action_min[:6]) \
            + self.action_min[:6]
        gripper = np.asarray([self._gripper_from_norm(row[6])], dtype=np.float32)
        return np.concatenate([arm, gripper], axis=0).astype(np.float32)

    def _rows_to_actions(self, rows: np.ndarray) -> list[dict]:
        """Map [H, A] model rows to per-step env action dicts."""
        out: list[dict] = []
        for row in rows:
            env_row = self._denormalize_row(row)
            if env_row.shape[0] == self.packed_dim:
                packed = env_row
            elif (self.num_arms == 2
                    and env_row.shape[0]
                    == self.robot_action_dim_info["arm_dim"][0]
                    + self.robot_action_dim_info["ee_dim"][0]
                    and self.dual_arm_mode == "first_arm"):
                pad = np.zeros(self.packed_dim - env_row.shape[0], dtype=np.float32)
                packed = np.concatenate([env_row, pad], axis=0)
            else:
                raise ValueError(
                    f"Model action dim {env_row.shape[0]} != env packed dim "
                    f"{self.packed_dim} (dual_arm_mode={self.dual_arm_mode!r}). "
                    "Use a RoboDojo-finetuned ckpt, or set dual_arm_mode: "
                    "first_arm for single-arm-ckpt smoke tests. See README.")
            action = unpack_robot_state(
                packed, self.action_type, self.robot_action_dim_info,
                source_type="obs")
            assert isinstance(action, dict)
            out.append({k: np.asarray(v, dtype=np.float32)
                        for k, v in action.items()})
        return out

    def get_action(self):
        if self._obs is None:
            raise RuntimeError("get_action() called before update_obs().")
        torch = self.torch
        obs = self._obs
        if self._chunk:
            return self._chunk
        views = torch.stack(
            [self._preprocess_view(obs["primary"]),
             self._preprocess_view(obs["wrist"])],
            dim=0).unsqueeze(0).to(self.device)
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"Packed state dim {state.shape[0]} != ckpt state_dim "
                f"{self.state_dim}. This ckpt was trained with a different "
                "proprio convention — fine-tune on RoboDojo data (see README).")
        norm = (state - self.proprio_mean) / (self.proprio_std + 1e-6)
        state_t = torch.from_numpy(norm).float().unsqueeze(0).to(self.device)
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.precision == "bf16" and self.device.type == "cuda"
                    else torch.no_grad())
        with torch.no_grad():
            with autocast:
                chunk = self.policy([obs["prompt"]], {"dinov3": views}, state_t)
        rows = np.nan_to_num(
            chunk.float().cpu().numpy()[0], nan=0.0, posinf=1.0, neginf=-1.0)
        rows = rows[:self.num_open_loop_steps]
        self._chunk = self._rows_to_actions(rows)
        return self._chunk

    def get_action_batch(self, env_idx_list=None):
        actions = self.get_action()
        n = len(env_idx_list) if env_idx_list is not None else 1
        return [actions for _ in range(n)]

    def reset(self):
        self._obs = None
        self._chunk = None
