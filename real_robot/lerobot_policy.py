"""Run a LeRobot policy (e.g. a pi0.5 fine-tune from the Hub) inside the runner.

Same call() interface as the websocket client, so runner.run() is unchanged:
reset / update_obs(obs) / get_action -> (T, 7) chunk in policy units.

  python3 -m real_robot.runner --backend piper --config real_robot/configs/piper_1_banana.json \\
      --lerobot richardshkim/pi05_piper_banana_v2_full --instruction "Put banana into basket."

Needs a LeRobot venv with the pi extra (docs/robot.md), not the repo venv.
"""
from __future__ import annotations

import os

import numpy as np

# torch.compile caches its kernels and autotune results here, so only the first
# compiled start on this machine pays for them (the default, /tmp, is wiped on
# reboot). Each start still traces the model again (~tens of seconds).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.expanduser("~/.cache/torchinductor"))


class LerobotPolicy:
    """Wraps a LeRobot pretrained policy + its saved pre/post-processors.

    camera_map: our camera name -> the policy's image key suffix
      (observation.images.<suffix>).
    extra_state: callable returning values appended to our 7-D state, for
      checkpoints trained with more (the banana checkpoint wants the 6-D end
      pose [x, y, z m, roll, pitch, yaw rad] after the joints and gripper).
    base_offset: rad added to joint 1 in the state the policy sees and
      subtracted from its joint-1 actions, for an arm whose base is turned
      relative to the training rig (ours faces the table at J1 ~ -90 deg; the
      banana checkpoint was trained at J1 ~ 0). The end pose is rotated about
      the base z axis to match: (x, y) by base_offset, yaw + base_offset
      (checked against piper_sdk forward kinematics).
    compile_mode: None runs eager; otherwise a torch.compile mode
      ("default", "max-autotune", ...) for the action sampler.
    num_steps: flow-matching denoising steps (None: the checkpoint's, 10).
    """

    def __init__(self, path: str, camera_map: dict[str, str], extra_state=None,
                 device: str = "cuda", robot_type: str = "piper_follower",
                 base_offset: float = 0.0, compile_mode: str | None = None,
                 num_steps: int | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = compile_mode is not None
            if compile_mode is not None:
                cfg.compile_mode = compile_mode
        if num_steps is not None:
            cfg.num_inference_steps = num_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=path)
        self.torch = torch
        self.device = torch.device(device)
        self.camera_map = camera_map
        self.extra_state = extra_state
        self.robot_type = robot_type
        self.base_offset = float(base_offset)
        self.state_dim = cfg.input_features["observation.state"].shape[0]
        missing = {f"observation.images.{v}" for v in camera_map.values()} - set(cfg.input_features)
        if missing:
            raise ValueError(f"policy has no inputs {sorted(missing)}; "
                             f"it wants {sorted(cfg.input_features)}")
        self.obs = None

    def _batch(self, obs: dict) -> dict:
        from lerobot.policies.utils import prepare_observation_for_inference

        state = np.asarray(obs["state"], dtype=np.float32).copy()
        state[0] += self.base_offset
        extra = obs.get("extra_state")  # sent along by PolicyProcess
        if extra is None and self.extra_state is not None:
            extra = self.extra_state()
        if extra is not None:
            extra = np.asarray(extra, dtype=np.float64).copy()
            if self.base_offset:
                extra = rotate_end_pose(extra, self.base_offset)
            state = np.concatenate([state, extra.astype(np.float32)])
        if state.shape != (self.state_dim,):
            raise ValueError(f"policy wants state ({self.state_dim},), got {state.shape}")
        raw = {"observation.state": state}
        for ours, theirs in self.camera_map.items():
            raw[f"observation.images.{theirs}"] = obs["images"][ours]
        batch = prepare_observation_for_inference(raw, self.device, obs["instruction"],
                                                  self.robot_type)
        return self.pre(batch)

    def call(self, func_name: str, **kw):
        if func_name == "reset":
            self.policy.reset()
            return None
        if func_name == "update_obs":
            self.obs = kw["obs"]
            return None
        if func_name == "get_action":
            with self.torch.inference_mode():
                chunk = self.policy.predict_action_chunk(self._batch(self.obs))
                chunk = self.post(chunk)
            out = chunk[0].float().cpu().numpy()  # (T, action_dim)
            out[:, 0] -= self.base_offset
            return out
        raise ValueError(f"unknown call {func_name!r}")

    def close(self) -> None:
        pass


class PolicyProcess:
    """A LerobotPolicy in a child process, with the same call() interface.

    In the robot process, piper_sdk's CAN reader thread (and the camera
    threads) compete with eager PyTorch for the GIL: inference took 682 ms
    there vs 358 ms alone (2026-10-07, RTX 3090). A child process has its own
    GIL. extra_state() is read here, next to the robot, and sent with the obs.
    """

    def __init__(self, extra_state=None, **policy_kwargs):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # CUDA cannot be forked
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, policy_kwargs), daemon=True)
        self.proc.start()
        self.extra_state = extra_state
        self.obs = None
        self._recv()  # wait for the model to load (raises its error if it failed)

    def _recv(self):
        if not self.conn.poll(600):
            raise TimeoutError("policy process did not answer in 600 s")
        ok, value = self.conn.recv()
        if not ok:
            raise RuntimeError(f"policy process: {value}")
        return value

    def call(self, func_name: str, **kw):
        if func_name == "update_obs":
            obs = dict(kw["obs"])
            if self.extra_state is not None:
                obs["extra_state"] = np.asarray(self.extra_state(), dtype=np.float64)
            self.obs = obs
            return None
        if func_name == "get_action":
            self.conn.send(("infer", self.obs))
        else:
            self.conn.send((func_name, None))
        return self._recv()

    def close(self) -> None:
        if self.proc.is_alive():
            try:
                self.conn.send(("stop", None))
            except (BrokenPipeError, OSError):
                pass
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.kill()


def _serve(conn, policy_kwargs) -> None:
    """Child process loop for PolicyProcess."""
    import traceback

    try:
        policy = LerobotPolicy(**policy_kwargs)
        conn.send((True, None))
    except Exception:  # noqa: BLE001  report load errors to the parent
        conn.send((False, traceback.format_exc(limit=3)))
        return
    while True:
        try:
            func, obs = conn.recv()
        except EOFError:
            return
        if func == "stop":
            return
        try:
            if func == "infer":
                policy.call("update_obs", obs=obs)
                conn.send((True, policy.call("get_action")))
            else:
                conn.send((True, policy.call(func)))
        except Exception:  # noqa: BLE001
            conn.send((False, traceback.format_exc(limit=3)))


def rotate_end_pose(pose: np.ndarray, theta: float) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] as if the base were turned by theta about z."""
    c, s = np.cos(theta), np.sin(theta)
    x, y = pose[0], pose[1]
    out = pose.copy()
    out[0], out[1] = c * x - s * y, s * x + c * y
    out[5] = (pose[5] + theta + np.pi) % (2 * np.pi) - np.pi
    return out
