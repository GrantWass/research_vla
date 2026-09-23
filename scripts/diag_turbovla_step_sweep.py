"""MAE vs training step: did the fine-tune move the model at all?

Scores the same windows with the same code path at each saved checkpoint, so a
0/N sim result can be split into "the fine-tune did nothing" and "it is learning
but did not get far enough":

    MAE falling with step  -> learning; the run is just short.
    MAE flat               -> the fine-tune is not doing anything.
    EMA better than raw    -> still in the unstable early phase, where the
                              instantaneous weights bounce and the moving
                              average carries the progress.

Compare against the printed hold-still baseline, not against zero: a model whose
MAE sits above hold-still is worse than not moving at all.

Deliberately excludes the released RoboTwin checkpoint the run initialised from.
It is 2-view, arms-first and ships 8-D proprio stats for a different robot, so
scoring it on ARX X5 data would measure the embodiment gap rather than our
training, and read as a misleading "before" number.

Usage (GPU box, turbovla-robodojo env):

    python scripts/diag_turbovla_step_sweep.py [N_EPISODES]

Env overrides: REPO_ROOT, DIAG_DATASET, DIAG_RUN (the run dir under
<turbovla>/results/Checkpoints).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

ROOT = os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
D = os.environ.get("DIAG_DATASET", f"{ROOT}/data/robodojo_tasks_joint/stack_bowls")
POLICY = f"{ROOT}/RoboDojo/XPolicyLab/policy/TurboVLA"
TASK = os.path.basename(D.rstrip("/"))  # dataset dir is named after the task
RUN = os.environ.get(
    "DIAG_RUN",
    f"{ROOT}/turbovla/results/Checkpoints/turbovla_robodojo_stack_bowls_ft",
)
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# The released RoboTwin ckpt is excluded: it is 2-view, arms-first, and ships 8-D
# proprio stats for a DIFFERENT robot, so scoring it on ARX X5 data measures the
# embodiment gap, not our training. 1000 vs 2000 is the like-for-like comparison.
CKPTS = [
    ("ours, step 1000 EMA", f"{RUN}/checkpoints/steps_1000_ema_pytorch_model.pt", "packed"),
    ("ours, step 1000 raw", f"{RUN}/checkpoints/steps_1000_pytorch_model.pt", "packed"),
    ("ours, step 2000 EMA", f"{RUN}/checkpoints/steps_2000_ema_pytorch_model.pt", "packed"),
    ("ours, step 2000 raw", f"{RUN}/checkpoints/steps_2000_pytorch_model.pt", "packed"),
]

with open(f"{D}/meta/info.json") as f:
    info = json.load(f)
e = pq.read_table(f"{D}/meta/episodes/chunk-000/file-000.parquet").to_pydict()
sys.path.insert(0, POLICY)
import model as adapter

CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
KEYS = ("left_arm_joint_state", "left_ee_joint_state",
        "right_arm_joint_state", "right_ee_joint_state")


_FRAME_CACHE: dict = {}


def frames(ep, t_offset):
    # Every checkpoint scores the identical windows, so decode each one once.
    key = (ep, round(t_offset, 6))
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    out = {}
    for cam in CAMS:
        v = info["video_path"].format(
            video_key=f"observation.images.{cam}",
            chunk_index=e[f"videos/observation.images.{cam}/chunk_index"][ep],
            file_index=e[f"videos/observation.images.{cam}/file_index"][ep])
        ts = e[f"videos/observation.images.{cam}/from_timestamp"][ep] + t_offset
        png = f"/tmp/sw_{cam}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ts), "-i",
                        f"{D}/{v}", "-frames:v", "1", png], check=True)
        out[cam] = np.asarray(Image.open(png).convert("RGB"))
    _FRAME_CACHE[key] = out
    return out


def gt_window(ep, start, chunk):
    t = pq.read_table(f"{D}/" + info["data_path"].format(
        chunk_index=e["data/chunk_index"][ep],
        file_index=e["data/file_index"][ep]), memory_map=True)
    idx = np.array(t.column("index").to_pylist())
    lo = e["dataset_from_index"][ep] + start
    sel = np.where((idx >= lo) & (idx < lo + chunk + 1))[0]
    if len(sel) < chunk:
        return None, None
    state = np.array(t.column("observation.state")[int(sel[0])].as_py(), dtype=np.float32)
    gt = np.array([t.column("action")[int(j)].as_py() for j in sel[:chunk]], dtype=np.float32)
    return state, gt


with open(f"{POLICY}/deploy.yml") as f:
    base = yaml.safe_load(f)

results = {}
hold = []
for label, path, layout in CKPTS:
    cfg = dict(base)
    cfg.update({"env_cfg_type": "arx_x5", "bench_name": "RoboDojo",
                "task_name": TASK, "action_type": "joint",
                "action_dim": 14, "seed": 0, "ckpt_name": label,
                "checkpoint_path": path, "action_layout": layout})
    m = adapter.Model(cfg)
    chunk = int(cfg.get("chunk_size", 50))
    errs = []
    first = not hold
    for ep in range(N_EP):
        for start in (0, 60, 120):
            state, gt = gt_window(ep, start, chunk)
            if gt is None:
                continue
            im = frames(ep, start / info["fps"])
            if first:
                hold.append(float(np.abs(np.tile(state, (len(gt), 1)) - gt).mean()))
            m.reset()
            m.update_obs({"images": im, "state": state, "instruction": e["tasks"][ep][0]})
            rows = m.get_action()
            pred = np.asarray([np.concatenate(
                [np.asarray(r[k], dtype=np.float32).reshape(-1) for k in KEYS if k in r])
                for r in rows], dtype=np.float32)[: len(gt)]
            errs.append(float(np.abs(pred - gt[: len(pred)]).mean()))
    results[label] = float(np.mean(errs))
    del m
    torch.cuda.empty_cache()  # 4 checkpoints back to back on one 16 GB card
    print(f"  {label:22s} MAE {results[label]:.4f}", flush=True)

print(f"\nwindows={len(hold)}")
print(f"{'checkpoint':24s} {'MAE':>8s}")
for label, _, _ in CKPTS:
    print(f"{label:24s} {results[label]:8.4f}")
print(f"{'hold still (baseline)':24s} {np.mean(hold):8.4f}")
