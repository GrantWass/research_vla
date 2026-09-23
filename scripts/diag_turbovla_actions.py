"""Score TurboVLA's predicted actions against ground truth, without the sim.

Loads the adapter exactly as deployed (the same deploy.yml an eval uses) and
compares predicted action chunks to the dataset's actions on episodes of the
task it was fine-tuned on. Answers the question a 0/20 sim result cannot:

    MAE(real cameras) << MAE(hold still)  -> the weights learned the task, so a
        sim failure points at the closed-loop/deployment path.
    MAE(real cameras) >= MAE(hold still)  -> the model did not learn the task
        (undertrained or bad recipe); the sim result is honest.
    MAE(real) ~= MAE(blacked out)         -> it is ignoring the cameras, which
        is a wiring bug (see scripts/diag_openvla_obs.py for the same check
        catching a quantization bug that blinded OpenVLA's vision backbone).

Usage (GPU box, turbovla-robodojo env, deploy.yml already pointed at the ckpt
via scripts/install_turbovla_deploy.sh):

    python scripts/diag_turbovla_actions.py [N_EPISODES]
"""

import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

ROOT = os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
D = os.environ.get(
    "DIAG_DATASET", f"{ROOT}/data/robodojo_tasks_joint/stack_bowls"
)
POLICY = f"{ROOT}/RoboDojo/XPolicyLab/policy/TurboVLA"
TASK = os.path.basename(D.rstrip("/"))  # dataset dir is named after the task
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5

e = pq.read_table(f"{D}/meta/episodes/chunk-000/file-000.parquet").to_pydict()
with open(f"{D}/meta/info.json") as _f:
    info = json.load(_f)

sys.path.insert(0, POLICY)
import model as adapter

with open(f"{POLICY}/deploy.yml") as _f:
    cfg = yaml.safe_load(_f)
cfg.update({
    "env_cfg_type": "arx_x5",
    "bench_name": "RoboDojo",
    "task_name": TASK,
    "action_type": "joint",
    "action_dim": 14,
    "seed": 0,
    "ckpt_name": "turbovla_ft_2000",
})
CHUNK = int(cfg.get("chunk_size", 50))
m = adapter.Model(cfg)

CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def frames(ep, t_offset):
    out = {}
    for key in CAMS:
        v = info["video_path"].format(
            video_key=f"observation.images.{key}",
            chunk_index=e[f"videos/observation.images.{key}/chunk_index"][ep],
            file_index=e[f"videos/observation.images.{key}/file_index"][ep],
        )
        ts = e[f"videos/observation.images.{key}/from_timestamp"][ep] + t_offset
        png = f"/tmp/tvd_{key}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", str(ts), "-i", f"{D}/{v}",
             "-frames:v", "1", png], check=True)
        out[key] = np.asarray(Image.open(png).convert("RGB"))
    return out


def gt_window(ep, start):
    t = pq.read_table(
        f"{D}/" + info["data_path"].format(
            chunk_index=e["data/chunk_index"][ep],
            file_index=e["data/file_index"][ep]),
        memory_map=True)
    idx = np.array(t.column("index").to_pylist())
    lo = e["dataset_from_index"][ep] + start
    sel = np.where((idx >= lo) & (idx < lo + CHUNK + 1))[0]
    if len(sel) < CHUNK:
        return None, None
    state = np.array(t.column("observation.state")[int(sel[0])].as_py(), dtype=np.float32)
    gt = np.array([t.column("action")[int(j)].as_py() for j in sel[:CHUNK]], dtype=np.float32)
    return state, gt


mae, hold, blind = [], [], []
for ep in range(N_EP):
    for start in (0, 60, 120):
        state, gt = gt_window(ep, start)
        if gt is None:
            continue
        im = frames(ep, start / info["fps"])
        instr = e["tasks"][ep][0]
        for label, images in [("real", im), ("black", {k: np.zeros_like(v) for k, v in im.items()})]:
            m.reset() if hasattr(m, "reset") else None
            m.update_obs({"images": images, "state": state, "instruction": instr})
            rows = m.get_action()
            # get_action returns per-step dicts; repack to RoboDojo order
            # [arm_0(6), ee_0(1), arm_1(6), ee_1(1)] to match the dataset action.
            pred = np.asarray(
                [
                    np.concatenate(
                        [
                            np.asarray(r[k], dtype=np.float32).reshape(-1)
                            for k in (
                                "left_arm_joint_state",
                                "left_ee_joint_state",
                                "right_arm_joint_state",
                                "right_ee_joint_state",
                            )
                            if k in r
                        ]
                    )
                    for r in rows
                ],
                dtype=np.float32,
            )[: len(gt)]
            err = float(np.abs(pred[: len(gt)] - gt[: len(pred)]).mean())
            (mae if label == "real" else blind).append(err)
        hold.append(float(np.abs(np.tile(state, (len(gt), 1)) - gt).mean()))

print(f"\nwindows={len(mae)} chunk={CHUNK}")
print(f"  MAE, real cameras   : {np.mean(mae):.4f}")
print(f"  MAE, cameras blacked: {np.mean(blind):.4f}")
print(f"  MAE, hold still     : {np.mean(hold):.4f}")
print("\nhold-still is the do-nothing baseline; real << hold means the model learned motion,")
print("real ~= blacked means it is ignoring the cameras.")
