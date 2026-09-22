"""Is OpenVLA actually using its cameras? Offline check against real demos.

Scores predicted vs ground-truth actions on RoboDojo demonstration frames under
observation variants (wrists swapped, wrists replaced by the head view, cameras
blacked out). It separates "this checkpoint is weak on the task" from "our
observation wiring is broken": if blacking out every camera barely moves the
predicted actions, the vision path is dead.

    conda activate openvla_oft
    python scripts/diag_openvla_obs.py [episodes] [4bit|8bit|bf16]

Needs the per-task joint dataset (see GPU_BOX_SETUP.md §7):
    python templates/turbovla_finetune/make_task_dataset.py \
      --source RoboDojo/.cache/robodojo_assets_repo/data/RoboDojo_lerobot_v30_video \
      --out data/robodojo_tasks_joint --task stack_bowls

Found this way (2026-09-22): 8-bit loading silently corrupted the FiLM vision
backbone (max action change 0.02 with all cameras black, vs 0.33 once the
backbone is excluded from quantization) — fixed in
patches/xpolicylab_openvla_oft_lowvram.patch.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

ROOT = os.path.expanduser("~/code/research_vla")
D = f"{ROOT}/data/robodojo_tasks_joint/stack_bowls"
POLICY = f"{ROOT}/RoboDojo/XPolicyLab/policy/OpenVLA_OFT"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
QUANT = sys.argv[2] if len(sys.argv) > 2 else "8bit"
CHUNK = 25

e = pq.read_table(f"{D}/meta/episodes/chunk-000/file-000.parquet").to_pydict()
with open(f"{D}/meta/info.json") as _f:
    info = json.load(_f)

sys.path.insert(0, f"{ROOT}/RoboDojo")
sys.path.insert(0, f"{POLICY}/openvla_oft")
from XPolicyLab.policy.OpenVLA_OFT import model as adapter

with open(f"{POLICY}/deploy.yml") as _f:
    cfg = yaml.safe_load(_f)
cfg.update(
    {
        "ckpt_name": "RoboDojo-sim-arx_x5-joint-1",
        "action_type": "joint",
        "env_cfg_type": "arx_x5",
        "bench_name": "RoboDojo",
        "task_name": "stack_bowls",
        "action_dim": 14,
        "seed": 0,
        "policy_name": "OpenVLA_OFT",
        "load_in_4bit": QUANT == "4bit",
        "load_in_8bit": QUANT == "8bit",  # bf16 needs a 24 GB+ GPU
    }
)
m = adapter.Model(cfg)


def frames(ep, t_offset=0.0):
    out = {}
    for slot, key in [
        ("cam_high", "cam_high"),
        ("cam_left_wrist", "cam_left_wrist"),
        ("cam_right_wrist", "cam_right_wrist"),
    ]:
        v = info["video_path"].format(
            video_key=f"observation.images.{key}",
            chunk_index=e[f"videos/observation.images.{key}/chunk_index"][ep],
            file_index=e[f"videos/observation.images.{key}/file_index"][ep],
        )
        ts = e[f"videos/observation.images.{key}/from_timestamp"][ep] + t_offset
        png = f"/tmp/ovd_{key}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(ts),
                "-i",
                f"{D}/{v}",
                "-frames:v",
                "1",
                png,
            ],
            check=True,
        )
        out[slot] = np.asarray(Image.open(png).convert("RGB"))
    return out


def gt_window(ep, start):
    t = pq.read_table(
        f"{D}/"
        + info["data_path"].format(
            chunk_index=e["data/chunk_index"][ep], file_index=e["data/file_index"][ep]
        ),
        memory_map=True,
    )
    idx = np.array(t.column("index").to_pylist())
    lo = e["dataset_from_index"][ep] + start
    sel = np.where((idx >= lo) & (idx < lo + CHUNK + 1))[0]
    state = np.array(
        t.column("observation.state")[int(sel[0])].as_py(), dtype=np.float32
    )
    gt = np.array(
        [t.column("action")[int(j)].as_py() for j in sel[:CHUNK]], dtype=np.float32
    )
    return state, gt


VARIANTS = {
    "as-wired": lambda im: im,
    "wrists-swapped": lambda im: {
        "cam_high": im["cam_high"],
        "cam_left_wrist": im["cam_right_wrist"],
        "cam_right_wrist": im["cam_left_wrist"],
    },
    "head-only (wrists=head)": lambda im: {k: im["cam_high"] for k in im},
    "head blacked out": lambda im: {**im, "cam_high": np.zeros_like(im["cam_high"])},
    "all black": lambda im: {k: np.zeros_like(v) for k, v in im.items()},
}

rows = {k: [] for k in VARIANTS}
preds = {k: [] for k in VARIANTS}
hold = []
for ep in range(N_EP):
    for start in (0, 60):
        state, gt = gt_window(ep, start)
        if len(gt) < CHUNK:
            continue
        im = frames(ep, t_offset=start / info["fps"])
        hold.append(np.abs(np.tile(state, (len(gt), 1)) - gt).mean())
        for name, fn in VARIANTS.items():
            obs = {"images": fn(im), "state": state, "instruction": e["tasks"][ep][0]}
            enc = adapter.encode_obs(
                obs, "joint", {"arm_dim": [6, 6], "ee_dim": [1, 1]}, e["tasks"][ep][0]
            )
            pred = np.asarray(m.infer(enc), dtype=np.float32)[: len(gt)]
            rows[name].append(np.abs(pred - gt).mean())
            preds[name].append(pred)

print(f"\nquant={QUANT} episodes={N_EP} windows={len(hold)} chunk={CHUNK}")
print(f"{'variant':26s} {'MAE vs GT':>10s}")
for name, vals in rows.items():
    print(f"{name:26s} {np.mean(vals):10.4f}")
print(f"{'baseline: hold still':26s} {np.mean(hold):10.4f}")
base = np.concatenate(preds["as-wired"])
print("\nmax |pred - pred(as-wired)| per variant (0 = observation ignored):")
for name in VARIANTS:
    if name == "as-wired":
        continue
    other = np.concatenate(preds[name])
    print(
        f"  {name:26s} max={np.abs(other - base).max():.5f}  mean={np.abs(other - base).mean():.5f}"
    )
