# Fine-tune TurboVLA on RoboDojo (`templates/turbovla_finetune/`)

Trains a RoboDojo-native TurboVLA checkpoint (dual_x5, 14-D state/action,
3 views) that `scripts/run_eval.sh --policy turbovla` can evaluate. Run the
training steps on the GPU box (Ubuntu 22.04 + CUDA); everything else is
Mac-safe.

## Why this works without a data rewrite

RoboDojo's downloadable `lerobot_v3.0_ee` data already uses the exact feature
schema TurboVLA's LeRobot training path expects:

| TurboVLA expects | RoboDojo `lerobot_v3.0_ee` provides |
|---|---|
| `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` | same (640×480, resized to 224 by the loader) |
| `observation.state` 14-D `[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]` | same (`dual_x5`) |
| `action` 14-D absolute joint targets | same |
| `task_index` → language string | same (standard LeRobot task metadata) |

The only glue is a **registry overlay**: `data_registry/data_config.py`
declares the `robodojo_arx_x5` mix + robot type, and the StarVLA registry
auto-discovers it from `turbovla/experiments/robodojo/`. Use the `*_ee`
format — plain `lerobot_v3.0` is joint-only (no gripper values).

## Recipe (GPU box)

```bash
# 0. Checkouts + overlay
bash setup.sh
bash scripts/install_turbovla_training.sh
bash scripts/install_adapter.sh turbovla

# 1. Data (120 GB, ee variant!) + training env (upstream recipe)
cd RoboDojo && bash scripts/RoboDojo/download_data.sh huggingface lerobot_v3.0_ee
cd ../turbovla && pip install -e ".[robotwin]" && pip install flash-attn==2.7.4.post1 --no-build-isolation
# model assets: DINOv3 (ViT-L recommended), bert-base-uncased, groundingdino_swint_ogc.pth

# 2. Train (4× RTX 4090 paper recipe: global batch 192, 55k steps)
cd /path/to/turbovla
ROBODOJO_DATA_ROOT=/data/RoboDojo_ee_lerobot_v30_video \
BERT_MODEL_PATH=/models/bert-base-uncased DINOV3_MODEL_PATH=/models/dinov3-vitl \
TURBOVLA_INIT_CKPT=/models/groundingdino_swint_ogc.pth \
RUN_ID=turbovla_robodojo_arx_x5_55k \
bash ../templates/turbovla_finetune/train.sh
# subset of tasks: ROBODOJO_TASKS=stack_bowls,push_T ... (default: all datasets found)
# smaller GPU:     PER_DEVICE_BATCH_SIZE=12 GRAD_ACCUM=4 NUM_PROCESSES=2 ...

# 3. Inference stats (same data root; writes the JSON the adapter consumes)
python ../templates/turbovla_finetune/compute_stats.py \
  --data-root /data/RoboDojo_ee_lerobot_v30_video \
  --out results/Checkpoints/turbovla_robodojo_arx_x5_55k/robodojo_stats.json

# 4. Deploy: copy deploy.robodojo.yml over the adapter config, set 2 paths
cp ../templates/turbovla_finetune/deploy.robodojo.yml \
   ../RoboDojo/XPolicyLab/policy/TurboVLA/deploy.yml
# edit: checkpoint_path -> .../checkpoints/steps_55000_ema_pytorch_model.pt
#       stats_path      -> .../robodojo_stats.json
# (or export TURBOVLA_CKPT / TURBOVLA_STATS instead of editing)

# 5. Evaluate — the OpenVLA comparison is now one flag
bash ../scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast
bash ../scripts/run_eval.sh --policy openvla  --task stack_bowls --mode smoke --fail-fast
```

## Files

| File | Role |
|---|---|
| `train.sh` | Env-var training launcher (discovers tasks, checks assets, `accelerate launch` of the upstream EMA recipe). GPU box only. |
| `data_registry/data_config.py` | `robodojo_arx_x5` mix + robot type (same modality keys/transforms as the paper's `robotwin50`). Installed into `turbovla/experiments/robodojo/`. |
| `configs/robodojo.yaml` | Training config: 14-D/14-D, horizon 50, 3 views @224, L1 + EMA + DeepSpeed ZeRO-2, 55k steps. |
| `compute_stats.py` | Proprio mean/std + action min/max over training frames → stats JSON for the adapter. |
| `deploy.robodojo.yml` | Adapter config matching this recipe (224/3 views/horizon 50/14-D). |

## Notes

- `train.sh` refuses to overwrite an existing `RUN_ID` output dir — bump the id per run.
- Checkpoints land as `<run>/checkpoints/steps_<N>_pytorch_model.pt` plus EMA
  siblings `steps_<N>_ema_pytorch_model.pt`; point the adapter at an **EMA** ckpt (same convention as the released RoboTwin ckpt).
- Gripper semantics: training uses `binary` normalization with threshold 0.49
  (paper recipe); the adapter's sign rule matches.
- Feature check first: if `train.sh` reports no datasets, your `ROBODOJO_DATA_ROOT`
  is one level off — it must directly contain per-task dirs with `meta/info.json`.
