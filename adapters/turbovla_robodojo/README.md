# TurboVLA adapter for RoboDojo (`adapters/turbovla_robodojo/`)

XPolicyLab policy adapter that serves **TurboVLA**
([H-EmbodVis/TurboVLA](https://github.com/H-EmbodVis/TurboVLA), 0.2B params,
direct V+L→A, 32 Hz / <1 GB VRAM on an RTX 4090) inside RoboDojo eval.
Owned by this repo; installed (copied) into the upstream checkout — the
upstream `RoboDojo/` tree is never edited by hand.

## Install (once per machine)

```bash
bash setup.sh                                          # pins openvla/, turbovla/, RoboDojo/
bash scripts/setup_policy.sh turbovla --robotwin-smoke # GPU box: adapter + env + ckpts + DINOv3/BERT
```

`setup_policy.sh` runs `install_adapter.sh` and this adapter's `install.sh`,
downloads DINOv3 ViT-L (gated: accept the license on Hugging Face and save a
read token first, see `GPU_BOX_SETUP.md` §4) and BERT. With `--robotwin-smoke`
it also installs `deploy.robotwin_smoke.yml` as the adapter's `deploy.yml`.

## Run (model swap is one flag)

```bash
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run   # Mac-safe
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast  # GPU box
bash scripts/run_eval.sh --policy openvla  --task stack_bowls --dry-run   # back to OpenVLA
```

## Status: wiring verified in Isaac Sim

- **Wiring is verified end to end.** With the released RoboTwin ckpt
  (`deploy.robotwin_smoke.yml`), a `stack_bowls` smoke run passes in Isaac Sim
  and both arms and grippers move, so observations, all 3 cameras, the 14-D
  dual-arm state and actions, and the server/client loop all work. The
  motions aren't task-meaningful because that ckpt was trained on a different robot.
- **Real weights come from `templates/turbovla_finetune/`**: train on RoboDojo's
  `lerobot_v3.0_ee` data (same LeRobot schema, no data rewrite), compute
  stats, copy `deploy.robodojo.yml` over this `deploy.yml`, and point `--ckpt`
  at the run dir.

Checkpoint handling:

| Ckpt | Result |
|---|---|
| RoboDojo-finetuned run dir | Full eval path (`action_layout: packed`). |
| RoboTwin `.safetensors` (released) | Loads: legacy module names are remapped, `learned_patch` position embeddings detected, `action_layout: arms_first` reorders RoboTwin's `[arm_0, arm_1, ee_0, ee_1]` to RoboDojo's `[arm_0, ee_0, arm_1, ee_1]`, and unnormalized grippers (stats `mask`) pass through. Smoke only. |
| LIBERO `.pth` | Fails fast with `Packed state dim ... != ckpt state_dim`, which is correct: the LIBERO proprio convention doesn't transfer. |
| Any 7-D single-arm ckpt + `dual_arm_mode: first_arm` | Drives arm 0, holds arm 1 at zero. Smoke only. |

The fail-fast errors name the exact knob to change, so a dimension mismatch
can never silently corrupt metrics.

## Files

| File | Role |
|---|---|
| `model.py` | `ModelTemplate` impl: ckpt load (EMA-aware, `.pth` or `.safetensors`, legacy key remap), N-view (head/left/right wrist) input, DINOv3 resize+normalize, proprio normalize, chunk infer, min/max arm denorm + gripper sign rule (mirrors upstream `turbovla/evaluation/policy.py`), `unpack_robot_state` to env dicts. |
| `deploy.yml` | All knobs (`dinov3_path`, `bert_path`, `checkpoint_path`, `stats_path`, `num_views`, `chunk_size`, `state_dim`, `action_dim`, `action_layout`, `dual_arm_mode`, ...). Null = env var / workspace default. |
| `deploy.robotwin_smoke.yml` | Sim-wiring smoke config for the released RoboTwin ckpt (`@SHARED@` filled in by `setup_policy.sh --robotwin-smoke`). |
| `deploy.py` | Env-side episode loop (`eval_one_episode[_batch]`), same as upstream `demo_policy`. |
| `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh` | Standard XPolicyLab launchers (same protocol as `SmolVLA`/`OpenVLA_OFT`). |
| `install.sh` | GPU-box env: `turbovla-robodojo` conda env, torch cu121, editable installs, `hf download H-EmbodVis/TurboVLA`. |
