# TurboVLA adapter for RoboDojo (`adapters/turbovla_robodojo/`)

XPolicyLab policy adapter that serves **TurboVLA**
([H-EmbodVis/TurboVLA](https://github.com/H-EmbodVis/TurboVLA), 0.2B params,
direct V+L→A, 32 Hz / <1 GB VRAM on an RTX 4090) inside RoboDojo eval.
Owned by this repo; installed (copied) into the upstream checkout — the
upstream `RoboDojo/` tree is never edited by hand.

## Install (once per machine)

```bash
bash setup.sh                              # pins openvla/, turbovla/, RoboDojo/
bash scripts/install_adapter.sh turbovla   # copy adapter -> RoboDojo/XPolicyLab/policy/TurboVLA/

# GPU box (Ubuntu 22.04 + CUDA):
cd RoboDojo/XPolicyLab/policy/TurboVLA
bash install.sh && conda activate turbovla-robodojo
```

## Run (model swap is one flag)

```bash
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run   # Mac-safe
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast  # GPU box
bash scripts/run_eval.sh --policy openvla  --task stack_bowls --dry-run   # back to OpenVLA
```

## Status: wiring vs. weights

- **Wiring is done**: `model.py` implements the `ModelTemplate` contract
  (`update_obs` / `get_action` / `reset`), serves TurboVLA action chunks
  open-loop (`num_open_loop_steps`), and packs/unpacks RoboDojo robot state.
- **Weights come from `templates/turbovla_finetune/`**: train on RoboDojo's
  `lerobot_v3.0_ee` data (same LeRobot schema the recipe expects — no data
  rewrite), compute stats, copy `deploy.robodojo.yml` over this `deploy.yml`,
  and point `--ckpt` at the run dir. Released LIBERO/RoboTwin ckpts do **not**
  transfer (different dims/proprio) and fail fast here by design.

What happens with a released ckpt today:

| Ckpt | Result |
|---|---|
| LIBERO `.pth` | Fails fast with `Packed state dim ... != ckpt state_dim` — correct: the LIBERO proprio convention does not transfer. |
| `dual_arm_mode: first_arm` | Drives arm 0 with a 7-D ckpt, holds arm 1 at zero. Smoke-test only, never benchmark with this. |
| RoboDojo-finetuned run dir | Full eval path. Collect RoboDojo demos → train TurboVLA → point `--ckpt` at the run dir + `stats_path` at its stats JSON. |

The fail-fast errors name the exact knob to change, so a dimension mismatch
can never silently corrupt metrics.

## Files

| File | Role |
|---|---|
| `model.py` | `ModelTemplate` impl: ckpt load (EMA-aware), DINOv3 resize+normalize, proprio normalize, chunk infer, min/max arm denorm + gripper sign rule (mirrors upstream `turbovla/evaluation/policy.py`), `unpack_robot_state` to env dicts. |
| `deploy.yml` | All knobs (`dinov3_path`, `bert_path`, `checkpoint_path`, `stats_path`, `num_views`, `chunk_size`, `state_dim`, `action_dim`, `dual_arm_mode`, ...). Null = env var / workspace default. |
| `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh` | Standard XPolicyLab launchers (same protocol as `SmolVLA`/`OpenVLA_OFT`). |
| `install.sh` | GPU-box env: `turbovla-robodojo` conda env, torch cu121, editable installs, `hf download H-EmbodVis/TurboVLA`. |

## Closest zero-shot candidate (unverified)

The RoboTwin `.safetensors` ckpt (3 views, 14-D joint actions) is dimensionally
closest to `dual_x5` joint mode (3 RoboDojo cameras, packed dim 14). It still
needs its `state_dim`/stats to line up — verify `deploy.yml` against the
ckpt config before trusting any rollout.
