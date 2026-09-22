# Multi-model RoboDojo eval: OpenVLA ↔ TurboVLA ↔ pi0.5 (and beyond)

Swapping the VLA model under RoboDojo is one flag. There are two layers:

1. **`policies/<name>.conf`** — model registry. Declares where the model
   lives, which XPolicyLab adapter serves it, which policy env it runs in
   (conda env, or `uv` for uv-managed adapters like pi0.5),
   and its default ckpt / action type / env cfg.
2. **`scripts/run_eval.sh`** — the only eval entry point you need. It sources
   the selected conf and forwards everything to
   `RoboDojo/scripts/robodojo.sh` (still the only harness that touches sim).

```
bash scripts/run_eval.sh --policy openvla  --task stack_bowls --dry-run
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run
bash scripts/run_eval.sh --policy pi05     --task stack_bowls --dry-run
make dry-run POLICY=pi05 TASK=stack_bowls
```

## Registered models

| `--policy` | What | Adapter | Env | Status |
|---|---|---|---|---|
| `openvla` (default) | OpenVLA-OFT 7B, LLM-centric baseline | Upstream `XPolicyLab/policy/OpenVLA_OFT` | `openvla_oft` (conda) | **Sim-verified** with the official ckpt `RoboDojo-sim-arx_x5-joint-1` (4-bit on 16 GB, see `GPU_BOX_SETUP.md`) |
| `turbovla` | TurboVLA 0.2B, direct V+L→A, 32 Hz / <1 GB VRAM (RTX 4090) | This-repo `adapters/turbovla_robodojo/` → installed to `XPolicyLab/policy/TurboVLA` | `turbovla-robodojo` (conda) | **Sim-verified wiring** with the released RoboTwin ckpt (`--robotwin-smoke`); train via `templates/turbovla_finetune/` for real evals |
| `pi05` | pi0.5 base VLA (Physical Intelligence, open-world generalization) | Upstream `XPolicyLab/policy/Pi_05` (openpi vendored inside) | `uv` (uv-managed, not conda) | **Sim-verified** with the official ckpt `RoboDojo-sim-arx_x5-joint-0` (stacked the bowls in the smoke run) |
| `demo` | Zero-action stub | Upstream `XPolicyLab/policy/demo_policy` | `RoboDojo` (conda) | Wiring smoke test only |

Paper: TurboVLA `arXiv:2607.27205`; code
`https://github.com/H-EmbodVis/TurboVLA` (pinned by `setup.sh` to `b29ab14`
in `turbovla/`); ckpts `H-EmbodVis/TurboVLA` on Hugging Face.

## Setup per model

Full bring-up (driver, simulator, verification) is in `GPU_BOX_SETUP.md`. The
per-policy step is one idempotent command that also carries the fixes the
upstream installers need:

```bash
bash setup.sh                                          # pins openvla/, turbovla/, RoboDojo/
bash scripts/setup_policy.sh openvla                   # env openvla_oft + official ckpt
bash scripts/setup_policy.sh pi05                      # uv env + seed-0 params only (~12 GB)
bash scripts/setup_policy.sh turbovla --robotwin-smoke # adapter + env + DINOv3/BERT (HF token needed)
bash scripts/setup_policy.sh demo
bash scripts/lowvram.sh apply                          # only on 16 GB GPUs
make smoke-all                                         # every policy through Isaac Sim
```

pi0.5 runs in a uv venv (`source RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/activate`),
not conda. Always run evals with the `RoboDojo` conda env active.

## Eval commands (GPU box)

```bash
# single task — same shape for every model, only --policy changes
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast
bash scripts/run_eval.sh --policy openvla  --task stack_bowls --mode smoke --fail-fast
bash scripts/run_eval.sh --policy pi05     --task stack_bowls --mode smoke --fail-fast

# pi0.5 training (GPU box, inside the upstream adapter — data first, then train):
cd RoboDojo/XPolicyLab/policy/Pi_05
bash process_data.sh RoboDojo cotrain arx_x5 joint
bash train.sh RoboDojo cotrain arx_x5 joint 0 0   # ckpt -> checkpoints/RoboDojo-cotrain-arx_x5-joint-0/

# full benchmark + leaderboard table
bash scripts/run_eval.sh --policy turbovla --mode benchmark --gpu-ids 0,1,3
cd RoboDojo && bash scripts/robodojo.sh summarize

# split machines (server binds 0.0.0.0, client dials it)
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode server --bind-host 0.0.0.0 --policy-port 9999
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode client --policy-host <IP> --policy-port 9999
```

### Windows GPU box as policy server (grant-pc)

The Windows box serves the policy end; the sim client runs on the Linux GPU
box. PC layout (all done, no reinstall needed): `C:\Users\grant\vla\`
holds `turbovla\` (pinned `b29ab14`), `RoboDojo\` (pinned `ee67a14` +
XPolicyLab, adapter installed), `models\` (BERT + TurboVLA release ckpts;
DINOv3 needs the license-gated download), and the `turbovla-robodojo`
conda env (torch 2.11 +cu128, CUDA live on the RTX 4070 Ti SUPER).

```powershell
# on grant-pc (PowerShell) — one command, same args as eval.sh:
C:\Users\grant\vla\serve_turbovla.ps1 -Task stack_bowls -Ckpt <RUN_DIR> -Port 5999
# on the Linux box — point the sim client at it over the tailnet:
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode client \
  --policy-host 100.100.161.16 --policy-port 5999
```

Launcher source: `scripts/serve_turbovla_windows.ps1` (copy to
`C:\Users\grant\vla\serve_turbovla.ps1` after edits). PC quirks it handles:
sanitizes the Codex `bin` junction out of PATH (`pip` dies on it with
WinError 448), creates a persistent `python3` shim for Git Bash
(`C:\Users\grant\vla\bin`), defaults `DINOV3_PATH`/`BERT_PATH` (overridable
per run via `$env:`). SSH into the box: `ssh winbox` (see `REMOTE_ACCESS.md`;
note a Mac-side VPN swallows the tailnet route — disconnect it first).

`--mode` is `eval` by default; `--dry-run` (alias) only prints the resolved
`robodojo.sh` call — Mac-safe, no sim. Any unrecognized flags pass through
to `robodojo.sh` verbatim (e.g. `--only`, `--dimension`, `--gpu-ids`).

## Adding the next model

1. `cp policies/demo.conf policies/<name>.conf`, fill in the fields
   (documented in `policies/README.md`). If the adapter already ships with
   XPolicyLab (like `Pi_05`), this step is the whole job — set `REPO_DIR=`
   empty and skip step 2 (`pi05.conf` is the example).
2. If it needs a new adapter: `cp -r adapters/turbovla_robodojo
   adapters/<name>_robodojo`, implement `model.py` against
   `XPolicyLab/model_template.py`, set the conf's `XPOLICYLAB_POLICY_DIR`,
   install with `scripts/install_adapter.sh <name>`.
3. `python3 -m unittest discover -s tests` — the suite checks every conf
   parses, every policy dry-runs, and every owned adapter matches the
   XPolicyLab contract.

## Why this shape

`RoboDojo/` and `openvla/`/`turbovla/` are upstream checkouts (git-ignored,
reproduced by `setup.sh`) — model-specific knowledge owned by this repo lives
in `policies/` + `adapters/` + `scripts/`, so upstream pulls never conflict
with a new model, and a new model never touches another model's files.
