# GRA Progress Report — VLA Policies on RoboDojo

## Status (as of 2026-09-15)
Thin reproducibility layer over upstream OpenVLA + TurboVLA + RoboDojo.
Committed: `fb6aa19` — registry, TurboVLA adapter, one-flag eval entry,
training template, Windows policy server. Suite: **54 tests pass, 1 skipped**
(`make test`, Mac-safe, system Python).

## Architecture
```
            +----------------+      Tailscale       +------------------+
            |  Linux GPU box |  <-----------------> |  Windows GPU PC  |
            |  RoboDojo sim  |   policy-host:port   |  TurboVLA brain  |
            |  (Isaac Sim)   |      :5999           |  (RTX 4070 Ti S) |
            +-------+--------+                      +--------+---------+
                    ^                                        ^
                    | run_eval.sh --policy <name>            | serve_turbovla_windows.ps1
            +-------+----------------------------------------+---------+
            |  this repo: policies/*.conf + adapters/ + templates/    |
            +----------------------------------------------------------+
OpenVLA-OFT (7B, upstream adapter) vs TurboVLA (0.2B, this-repo adapter).
```

## Methods
RoboDojo owns the benchmark, the VLA repos own the models, and this repo
owns the glue: a model registry (`policies/*.conf`), an XPolicyLab adapter
for TurboVLA (`adapters/turbovla_robodojo/`), and a single eval entry
(`scripts/run_eval.sh`) that forwards to `RoboDojo/scripts/robodojo.sh`.
Swapping models is one flag; `--dry-run` resolves the full command
Mac-safe without launching sim. TurboVLA training reuses RoboDojo's
`lerobot_v3.0_ee` data as-is (14-D `dual_x5` state/action, 3 views —
same schema the upstream recipe expects), so no data rewrite.

## Evidence (all reproduced on Mac, no GPU)
- `make test` → 54 pass, 1 skipped
- `make policies` → demo / openvla / turbovla listed
- Dry-run resolves all 3 policies × `stack_bowls`, `pick_place_milk`
  (e.g. turbovla → `XPolicyLab/policy/TurboVLA`, ckpt
  `turbovla_robodojo_arx_x5_55k`, env `turbovla-robodojo`)
- `make inventory` → 54 runnable tasks; `make doctor` → inventory PASS,
  sim-asset checks FAIL as expected on Mac (assets are git-ignored)
- Fail-fast boundaries verified: `train.sh` names missing `BERT_MODEL_PATH`;
  `compute_stats.py` discovers tasks then requires GPU-box `lerobot` dep;
  `model.py`/`data_config.py` require GPU-box `XPolicyLab`/`starVLA` imports

## Results (pending GPU — table skeleton for next period)
| Task | OpenVLA-OFT | TurboVLA (finetuned) | Notes |
|---|---|---|---|
| stack_bowls | pending ckpt | pending train | smoke first |
| pick_place_milk | pending ckpt | pending train | — |
| full 54-task benchmark | — | — | `robodojo.sh summarize` |

## Blockers / Next
1. grant-pc offline (~12h, Tailscale) — wake + `tailscale up`, then server smoke
2. DINOv3 weights — license-gated download to `C:\Users\grant\vla\models\`
   (BERT + TurboVLA release ckpts already cached on PC)
3. Train `turbovla_robodojo_arx_x5_55k` (55k steps, 4×4090 recipe; single-GPU
   override documented) → `compute_stats.py` → point adapter at EMA ckpt
4. Run smoke → benchmark → leaderboard table
