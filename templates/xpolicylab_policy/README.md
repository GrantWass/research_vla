# XPolicyLab policy template (`my_policy`)

Mirrors `XPolicyLab/policy/demo_policy/` (zero-action reference). To add a policy:

```bash
cp -r templates/xpolicylab_policy/my_policy RoboDojo/XPolicyLab/policy/<YourPolicy>
cd RoboDojo/XPolicyLab/policy/<YourPolicy>
# 1. grep TODO and fill in model weights / env / deps
# 2. install the policy env
bash install.sh
# 3. Isaac-free wiring check (runs on this Mac)
cd ../../..   # back to RoboDojo root
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/<YourPolicy> \
  --task stack_bowls --ckpt demo --policy-env <your-env> --dry-run
# 4. runtime check on the GPU box (EVAL_ENV_TYPE=debug = offline, no simulator)
```

## Files

| File | Role |
|---|---|
| `deploy.yml` | Policy identity + runtime contract. `protocol: ws` (eval client default), `eval_batch: false` unless the policy serves batches. `setup_eval_policy_server.sh` overrides host/port/bench/task/ckpt at launch. |
| `eval.sh` | Same-machine entry: starts policy server on a free port, waits via `wait_for_policy_server.sh`, then runs the sim client. Split-machine flows use the two `setup_eval_*` scripts directly. |
| `setup_eval_policy_server.sh` | Activates the **policy** conda env, `exec`s `setup_policy_server.py` with `--config_path deploy.yml --overrides ...`. Runs with CWD = policy dir. |
| `setup_eval_env_client.sh` | Activates the **sim** env and delegates to XPolicyLab `setup_env_client.sh`, which launches `src/eval_client/main.py` in RoboDojo. |
| `model.py` | Adapter: `ModelTemplate` subclass (`__init__`, `update_obs[_batch]`, `get_action[_batch]`, `reset`). Replace zero-action `get_action` with real inference. |
| `deploy.py` | Eval loop (`eval_one_episode[_batch]`): reset → obs → action chunk → `take_action`. Keep unless the policy needs custom stepping. |
| `install.sh` | Creates the policy conda env. Keep policy deps isolated from the `RoboDojo` sim env. |

## Gotchas

- `robodojo.sh eval` does **not** call `eval.sh` directly; it uses the
  `setup_eval_*` launchers with CWD = policy directory (keep paths relative).
- `deploy.yml` `ckpt_name`/`action_type` flow into result paths via
  `additional_info="ckpt_name=...,action_type=..."` — keep them accurate.
- For split-machine/Docker eval the server must bind `0.0.0.0`, not localhost.
