# `policies/` — model registry for RoboDojo eval

One small file per VLA model. `scripts/run_eval.sh` sources the selected file
and translates it into a `RoboDojo/scripts/robodojo.sh` call, so swapping
models is just `--policy <name>` (or `make dry-run POLICY=<name> TASK=...`).

## Adding a new model

1. Copy `demo.conf` to `<name>.conf` and fill in the fields below.
2. If the model needs a new XPolicyLab adapter, add it under
   `adapters/<name>_robodojo/` (copy `adapters/turbovla_robodojo/`) and note
   the install step in the conf's comments. If the adapter already ships
   with XPolicyLab (like `Pi_05` for `pi05.conf`), skip this — leave
   `REPO_DIR` empty and `scripts/install_adapter.sh <name>` is a no-op.
3. `python3 -m unittest discover -s tests` must still pass (it checks every
   `*.conf` parses and resolves).

## Fields

| Key | Meaning |
|---|---|
| `POLICY_NAME` | Must equal the conf basename (`openvla` ↔ `openvla.conf`). |
| `DESCRIPTION` | One line for `run_eval.sh --list`. |
| `REPO_DIR` | Model source checkout at the workspace root (empty if none). Pinned by `setup.sh`. |
| `XPOLICYLAB_POLICY_DIR` | Adapter dir relative to `RoboDojo/` (becomes `--policy-dir`). |
| `POLICY_ENV` | Policy runtime env (becomes `--policy-env`). Usually a conda env name; use `uv` for uv-managed adapters like `Pi_05` (`robodojo.sh` accepts "conda env, uv, or env path"). |
| `DEFAULT_CKPT` | Used when `--ckpt` is omitted. |
| `ACTION_TYPE` | `joint` or `ee` (becomes `--action-type` unless `--action-type` is passed). |
| `ENV_CFG` | Becomes `--env-cfg` unless overridden. |

## Precedence

CLI flags beat conf values beat builtin defaults: `--ckpt` >
`DEFAULT_CKPT`, `--action-type` > `ACTION_TYPE`, `--env-cfg` > `ENV_CFG`
(`arx_x5` / `ee` when the conf omits them). So a conf holds the model's
normal setup, and one-off runs override it without editing files:

```bash
bash scripts/run_eval.sh --policy turbovla --task stack_bowls \
  --env-cfg dual_x5 --action-type ee --dry-run
```
