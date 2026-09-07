# Contributing to research_vla

This repo is a thin workspace over upstream clones (`openvla/`, `RoboDojo/` — both
git-ignored). We own: `tests/`, `templates/`, docs, `setup.sh`, `Makefile`, CI.

## Setup

```bash
bash setup.sh   # clones at pinned commits + inits XPolicyLab submodule
make test       # must pass before every PR
```

## Branches & commits

- Branches: `<user>/<short-description>`, e.g. `sam/task-wipe-table`.
- Commits: `[Scope] type: description` — scopes `Tests | Templates | Task | Policy | Docs | scripts | fix | chore`;
  types `feat | fix | update | docs | chore`. (Matches the RoboDojo convention.)
- Never commit: upstream checkouts, `Assets/`, `eval_result/`, `checkpoints/`,
  weights (`*.pt/*.safetensors/*.hdf5`), `.hf_token`/`.env`, `wandb/`.
  Pre-commit blocks large files; `.gitignore` covers the rest.

## What goes where

- **New benchmark task** → copy `templates/robodojo_task/`, follow its README checklist.
  Upstream task code belongs in the RoboDojo checkout, not committed here —
  PR the task to `RoboDojo-Benchmark/RoboDojo` and note the commit here.
- **New policy adapter** → copy `templates/xpolicylab_policy/my_policy/`, same deal
  (PR to `XPolicyLab/XPolicyLab`).
- **Fine-tune runs** → `templates/openvla_finetune/`; checkpoints stay on
  HuggingFace Hub / shared storage, never in git. Record run dirs + dataset
  revisions in the PR description so teammates can reproduce.
- **Workspace changes** (tests, templates, docs, scripts) → PR here with the
  checklist below filled in.

## PR checklist

- [ ] `make test` passes (paste output or link CI run)
- [ ] New/changed template covered by `tests/test_templates.py`
- [ ] No secrets, weights, or generated artifacts in the diff (`git status` clean of them)
- [ ] Docs updated if behavior changed (`OPENVLA_ROBODOJO_SETUP.md` or template README)

## Secrets

- HuggingFace: `huggingface-cli login` or `~/.hf_token` (never in-repo).
- W&B entity/project passed as env vars to `finetune_lora.sh`, never hardcoded.
