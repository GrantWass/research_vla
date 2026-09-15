# research_vla

Workspace for Vision-Language-Action robot research: **OpenVLA** and
**TurboVLA** (the policies) evaluated in **RoboDojo** (the benchmark, built
on NVIDIA Isaac Sim + Isaac Lab). Swapping models is one flag:
`bash scripts/run_eval.sh --policy turbovla --task stack_bowls --dry-run`.
This repo is a thin layer over upstream clones — we own the policy registry,
adapters, tests, templates, and docs; the simulators and models live upstream.

> Hardware note: full simulation needs a Linux box with an NVIDIA RTX GPU
> (Ubuntu 22.04, driver 570/580, 32 GB RAM, 16 GB VRAM). Everything marked
> "Mac-safe" below runs on Apple Silicon with system Python and no GPU.

## What everything is

| Path | What it does |
|---|---|
| `openvla/` | Upstream OpenVLA clone (git-ignored). Training, LoRA fine-tune (`vla-scripts/`), REST deploy server, LIBERO/Bridge eval scripts. Pinned by `setup.sh`. |
| `turbovla/` | Upstream TurboVLA clone (git-ignored). 0.2B direct V+L→A VLA (32 Hz, <1 GB VRAM on RTX 4090). Training/eval in `turbovla/` + `experiments/`; released ckpts `H-EmbodVis/TurboVLA`. Pinned by `setup.sh`. |
| `RoboDojo/` | Upstream benchmark clone (git-ignored). 54 runnable sim tasks, sim stack (`env/`), configs (`env_cfg/`), eval entry `scripts/robodojo.sh`, policy adapters in `XPolicyLab/` (incl. `OpenVLA_OFT`, `demo_policy`). Pinned by `setup.sh`. |
| `policies/` | Model registry: one `*.conf` per VLA (`openvla`, `turbovla`, `demo`). `scripts/run_eval.sh --policy <name>` resolves adapter dir, conda env, ckpt, action type. See `POLICIES_ROBODOJO.md`. |
| `adapters/turbovla_robodojo/` | This-repo XPolicyLab adapter serving TurboVLA in RoboDojo. Install: `bash scripts/install_adapter.sh turbovla`. |
| `tests/` | CPU-only smoke suite (stdlib `unittest`, Mac-safe). Layout checks, task-inventory validation (54 runnable, names match, real rewards), eval dry-run wiring, policy contract, template self-checks. Run: `make test`. |
| `templates/robodojo_task/` | Skeleton for a new benchmark task (`my_task.yml` + `my_task.py` + checklist). Copy into the RoboDojo checkout, rename, PR upstream. |
| `templates/xpolicylab_policy/` | Skeleton for a new policy adapter (`eval.sh`, `deploy.yml`, server/client launchers, `model.py`, eval loop, `install.sh`). Mirrors `demo_policy`. |
| `templates/openvla_finetune/` | LoRA fine-tune launcher (`finetune_lora.sh`, env-var configured) + checklist (routes, dataset registration, data-collection rules, sanity checks). GPU box only. |
| `templates/turbovla_finetune/` | TurboVLA-on-RoboDojo training: env-var launcher (`train.sh`), registry overlay (`data_registry/`), recipe (`configs/robodojo.yaml`), stats (`compute_stats.py`), eval-ready `deploy.robodojo.yml`. GPU box only. |
| `OPENVLA_ROBODOJO_SETUP.md` | Full guide: what was pulled, why RoboDojo, Mac-verified steps, Linux GPU setup, every eval command, troubleshooting. |
| `POLICIES_ROBODOJO.md` | Multi-model eval: registry, per-model setup, all `run_eval.sh` commands, adding the next model. |
| `TURBOVLA_ROBODOJO_SIMPLE.md` | Plain-language overview: the course, the brain, and the glue between them. |
| `PROGRESS_REPORT.md` | GRA status: wiring evidence, results skeleton, blockers/next steps. |
| `REMOTE_ACCESS.md` | SSH into the Windows box from any network (OpenSSH server, key auth, Tailscale, troubleshooting). |
| `CONTRIBUTING.md` | Branch/commit conventions, where each kind of work belongs, secrets handling, PR checklist. |
| `setup.sh` | Reproduces the upstream checkouts at pinned commits + inits the XPolicyLab submodule. `--check` verifies pins without network writes. |
| `Makefile` | Shortcuts: `test`, `setup`, `check`, `doctor`, `inventory`, `dry-run TASK=<t>`, `finetune-help`, `clean`. |
| `.github/workflows/smoke.yml` | CI: restores checkouts via `setup.sh`, runs the smoke suite on every push/PR. |
| `.github/pull_request_template.md` | PR front-matter: summary, test plan, checklist. |
| `.pre-commit-config.yaml` | Whitespace/YAML hygiene, ruff on `tests/`+`templates/`+`adapters/`, blocks >1 MB files (no weights in git). |

## Quickstart

```bash
bash setup.sh          # clone upstream repos at pinned commits (once per machine)
bash scripts/install_adapter.sh turbovla   # install this-repo TurboVLA adapter (once)
make test              # smoke suite — must pass before every PR
make check             # verify checkouts still match pins
make policies          # list registered VLA models
make doctor            # RoboDojo health check (Mac-safe: skips Isaac/conda/policy)
make inventory         # list the 54 runnable tasks
make dry-run POLICY=turbovla TASK=stack_bowls   # resolve an eval command without launching sim
```

## Common workflows

- **Evaluate a policy (GPU box):** `bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast` (swap `turbovla`→`openvla` for the baseline) — full command reference in `POLICIES_ROBODOJO.md`; harness details in `OPENVLA_ROBODOJO_SETUP.md` §5.
- **New benchmark task:** copy `templates/robodojo_task/`, follow its README, validate with `make test` + dry-run, PR the task to `RoboDojo-Benchmark/RoboDojo`.
- **New policy adapter:** copy `templates/xpolicylab_policy/my_policy/`, fill in the TODOs, PR to `XPolicyLab/XPolicyLab`.
- **Fine-tune OpenVLA (GPU box):** follow `templates/openvla_finetune/README.md`, then `cd openvla && bash ../templates/openvla_finetune/finetune_lora.sh`.

## Contributing

Read `CONTRIBUTING.md`. TL;DR: `make test` before every PR, never commit checkouts/weights/secrets (`pre-commit` enforces the large-file part), record checkpoint locations + dataset revisions for any training run.
