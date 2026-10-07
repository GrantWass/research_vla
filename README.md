# research_vla

Vision-Language-Action policies (pi0.5, OpenVLA, TurboVLA), evaluated in
simulation (RoboDojo on Isaac Sim) and run on two real AgileX Piper arms. This
repo is a thin layer over upstream checkouts. We own the policy registry,
adapters, the real-robot code, serving experiments, tests and docs.

Everything runs on one machine: the Alienware workstation (Ubuntu 22.04,
RTX 3090 24 GB) that drives the arms.

## Docs

| | |
|---|---|
| [docs/robot.md](docs/robot.md) | The Piper rig: hardware map, safety, teleop, running policies, next steps |
| [docs/sim.md](docs/sim.md) | RoboDojo setup, the policy registry, evals, fine-tuning, troubleshooting |
| [docs/serving.md](docs/serving.md) | Serving many robots from one policy server: strategies, results, tools |
| [docs/findings.md](docs/findings.md) | What we learned and the evidence behind it |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Conventions and PR checklist |

## Layout

| Path | What |
|---|---|
| `real_robot/` | Piper backend, cameras, safety filter, control loop, teleop, checks (`docs/robot.md`) |
| `policies/` | One `*.conf` per sim policy; `scripts/run_eval.sh --policy <name>` resolves it |
| `adapters/turbovla_robodojo/` | Our XPolicyLab adapter for TurboVLA |
| `patches/` | Fixes to the upstream checkouts, applied by the setup scripts; `make test` checks they still apply |
| `scripts/` | `setup_policy.sh`, `run_eval.sh`, `install_*.sh`, `lowvram.sh`, `serve_bench/` |
| `templates/` | Skeletons: new RoboDojo task, new policy adapter, OpenVLA LoRA and TurboVLA fine-tunes |
| `tests/` | CPU-only `unittest` suite; no GPU, Isaac or robot needed |
| `openvla/`, `turbovla/`, `RoboDojo/` | Upstream checkouts (git-ignored), recreated at pinned commits by `setup.sh` |

## Quickstart

```bash
bash setup.sh        # upstream checkouts at pinned commits
make test            # must pass before every PR
make help            # every shortcut
```
