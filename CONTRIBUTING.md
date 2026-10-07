# Contributing

## Conventions

- Branches: `<user>/<short-description>`. Commits: `[Scope] type: description`.
  Scopes: `Robot | Sim | Policies | Serving | Templates | Tests | Docs | chore`.
  Types: `feat | fix | update | docs | chore`.
- Upstream code (`turbovla/`, `RoboDojo/`) is never committed. Change it
  through a file in `patches/` applied by a setup script, or PR it upstream.
- Never commit weights, datasets, eval outputs, run logs (`real_robot_logs/`,
  `verify/`) or secrets (`.env`, HF tokens). `.gitignore` and pre-commit's
  1 MB limit catch most of these.
- Record the checkpoint location and dataset revision of any training run in
  its PR.

## Where things go

- A fact about the rig, or a procedure: `docs/robot.md` or `docs/sim.md`.
- A result or lesson: `docs/findings.md`, short, with the evidence.
- New sim policy: `policies/<name>.conf` (+ `adapters/` if needed), see `docs/sim.md`.
- New task or adapter for upstream: start from `templates/`.

## PR checklist

- [ ] `make test` passes
- [ ] No weights, logs or secrets in the diff
- [ ] Docs updated if behavior changed
