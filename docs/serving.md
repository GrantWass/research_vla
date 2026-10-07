# Multi-robot serving

Goal: serve N robots from one pi0.5 server on one GPU. Upstream serves them
one at a time. `Pi_05.get_action_batch()` loops `policy.infer()` once per
robot, and the websocket server takes a global lock on each call. Everything
here is off by default and switched on by env vars at server start.

## Strategies

| ID | Env | What it does |
|---|---|---|
| S0 | (defaults) | upstream: one forward per robot, observe and answer as two round trips |
| S1 | `PI05_BATCHED_INFER=1` | robots on one connection share one forward |
| S2 | S1 + `PI05_ATOMIC=1` | observe and answer in a single call under one lock |
| S3 | S2 + `XPL_SERVER_BATCH=1` | robots on **separate connections** are coalesced into one forward (continuous batching) |

Knobs:
- `PI05_MAX_BATCH`: caps one forward.
- `XPL_BATCH_WINDOW_MS` and `XPL_BATCH_MAX`: hold a batch open for a window, and cap S3 batches.
- `PI05_WARMUP_BATCHES=1,2,...`: compiles those batch sizes at startup. Each new size otherwise stalls every robot for 9-13 s mid-run. The warmup also probes VRAM and caps the batch at the first size that fails.
- `PI05_NOISE_SEED`: fixes the sampler noise, for equivalence tests.
- `PI05_TIMING_LOG`: writes a JSONL row per forward.

The code is in `patches/xpolicylab_pi05_batched_infer.patch` and
`patches/xpolicylab_server_batch_window.patch`, both applied by
`scripts/setup_policy.sh pi05`.

## What we measured

- **Batching doesn't change outcomes.** In a paired run (N=6, three seeds × 25
  scenes, 150 episodes) S0 and S2+warmup scored 48/75 vs 46/75, McNemar
  p=0.85. Batched and solo answers differ by 2-16% of what changing the noise
  seed alone does.
- **It is cheaper per robot, up to about 38%.** A forward costs about
  47 ms fixed plus 77 ms per robot (B=1..6: 124, 211, 291, 366, 430, 513 ms).
  S2 used 76% fewer calls than S0 at N=6.
- **It doesn't make RoboDojo faster.** The GPU is about 2% busy, so the sim
  is the bottleneck. Wall-clock differences between runs follow failed
  episodes, which run the full 800 steps.
- **Separate clients:** with no think time, FIFO tops out at 8.4 req/s and
  continuous batching reaches 12.0 req/s (p50 -30%). A 10 ms window helps at
  2-6 clients, because closed-loop clients otherwise phase-lock and never
  meet in a batch. At light load every mode ties. Default: `XPL_SERVER_BATCH=1`, window 0.
- **Shared state is the real hazard.** The model keeps one observation
  buffer, so two clients could interleave observe and answer. `PI05_ATOMIC`
  fixes that.
- **Statistics:** 12 episodes per arm can only detect gaps of about 57
  points. Use paired seeds and McNemar, not raw success rates.

## Tools (`scripts/serve_bench/`)

| File | Purpose |
|---|---|
| `sim_battery.sh NAME N TASK EPS SEED S0\|S1\|S2\|S2W` | one RoboDojo run with N envs; appends to a manifest |
| `live_battery.sh equivalence\|fleet` | standalone server, no sim, across server modes |
| `replay_client.py` | synthetic robots against a live server (`--scenario equivalence\|fleet`) |
| `compare_runs.py [--pairwise]` | comparison tables from a manifest plus timing logs |
| `fleet_report.py`, `equivalence_report.py` | tables for the two live scenarios |
| `simulate.py`, `schedulers.py` | fleet simulator replaying measured timings (sequential / round-robin / EDF) |
| `metrics.py`, `stats.py` | request records, Wilson/Fisher/McNemar |

Raw data from the runs above is committed in `scripts/serve_bench/data/`
(`runs_2026_09_30.json` is the manifest for the first battery), so
`python3 scripts/serve_bench/compare_runs.py --pairwise` regenerates the tables.

## Open directions

- A setup where inference really is the bottleneck (more robots, a heavier
  policy, a higher control rate), so scheduling policy can be measured end to end.
  The simulator predicts batching starts to matter around 6 robots with mixed
  chunk lengths, and 10 with uniform ones.
- Smoother chunk boundaries. The real-robot runner already executes
  asynchronously (`real_robot.runner --async`: the arm keeps moving while the
  next chunk is computed). PAINT (arXiv:2606.19774) would add a flow-inversion
  start noise for smoother transitions, at about 2x inference cost.
- Real robots: two Pipers sharing one server on the 3090.
