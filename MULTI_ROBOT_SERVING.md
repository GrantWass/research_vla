# Multi-robot inference serving on one GPU

Goal: serve N robots from one pi0.5 policy server on a single 16 GB GPU
(RTX 4070 Ti SUPER, driver 580), tested in RoboDojo with 2 sim envs sharing
one server. Every code/infra change for this goal is logged below.

## Strategies under comparison

| ID | Name | Flags | What it does, plainly |
|---|---|---|---|
| S0 | sequential | (defaults) | One robot per brain call, observe+answer as two network trips. Upstream behavior. |
| S1 | batched | `PI05_BATCHED_INFER=1` | All waiting robots answered in one brain call. |
| S2 | atomic+batched | `PI05_BATCHED_INFER=1 PI05_ATOMIC=1` | Batched, plus observe+answer in a single trip (no cross-talk, fewer trips). |
| S3 | server-side batching | S2 + `XPL_SERVER_BATCH=1` [`XPL_BATCH_WINDOW_MS`, `XPL_BATCH_MAX`] | Robots on **separate connections** share one brain call: the server coalesces `infer_obs` requests that arrive while a forward is running (or within the window). |

Shared knobs: `PI05_WARMUP_BATCHES=1,2,...` compiles those batch sizes on the
first call (each new size otherwise stalls every robot ~9-13 s mid-run);
`PI05_NOISE_SEED` fixes the sampler's start noise for equivalence tests.

Tooling: `scripts/serve_bench/` (`metrics.py` records, `schedulers.py`
policies, `simulate.py` fleet sim with our measured timings,
`replay_client.py` live-server scenarios incl. `equivalence` and `fleet`,
`live_battery.sh` / `sim_battery.sh` battery drivers, `compare_runs.py`,
`fleet_report.py`, `equivalence_report.py` tables, `stats.py` CIs and
significance tests, `report.py` summaries). Tests: `tests/test_serve_bench.py`
(server-batcher tests run where the patch is applied, i.e. on the box).

## Baseline (2026-09-30, no code changes yet)

Single-robot path, verified in this repo:

- `RoboDojo/XPolicyLab/policy/Pi_05/model.py:128-139` `get_action_batch()`
  stacks obs then **loops `policy.infer()` per env** (sequential N x 3B forwards).
- `RoboDojo/XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/policy.py:68-84`
  `Policy.infer()` **does support batched input** (`is_batched` via state ndim>1).
  The batching is defeated by the loop above.
- `RoboDojo/XPolicyLab/client_server/ws/model_server.py:398-400`
  `_call_model_method()` holds a global `_model_lock`: concurrent WS clients
  serialize anyway.
- `RoboDojo/XPolicyLab/policy/Pi_05/deploy.yml:9` `eval_batch: true`, so
  `src/eval_client/main.py:260-321` honors `num_envs`. `patches/robodojo_lowvram_sim.patch:9-13`
  pins `sim_config.yml num_envs: 10 -> 1` (fairness + VRAM). Single-env VRAM:
  pi05 14.4 GB + sim 5.7 GB (`GPU_BOX_SETUP.md` verified table).
- Action chunking desyncs robots: `XPolicyLab/policy/Pi_05/deploy.py:30`
  iterates `chunk_size` steps per infer; batch loop already tracks
  `get_running_env_idx_list()` for finished envs.

Target algorithm: request queue + batched forward (Triton-style):

```
robots -> obs queue -> [up to B_max or T_wait ms] -> 1x batched infer -> slice -> reply
```

with continuous batching (evict done envs, keep running ones).

## Measured comparison (12 runs, 2026-09-30)

Regenerate with `python3 scripts/serve_bench/compare_runs.py --pairwise`
(raw logs in `scripts/serve_bench/data/`, per-episode results backfilled from
each run's `_result.json`). `batch mix` is how many robots shared each brain
call. `gpu busy` is the share of wall time the brain was actually working;
`ms/chunk` is the GPU cost of serving one robot one action chunk. `95% CI` is
Wilson. The second table is steady-state median ms per forward by batch size
(JAX compiles excluded and counted). The third tests each run against the S0
run with the same task and N: Fisher (unpaired) and, because runs with the
same seed replay the same scene layouts, an exact McNemar test on the scenes
where the two strategies disagreed.

| run | strat | N | eps | succ | 95% CI | score | wall_s | peak_MiB | calls | batch mix | robot_chunks | gpu_s | gpu_busy_% | ms/chunk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_S0_N2 | S0 sequential | 2 | 12 | 8/12 | 39%-86% | 71.7 | 651 | 13577 | 135 | 1x135 | 135 | 16.9 | 2.6 | 125.1 |
| long_S1_N2 | S1 batched | 2 | 12 | 5/12 | 19%-68% | 47.9 | 770 | 13367 | 94 | 1x33 2x61 | 155 | 17.0 | 2.2 | 109.7 |
| long_S2_N2 | S2 atomic+batched | 2 | 12 | 7/12 | 32%-81% | 63.3 | 732 | 13382 | 90 | 1x39 2x51 | 141 | 15.6 | 2.1 | 110.9 |
| long_S0_N3 | S0 sequential | 3 | 12 | 8/12 | 39%-86% | 70.4 | 629 | 13765 | 154 | 1x154 | 154 | 19.2 | 3.0 | 124.5 |
| long_S1_N3 | S1 batched | 3 | 12 | 8/12 | 39%-86% | 71.7 | 612 | 13759 | 58 | 1x11 2x9 3x38 | 143 | 14.3 | 2.3 | 100.2 |
| long_S2_N3 | S2 atomic+batched | 3 | 12 | 6/12 | 25%-75% | 56.3 | 622 | 13774 | 58 | 1x3 2x20 3x35 | 148 | 14.8 | 2.4 | 100.0 |
| scale_N5_S2 | S2 atomic+batched | 5 | 10 | 7/10 | 40%-89% | 73.0 | 431 | 14172 | 28 | 1x3 2x5 3x7 5x13 | 99 | 9.1 | 2.1 | 91.6 |
| scale_N6_S2 | S2 atomic+batched | 6 | 6 | 3/6 | 19%-81% | 57.5 | 300 | 14056 | 12 | 3x4 4x3 6x5 | 54 | 4.8 | 1.6 | 89.4 |
| cap2_N4_S2 | S2 atomic+batched cap2 | 4 | 4 | 3/4 | 30%-95% | 78.8 | 212 | 14403 | 21 | 1x3 2x18 | 39 | 4.2 | 2.0 | 106.7 |
| blocks_S0_N2 | S0 sequential | 2 | 4 | 0/4 | 0%-49% | 7.5 | 210 | 13383 | 43 | 1x43 | 43 | 5.4 | 2.6 | 124.6 |
| blocks_S2_N2 | S2 atomic+batched | 2 | 4 | 0/4 | 0%-49% | 11.2 | 211 | 13367 | 21 | 2x21 | 42 | 4.4 | 2.1 | 105.9 |
| sanity_N1_S0 | S0 sequential | 1 | 2 | 1/2 | 9%-91% | 57.5 | 187 | 13152 | 21 | 1x21 | 21 | 2.6 | 1.4 | 125.9 |

| run | compiles | B=1 | B=2 | B=3 | B=4 | B=5 | B=6 |
|---|---|---|---|---|---|---|---|
| long_S0_N2 | 1 | 125 |  |  |  |  |  |
| long_S1_N2 | 2 | 124 | 212 |  |  |  |  |
| long_S2_N2 | 2 | 124 | 211 |  |  |  |  |
| long_S0_N3 | 1 | 124 |  |  |  |  |  |
| long_S1_N3 | 3 | 124 | 211 | 291 |  |  |  |
| long_S2_N3 | 3 | 124 | 211 | 291 |  |  |  |
| scale_N5_S2 | 4 | 124 | 212 | 291 |  | 430 |  |
| scale_N6_S2 | 4 |  |  | 292 | 366 |  | 513 |
| cap2_N4_S2 | 2 | 124 | 211 |  |  |  |  |
| blocks_S0_N2 | 1 | 125 |  |  |  |  |  |
| blocks_S2_N2 | 1 |  | 212 |  |  |  |  |
| sanity_N1_S0 | 1 | 126 |  |  |  |  |  |

| run | vs | succ | baseline | Fisher p | detectable gap at this n | paired: only-base / only-run | McNemar p | score diff (95% CI) |
|---|---|---|---|---|---|---|---|---|
| long_S1_N2 | long_S0_N2 | 5/12 | 8/12 | 0.41 | 57% | 5 / 2 of 12 | 0.45 | -0.24 (-0.60, +0.12) |
| long_S2_N2 | long_S0_N2 | 7/12 | 8/12 | 1.00 | 57% | 2 / 1 of 12 | 1.00 | -0.08 (-0.33, +0.16) |
| long_S1_N3 | long_S0_N3 | 8/12 | 8/12 | 1.00 | 57% | 1 / 1 of 12 | 1.00 | +0.01 (-0.19, +0.22) |
| long_S2_N3 | long_S0_N3 | 6/12 | 8/12 | 0.68 | 57% | 2 / 0 of 12 | 0.50 | -0.14 (-0.35, +0.07) |
| blocks_S2_N2 | blocks_S0_N2 | 0/4 | 0/4 | 1.00 | 99% | 0 / 0 of 4 | 1.00 | +0.04 (-0.04, +0.11) |


What the numbers say, plainly:

- **The brain is almost idle.** GPU busy is 1.4-3.0% of wall time in every
  run. The simulator, not pi0.5, is the bottleneck, so no serving strategy
  can meaningfully change wall-clock at these sizes.
- **Batching does make serving cheaper per robot.** Cost per robot chunk
  falls from ~125 ms (sequential) to ~110 ms at N=2, ~100 ms at N=3, ~92 ms
  at N=5 and ~89 ms at N=6 - up to ~28% cheaper. It also cuts brain calls
  by 62% at N=3 (154 -> 58).
- **Wall-clock differences are confounded, not causal.** Failed episodes run
  the full 800 steps while successes end early, so lower-success runs take
  longer. Never compare wall time across strategies without matching
  episode counts and success rates.
- **These runs cannot tell strategies apart on success, either way.** Every
  12-episode interval spans ~45 points, and 12 episodes per arm can only
  detect a ~57-point gap. The 8/12 vs 5/12 "drop" for S1 at N=2 has Fisher
  p=0.41. Paired by scene, the batched arms lost 10 scenes the baseline won
  and won 4 it lost (pooled McNemar p=0.18): a lean, not a finding. The
  large paired run below is what settles it.
- **Cost per forward is ~47 ms fixed + ~77 ms per robot** (B=1..6: 124,
  211, 291, 366, 430, 513 ms). The GPU is already compute-bound at one
  robot, so batching only amortizes the fixed part: per-robot cost tends to
  ~77 ms, at most ~38% below solo. That is the ceiling on what any batching
  scheme can buy on this card.
- **Six robots fit on one 16 GB card**, peaking at 14.2 GB, and the batch
  engine held full 6-robot batches (5 of 12 calls). Per-robot-chunk cost
  keeps falling with N, so more robots get cheaper, not more expensive.
- **`PI05_MAX_BATCH=2` is honoured** (18 of 21 calls were pairs, never 3+)
  and is the knob to use if a future batch would not fit in VRAM.
- **`stack_blocks` scores 0/4 under both strategies** (score 7.5 vs 11.2),
  with S2 halving calls (43 -> 21). The task is too hard for this policy to
  be a useful serving signal - it is a floor, not a comparison.
- **Single-robot baseline is intact**: 126 ms per call, 1/2 success, 2.6 GB
  of the peak. No regression from the batching path.

## Change log

| Date | Change | Files | Result |
|---|---|---|---|
| 2026-09-30 | One-time UEFI boot Windows -> Ubuntu via `bcdedit /set {fwbootmgr} bootsequence {e7c18e2e-b666-11f1-88ae-806e6f6e6963}` + `shutdown /r`. No boot-order default changed (fallback stays Windows). | none (infra) | `grant-linux` up, RTX 4070 Ti SUPER 16 GB driver 580, GPU idle 150 MiB, `.lowvram.env` active (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`) |
| 2026-09-30 | Created this log per request. | `MULTI_ROBOT_SERVING.md` | — |
| 2026-09-30 | 2-robot pi0.5 eval: temp `sim_config.yml num_envs 1->2`, `run_eval.sh --policy pi05 --task stack_bowls --eval-num 2 --seed 0`. No code change (sequential loop + model lock as baselined). | `RoboDojo/env_cfg/sim/sim_config.yml` (temp, on GPU box; repo pin stays 1) | PASS: 2 envs shared 1 server, wall 165 s, success 1/2 (ep0 success 306 frames, ep1 fail 801), `_result.json` success_rate 0.5 score 50.0. Peak ~12.5 GB / 16 GB. No OOM. |
| 2026-09-30 | Serving-middleware bench harness (metrics + schedulers + report). Metric set from Armory arXiv:2608.00337 (starvation rate, task throughput, infer vs batch) + vLLM/SGLang/Triton/AIPerf + MLPerf (E2E p50/p99, TTFAC=TTFT analog, inter-chunk latency=ITL analog, queue p99, req/s, actions/s) + RoboDojo (success_rate, score). Schedulers: sequential (status quo), round_robin, earliest_deadline (Armory EDF). | `scripts/serve_bench/metrics.py`, `scripts/serve_bench/schedulers.py`, `scripts/serve_bench/report.py`, `tests/test_serve_bench.py` | `python3 -m unittest tests.test_serve_bench` 7/7 OK (Mac-safe, stdlib). Openpi already emits `policy_timing.infer_ms`; server adds `latency_ms` — harness consumes both, no model change. |
| 2026-09-30 | 3-robot pi0.5 eval: temp `sim_config.yml num_envs 1->3`, `run_eval.sh --policy pi05 --task stack_bowls --eval-num 3 --seed 0`. Same sequential serving as baseline. | `RoboDojo/env_cfg/sim/sim_config.yml` (temp, GPU box; restored to 1 after) | PASS: 3 envs shared 1 server, wall 165 s (same as 2-robot), success 3/3, score 100.0. Mid-run VRAM 13.7 GB / 16 GB at 43% util, no OOM. Restored `num_envs: 1`; box `git status` clean. |
| 2026-09-30 | Synced bench harness to GPU box (`scp` serve_bench + test). | same 4 files (both checkouts) | `python3 -m unittest tests.test_serve_bench` 7/7 OK on `grant-linux`. |
| 2026-09-30 | First middleware variant: batched pi0.5 forward. `get_action_batch()` tries one `policy.infer(stacked_obs)` behind `PI05_BATCHED_INFER=1` (default 0 = sequential), `PI05_MAX_BATCH` caps one forward, silent-sequential fallback + first-batch stdout marker + `last_batch_info()` for the harness. Upstream pin untouched (patch file). | `patches/xpolicylab_pi05_batched_infer.patch`, `scripts/setup_policy.sh` (apply in `setup_pi05`), `tests/test_serve_bench.py`+`test_gpu_box_tooling.py` (`PATCH_TARGETS`) | `make test` 76 OK. Patch `--check` clean on Mac (not applied there). |
| 2026-09-30 | A/B on `grant-linux` (`stack_bowls`, temp `num_envs`, restored to 1 after): control patch-on/batch-off seed1/N2 → 2/2 wall 123 s; batched seed1/N2 → 2/2 wall 140 s (`batch_size=2 forwards=1`, no fallback); batched seed0/N3 → 1/3 score 43.3 wall 202 s vs sequential seed0/N3 3/3 wall 165 s. | patch applied on box only (default OFF = safe) | Batched path verified functional (1 forward serves N). No speedup at N=2..3; N=3 outcome gap is within single-run sampling variance (stochastic flow-matching; baseline 50%) — NOT a regression claim. Missing: per-request `infer_ms` distribution + repeated seeds + `PI05_MAX_BATCH` sweep. |
| 2026-09-30 | Timing hook in the patch: every model forward appends `{mode, batch, infer_ms}` JSONL to `PI05_TIMING_LOG` (unset = zero overhead). Sweep running on box: seq vs batched at N=2/N=3, seeds 0+1, plus `PI05_MAX_BATCH=2` at N=3, each with wall time + VRAM trace. | `patches/xpolicylab_pi05_batched_infer.patch` | `make test` 76 OK. Results pending. |
| 2026-09-30 | Sweep done (9 runs total incl. earlier A/B). Task scores look the same either way: seq 8/10 episodes, batched 9/13 — small numbers, coin-flip range, no winner. Brain-time per forward (steady): one-robot 125 ms, batch-of-2 212 ms (~106/robot), batch-of-3 291 ms (~97/robot) — batching is ~20% cheaper per robot but each round takes longer to come back, so wall-clock ties (N=2: 143 s both; N=3 gaps track failed episodes running full length, not serving speed). Batches observed at sizes 1-3 (robots drift out of sync — the Armory paper's exact point). Memory identical per N (13.4 GB N=2, 13.7 GB N=3; ~2.5 GB headroom). `PI05_MAX_BATCH=2` behaves like full batch. | patch v3 on box; `num_envs` back to 1 | Batched serving is safe and slightly cheaper per robot, but at 2-3 robots it doesn't move the needle. Bigger fleets (or the EDF scheduler) is where it should pay off. |
| 2026-09-30 | DRTC ideas implemented as a simulator: replays our real timing logs, compares schedulers, injects lost messages. Finding: at our measured speeds with 2-6 robots nobody starves regardless of scheduler (matches the box). Stress test (10 robots, short plans): one-at-a-time serving starves 30% of steps, batched schedulers ~0%. Lost messages hurt as expected. | `scripts/serve_bench/simulate.py`, `tests/test_serve_bench.py` (4 new tests) | `make test` 80 OK both sides. Lesson: scheduling only matters once the fleet outgrows the brain — our box isn't there yet at N=3. |
| 2026-09-30 | PAINT assessed (not built): the hook it needs — choosing the plan's random starting noise — already exists in our code (`sample_actions(noise=...)`, `Policy.infer(noise=...)`). But it only pays off with async execution (brain works while robot moves), and our eval loop is strictly take-turns. Prototype needs: async loop + backward-inversion (~2x cost per call). | none (assessment) | Next in line if we want smoother motion under serving delays. |
| 2026-09-30 | Longer verification: 4 robots sharing one brain on `stack_bowls` (4 episodes) — sequential and batched both score 3/4 (78.8), identical. One forward serves all 4 (`batch_size=4 forwards=1`); peak memory 13.7 GB, same as 3 robots. Second task `push_T`: pi0.5 scores 0/4 either way (too hard for this checkpoint — no signal on quality, but serving ran clean cross-task). Timing pattern holds everywhere: 125 ms one-at-a-time, ~212 ms per shared answer. | temp `num_envs` on box (restored to 1) | Serving mode doesn't change results at any size tested (2-4 robots, 2 tasks). 4 robots fit comfortably; 5-6 likely fit too. |
| 2026-09-30 | Stagger test (fresh seed 2, N=2, 6 episodes = 3 waves; early finishers restart mid-episode so later waves run out of sync): seq and batched both 4/6 (scores 69.2 vs 71.7). Batch mix over the run: 38 shared answers, 2 solo — staggered robots still mostly ask together (same answer rhythm re-syncs them). | temp `num_envs` on box (restored to 1) | Different views per robot are no problem: each robot's cameras+instruction stay its own, answers are sliced back per robot. Stagger changes nothing. |
| 2026-09-30 | "Does asking together make scheduling pointless?" Partly. Fixed a real sim bug first (its clock skipped GPU busy time, so speed never mattered — the first sim's "ties everywhere" was the bug talking). Corrected numbers with our measured timings: same-plan fleets tie until N=10 (then one-at-a-time starves 40% of steps, batched 4%). Mixed fleets (half long-plan, half short-plan) split earlier — N=6: 44% vs ~16%, deadline-first edging plain batching at N=10 (40% vs 45%). | `scripts/serve_bench/simulate.py` (clock fix + per-robot plan lengths) | The sync dampens the algo's value but doesn't kill it: scheduling matters at scale and with mixed robots. Our box (same task, ≤4 robots) lives in the ties-everywhere zone. |
| 2026-09-30 | Heterogeneous test without a second sim (wouldn't fit in 16 GB): live standalone server + replay client sending mixed-persona batches (different states, instructions, pixels). Mixed pairs cost the same as same-task (~207 ms) and each persona gets its own correct answer back (A-means ~0.15, B-means ~0.53, matching solo runs). Staggered two-client test: alone ~122 ms, overlapped ~235 ms — separate clients queue at the server lock, nearly 2x. | throwaway `/tmp/replay.py` + `/tmp/cmp.py` on box (not committed); server shut down after, GPU idle | Proves the next bottleneck: batching helps robots sharing one connection, but separate clients still queue. Server-side batching window (planned change #2) is the fix. |
| 2026-09-30 | Found and fixed a real sharing bug: the server locks each request separately, but the brain keeps one shared notepad — two clients could land one's observations under another's answer (proven: B's answers slid from ~0.53 into A's ~0.2 range with a 50 ms gap). Fix: single atomic call does observe+answer under one lock (plus halves network round trips). Same-connection evals were never affected. Verified: concurrent hammer stays clean, full atomic eval 2/2 score 100 wall 141 s. | `patches/xpolicylab_pi05_batched_infer.patch` (now covers `model.py` + `deploy.py`), off by default (`PI05_ATOMIC=1` to use) | `make test` 80 OK; pin untouched. Lesson: sharing hardware means checking shared state, not just speed. |
| 2026-09-30 | Long runs, S0 vs S2, 12 episodes each (`stack_bowls`): N=2 S0 8/12 (651 s) vs S2 7/12 (732 s); N=3 S0 8/12 (629 s) vs S2 6/12 (622 s). Outcomes all sit in the 50-65% coin-flip zone — strategy doesn't move success. What moves: brain calls. S2 needed 32% fewer forwards at N=2 (92 vs 136) and 61% fewer at N=3 (61 vs 155), same answers. Batch mixes: N=2 {1:40, 2:52}, N=3 {1:4, 2:21, 3:36}. Steady timings: 125 ms solo, ~211 ms pairs, ~290 ms triples. | temp `num_envs` (restored to 1); `scripts/serve_bench/replay_client.py` committed for live-server tests | Verdict: serving wins are efficiency (fewer calls, ~20% cheaper per robot), not task scores. Framework complete: metrics, schedulers, sim, replay, report, 83 tests. |
| 2026-09-30 | **Corrected two numbers above.** Reading results via `ls -td ... | head -1` picked the wrong directory while runs were still being written. All 12 runs re-verified against explicit directories: N=2 S0 is 8/12 (not 6/12) and N=3 S0 scores 70.4 (not 71.7). Same success counts, so no conclusion changes. | `scripts/serve_bench/compare_runs.py` + `runs_2026_09_30.json` now read results by explicit name | Lesson: pin the exact result directory; never trust "newest directory" while a battery is running. |
| 2026-09-30 | Second battery, 8 more runs (~55 min GPU) to compare baselines: full S0/S1/S2 matrix at N=2 and N=3 (12 eps each), scaling to N=5 and N=6, a second task (`stack_blocks`), a batch-cap run, and a single-robot sanity run. Results in the table below. | temp `num_envs`; `PI05_MAX_BATCH=2`; raw timing logs kept in `scripts/serve_bench/data/` | 88 tests OK. Reproduce the table with `python3 scripts/serve_bench/compare_runs.py`. |

| 2026-09-30 | **Corrections and tooling review.** (1) `compare_runs.py` read timing logs from `/tmp` by default, so the documented regenerate command only worked on a machine that happened to have them there; now defaults to `data/`. (2) `replay_client.make_obs` seeded pixels with `hash()`, which Python salts per process, so "identical" observations differed between runs; now deterministic. (3) Success numbers now carry Wilson CIs, Fisher and paired McNemar tests, and a detectable-gap column. | `compare_runs.py`, `stats.py` (new), `replay_client.py`, `runs_2026_09_30.json` (per-episode backfill) | 12-ep runs can only detect ~57-pt gaps; no strategy difference is significant (table above). |
| 2026-09-30 | **Batched answers are numerically equivalent to solo answers.** Server under `PI05_NOISE_SEED` (same start noise for every robot), 3 iterations x mixed 3-robot batches: batched vs solo max abs diff 0.012-0.044; repeating a solo call gives exactly 0.0; different robots differ by 14.5 (so the check is not vacuous). Yardstick: changing only the noise seed moves the same robot's answer 0.28-0.76. Batch drift is **2-16% of ordinary sampling spread** (bf16 kernels pick different reduction orders per batch shape). | `PI05_NOISE_SEED` in `patches/xpolicylab_pi05_batched_infer.patch`; `replay_client.py --scenario equivalence`; `equivalence_report.py`; data in `data/equivalence_2026_09_30/` | Batching cannot plausibly change task outcomes; any success gap is sampling noise. Caveat: synthetic observations (random pixels), not sim frames. |
| 2026-09-30 | **Compile stalls measured and fixed.** Every new batch size costs a one-time JAX compile: 9.0-9.8 s (B=1), 10.0-10.4 (B=2), 10.6-11.4 (B=3), 11.3 (B=4), 11.8-12.0 (B=5), 12.6 (B=6). During a run, all waiting robots stall for it (a chunk lasts ~2 s). `PI05_WARMUP_BATCHES=1,2,...` compiles the listed sizes on the first call from the first real observation. | `patches/xpolicylab_pi05_batched_infer.patch` (`_warmup_batches`) | Startup cost moves to t=0 instead of mid-episode. `compare_runs.py` ignores warmup rows. |
| 2026-09-30 | **Server-side batching (planned change #1) implemented.** `XPL_SERVER_BATCH=1` routes `infer_obs` CALLs through a coalescer: requests arriving while a forward runs join the next one (continuous batching); `XPL_BATCH_WINDOW_MS` additionally holds an idle batch open; `XPL_BATCH_MAX` caps it. Errors propagate to every waiter; replies carry `server_batch`/`server_infer_ms`. Default off. | `patches/xpolicylab_server_batch_window.patch` (applied by `setup_policy.sh`), `tests/test_serve_bench.py::TestServerBatchWindow` (6 tests, pass on box) | Fleet comparison below. |
| 2026-09-30 | **Fleet battery: 1-8 separate clients x 3 think times x 4 server modes** (no sim; 30 calls per robot after warmup; `live_battery.sh fleet`). Saturated (think 0): FIFO caps at **8.4 req/s** whatever the fleet size, p50 grows linearly (8 clients 947 ms); continuous batching reaches **12.0 req/s**, p50 663 ms (-30%). 10 ms window best at 2-6 clients (2: 216 vs 232 ms; 4: 363 vs 388) because closed-loop clients otherwise **phase-lock into alternation** and never meet in a batch; but it adds ~11 ms at low load (25 ms window: ~26 ms) and never wins at think >= 0.5 s. At think 2 s (light load) all modes tie at ~123 ms p50. | `fleet_report.py`; data `data/fleet_2026_09_30/` (first pass) and `data/fleet_capped_2026_09_30/` (after the cap fix) | **Default: `XPL_SERVER_BATCH=1`, window 0.** Batching only matters when the brain is the bottleneck; at RoboDojo's real load (~2% busy) nothing changes. |
| 2026-09-30 | **Batch-of-8 OOMs under the low-VRAM profile, and the fallback hid it.** First pass, 8 clients with a window: each 8-batch hit `RESOURCE_EXHAUSTED` (fraction 0.45 = 7.2 GB), fell back to 8 sequential forwards, p50 **11,021 ms**. Fix: warmup doubles as a VRAM probe; the first size that fails caps later forwards (`[Pi_05] batch 8 does not fit; capping at 7`). Rerun: p50 678 ms, 0 fallbacks. | `patches/xpolicylab_pi05_batched_infer.patch` | Ceiling on this box with the sim profile: **B=7**. `compare_runs.py --pairwise` now shows a `fallbacks` column so this cannot hide again. |
| 2026-10-01 | **Large paired sim comparison settles success: no effect.** S0 vs S2W (atomic+batched+warmup), N=6, `stack_bowls`, seeds 0/1/2 x 25 episodes, same scenes under both = 150 episodes. Pooled 48/75 vs 46/75; discordant scenes 15 vs 13, McNemar p=0.85. Per seed: 20/20, 18/12, 10/14 (opposite leans cancel). S2W: 233 vs 905 calls (-76%), ~90 vs 124 ms per robot chunk, 0 compiles mid-run, 0 fallbacks, peak 14.2 GB both, wall equal. Also found: RoboDojo caps at 25 episodes per layout seed but writes the requested count to `eval_time`. | `sim_battery.sh` (records the actual episode count), `compare_runs.py` (counts per-episode rows; matches baselines by seed; pooled McNemar); data `data/sim_big_2026_09_30/` | Batching is outcome-neutral; 37% of scenes flip run to run anyway. Earlier "batched looks worse" lean (10 vs 4) was noise. |

## Planned changes

Done (see change log): (1) batched `get_action_batch` behind
`PI05_BATCHED_INFER`, with `PI05_MAX_BATCH`, timing log and sequential
fallback; atomic `infer_obs`/`infer_obs_batch` behind `PI05_ATOMIC`; 2/3/4/5/6
robot evals plus a 12-run comparison; `sample_actions` measured batch-safe for
flow-matching and per-robot prompts, with VRAM tracked by `nvidia-smi -lms`.
(3) 2-robot smoke, many times over; `num_envs` reverted to 1 for headline runs.

Still open:

1. Done (S3 above): server-side batching for separate clients.
2. A setup where inference actually is the bottleneck, so scheduler choices
   can be measured end to end. At N<=6 in RoboDojo the brain is ~2% busy, so
   queueing policy is untestable here - it needs many more robots, a heavier
   policy, or a different control rate.
3. PAINT-style action inversion for smoother motion under queueing delay
   (needs an async `sample_actions(noise=...)` entry point; inversion measured
   at ~2x cost, so it must be weighed against the latency it removes).

## Paper ideas backlog (summer/fall 2026)

| Paper | Idea in plain words | Fits us? |
|---|---|---|
| PAINT (`arXiv:2606.19774`, Jun 2026) | When the brain answers late, the robot has already moved, so the new plan starts from the wrong place and motion gets jerky. PAINT fixes this by picking a better random starting point for the plan (found by running the flow backwards), with no retraining and no changes to the policy. Works with any flow-matching policy like pi0.5. | Yes — built for exactly our delay problem; testable in the dojo as an async-execution upgrade. |
| DRTC (`jackvial/drtc`, 2026) | Practical tricks for serving a policy over a network (they also use Tailscale): merge messages so the newest plan always wins, cool down after lost/delayed messages, and test by deliberately breaking the connection. | Yes — cheap to adopt; matches our remote-GPU setup exactly. |
| FlashVLA (`arXiv:2608.27384`, Aug 2026) | Rebuilds the action generator so it streams out one usable piece per step (30+ answers/sec on one GPU) instead of one big chunk per slow call. | Not directly — needs model changes, not just serving changes. Keep as a future direction. |

## How to run the 2-robot test (reference box)

```bash
ssh linuxbox
cd ~/code/research_vla
bash scripts/lowvram.sh status   # expect applied
bash scripts/run_eval.sh --policy pi05 --task stack_bowls --eval-num 2 --seed 0 -- --num_envs 2
# in second shell:
nvidia-smi --query-gpu=memory.used --format=csv -lms 500
```

Outputs: `RoboDojo/eval_result/RoboDojo/<task>/<policy>/arx_x5/<seed>_.../_result.json` + one mp4 per camera. PASS = episode ran; check head-camera mp4 for arm motion.
