# Guiding a weak pi0.5 by sampling and selecting (2026-11-10, offline)

Goal: make a weak checkpoint finish the task at low latency, without hand-written recoveries.
Idea: pi0.5 is a flow model, so it can propose many action chunks; a cheap selector picks one.

Setup: checkpoint `2ep/004033`. Test frames are from the 6 demos it never trained on
(episodes 7, 25, 35, 50, 70, 90; its `train_config.json` lists the rest): 170 frames, 6 at named
points around the grasp/carry/release plus every 30 frames. Per frame: 5 instructions x 4 noise
scales x 16 samples = 320 candidates. Metric: joint RMSE to the demo over the first 0.5 s
(gripper weighted 1), and whether the chunk closes the gripper when the demo does.
Results: `results/verifier_*.json`, `results/jev_lora_3d_*.json`.

## 1. JEV-9B 3-D buckets: final (stop here)

| | reach dir | sideways dir | height dir | xy after one move | latency |
|---|---|---|---|---|---|
| JEV-9B 3-D (served, fp8) | 99.5% | 75% | 100% | 5.4 -> 1.68 cm | 500 ms |
| JEV-lite `offset_ens5` | 98% | 87% | 100% | 5.4 -> 1.36 cm | 30 ms |

Same accuracy class, 17x slower. Wired as `--recover align --offset-model jev` but not the way forward.

## 2. Coverage: pi0.5 does propose the right action

- On every frame where the demo closes on the cube, at least one candidate closes too (100%, even with
  only the main instruction). The live "hover" is not a lack of good proposals.
- Best of 16 (oracle) halves the error: 0.059 -> 0.029; best of all 320: 0.022.
- Noise scale: 0.5 gives the best picks; 1.5-2.0 produce wild chunks (2.0 random error 0.24) but
  close the gripper much more often (close rate 38% at 0.5, 99% at 2.0).
- Sub-step instructions ("move down to...", "grasp...") widen coverage a little (oracle 0.029 -> 0.023)
  but no selector exploited it.

## 3. Latency: extra samples are nearly free

Run the VLM prefix once, repeat its KV cache for N noises (`pi05_samples.sample_shared`; matches the
batched path to 0.004 rad):

| samples | batched | shared prefix |
|---|---|---|
| 1 | 373 ms | 376 ms |
| 8 | 788 ms | 376 ms |
| 16 | 1321 ms | 430 ms |
| 32 | - | 722 ms |

The verifier scores all candidates in < 1 ms. Best-of-16 costs ~15% over one sample.

## 4. Selection: the bottleneck

| selector (main instr, 16 samples) | error | closes when demo closes | closes when it shouldn't |
|---|---|---|---|
| single sample | 0.059 | 66% | 26% |
| medoid (consensus), scale 0.5 | 0.049 | 55% | 17% |
| verifier v2, scale 0.5 | 0.049 | 61% | 17% |
| verifier v2, scales 0.5+1 | 0.051 | 74% | 25% |
| JEV-lite phase gate + medoid | 0.050-0.056 | 50-61% | 18-41% |
| oracle | 0.029 | - | - |

- Verifiers (`real_robot/verifier.py`): DINOv2 grid features (JEV-lite's) + 0.5 s of the chunk,
  listwise loss toward softmax(-distance/tau).
  - v1: demo-derived candidates (time shifts, hold still, noise, other episodes).
  - v2: + gripper weighted, true "hover" candidate, grasp frames x5. Prefers the demo over hovering
    78% in the grasp phase (v1 58%); ranks the demo first among 17 in 51% (chance 6%).
  - v3: trained on pi0.5's own samples (40 training episodes, scales 0.5/1): collapses on unseen
    scales (picks scale-2 chunks, closes 92% always). v3 from v2: no better than v2.
- Every selector captures ~15-18% of the error; none closes the oracle gap or fixes the grasp
  decision. The trained verifier is no better than consensus: it learned "typical", not "progress".

## 5. Progress scorer (`real_robot/progress.py`)

P(image features now, future joint+gripper state) -> task progress (demo time / length), trained on
demos only; corrupted futures score below standing still. A candidate is scored by the state it
reaches after 15 steps (0.5 s).

Ranking on unseen episodes, demo future vs alternatives:

| phase | demo > hover | demo > noisy | demo > backward |
|---|---|---|---|
| all | 86% | 94% | 90% |
| grasp | **96%** (verifier v2: 78%) | 93% | 96% |

As a selector over pi0.5 samples (main instruction, 16 at scale 0.5): error 0.055 (single 0.059),
closes when it should 68% (66%), closes when it shouldn't **18%** (26%). With wilder candidates
(scales up to 2) it picks far-out chunks that close everywhere (dist 0.15): it rewards progress
without knowing what is reachable, so it needs the consensus filter (keep the most typical half,
then max progress: 0.049-0.053, but closing gains vanish).

Reading: the scorer knows what progress looks like, but within pi0.5's typical samples the
differences in 0.5 s are small; the candidates that differ a lot are the unreliable ones.

## 6. Latency breakdown (3090, shared with another policy server)

| denoise steps | 1 sample | 16 samples |
|---|---|---|
| 10 (default) | 387 ms | 439 ms |
| 5 | 243 ms | 268 ms |
| 3 | 185 ms | 200 ms |

Preprocessing is 3 ms; weights are already bf16. Profiled: a denoise step is ~1,070 kernel
launches, 5.5 ms of GPU time but 31 ms wall, so eager pi0.5 is CPU launch-bound.

### CUDA graph of the denoise loop (`pi05_samples.GraphDenoiser`)

Captures the N-sample loop once; per call the prefix (eager, 79 ms) is copied into static buffers.
pi0.5's `embed_suffix` builds a constant mask with `torch.tensor(list, device=cuda)` each step,
which breaks capture; constants are memoised during capture. Output identical to eager (max diff 0).

| samples | eager 10 steps | graph 10 steps | graph 5 steps |
|---|---|---|---|
| 1 | 431 ms | **127 ms** | **103 ms** |
| 2 | - | 146 ms | 113 ms |
| 4 | - | 181 ms | 129 ms |
| 8 | - | 250 ms | 164 ms |
| 16 | 449 ms | 403 ms | 242 ms |

At 16 the GPU work dominates (attention re-reads the repeated prefix cache per sample). The prefix
itself did not capture (something inside it also invalidates capture) and is mostly real compute.

### Fewer denoise steps do not hurt (170 unseen frames, 8 samples each, same noise)

| steps | error to demo | closes when it should | when it shouldn't | change vs 10 steps |
|---|---|---|---|---|
| 10 | 0.0588 | 64% | 29% | - |
| 5 | 0.0561 | 66% | 29% | 0.006 |
| 3 | 0.0540 | 69% | 30% | 0.012 |
| 2 | 0.0521 | 73% | 30% | 0.019 |

Fewer steps land slightly closer to the demo (less sampling noise).

Correction: the 431 ms "single sample" above is the uncompiled path. The live server already runs
`--compile reduce-overhead` (CUDA graphs): **107 ms** at 10 steps, **82 ms** at 3. So the graph
work buys nothing for one sample; it matters for best-of-N.

### Wired in (`lerobot_policy.LerobotPolicy(best_of=N)`, `--best-of`, `make ... BEST_OF= DENOISE=`)

| live setting | per chunk |
|---|---|
| compiled, 10 steps, 1 sample (today) | 107 ms |
| compiled, 3 steps | 82 ms |
| best-of-4, 3 steps, compiled sampler re-run per sample | 250 ms |
| best-of-4, 3 steps, shared prefix + graph (used) | **128 ms** |
| best-of-8, 3 steps | 149 ms |
| best-of-4 / 8, 5 steps | 150 / 183 ms |

Selection is the medoid (consensus); tests pass (`tests.test_real_robot` 36, `test_real_robot_jev` 34).

## Takeaways

1. Proposals are good enough; the open problem is a scorer that knows which proposal makes progress.
   Imitation distance to one demo is a weak proxy (several actions are valid), and the oracle number
   is optimistic for the same reason.
2. Sampling cost is solved (shared prefix). A live best-of-16 is affordable on the 3090.
3. Do not train the scorer on a narrow slice of the candidate distribution (v3).

## Next

- Closed-loop is the real test: 3-5 live trials of best-of-16, scale 0.5, medoid or v2
  (`real_robot/guided_policy.py`, not yet wired into the runner) vs plain 004033.
- A progress scorer instead of imitation: label each frame with time-to-success from demos and failed
  live attempts, and score a candidate by the predicted progress of where it ends up (needs a small
  forward model of the end-effector/gripper state, which the arm's kinematics give for free).
- Gripper decision separately: closing is a binary choice; a dedicated "close now?" head from the
  gripper traces may beat any whole-chunk scorer.

## Files

`real_robot/pi05_samples.py` (sampler, shared prefix), `verifier.py` (train / train-samples / eval),
`select_eval.py` (selector comparison), `guided_policy.py` (runtime wrapper), `jev_offset3d.py` +
`offset3d_eval.py` (JEV-9B 3-D), `scripts/strip_adapter.py`.
