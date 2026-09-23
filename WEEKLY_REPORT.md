# Weekly Report — VLA Policies on RoboDojo

**Date:** 2026-09-23 · **Author:** Grant Wasserman

Three vision-language-action policies now run on one 16 GB GPU box against a shared
RoboDojo task, driven by a single flag. The comparison finally measures what it
claims to: two confounds that invalidated every earlier number were found and
removed. **pi0.5 reaches 50% success; OpenVLA-OFT and both TurboVLA checkpoints
score zero** — and we established *why* each zero happens rather than assuming.

---

## 1. Results

Task `stack_bowls`, ARX X5 dual-arm robot, joint actions, seed 0, 20 episodes each,
run sequentially on the same machine with identical rendering.

| Policy | Size | Checkpoint | Trained on | Success | Score |
|---|---|---|---|---:|---:|
| pi0.5 | 3B | official RoboDojo | RoboDojo, all tasks | **10/20 (50%)** | 56.8 |
| OpenVLA-OFT | 7B | official RoboDojo, 4-bit | RoboDojo, all tasks | 0/20 | 0.0 |
| TurboVLA | 0.2B | released RoboTwin | RoboTwin, different robot | 0/20 | 0.0 |
| TurboVLA (ours) | 0.2B | fine-tuned, step 2000 | `stack_bowls` only | 0/20 | 0.0 |

The last row is a **specialist**, not a peer of the others: fine-tuned on this one
task, while the first three are generalists tested on one of many tasks they cover.

RoboDojo awards partial credit for sub-goals, so the score column carries
information the success column does not. pi0.5 accumulated 56.8 points; the three
zero rows scored **exactly 0.0 on every episode**, meaning the arms never reached
the bowls at all.

### The two confounds (why earlier numbers were withdrawn)

Both lived in shared simulator config — not in any model's code — and neither
produced an error.

1. **Antialiasing was off, so every camera frame was speckled.** It saved 0.37 GB.
   But that noise is a distribution shift for a vision policy, not a cosmetic
   setting: every model was acting on images unlike its training data.
2. **pi0.5 alone ran ten parallel simulations.** Its adapter sets
   `eval_batch: true`, which the shared `num_envs: 10` honored, while the other
   three ran one environment. One model had 10× the environment throughput of its
   competitors. `num_envs` is now pinned to 1.

Every number produced before these fixes is withdrawn, including a 5-episode table
already written into the repo docs.

---

## 2. Can we correctly map inputs into the simulator?

This was the primary question, and the answer is **yes for OpenVLA, verified two
ways** — one static, one empirical.

### Proprioception: verified against the checkpoint's own statistics

The checkpoint ships `dataset_statistics.json` describing the 14-D proprio vector
it was trained on. Its gripper values sit at indices 6 and 13, which is exactly
RoboDojo's packed layout `[arm_0(6), ee_0(1), arm_1(6), ee_1(1)]`. Comparing what
we feed against what it expects, element by element:

| | idx 0–5 (left arm) | idx 6 (left grip) | idx 7–12 (right arm) | idx 13 (right grip) |
|---|---|---:|---|---:|
| Checkpoint expects | −0.20, 0.925, 0.687, −0.34, 0.066, 0.004 | 0.772 | 0.173, 0.822, 0.619, −0.36, −0.063, 0.004 | 0.786 |
| We feed | −0.25, 1.039, 0.764, −0.35, 0.052, −0.067 | 0.694 | 0.188, 0.865, 0.677, −0.46, −0.076, 0.105 | 0.700 |

Same signs, same magnitudes, grippers in the right slots. (Small differences are
expected: the checkpoint's stats cover all tasks, ours cover one.) **No arm swap,
no permutation, no unit mismatch.**

### Images: correct order, and independently proven usable

The adapter maps `cam_high → full_image`, `cam_left_wrist → left_wrist_image`,
`cam_right_wrist → right_wrist_image`, which is the ALOHA convention the checkpoint
(`...3img--proprio--film...`) was trained with. Upstream assembles wrist images by
dict insertion order, and our adapter emits left before right.

The strongest evidence is indirect but decisive: **pi0.5 succeeds 50% of the time
through the same simulator, the same camera pipeline, and the same harness.** The
images reaching policies are good. A rendering or camera-plumbing fault would sink
pi0.5 too.

### Action decoding

Predicted actions are unpacked back into the env's dict format
(`left_arm_joint_state`, `left_ee_joint_state`, `right_arm_joint_state`,
`right_ee_joint_state`) through RoboDojo's own `unpack_robot_state`, so the return
path uses the benchmark's function rather than our own re-implementation.

---

## 3. Why OpenVLA-OFT never succeeds on an official checkpoint

Short answer: **the checkpoint has learned a largely vision-independent policy.**
It predicts plausible motion from proprioception and the instruction, but does not
use the cameras enough to find the bowls.

`scripts/diag_openvla_obs.py` scores predicted vs ground-truth actions on real
recorded episodes under observation variants — no simulator involved
(5 episodes, 10 windows, 25-step action chunks):

| Variant | MAE, 4-bit | MAE, 8-bit |
|---|---:|---:|
| as-wired | 0.0913 | 0.0906 |
| wrists swapped | 0.0856 | 0.0850 |
| head camera only | 0.0861 | 0.0856 |
| head blacked out | 0.0938 | 0.0929 |
| **all cameras black** | **0.0882** | **0.0875** |
| hold still (do-nothing baseline) | 0.1349 | 0.1349 |

Read it in two steps:

- **It beats hold-still** (0.091 vs 0.135), so it is predicting real motion, not
  freezing. The model works.
- **Blacking out every camera does not make it worse** (0.088 vs 0.091 — very
  slightly *better*). Neither does swapping the wrist cameras. Whatever it is
  predicting from, it is not mainly the images.

A policy like this can score a respectable open-loop action MAE, because much of a
demonstration trajectory is predictable from the current joint configuration. It
cannot stack bowls, because that requires knowing where the bowls actually are.
Hence 0.0 partial credit on all 20 episodes: the arms never arrive.

**This matches the published result.** From Table 1 of the RoboDojo paper
([arXiv:2607.04434](https://arxiv.org/abs/2607.04434)), captioned:

> "RoboDojo Simulation Benchmark Leaderboard. Each cell reports score / success
> rate for a capability dimension."

| Policy | Score / success rate (average) |
|---|---|
| OpenVLA-OFT | **0.21 / 0.02%** |
| pi0.5 (best policy in the table) | 11.41 / 6.91% |
| Human expert (teleop) | 80.42 / 76.03% |

OpenVLA-OFT succeeding roughly **2 times in 10,000** is the published behavior of
this checkpoint. Our 0/20 is exactly what that predicts, and is not a setup fault.
Note also that even the *best* policy in the paper reaches under 7% success against
76% for a human — this benchmark is hard by design.

**It is not a quantization artifact.** 4-bit and 8-bit agree to within ~0.001 MAE.

### Two honest caveats

- **This MAE test cannot validate camera mapping.** Because the checkpoint is
  camera-insensitive, a wrong camera assignment would look identical to a right
  one. The proprio comparison in §2 is what establishes the input mapping — not
  this table. *(An earlier note in this repo over-read these numbers as proof the
  observation path was correct; that claim has been corrected.)*
- **The unquantized control is still missing.** bf16 needs ~15.4 GB and will not
  fit alongside the display on a 16 GB card. We have ruled out 4-bit vs 8-bit as
  the cause, but not quantization in general.

### The bug this method caught

Running OpenVLA in 8-bit, blacking out every camera changed the predicted actions
by at most **0.02** — the vision backbone was being quantized into uselessness and
the policy was acting fully blind. Excluding it from quantization fixed it.

That bug is the argument for the whole method: a blind vision backbone produces no
error, no warning, and a plausible-looking score. It is indistinguishable from "this
model is weak" unless something explicitly tests whether the cameras matter.

---

## 4. How we set up training / fine-tuning

We fine-tuned the 0.2B TurboVLA on `stack_bowls` alone to test a small specialist
against large generalists.

**Data.** RoboDojo ships one combined LeRobot dataset covering every task —
3,500 episodes across 35 tasks, 1,856,102 frames, **117 GB** of LFS objects.
`templates/turbovla_finetune/make_task_dataset.py` carves a single task out of it
without copying frame data: it symlinks the data and video files that task's
episodes reference and rewrites only the metadata, so only that task's LFS objects
need fetching. This took the requirement from **117 GB to 3.1 GB**
(100 episodes, 44,774 frames). It also
writes the GR00T `modality.json` the trainer needs, and refuses to run on 16-D
end-effector data rather than silently mislabeling it as 14-D joints.

**Normalization.** `compute_stats.py` computes per-dimension statistics with pyarrow
only (dropping a heavyweight `lerobot` dependency).

**Initialization.** Rather than starting from GroundingDINO as the stock recipe does,
`patches/turbovla_full_ckpt_init.patch` adds `TURBOVLA_INIT_FULL=1`, which initializes
from a released TurboVLA checkpoint — remapping legacy key names and refusing to
proceed below 80% tensor match. **878 of 878 tensors loaded.** The guard matters: a
silent partial match looks like slow convergence, not an error.

**Two upstream bugs had to be fixed to train at all:**

- The GR00T LeRobot loader built video paths from the **data** file index, so any
  dataset whose videos aren't numbered like its parquet shards read the wrong
  episode's frames (`patches/turbovla_lerobot_video_index.patch`).
- `train.sh` copied itself into the run directory *after* `cd`-ing elsewhere, so the
  provenance copy was never made.

**Deployment.** `scripts/install_turbovla_deploy.sh` renders the deploy config from a
repo template with real checkpoint/stats paths, instead of hand-editing the installed
file.

### Result: undertrained, not miswired

The run reached 2,000 steps at effective batch 16 before the host's OOM killer
stopped it during a checkpoint save.

| | Our run | Recipe default |
|---|---:|---:|
| Steps | 2,000 | 55,000 |
| Effective batch | 16 | 48 |
| Samples seen | 32,000 | 2,640,000 |

Against 44,774 frames, 32,000 samples is **0.71 epochs** — never one full pass — and
only 1,000 steps past the 1,000-step warmup. That is **~1.2% of the intended
training**.

`scripts/diag_turbovla_actions.py` confirms the diagnosis rather than inferring it:

| Variant | MAE vs ground truth |
|---|---:|
| real cameras | 0.339 |
| cameras blacked out | 0.434 |
| **hold still (do nothing)** | **0.212** |

The model is **worse than doing nothing**, so the weights have not learned the task.
But its predictions *do* respond to the cameras (0.339 < 0.434), so the observation
pipeline is wired correctly. **Undertrained, not miswired** — two causes that look
identical from the 0/20 alone.

### Is it improving at all?

Yes — just far too slowly to show up as a success.
`scripts/diag_turbovla_step_sweep.py` scores the same windows at both saved steps:

| Checkpoint | MAE |
|---|---:|
| step 1000, EMA | 0.3476 |
| **step 2000, EMA** | **0.3394** |
| step 1000, raw weights | 0.3907 |
| step 2000, raw weights | 0.4146 |
| hold still (baseline) | 0.2116 |

Two things to read here.

**The EMA weights are improving, at about 0.008 MAE per 1,000 steps.** The direction
is right and the optimization is working. But the gap still to close just to match
*doing nothing* is 0.34 − 0.21 = 0.13. At the observed rate that is on the order of
10⁴ more steps — the same order as the recipe's own 55,000. (A crude linear
extrapolation; real learning curves decelerate, so treat it as a rough floor, and a
larger batch would cover the same ground in fewer steps.)

**The raw weights got *worse* from step 1000 to 2000** (0.391 → 0.415) and are
consistently worse than the EMA copy. That is the signature of a model still early
in optimization, where the instantaneous weights bounce around and the moving
average is what carries the progress. It is further evidence that the run stopped in
the unstable early phase rather than converging to a bad solution.

So the honest answer to "why did it not improve" is that **it did improve, just
barely**: MAE fell 2.4% in relative terms over those 1,000 steps, which is about 6%
of the distance from where it started to merely matching the do-nothing baseline.

---

## 5. How we got each model working on RoboDojo

All four run through one entry point: `bash scripts/run_eval.sh --policy <name> --task <task>`.

| Policy | What it took |
|---|---|
| **demo** | Zero-action stub. Proves the harness, isolates wiring bugs from model bugs. |
| **pi0.5** (3B, JAX) | Official params-only checkpoint. Base conda env lacked PyYAML (the launcher reads `deploy.yml` with it); JAX `RESOURCE_EXHAUSTED` on restore until the memory fraction was made overridable. |
| **OpenVLA-OFT** (7B) | Env name is `openvla_oft`, not `openvla-oft`. `flash-attn` needs `CUDA_HOME`, so a prebuilt wheel is pinned. numpy 2 / opencv 5 / protobuf 5 / accelerate 1.x conflicts pinned. Two upstream loader bugs fixed: the FiLM backbone was left on CPU, and 4-bit packing broke its `state_dict`. |
| **TurboVLA** (0.2B) | Adapter written in-repo (`adapters/turbovla_robodojo/`): upstream shipped no `deploy.py`. Handles safetensors, legacy key remapping, 3 camera views, dual-arm denormalization, and both action layouts (RoboDojo packed vs RoboTwin arms-first). RoboDojo-trained checkpoints also carry a `model.` key prefix the released ones don't. |

**Cross-cutting fixes:**

- The NVIDIA driver must stay at **580**; Isaac Sim 5.1 segfaults in
  `librtx.scenedb.plugin.so` on 595.
- RoboDojo's `install.sh` submodule step uses `--remote`, which drifts off the pins.
- Launching an eval non-interactively (over SSH / `nohup`) failed four ways, each now
  fixed in `run_eval.sh`: conda not on PATH for the policy server; the eval *client*
  calls bare `python` and activates no env; Isaac Sim's EULA prompt aborts without a
  TTY; and the deployed config was being reverted (below).

**A live hazard worth flagging.** Running `make test` silently reverted the deployed
model config to the stock released weights — the suite force-syncs the adapter, and
`deploy.yml` sits among the adapter's code. Confirmed from the file's mtime moving
mid-evaluation. This run survived only because the policy server had already loaded
the model; **an eval launched minutes later would have reported numbers for the wrong
checkpoint, with nothing in the log to say so.** Config is now protected from that
sync, and the protection is itself tested.

---

## 6. Adapting to 16 GB VRAM

The box is an RTX 4070 Ti SUPER (16 GB) with 32 GB system RAM. A 7B policy *and*
Isaac Sim must share the card. `bash scripts/lowvram.sh apply` is the reversible
profile.

| Measure | Effect |
|---|---|
| Sim render profile: no reflections, no global illumination, no translucency, texture budget capped | Largest single saving |
| **Antialiasing kept ON (DLAA)** | Costs 0.37 GB — non-negotiable, see §1 |
| `num_envs: 1`, PhysX buffers sized to match | Saves memory *and* makes the comparison fair |
| OpenVLA in 4-bit NF4, vision backbone and projector excluded from quantization | 15.2 GB → 9.2 GB |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` (pi0.5/JAX) | 0.3 too small to restore; 0.55 caused Vulkan OOM in the sim |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces fragmentation |

Measured peak usage (policy + sim): demo 5.7 GB · TurboVLA 7.5 GB · pi0.5 14.4 GB ·
OpenVLA 14.9 GB. These were taken *before* antialiasing was restored, so add roughly
0.4 GB to each for the current profile — which is what makes OpenVLA the tight one.

### Limitations this imposes

- **OpenVLA runs 4-bit, not bf16.** Quantization changes its outputs. The card
  exposes 15.55 GiB usable; loading bf16 got to 15.11 GiB and then failed asking for
  another 252 MiB, with the display manager holding part of the remainder. So the
  unquantized control needs a bigger GPU or a headless box, not a tweak.
- **Reduced rendering**, though antialiasing is restored after learning what turning
  it off cost.
- **Training is RAM-bound, not just VRAM-bound.** Batch 8 OOMs the GPU (we use 4 ×
  grad-accum 4), and the 32 GB of system RAM is what actually killed the training run
  — the OOM killer struck during a checkpoint save, at ~1.7 GB per checkpoint plus
  DeepSpeed ZeRO-2 state.
- **These are not benchmark numbers.** One task, one seed, 20 episodes, reduced
  rendering, quantized OpenVLA. The pi0.5-vs-rest gap is large enough to survive that;
  the ordering of the three zeros is not meaningful.

---

## 7. Next steps

Ranked by what changes a conclusion rather than confirming one.

1. **Train TurboVLA properly — highest value.** 20,000 steps ≈ 4.5 hours on this box,
   reaching ~7 epochs. Only then does the specialist-vs-generalist comparison mean
   anything. Requires fixing the checkpoint-save RAM ceiling first (save less often,
   or offload optimizer state).
2. **Get the unquantized OpenVLA control.** bf16 overran the card by ~250 MB, so
   going headless *might* just fit it — but it is marginal enough that the reliable
   route is running the policy server on a larger GPU
   (`robodojo.sh server --bind-host 0.0.0.0`). This closes the one open question in
   §3 — whether quantization contributes to the camera-insensitivity. Note the
   diagnostic needs no simulator, so any 24 GB machine can answer it.
3. **Probe *why* OpenVLA ignores its cameras.** The FiLM conditioning path is the
   prime suspect. If the released checkpoint genuinely learned a vision-independent
   policy, that is a finding worth writing up on its own.
4. **Add tasks and seeds** to turn a demonstration into a measurement. Screening
   suggested `stack_bowls` is on the easier end; a 3-task × 3-seed grid would give
   error bars.
5. **Operational:** rotate the HuggingFace token (it was pasted into a chat log), and
   finish the Windows-side scheduled task that keeps Ubuntu first in the boot order.

---

## Appendix: reproducibility

Everything above is in the repository; a fix that exists only on one machine does not
count. The suite is 72 cases (**71 pass, 1 skipped**, plus 30 subtests): it runs on a
laptop with no GPU, and is additionally run on the GPU box against the real upstream
checkouts, since the patch-application tests are vacuous without them.

- `GPU_BOX_SETUP.md` — fresh Linux → driver → Isaac Sim → all four policies, with
  measured VRAM and troubleshooting for every failure hit.
- `scripts/setup_policy.sh` — one idempotent command per policy.
- `scripts/lowvram.sh` — `apply` / `revert` / `status` for the 16 GB profile.
- `patches/` — five upstream fixes, each with its cause recorded; tests check they
  still apply to the pinned checkouts.
- `scripts/diag_openvla_obs.py`, `scripts/diag_turbovla_actions.py`,
  `scripts/diag_turbovla_step_sweep.py` — offline action diagnostics, no simulator
  required.
- `templates/turbovla_finetune/` — dataset carving, stats, training recipe.
