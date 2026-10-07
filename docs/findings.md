# Findings

What we learned, and the evidence behind it. Sim results are on RoboDojo
`stack_bowls`, `env_cfg=arx_x5`, joint actions, seed 0.

## Policy comparison in sim

| Policy | Checkpoint | Trained on | Success (20 eps) |
|---|---|---|---:|
| pi0.5 3B | official, bf16 | RoboDojo, all tasks | **10/20** |
| OpenVLA-OFT 7B | official, 4-bit | RoboDojo, all tasks | 0/20 |
| TurboVLA 0.2B | released | RoboTwin (different robot) | 0/20 |
| TurboVLA 0.2B | ours, 2,000 steps | `stack_bowls` only | 0/20 |

This matches the RoboDojo paper (arXiv:2607.04434, Table 1): pi0.5 is the best
policy at 6.91% success over the whole benchmark, and OpenVLA-OFT scores 0.02%.

## A 0/N score needs an offline check before you trust it

Score predicted actions against the demo's ground truth on held-out episodes
(MAE), next to a **hold-still baseline** and with the **cameras blacked out**.
That separates three cases: the model never learned the task, the model
ignores vision, or the deployment path is wrong.

**OpenVLA-OFT ignores the cameras.** It beats hold-still (MAE 0.091 vs 0.135),
but blacking out every camera (0.088) or swapping the wrists (0.086) doesn't
make it worse. It predicts from proprioception and the instruction, which is
enough for a low MAE but not enough to find a bowl. pi0.5 reaches 50% through
the same camera pipeline, so the pipeline is fine. 4-bit and 8-bit agree to
within 0.001, so this isn't a quantization artifact. The input mapping was
checked separately against the checkpoint's own `dataset_statistics.json`:
grippers at indices 6 and 13 of a 14-D state, with `cam_high` as the primary
image.

**Our TurboVLA fine-tune is undertrained, not miswired.** 2,000 steps at
batch 16 is 0.71 epochs, about 1.2% of the recipe's 55k steps × batch 48. It
is worse than doing nothing (0.339 vs hold-still 0.212), but it does use the
cameras (0.434 blacked out). Between step 1,000 and step 2,000 the EMA weights
improved by about 0.008 MAE, while the raw weights got worse. That pattern
means a run stopped early in optimization, not one that converged badly.

## Sim pitfalls that invalidated earlier numbers

- **Antialiasing off** made every camera frame speckled. For a vision policy
  that is a distribution shift, not a cosmetic setting. Keep DLAA on.
- **Unequal parallel envs:** pi0.5's adapter sets `eval_batch: true`, so it
  ran 10 sims while the others ran 1. `num_envs` is pinned to 1.
- **Quantizing OpenVLA's vision backbone** makes it blind: silently in 8-bit,
  fatally in 4-bit. Quantize the LLM only (`llm_int8_skip_modules`).
- **Reading "the newest result directory"** while a battery is still running
  picks the wrong run. Pin exact result paths.

## Multi-robot serving

Batching robots onto one pi0.5 server is outcome-neutral and cheaper per
robot, but in RoboDojo the GPU is about 2% busy, so it doesn't change
wall-clock time. Details: `docs/serving.md`.

## Real robot (Piper)

- No public policy drives a Piper well out of the box. A useful policy needs
  our own teleop demos and a fine-tune (`docs/robot.md`).
- Frames must be the newest ones. Reading the camera on demand returned frames
  queued during inference, and after a 17° base turn the policy still saw the
  old pose. The camera readers now drain each camera in a background thread.
