# jev-improve: JEV around pi0.5 on the Piper arm

Work log and results for turning JEV (autotrust JEV-9B, a vision-language "verifier" model) from a
proof of concept into a working layer around our pi0.5 policy on the Piper arm. Task throughout:
*"pick up the Rubik's cube and put it in the bowl"*. Dates: 2026-11-04 to 2026-11-10.

Code lives in `real_robot/` on branch `robot/jev-pipeline` (robot PC checkout `~/research_vla`).
`docs/jev.md` is the running runbook; this folder is the write-up.

| file | what |
|---|---|
| [README.md](README.md) | this page: the story, headline numbers, status |
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the pieces fit, every module and flag |
| [RESULTS.md](RESULTS.md) | every experiment with its numbers, including what failed or was withdrawn |
| [RUNBOOK.md](RUNBOOK.md) | commands: live A/B, data collection, retraining, servers |
| [LESSONS.md](LESSONS.md) | what went wrong live and in the data, and what it taught |
| [NEXT.md](NEXT.md) | what to do next, in order |

## The story in one page

1. **The problem.** The pi0.5 LoRA fine-tune went 20/20 live, leaving nothing for a verifier to
   catch. We trained deliberately weak checkpoints. Live, the useful one (`2ep/004033`) lines up over
   the cube and then **hovers without closing**; `weak/001210` is about as good as holding still.
2. **JEV-9B as verifier + servo (first pass).** Yes/no checks ("holding?", "in the bowl?") at gripper
   events, and a direction-bucket servo with a fine-tuned JEV adapter (96% offline). Live, the servo
   failed: left/right never converged and up/down drove the arm into the table (the controller
   faulted, the arm dropped). Added a workspace guard. A simple **`--recover descend`** (straight down,
   close, check) **grasped on its first live try**. Then JEV-9B scored the cube **held over the bowl,
   never released, as placed** (P 0.82).
3. **JEV-lite.** A 25M-parameter judge (frozen DINOv2-small + MLP) trained on phase labels that come
   free from the demos' gripper traces. On held-out frames it never calls a held-over-bowl cube placed
   (JEV-9B: 35/60), is more accurate on every check, and runs in ~15-26 ms instead of ~170 ms per
   question, so it can watch every step. It is now the default judge.
4. **Monitor rules.** Hover = gripper open and arm still for 3 s (catches all 7 live hovers in replay,
   no false triggers); never released is never success; every judgement saves its frames for retraining.
5. **Where to grasp, how far: 3-D buckets.** A model on the same features predicts the offset to where
   the demo grasped (reach, sideways, height), as a regression ensemble and as JEV-style 7-bucket
   answers per axis. One correction takes the gripper from ~4-5 cm to ~1-1.7 cm off. `--recover align`
   uses it. The demos never overshoot or sidestep, so some buckets have no data: an arm collection
   script fills them.
6. **JEV-9B taught the 3-D buckets** (`jev_lora_3d.py`): zero-shot 22-61% bucket accuracy; after a
   third of an epoch 69-73%, direction 99-100% on reach and height, still poor sideways (training).
7. **Transfer.** A second recording session with a different wrist-camera mount drops JEV-lite to 70%;
   training on both sessions brings it to 87%. Cross-object transfer can't be measured: the on-disk
   "general" datasets' other objects are 20-frame stubs, and their full episodes are the cube demos
   duplicated under other instructions (which also leaked test data until the split was fixed).

## Headline numbers

| | before | now |
|---|---|---|
| cube held over the bowl, judged "placed" | JEV-9B 35/60 (live: P 0.82) | JEV-lite **0/60** |
| place check accuracy (held-out) | 81.7% | **98.3%** |
| grasp check accuracy (held-out) | 74.6% | **96.7%** |
| judge latency | ~170 ms per question | **~26 ms** for both |
| hover caught | at the 40-100 s step limit | **21-26 s**, 7/7, 0 false |
| distance to grasp after one correction (held-out, 3-15 cm away) | — | **4.1 → 1.3 cm** |
| JEV-9B 3-D bucket direction, reach / height | 73% / 35% zero-shot | **99% / 100%** (1/3 epoch) |

## Status (2026-11-10)

- Live-tested: verifier + `--recover descend` (1 trial, grasped). Everything after that is offline
  and fake-arm tested only: JEV-lite in the loop, hover rule, `--recover align`, data collection.
- JEV-9B 3-D bucket fine-tune running on the robot PC (`outputs/jev_lora_3d/`); a watcher restarts the
  original JEV server when it ends.
- Commits on `robot/jev-pipeline`: `3467072` (JEV-9B verifier, servo, descend), `807fa8a` (JEV-lite,
  hover, never-released), `643f62a` (grasp offset, align), `1d08952` (ensemble, 3-D buckets,
  flywheel, collection), `e239c9d` (JEV-9B 3-D fine-tune, cross-dataset eval, leak-proof split).
- The robot PC is shared (another user's policy server and uncommitted edits): commits include only
  this work's hunks.
