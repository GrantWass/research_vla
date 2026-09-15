# TurboVLA on RoboDojo — Simple Overview

## The goal in one sentence
Let RoboDojo (the robot test course) swap brains with one flag: OpenVLA or TurboVLA.

## The pieces
- **RoboDojo** = the obstacle course. 54 tabletop tasks (stack bowls, push things) running in a physics simulator. It only runs on a Linux machine with an NVIDIA GPU.
- **TurboVLA** = a small, fast robot brain (0.2B params). It looks at camera images + a typed command and outputs arm movements, about 32 times per second.
- **OpenVLA** = the bigger, slower baseline brain (7B). We keep it so we can compare.

## Why they didn't just plug together
Three mismatches:
1. **Different languages.** RoboDojo speaks "XPolicyLab policy server"; TurboVLA speaks its own training script. Someone had to translate.
2. **Different bodies.** TurboVLA's released weights were trained on other robots (LIBERO one-arm, RoboTwin two-arm). Our course uses a dual-arm setup with 14 numbers for joints + grippers and 3 cameras. Plug the wrong weights in and the sizes don't line up.
3. **Different machines.** The fast GPU we had is a Windows PC; the simulator needs Linux. So the brain and the course live on different computers and talk over the network.

## What we built (the glue)
1. **Pinned the ingredients** (`setup.sh`). Froze TurboVLA and RoboDojo at known-good versions so results are repeatable.
2. **A translator** (`adapters/turbovla_robodojo/`). A small program that:
   - loads TurboVLA weights,
   - takes RoboDojo's camera + joint readings,
   - runs the brain,
   - hands back arm movements RoboDojo understands.
   If the sizes don't match, it stops with a clear error instead of silently faking results.
3. **A name tag** (`policies/turbovla.conf`). One file that says where the brain lives, which translator to use, and which settings are defaults.
4. **One steering wheel** (`scripts/run_eval.sh`). One command runs either brain:
   `run_eval.sh --policy turbovla --task stack_bowls` vs `--policy openvla`. A `--dry-run` flag just prints what *would* run, so it works on a Mac with no GPU.
5. **A training recipe** (`templates/turbovla_finetune/`). Since no ready-made TurboVLA weights fit our course, we wrote the recipe to teach it: reuse RoboDojo's existing demonstration data as-is (no reformatting), train ~55k steps, compute normalization stats, then point the translator at the new weights.
6. **A Windows serving script** (`scripts/serve_turbovla_windows.ps1`). One command starts the brain on the Windows GPU PC; the Linux simulator dials it over Tailscale. It also works around two Windows-only annoyances (a broken PATH entry and a missing `python3` command).

## What "ready" means today
- ✅ Either brain can be selected with one flag; the command resolves correctly (covered by 54 automated tests).
- ✅ The translator, registry, training recipe, and Windows launcher all exist and are documented.
- ⏳ Full proof still needs: download one gated file (DINOv3 vision weights), run the training recipe on a GPU, then run a real scored rollout (sim on Linux + brain on Windows).

## The one-line mental model
RoboDojo owns the course, TurboVLA owns the brain, and this repo owns the three pieces of glue between them: the translator, the name tag, and the steering wheel.
