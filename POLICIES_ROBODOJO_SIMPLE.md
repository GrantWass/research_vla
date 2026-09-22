# TurboVLA + pi0.5 on RoboDojo — Simple Overview

## The goal in one sentence
Let RoboDojo (the robot test course) swap brains with one flag: OpenVLA, TurboVLA, or pi0.5.

## The pieces
- **RoboDojo** = the obstacle course. 54 tabletop tasks (stack bowls, push things) running in a physics simulator. It only runs on a Linux machine with an NVIDIA GPU.
- **TurboVLA** = a small, fast robot brain (0.2B params). It looks at camera images + a typed command and outputs arm movements, about 32 times per second.
- **pi0.5** = a bigger, generalist robot brain from Physical Intelligence. Built for open-world generalization — handling new objects and scenes it wasn't explicitly trained on.
- **OpenVLA** = the bigger, slower baseline brain (7B). We keep it so we can compare.

## Why they didn't just plug together
Three mismatches, shared by both brains:
1. **Different languages.** RoboDojo speaks "XPolicyLab policy server"; the brains speak their own training scripts. Someone has to translate.
2. **Different bodies.** Released weights were trained on other robots. Our course uses a dual-arm setup with 14 numbers for joints + grippers and 3 cameras. Plug the wrong weights in and the sizes don't line up.
3. **Different machines.** The fast GPU we had is a Windows PC; the simulator needs Linux. So the brain and the course can live on different computers and talk over the network.

The key difference is *who wrote the translator*:
- **TurboVLA: we wrote it.** No RoboDojo adapter existed, so we built one from scratch.
- **pi0.5: it came in the box.** RoboDojo's XPolicyLab already ships a `Pi_05` adapter (with the openpi code vendored inside), so there was nothing to translate — we just had to point at it.

## What we built (the glue)
1. **Pinned the ingredients** (`setup.sh`). Froze TurboVLA and RoboDojo (plus its XPolicyLab submodule, which contains the pi0.5 adapter) at known-good versions so results are repeatable.
2. **A translator — TurboVLA only** (`adapters/turbovla_robodojo/`). A small program that:
   - loads TurboVLA weights,
   - takes RoboDojo's camera + joint readings,
   - runs the brain,
   - hands back arm movements RoboDojo understands.
   If the sizes don't match, it stops with a clear error instead of silently faking results.
   pi0.5 skips this step entirely — its translator already lives at `RoboDojo/XPolicyLab/policy/Pi_05`.
3. **Name tags** (`policies/turbovla.conf`, `policies/pi05.conf`). One small file per brain that says where it lives, which translator to use, and what the defaults are. TurboVLA's tag points at our adapter + a conda env; pi0.5's tag points at the upstream adapter + a `uv` env (a different Python toolbox — no `conda activate`, just `source openpi/.venv/bin/activate`).
4. **One steering wheel** (`scripts/run_eval.sh`). One command runs any brain:
   `run_eval.sh --policy turbovla --task stack_bowls` vs `--policy pi05` vs `--policy openvla`. A `--dry-run` flag just prints what *would* run, so it works on a Mac with no GPU.
5. **Training recipes — one per brain, in different places.**
   - TurboVLA (`templates/turbovla_finetune/`): since no ready-made weights fit our course, we wrote the recipe ourselves — reuse RoboDojo's demo data as-is (no reformatting), train ~55k steps, compute normalization stats, point the translator at the new weights.
   - pi0.5 (inside the upstream adapter): no new recipe — run the two scripts it ships with, `process_data.sh` then `train.sh`, and you get a `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/` folder to point at.
6. **A Windows serving script — TurboVLA only** (`scripts/serve_turbovla_windows.ps1`). One command starts the TurboVLA brain on the Windows GPU PC; the Linux simulator dials it over Tailscale. It also works around two Windows-only annoyances (a broken PATH entry and a missing `python3` command). pi0.5 uses the standard server/client flags instead of a custom script.

## What "ready" means today
- ✅ Any brain can be selected with one flag; the command resolves correctly (covered by 54 automated tests).
- ✅ The TurboVLA translator, both name tags, both training recipes, and the Windows launcher all exist and are documented.
- ✅ pi0.5 needs no install step (`install_adapter.sh` is a no-op for it) — just install its `uv` env on the GPU box and train.
- ⏳ Full proof still needs: download gated weight files (DINOv3 for TurboVLA), run the training recipe on a GPU, then run a real scored rollout (sim on Linux + brain on Windows/Linux).

## The one-line mental model
RoboDojo owns the course, TurboVLA and pi0.5 own the brains, and this repo owns the glue: a custom translator for TurboVLA, a name tag + steering wheel shared by both, and a recipe to teach each brain our course.
