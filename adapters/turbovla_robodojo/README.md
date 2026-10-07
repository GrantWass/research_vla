# TurboVLA adapter for RoboDojo

XPolicyLab adapter that serves **TurboVLA**
([H-EmbodVis/TurboVLA](https://github.com/H-EmbodVis/TurboVLA): 0.2B params,
direct V+L→A, 32 Hz, under 1 GB of VRAM) in RoboDojo. It is installed by
copying it into the upstream checkout; never edit `RoboDojo/` by hand.

```bash
bash scripts/setup_policy.sh turbovla --robotwin-smoke   # adapter + env + released ckpt + DINOv3/BERT (HF token needed)
bash scripts/run_eval.sh --policy turbovla --task stack_bowls --mode smoke --fail-fast
```

**Status:** wiring is verified end to end in Isaac Sim with the released
RoboTwin checkpoint (`deploy.robotwin_smoke.yml`): 3 cameras, 14-D dual-arm
state and actions, both arms move. That checkpoint is for a different robot, so
it can't score. Real results need a checkpoint trained on RoboDojo data. Our
2,000-step fine-tune was far undertrained (`docs/findings.md`). The training
recipe was removed from this repo; it is in git history before the cleanup
branch and in upstream `turbovla/`.

| Checkpoint | Behavior |
|---|---|
| RoboDojo-trained run dir | full eval path (`action_layout: packed`) |
| RoboTwin `.safetensors` (released) | loads: legacy names remapped, `action_layout: arms_first` reorders `[arm_0, arm_1, ee_0, ee_1]`. Smoke only |
| LIBERO `.pth` | fails fast on the state-dim mismatch, which is correct |
| 7-D single-arm + `dual_arm_mode: first_arm` | drives arm 0, holds arm 1. Smoke only |

| File | Role |
|---|---|
| `model.py` | checkpoint load (EMA-aware, legacy key remap), N-view input, DINOv3 preprocessing, normalization, chunk inference, unpacking to env dicts |
| `deploy.yml` | every knob (paths, `num_views`, `chunk_size`, dims, `action_layout`, `dual_arm_mode`); null = env var / default |
| `deploy.robotwin_smoke.yml` | wiring-smoke config for the released checkpoint |
| `deploy.py`, `eval.sh`, `setup_eval_*.sh` | standard XPolicyLab episode loop and launchers |
| `install.sh` | `turbovla-robodojo` conda env + released checkpoints |
