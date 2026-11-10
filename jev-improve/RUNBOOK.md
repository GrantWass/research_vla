# Runbook

All commands from `~/research_vla` on the robot PC. Training/data commands use the lerobot venv
(`~/lerobot_robot_piper/.venv/bin/python`); JEV-9B training uses `~/jev/.venv/bin/python`.
Live runs: hand on the e-stop, home the arm before every trial.

## Servers

```bash
make policy-server CKPT_HF=outputs/cube_grid_rel_lora32_2ep_20261031-1538/checkpoints/004033/pretrained_model_merged
ADAPTER=$PWD/outputs/jev_lora_dir/adapter_vllm make jev-server   # only for --judge jev|both or the JEV-9B servo
```

## Live A/B (10 trials per arm, same cube/bowl positions)

```bash
python -m real_robot.arm home --arms 1
make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 JEV=off
make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 ATTEMPTS=2 JEV_FLAGS="--recover descend"
make jev-live CKPT_HF=<ckpt> CONFIG=real_robot/configs/piper_1_clamp.json INSTR="pick up the Rubik's cube and put it in the bowl" STEPS=1200 SPEED=1 ATTEMPTS=2 JEV_FLAGS="--recover align"
```

Add `--judge both` to require JEV-9B to agree on placements; `--jev-dry` to judge without acting.
Each run writes `real_robot_logs/<stamp>-jev_*/jev_report.json` and `attemptN/judged/`.

Knobs: `--grasp-thr 0.35`, `--place-thr 0.6`, `--hover-s 3`, `--grasp-z 0.151`, `--z-margin 0.015`,
`--servo-lift 0.04`, `--align-deadband 0.015`, `--align-max 0.06`, `--align-max-spread 0.01`,
`--lite-model outputs/jevlite/model_dinos_g4_aug`, `--offset-model outputs/jevlite/offset_ens5`.

## After live runs: retrain JEV-lite on what happened

```bash
python -m real_robot.jevlite_live template --logs real_robot_logs --out real_robot_logs/live_labels.csv
# fill grasped / placed / released (y/n) per attempt
python -m real_robot.jevlite_live build --csv real_robot_logs/live_labels.csv --out outputs/jevlite/live_feats.npz
python -m real_robot.jevlite train --feats outputs/jevlite/feats_dinos_g4.npz --grid 4 \
    --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz \
    --live outputs/jevlite/live_feats.npz --out outputs/jevlite/model_dinos_g4_live
```

## Arm data collection for the offset model (~5 min, 75 poses)

Put the arm at a good grasp pose over the cube (gripper open), then:

```bash
python -m real_robot.jevlite_collect --config real_robot/configs/piper_1_clamp.json --out outputs/jevlite/collect_$(date +%Y%m%d-%H%M)
python -m real_robot.jevlite_offset train --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
    --feats outputs/jevlite/feats_dinos_g4.npz --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz \
    --collected outputs/jevlite/collect_<stamp> --members 5 --out outputs/jevlite/offset_ens5_collected
```

## Rebuild everything from the demos

```bash
python -m real_robot.jevlite_data --dataset demos/cube_grid_all_pi05 --out outputs/jevlite/frames
python -m real_robot.jevlite features --frames outputs/jevlite/frames --out outputs/jevlite/feats_dinos_g4.npz --grid 4
for s in 1 2; do python -m real_robot.jevlite features --frames outputs/jevlite/frames --out outputs/jevlite/feats_dinos_g4_aug$s.npz --grid 4 --aug-seed $s; done
python -m real_robot.jevlite train --feats outputs/jevlite/feats_dinos_g4.npz --grid 4 \
    --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz --out outputs/jevlite/model_dinos_g4_aug
python -m real_robot.jevlite_offset train --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
    --feats outputs/jevlite/feats_dinos_g4.npz --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz \
    --members 5 --out outputs/jevlite/offset_ens5
python -m real_robot.jevlite_offset train --buckets ... --out outputs/jevlite/offset_buckets
python -m real_robot.jevlite_compare --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
    --lite outputs/jevlite/model_dinos_g4_aug --out outputs/jevlite/compare.json   # needs the JEV server
python -m real_robot.jevlite eval --model outputs/jevlite/model_dinos_g4_aug --feats <other dataset feats>
```

Times on the 3090 (sharing it): frames ~6 min per 100 demos, features ~1.5 min per pass, heads seconds.

## JEV-9B 3-D bucket fine-tune

```bash
python -m real_robot.jev_lora_3d --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
    --adapter x --base x --out outputs/jev_lora_3d --prepare-only            # lerobot venv
~/jev/.venv/bin/python -m real_robot.jev_lora_3d --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
    --adapter outputs/jev_lora_dir/adapter_vllm --base ~/jev/Qwen3.5-9B --out outputs/jev_lora_3d
```

Needs the JEV vLLM server stopped (~19 GB with the other user's policy server); ~5.5 h per epoch.

## Tests

```bash
.venv/bin/python -m unittest tests.test_real_robot_jev   # ~7 min; collection test needs Pillow (lerobot venv)
```
