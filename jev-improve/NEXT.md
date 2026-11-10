# Next steps, in order

1. **Read the JEV-9B 3-D results** (`outputs/jev_lora_3d/train_report.json`). If sideways direction stays
   poor, it is the same coverage gap as JEV-lite: wait for collected data (step 3).
2. **Live A/B, 10 trials each** (RUNBOOK): `JEV=off` vs `--recover descend` vs `--recover align`, checkpoint
   `004033`, config `piper_1_clamp.json`, homing before each trial. Record the real outcome per attempt.
   This is the first live test of JEV-lite in the loop, the hover rule and align.
3. **Arm collection** (`jevlite_collect`, ~5 min per cube position, 3-5 positions): fills the overshoot /
   sideways buckets. Retrain the offset ensemble and buckets with `--collected`; re-run JEV-9B 3-D with the
   collected frames added.
4. **Flywheel**: label the A/B attempts (`jevlite_live template/build`), retrain JEV-lite with `--live`,
   compare on the next session.
5. **Per-rig data**: when the camera mount or scene changes, record ~50 demos; train on all sessions
   (`model_2sess_g4` recipe).
6. **Real multi-object demos** (full-length, not stubs) to measure and train cross-object judging.
7. Optional: serve the JEV-9B 3-D adapter (strip `lm_head.base_layer` like `adapter_vllm`) and add
   `--offset-model jev` so align can ask JEV-9B directly; compare to JEV-lite closed-loop.
