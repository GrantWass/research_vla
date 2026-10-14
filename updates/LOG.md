# Update log (newest first). Plain text. The visual doc is notebook.html (separate).
# One entry per paper. Dates are when we read it; paper dates are in the entry.

------------------------------ 2026-10-13 · FirmVLA ------------------------------
Paper: FirmVLA: Sequence-SLO Serving for Heterogeneous VLA Fleets (Oct 4, 2026; ODI 2026 workshop poster, UT Dallas)
Link: https://neurips.cc/virtual/2026/170025 (OpenReview: https://openreview.net/forum?id=lW9ZhfLCOK)
Read: abstract only

1. Insight: Replaces best-effort scheduling with guarantees. Each robot states an (m,k,r) SLO (served at least m of every k refills, never more than r misses in a row) and the system builds an exact repeating admission schedule with reservations.
2. Results (reported): zero SLO violations on 6 fleets of 2 to 4 policies with 14 to 122 admitted robots, and in 14 trials across start times at 50-100% load. DBP, EDF, round robin and Armory Lookahead violate at high load. Robots that keep leftover actions through missed refills succeed 79.8% vs 41.2% when they discard them (3,584 task episodes, four policies).
3. Limits: workshop poster; a precomputed schedule implies a known fleet (our reading); unknown on real robots.
4. Next: read the full paper if it appears; try "keep leftover actions on a miss" on our robot.

------------------------------ 2026-10-13 · Robion ------------------------------
Paper: Efficient Vision-Language-Action Management and Serving for Robot Factories (Sep 14, 2026)
Link: https://arxiv.org/abs/2609.12075
Read: first two thirds plus references; evaluation and conclusion not read

1. Insight: VLA stages are millisecond-scale and underuse the GPU (pi0 tensor cores: 52% VLM stage, 15% action stage at batch 4). Running the VLM and action diffusion transformer on two GPU streams in lockstep, with SM partitioning for the VLM stream, beats both a monolithic pipeline and stage-disaggregated vLLM-Omni. Earliest-SLO-first scheduling plus an integer-program traffic controller handle several models across GPUs.
2. Results (reported): 6.7x robot load vs vLLM-Omni and 1.5x vs monolithic at 98% SLO attainment. 8 models on 4 GPUs serve up to 64 robots. vLLM-Omni's per-step inspection costs 30-35% of latency. Cloud is 18-24% lower in SLO attainment than edge at 24 robots/model. About 6.8x lower hardware cost than Jetson AGX Thor on the robot. Evaluated on LIBERO.
3. Limits: limitations section not read. Does not use Armory-style chunk state. Criticism of Armory (ignores per-request SLOs, treats VLA as one pipeline) not checked against Armory.
4. Next: read its evaluation section; check whether the edge-vs-cloud gap holds at our fleet size.

------------------------------ 2026-10-13 · Armory ------------------------------
Paper: Action Chunk Scheduling for Batched Robot Policy Serving (Aug 2026)
Link: https://arxiv.org/abs/2608.00337
Read: full text via summary

1. Insight: When robots consume action chunks at different rates, naive batching heuristics fail. Urgency depends on each robot's remaining action buffer. The Lookahead scheduler simulates forward over a server-side mirror of each robot's executed action, received action, and queued chunks.
2. Results (reported): up to 18% higher system throughput in real-world runs; about 2x fast-tier throughput over EDF with one fast robot. pi0.5 on one L40S with 10 AgileX PiPER arms. pi0.5 takes about 73 ms at batch 1 and 211 ms at batch 5. Practical up to 50 ms one-way delay.
3. Limits: plain batching on one GPU only (no continuous batching, PagedAttention, distributed inference). Real-world results rest on 3 seeds and 1-minute rollouts. Lookahead depth 2 or 3 did not help. Stale chunks hurt dynamic manipulation more than quasi-static tasks.
4. Next: check whether our server-side batching has an equivalent of time-until-the-robot-runs-dry.

------------------------------ 2026-10-13 · ROSA ------------------------------
Paper: ROSA: A Robotics Foundation Model Serving System for Robot Factories (Jul 2026)
Link: https://arxiv.org/abs/2607.01088
Read: full text via summary

1. Insight: Serving a factory means a shared GPU pool, multi-model pipelines (action, planner, safety, monitor) and a scheduler that maximizes productivity under SLOs, not per-request latency. ILP packing, profiling-guided batching, request-rate capping, failover.
2. Results (reported): up to 12.06x factory productivity vs the best dedicated baseline; 2.44x SLO-qualified throughput vs shared baselines; request-rate capping 3.18x at 64 robots; batch 16 gives about 3-4x the throughput of batch 1. 8 H200 GPUs, up to 64 virtual robots replaying real observations, GR00T N1.6 and pi0.5, plus one real Franka Panda.
3. Limits: assumes a known, static, profiled workload and synchronous execution (also Armory's critique). Scale results are synthetic. Needs a stable wired or Wi-Fi 7 network.
4. Next: look at its request-rate capping; test whether capping helps at our scale.

------------------------------ 2026-10-13 · vla.cpp ------------------------------
Paper: vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models (Jun 6, 2026)
Link: https://arxiv.org/abs/2606.08094
Read: full text via summary

1. Insight: A portable C++ runtime on ggml/llama.cpp can run many VLA architectures with per-model adapters and a numerical check against reference actions. Flow-matching policies cache conditioning across solver steps within a request. Batch-1 inference is compute-bound.
2. Results (reported): 11 models supported (incl. OpenVLA-OFT, pi0, pi0.5, GR00T, SmolVLA, BitVLA). 10 of 11 within 3.2 points of reference on LIBERO-Object (GR00T-N1.6: 88% vs 99%, cause unresolved). Tested on RTX 3060/3090/4070/5070, Jetson Orin, Apple M4. 1.02x to 6.21x faster than eager PyTorch. BitVLA 200/200 on an 8 GB Orin Nano. SmolVLA on Orin Nano: 49/50 executing 16 actions per chunk vs 19/50 executing 50.
3. Limits: no batching and no multi-robot serving; synchronous requests only. Robot studies are small and quasi-static.
4. Next: check whether it is worth running on our Piper PC 3090; use the execution-horizon result to design our staleness sweep on stack_bowls.
