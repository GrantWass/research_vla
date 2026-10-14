# Update log (newest first). Plain text. The visual doc is notebook.html (separate).
# One entry per paper. Dates are when we read it; paper dates are in the entry.

------------------------------ 2026-10-13 · FirmVLA ------------------------------
Paper: FirmVLA: Sequence-SLO Serving for Heterogeneous VLA Fleets (Oct 4, 2026, UT Dallas)
Link: https://openreview.net/forum?id=lW9ZhfLCOK

1. Insight: Replaces best-effort scheduling with guarantees. Each robot states an (m,k,r) SLO (served at least m of every k refills, never more than r misses in a row). Admission solves an exact integer program that assigns each 500 ms GPU slot to one policy and reserves places for robots; the server serves reservations first, then fills spare places by distance-based priority. Where misses fall matters, and tolerance is policy-dependent: with 10 actions executed per refill, the first stall comes at a miss run of 5 for SmolVLA and pi0.5, 3 for X-VLA, 1 for GR00T N1.7.
2. Results (reported): zero SLO violations on 6 fleets of 2 to 4 policies (pi0.5, SmolVLA, X-VLA, GR00T N1.7) with 14 to 122 admitted robots, and in 14 trials at 50-100% load. At 100% load on one SmolVLA server: DBP 65 violating windows, Armory Lookahead / EDF / round robin 784 each, same work done. Admission matches exhaustive search on 565 small fleets; 22 ms median at 10 robots, 76 ms at 80. Keeping leftover actions: 79.8% success vs 41.2% when discarded (3,584 LIBERO episodes). GR00T N1.7: 62/64 under miss pattern 0011 vs 28/64 under 1001 at equal served fraction. On a GB10 (DGX Spark chip) vs RTX PRO 6000: pi0.5 capacity 2 vs 12, SmolVLA 14 vs 58, cross-GPU fleet boundary 29 vs 122 robots.
3. Limits: LIBERO simulation only, no physical robots described. Needs a known fleet and a single GPU; one policy per slot. Armory baseline is the authors' adaptation (depth-two lookahead, same slots). At 160 robots only one of three admission cases finished in 1 s. 1000 ms GPU stalls cause violations that recover in 1.0 to 3.5 s.
4. Next: use its miss-pattern method on stack_bowls with pi0.5 (which refill patterns does it tolerate?); pick the execution horizon H and compute how many missed refills the action reserve (L - H) absorbs.

------------------------------ 2026-10-13 · Robion ------------------------------
Paper: Efficient Vision-Language-Action Management and Serving for Robot Factories (Sep 14, 2026)
Link: https://arxiv.org/abs/2609.12075

1. Insight: VLA stages are millisecond-scale and underuse the GPU (pi0 tensor cores: 52% VLM stage, 15% action stage at batch 4). Running the VLM and action diffusion transformer on two GPU streams in lockstep, with SM partitioning for the VLM stream, beats both a monolithic pipeline and stage-disaggregated vLLM-Omni. Earliest-SLO-first scheduling plus an integer-program traffic controller handle several models across GPUs.
2. Results (reported): 6.7x robot load vs vLLM-Omni and 1.5x vs monolithic at 98% SLO attainment. 8 models on 4 GPUs serve up to 64 robots. vLLM-Omni's per-step inspection costs 30-35% of latency. Cloud is 18-24% lower in SLO attainment than edge at 24 robots/model. About 6.8x lower hardware cost than Jetson AGX Thor on the robot. Evaluated on LIBERO.
3. Limits: Does not use Armory-style chunk state. Criticism of Armory (ignores per-request SLOs, treats VLA as one pipeline) not checked against Armory.
4. Next: check whether the edge-vs-cloud gap holds for pi0.5 at our fleet size.

------------------------------ 2026-10-13 · Armory ------------------------------
Paper: Action Chunk Scheduling for Batched Robot Policy Serving (Aug 2026)
Link: https://arxiv.org/abs/2608.00337

1. Insight: When robots consume action chunks at different rates, naive batching heuristics fail. Urgency depends on each robot's remaining action buffer. The Lookahead scheduler simulates forward over a server-side mirror of each robot's executed action, received action, and queued chunks.
2. Results (reported): up to 18% higher system throughput in real-world runs; about 2x fast-tier throughput over EDF with one fast robot. pi0.5 on one L40S with 10 AgileX PiPER arms. pi0.5 takes about 73 ms at batch 1 and 211 ms at batch 5. Practical up to 50 ms one-way delay.
3. Limits: plain batching on one GPU only (no continuous batching, PagedAttention, distributed inference). Real-world results rest on 3 seeds and 1-minute rollouts. Lookahead depth 2 or 3 did not help. Stale chunks hurt dynamic manipulation more than quasi-static tasks.
4. Next: check whether our server-side batching has an equivalent of time-until-the-robot-runs-dry.

------------------------------ 2026-10-13 · ROSA ------------------------------
Paper: ROSA: A Robotics Foundation Model Serving System for Robot Factories (Jul 2026)
Link: https://arxiv.org/abs/2607.01088

1. Insight: Serving a factory means a shared GPU pool, multi-model pipelines (action, planner, safety, monitor) and a scheduler that maximizes productivity under SLOs, not per-request latency. ILP packing, profiling-guided batching, request-rate capping, failover.
2. Results (reported): up to 12.06x factory productivity vs the best dedicated baseline; 2.44x SLO-qualified throughput vs shared baselines; request-rate capping 3.18x at 64 robots; batch 16 gives about 3-4x the throughput of batch 1. 8 H200 GPUs, up to 64 virtual robots replaying real observations, GR00T N1.6 and pi0.5, plus one real Franka Panda.
3. Limits: assumes a known, static, profiled workload and synchronous execution (also Armory's critique). Scale results are synthetic. Needs a stable wired or Wi-Fi 7 network.
4. Next: look at its request-rate capping; test whether capping helps at our scale.

------------------------------ 2026-10-13 · vla.cpp ------------------------------
Paper: vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models (Jun 6, 2026)
Link: https://arxiv.org/abs/2606.08094

1. Insight: A portable C++ runtime on ggml/llama.cpp can run many VLA architectures with per-model adapters and a numerical check against reference actions. Flow-matching policies cache conditioning across solver steps within a request. Batch-1 inference is compute-bound.
2. Results (reported): 11 models supported (incl. OpenVLA-OFT, pi0, pi0.5, GR00T, SmolVLA, BitVLA). 10 of 11 within 3.2 points of reference on LIBERO-Object (GR00T-N1.6: 88% vs 99%, cause unresolved). Tested on RTX 3060/3090/4070/5070, Jetson Orin, Apple M4. 1.02x to 6.21x faster than eager PyTorch. BitVLA 200/200 on an 8 GB Orin Nano. SmolVLA on Orin Nano: 49/50 executing 16 actions per chunk vs 19/50 executing 50.
3. Limits: no batching and no multi-robot serving; synchronous requests only. Robot studies are small and quasi-static.
4. Next: check whether it is worth running on our Piper PC 3090; use the execution-horizon result to design our staleness sweep on stack_bowls.
