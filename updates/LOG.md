# Update log (newest first). Plain text. The visual doc is notebook.html (separate).
# One entry per paper. Dates are when we read it; paper dates are in the entry.

------------------------------ 2026-10-13 · OxyGen ------------------------------
Paper: OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism (Mar 2026, rev. May 2026, Tsinghua AIR)
Link: https://arxiv.org/abs/2603.14371

1. Insight: One on-device pi0.5-style model can run action generation and language generation from the same observation. Prefill the observation once and share its KV cache between the action and language experts; batch in-flight language requests from different frames under a fixed per-frame token budget so the action deadline holds.
2. Results (reported): Up to 3.7x over isolated execution; over 200 tokens/s language with 70 Hz actions. RTX 4090 action rate stays near 60 Hz as decoding grows (baseline 49.9 to 19.1 Hz); about 27 Hz on Jetson Thor. Real humanoid with onboard Thor: language 822.3 to 195.4 ms, total inference 1030 to 393 ms. Memory +15% (6.43 to 7.41 GB), energy per request down up to 78%. Naive parallelism nearly doubles peak memory. LIBERO success within about 0.8 points of openpi.
3. Limits: One model and expert layout (action + language), single robot, no SLOs, no multi-robot or cloud serving. Token budget is calibrated offline per device and rate. Measures inference speed, not closed-loop rollouts.
4. Next: only relevant if our robot needs language output or a planner in the loop with pi0.5; the shared-prefill idea is the part to reuse.

------------------------------ 2026-10-13 · Offload or Overload ------------------------------
Paper: Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads (Mar 18, 2026, Microsoft and UC Berkeley)
Link: https://arxiv.org/abs/2603.18284

1. Insight: Measures where manipulation and navigation workloads should run. For pi0.5 the deciding factors are GPU class (accuracy collapses on an Orin AGX), added network delay, video compression, and how the GPU is shared.
2. Results (reported): pi0.5 fine-tuned on a bimanual SO-101 (tape handover, 70 s, real-time chunking). Action-chunk latency about 60 ms A100, 144 ms Thor, 440 ms Orin AGX; accuracy 80%, 80%, 30%. DGX Spark accuracy 80% to 70% with 10 ms mean (SD 15) added one-way delay; A100 held 80%. Three cameras lossless 640x480 at 30 FPS need about 100 Mbps; H.264 at 12 or 6.5 Mbps lowered accuracy. A100 batch 4: pi0.5 1.6x faster than sequential. Time-slicing raises pi0.5 latency 75%, unthrottled MPS 230%; capping VLMaps at 10% of SMs held it to 3% (VLMaps frame time +261%). Onboard stack needs about 50 GB.
3. Limits: Up to four concurrent requests, one fine-tuned task on one arm setup, emulated delay from a fixed distribution, one Wi-Fi access point, battery-life gain is modeled. No scheduler.
4. Next: reuse the delay-injection and accuracy comparison for our own arms; check how a 3090 and a DGX Spark compare on chunk latency.

------------------------------ 2026-10-13 · VLA-Perf ------------------------------
Paper: How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf (Feb 20, 2026, NVIDIA Research)
Link: https://arxiv.org/abs/2602.18397

1. Insight: An analytical roofline model of VLA inference (compute vs memory per operator, plus network). The action expert is memory-bound on every GPU; vision and VLM are compute-bound except on Jetson Thor. Gives a ceiling on control rate per hardware and network before building anything.
2. Results (reported): pi0 at batch 1, no network: Thor 19.0 Hz, RTX 4090 32.2, A100 61.7, H100 162.5, B100 314.4. Model scaling: B100 9.6 Hz at 81B; 4090 out of memory at 16.7B. 10 to 50 denoising steps costs 2.15x latency; chunk 50 to 250 adds 11%. Diffusion head 102x faster than autoregressive decoding at chunk 50. RTX 4090 over Wi-Fi beats on-device Thor; B100 beats on-device even over 5G. Async inference: 2.63x on Wi-Fi 7, 5.99x on 5G. Device/server model splits are usually slower than device only. Validated at 73-83% of roofline on a Triton pi0, 4090.
3. Limits: Optimistic upper bound (ignores launch overhead); batch size 1 only; no scheduling or multi-robot contention; pi0.5 not modeled; staleness effect on control quality left open.
4. Next: use it as the sanity ceiling when we measure pi0.5 on the 3090 and DGX Spark; compare measured against the roofline.

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
