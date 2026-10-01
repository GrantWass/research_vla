"""Serving-bench metrics: request records + percentile aggregation.

Stdlib only (Mac-safe, no GPU/Isaac). Metric set follows:
- Armory (arXiv:2608.00337, pi0.5 multi-robot serving): starvation rate,
  task throughput (success/min), infer latency vs batch size.
- vLLM/SGLang/Triton + MLPerf Inference: E2E latency p50/p99, TTFT,
  ITL/TPOT, queue time, throughput.
- RoboDojo native: success_rate, score.

VLA adaptations: TTFT -> TTFAC (time-to-first-action-chunk),
token throughput -> action throughput (actions/s).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field


@dataclass
class RequestRecord:
    """One served inference request (one robot's one query)."""

    request_id: str
    robot_id: str
    scheduler: str = "sequential"
    batch_size: int = 1
    queue_ms: float = 0.0  # wait for batch slot (vLLM request_queue_time analog)
    infer_ms: float = 0.0  # model forward only (openpi policy_timing.infer_ms)
    e2e_ms: float = 0.0  # obs send -> action received (CALL_RESULT latency_ms)
    chunk_len: int = 0  # actions returned in this chunk
    starved: bool = False  # Armory: no action available at control step
    success: bool | None = None  # task outcome when known
    score: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ttfac_ms(self) -> float:
        """Time-to-first-action-chunk (TTFT analog)."""
        return self.queue_ms + self.infer_ms


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = math.ceil(pct / 100.0 * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(rank, len(sorted_vals) - 1))]


@dataclass
class BenchSummary:
    n_requests: int = 0
    n_robots: int = 0
    wall_s: float = 0.0
    e2e_p50_ms: float | None = None
    e2e_p99_ms: float | None = None
    ttfac_p50_ms: float | None = None
    ttfac_p99_ms: float | None = None
    queue_p99_ms: float | None = None
    infer_p50_ms: float | None = None
    req_per_s: float | None = None
    actions_per_s: float | None = None
    starvation_rate: float | None = None
    success_rate: float | None = None
    mean_batch: float | None = None
    peak_vram_mib: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(records: list[RequestRecord], wall_s: float = 0.0,
              peak_vram_mib: float | None = None) -> BenchSummary:
    """Aggregate request records into serving metrics."""
    s = BenchSummary(n_requests=len(records), wall_s=wall_s,
                     peak_vram_mib=peak_vram_mib)
    if not records:
        return s
    s.n_robots = len({r.robot_id for r in records})
    e2e = sorted(r.e2e_ms for r in records)
    ttfac = sorted(r.ttfac_ms for r in records)
    queue = sorted(r.queue_ms for r in records)
    infer = sorted(r.infer_ms for r in records)
    s.e2e_p50_ms = _percentile(e2e, 50)
    s.e2e_p99_ms = _percentile(e2e, 99)
    s.ttfac_p50_ms = _percentile(ttfac, 50)
    s.ttfac_p99_ms = _percentile(ttfac, 99)
    s.queue_p99_ms = _percentile(queue, 99)
    s.infer_p50_ms = _percentile(infer, 50)
    if wall_s > 0:
        s.req_per_s = len(records) / wall_s
        s.actions_per_s = sum(r.chunk_len for r in records) / wall_s
    s.starvation_rate = sum(1 for r in records if r.starved) / len(records)
    known = [r for r in records if r.success is not None]
    if known:
        s.success_rate = sum(1 for r in known if r.success) / len(known)
    s.mean_batch = sum(r.batch_size for r in records) / len(records)
    return s


def write_jsonl(path: str, records: list[RequestRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_jsonl(path: str) -> list[RequestRecord]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(RequestRecord(**{k: d.get(k, v) for k, v in {
                    "request_id": "", "robot_id": "", "scheduler": "sequential",
                    "batch_size": 1, "queue_ms": 0.0, "infer_ms": 0.0,
                    "e2e_ms": 0.0, "chunk_len": 0, "starved": False,
                    "success": None, "score": None, "extra": {}}.items()}))
    return out
