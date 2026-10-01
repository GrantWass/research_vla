#!/usr/bin/env python3
"""Render a serve-bench JSONL file as a markdown table (Mac-safe).

Usage: python3 scripts/serve_bench/report.py <records.jsonl> [--wall-s S] [--peak-vram MIB]
"""
from __future__ import annotations

import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))

from metrics import read_jsonl, summarize


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    path = argv[0]
    wall_s = 0.0
    peak = None
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--wall-s" and i + 1 < len(args):
            wall_s = float(args[i + 1]); i += 2
        elif args[i] == "--peak-vram" and i + 1 < len(args):
            peak = float(args[i + 1]); i += 2
        else:
            i += 1
    recs = read_jsonl(path)
    s = summarize(recs, wall_s=wall_s, peak_vram_mib=peak)
    rows = [
        ("requests", s.n_requests), ("robots", s.n_robots),
        ("wall_s", round(s.wall_s, 1)),
        ("e2e_p50_ms", _r(s.e2e_p50_ms)), ("e2e_p99_ms", _r(s.e2e_p99_ms)),
        ("ttfac_p50_ms", _r(s.ttfac_p50_ms)), ("ttfac_p99_ms", _r(s.ttfac_p99_ms)),
        ("queue_p99_ms", _r(s.queue_p99_ms)), ("infer_p50_ms", _r(s.infer_p50_ms)),
        ("req_per_s", _r(s.req_per_s)), ("actions_per_s", _r(s.actions_per_s)),
        ("starvation_rate", _r(s.starvation_rate)),
        ("success_rate", _r(s.success_rate)),
        ("mean_batch", _r(s.mean_batch)), ("peak_vram_mib", s.peak_vram_mib),
    ]
    print("| metric | value |")
    print("|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    return 0


def _r(v):
    return round(v, 2) if isinstance(v, float) else v


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
