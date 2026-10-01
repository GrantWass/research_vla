"""Aggregate measured runs into one comparison table.

Reads a manifest of runs (scripts/serve_bench/runs_*.json) plus the raw
PI05_TIMING_LOG jsonl files captured on the GPU box, and reports the two
things that actually decide whether serving strategy matters:

- gpu_busy_pct: sum(infer_ms) / wall_s. How much of the run the brain was
  actually working. Low numbers mean the simulator is the bottleneck and
  serving strategy cannot move wall-clock time.
- ms_per_robot_chunk: total GPU time / robot-chunks served. The real cost
  of serving one robot one action chunk.

Timing logs are optional; runs without them still report scores/wall.
Calls >= 2 s are JAX compiles (one per new batch shape), counted apart from
steady-state cost. Success carries a Wilson 95% CI; `--pairwise` adds a
Fisher exact test of every run against the S0 run at the same task and N.

Manifest entry fields: name, strategy, task, seed, n_envs, eval_time,
success_rate, score, wall_s, peak_vram_mib, timing_log (basename or path).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

from stats import (fisher_exact_p, mcnemar_exact_p, median, min_detectable_gap, paired_episodes,
                   wilson_ci)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
COMPILE_MS = 2000.0


def load_timing(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_stats(run: dict, log_dir: str = DATA_DIR) -> dict:
    """Aggregate one run: calls, batch mix, GPU busy %, cost per robot chunk."""
    log = run.get("timing_log", "")
    path = log if os.path.isabs(log) and os.path.exists(log) else os.path.join(
        log_dir, os.path.basename(log))
    if not log:
        path = ""
    raw = load_timing(path)
    timed = [r for r in raw
             if isinstance(r.get("infer_ms"), (int, float)) and r["infer_ms"] > 0
             and r.get("mode") != "warmup"]
    rows = [r for r in timed if r["infer_ms"] < COMPILE_MS]
    by_batch: dict[int, list[float]] = {}
    for r in rows:
        by_batch.setdefault(int(r.get("batch") or 1), []).append(r["infer_ms"])
    calls = len(rows)
    gpu_ms = sum(r["infer_ms"] for r in rows)
    batches = Counter(int(r["batch"]) for r in rows if r.get("batch"))
    robot_chunks = sum(size * count for size, count in batches.items())
    wall_s = run.get("wall_s") or 0
    return {
        "calls": calls,
        "compiles": len(timed) - calls,
        # batched forward failed (e.g. OOM) and silently went sequential
        "fallbacks": sum(1 for r in raw if r.get("mode") == "fallback"),
        "median_ms_by_batch": {b: median(v) for b, v in sorted(by_batch.items())},
        "robot_chunks": robot_chunks,
        "batches": dict(sorted(batches.items())),
        "gpu_s": gpu_ms / 1000.0,
        "gpu_busy_pct": (gpu_ms / 1000.0 / wall_s * 100.0) if wall_s else None,
        "ms_per_robot_chunk": (gpu_ms / robot_chunks) if robot_chunks else None,
    }


def _successes(run: dict) -> tuple[int, int]:
    # Per-episode rows win: RoboDojo echoes the REQUESTED eval_time even when
    # it runs fewer episodes (54 requested on seed 2 ran 25).
    eps = run.get("episodes")
    if eps:
        return sum(1 for e in eps if e[1]), len(eps)
    n = int(run.get("eval_time") or 0)
    return round(run.get("success_rate", 0) * n), n


def table(runs: list[dict], log_dir: str = DATA_DIR) -> str:
    head = ("| run | strat | N | eps | succ | 95% CI | score | wall_s | "
            "peak_MiB | calls | batch mix | robot_chunks | gpu_s | "
            "gpu_busy_% | ms/chunk |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in runs:
        s = run_stats(r, log_dir)
        n_eps = len(r["episodes"]) if r.get("episodes") else r.get("eval_time")
        k, n = _successes(r)
        succ = f"{k}/{n}" if n else "-"
        lo, hi = wilson_ci(k, n)
        ci = f"{lo:.0%}-{hi:.0%}" if n else "-"
        mix = " ".join(f"{k}x{v}" for k, v in s["batches"].items()) or "-"
        lines.append(
            "| {name} | {strategy} | {n_envs} | {eps} | {succ} | {ci} | {score} | "
            "{wall} | {vram} | {calls} | {mix} | {chunks} | {gpu} | {busy} | "
            "{mperc} |".format(
                name=r.get("name", "?"), strategy=r.get("strategy", "?"),
                n_envs=r.get("n_envs", "?"), eps=n_eps or "-", succ=succ, ci=ci,
                score=round(r.get("score", 0), 1), wall=r.get("wall_s", "-"),
                vram=r.get("peak_vram_mib", "-"), calls=s["calls"], mix=mix,
                chunks=s["robot_chunks"], gpu=round(s["gpu_s"], 1),
                busy=round(s["gpu_busy_pct"], 1) if s["gpu_busy_pct"] else "-",
                mperc=round(s["ms_per_robot_chunk"], 1)
                if s["ms_per_robot_chunk"] else "-"))
    return head + "\n".join(lines)


def latency_table(runs: list[dict], log_dir: str = DATA_DIR) -> str:
    """Steady-state median ms per forward, by how many robots it served."""
    sizes = sorted({b for r in runs
                    for b in run_stats(r, log_dir)["median_ms_by_batch"]})
    head = ("| run | compiles | fallbacks | " + " | ".join(f"B={b}" for b in sizes)
            + " |\n|---|---|---|" + "---|" * len(sizes) + "\n")
    lines = []
    for r in runs:
        s = run_stats(r, log_dir)
        cells = [f"{s['median_ms_by_batch'][b]:.0f}"
                 if b in s["median_ms_by_batch"] else "" for b in sizes]
        lines.append(f"| {r.get('name')} | {s['compiles']} | {s['fallbacks']} | "
                     + " | ".join(cells) + " |")
    return head + "\n".join(lines)


def pairwise(runs: list[dict]) -> str:
    """Each non-S0 run vs the S0 run with the same task and N."""
    def key(r, seed=True):
        return (r.get("task"), r.get("n_envs")) + ((r.get("seed"),) if seed else ())

    s0 = [r for r in runs if str(r.get("strategy", "")).startswith("S0")]
    exact = {key(r): r for r in s0}
    loose = {key(r, False): r for r in s0}
    pooled: dict[str, list] = {}
    head = ("| run | vs | succ | baseline | Fisher p | detectable gap at "
            "this n | paired: only-base / only-run | McNemar p | "
            "score diff (95% CI) |\n|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in runs:
        b = exact.get(key(r)) or loose.get(key(r, False))
        if b is None or b is r:
            continue
        k1, n1 = _successes(r)
        k0, n0 = _successes(b)
        paired = "| - | - | - |"
        if r.get("episodes") and b.get("episodes") and r.get("seed") == b.get("seed"):
            p = paired_episodes(b["episodes"], r["episodes"])
            pooled.setdefault(str(r.get("strategy")), []).append(p)
            lo, hi = p["score_diff_ci"]
            paired = (f"| {p['only_a']} / {p['only_b']} of {p['pairs']} | "
                      f"{p['mcnemar_p']:.2f} | {p['score_diff']:+.2f} "
                      f"({lo:+.2f}, {hi:+.2f}) |")
        lines.append(f"| {r['name']} | {b['name']} | {k1}/{n1} | {k0}/{n0} | "
                     f"{fisher_exact_p(k1, n1, k0, n0):.2f} | "
                     f"{min_detectable_gap(min(n0, n1)):.0%} " + paired)
    for strat, ps in pooled.items():
        if len(ps) < 2:
            continue
        a, b2 = sum(p["only_a"] for p in ps), sum(p["only_b"] for p in ps)
        n = sum(p["pairs"] for p in ps)
        lines.append(f"| **pooled {strat}** | S0, {len(ps)} matched runs | | | | "
                     f"{min_detectable_gap(n):.0%} | {a} / {b2} of {n} | "
                     f"{mcnemar_exact_p(a, b2):.2f} | |")
    return head + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?",
                    default=os.path.join(HERE, "runs_2026_09_30.json"))
    ap.add_argument("--log-dir", default=DATA_DIR)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pairwise", action="store_true",
                    help="add latency-by-batch and Fisher tests vs S0")
    args = ap.parse_args(argv)
    with open(args.manifest) as fh:
        text = fh.read()
    # .json = list (curated manifest); .jsonl = sim_battery.sh output
    runs = (json.loads(text) if args.manifest.endswith(".json")
            else [json.loads(line) for line in text.splitlines() if line.strip()])
    if args.json:
        print(json.dumps([{**r, **run_stats(r, args.log_dir)} for r in runs],
                         indent=2))
    else:
        print(table(runs, args.log_dir))
        if args.pairwise:
            print("\n" + latency_table(runs, args.log_dir))
            print("\n" + pairwise(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
