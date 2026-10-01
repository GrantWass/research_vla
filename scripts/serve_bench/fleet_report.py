"""Pivot live_battery.sh fleet results into one comparison table.

Input: fleet_summary.jsonl (one replay_client `--scenario fleet` summary per
line). For each (think time, clients) cell and server mode it reports e2e
p50/p99 and served req/s. `idle%` is the share of a robot's loop spent
waiting on the brain, e2e_p50 / (e2e_p50 + think), which is what a
take-turns robot experiences as stalled time.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("fifo", "cont", "win10", "win25")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def table(rows: list[dict]) -> str:
    cells = {(r["think_s"], r["clients"], r["label"]): r for r in rows}
    modes = [m for m in MODES if any(k[2] == m for k in cells)]
    head = ("| think_s | clients | " + " | ".join(
        f"{m} p50/p99 ms | {m} req/s | {m} idle%" for m in modes)
        + " |\n|---|---|" + "---|---|---|" * len(modes) + "\n")
    lines = []
    for think, k in sorted({(t, c) for t, c, _ in cells}):
        parts = []
        for m in modes:
            r = cells.get((think, k, m))
            if r is None:
                parts.append(" | | ")
                continue
            p50 = r["e2e_p50_ms"]
            idle = p50 / (p50 + think * 1000.0)
            parts.append(f"{p50:.0f}/{r['e2e_p99_ms']:.0f} | "
                         f"{r['req_per_s']:.1f} | {idle:.0%}")
        lines.append(f"| {think} | {k} | " + " | ".join(parts) + " |")
    return head + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", nargs="?", default=os.path.join(
        HERE, "data", "fleet_2026_09_30", "fleet_summary.jsonl"))
    args = ap.parse_args(argv)
    print(table(load(args.summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
