"""Judge batched-vs-solo drift against noise-seed spread.

Inputs: two `replay_client.py --scenario equivalence` outputs from servers
started with different PI05_NOISE_SEED values. For each robot observation:

- drift: max |batched - solo| under one seed (numerical, from batch shape).
- spread: max |solo(seed a) - solo(seed b)|, i.e. how much the answer moves
  when the sampler's start noise changes, which the policy treats as equally
  valid. Batching is harmless when drift is a small fraction of spread.
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> dict[tuple[int, str], dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {(r["iter"], r["robot"]): r for r in rows if "solo" in r}


def compare(a: dict, b: dict) -> list[dict]:
    out = []
    for key in sorted(set(a) & set(b)):
        sa, sb = a[key]["solo"], b[key]["solo"]
        spread = max(abs(x - y) for x, y in zip(sa, sb))
        drift = max(a[key]["batch_vs_solo"], b[key]["batch_vs_solo"])
        out.append({"iter": key[0], "robot": key[1], "drift": drift,
                    "spread": spread,
                    "ratio": drift / spread if spread else float("inf")})
    return out


def main(argv: list[str]) -> int:
    rows = compare(load(argv[0]), load(argv[1]))
    print("| obs | drift (batch vs solo) | spread (seed 7 vs 8) | drift/spread |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['robot']}#{r['iter']} | {r['drift']:.4f} | "
              f"{r['spread']:.4f} | {r['ratio']:.1%} |")
    worst = max(r["ratio"] for r in rows)
    print(f"\nworst drift/spread: {worst:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
