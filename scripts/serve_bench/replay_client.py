"""Live-server replay scenarios (heterogeneous serving tests).

Needs a running pi0.5 policy server (GPU box) + `websockets` + `numpy`;
both imports are lazy so Mac CI (which has neither) can still use the
pure helpers and run the tests.

Scenarios:
- mixed_batch: one client sends same-persona and mixed-persona batches.
  Checks batched serving keeps each robot's answer its own.
- contention: two clients (threads) share one server with staggered
  starts, measuring queueing through the server lock.
- equivalence: with the server started under PI05_NOISE_SEED (same start
  noise for every robot), every robot's answer from a mixed batch must
  match its one-robot answer for the same observation. Reports max abs
  difference per robot; this is the correctness gate for batching.
- fleet: K separate clients (one connection each, like K robot processes)
  run closed loops of atomic `infer_obs` with a think time standing in for
  executing the chunk. Compare server modes (XPL_SERVER_BATCH /
  XPL_BATCH_WINDOW_MS) by restarting the server between runs.

Personas differ the way real robots do: different joint states,
different instructions, different pixels.

Every served call is recorded as a metrics.RequestRecord JSONL row, so
`report.py` summarizes live traffic exactly like simulated traffic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))

from metrics import RequestRecord  # noqa: E402 (stdlib-only, always importable)

PERSONAS = {
    "A": {"state": list(map(float, range(-7, 7))),
          "instruction": "stack the bowls"},
    "B": {"state": [float(v) + 0.5 for v in range(7, -7, -1)],
          "instruction": "push the T block"},
}
STATE_DIM = 14
IMG_HW = (224, 224, 3)
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def make_obs(persona: str, seed: int = 0) -> dict:
    """Build one encoded observation (server also accepts raw env obs)."""
    import numpy as np  # lazy: GPU-box/server-test dependency only

    # Not hash(): str hashing is salted per process, so pixels would differ
    # between runs and between the solo and batched halves of a comparison.
    rng = np.random.default_rng([ord(c) for c in persona] + [seed])
    p = PERSONAS[persona]
    state = np.asarray(p["state"][:STATE_DIM], dtype=np.float32)
    return {
        "state": state,
        "images": {k: rng.integers(0, 256, IMG_HW).astype(np.uint8)
                   for k in CAMERAS},
        "instruction": p["instruction"],
    }


def to_record(request_id: str, robot_id: str, scheduler: str, e2e_ms: float,
              batch_size: int, infer_ms: float | None = None,
              chunk_len: int = 0) -> RequestRecord:
    queue_ms = max(0.0, e2e_ms - (infer_ms or 0.0))
    return RequestRecord(request_id=request_id, robot_id=robot_id,
                         scheduler=scheduler, batch_size=batch_size,
                         queue_ms=queue_ms, infer_ms=infer_ms or 0.0,
                         e2e_ms=e2e_ms, chunk_len=chunk_len)


def _client(url: str, evaluation_id: str, trial_id: str):
    from client_server.ws.model_client import WsModelClient  # lazy

    return WsModelClient(url=url, evaluation_id=evaluation_id,
                         trial_id=trial_id)


def scenario_mixed_batch(url: str, iters: int = 6) -> list[RequestRecord]:
    """Alternate same-persona and mixed-persona batches on one client."""
    recs: list[RequestRecord] = []
    with _client(url, "replay", "replay-mixed") as cli:
        cli.call(func_name="reset")
        for i in range(iters):
            pair = ["A", "B"] if i % 2 else ["A", "A"]
            t0 = time.perf_counter()
            cli.call(func_name="update_obs_batch",
                     obs=[make_obs(p, seed=i) for p in pair])
            res = cli.call(func_name="get_action_batch", obs=[0, 1])
            e2e = (time.perf_counter() - t0) * 1000.0
            info = cli.call(func_name="last_batch_info")
            n_actions = sum(len(a) for a in res) if res else 0
            recs.append(to_record(f"mixed-{i}", " robots-AB", "batched",
                                  e2e, info.get("batch_size", 2),
                                  chunk_len=n_actions))
    return recs


def scenario_contention(url: str, iters: int = 20, gap_s: float = 3.0,
                        out: str = "") -> list[RequestRecord]:
    """Two staggered solo clients sharing one server (threads)."""
    import threading

    recs: list[RequestRecord] = []
    lock = threading.Lock()

    def hammer(persona: str, delay: float):
        time.sleep(delay)
        with _client(url, "replay", f"replay-{persona}") as cli:
            cli.call(func_name="reset")
            for i in range(iters):
                t0 = time.perf_counter()
                cli.call(func_name="update_obs",
                         obs=make_obs(persona, seed=i))
                cli.call(func_name="get_action")
                e2e = (time.perf_counter() - t0) * 1000.0
                with lock:
                    recs.append(to_record(f"{persona}-{i}", persona,
                                          "sequential", e2e, 1))

    threads = [threading.Thread(target=hammer, args=("A", 0.0)),
               threading.Thread(target=hammer, args=("B", gap_s))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if out:
        from metrics import write_jsonl

        write_jsonl(out, recs)
    return recs


def flatten(x) -> list[float]:
    """Actions come back as arrays or per-arm dicts/lists; flatten in order."""
    if isinstance(x, dict):
        return [v for key in sorted(x) for v in flatten(x[key])]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [v for item in x for v in flatten(item)]
    return [float(x)]


def max_abs_diff(a, b) -> float:
    a, b = flatten(a), flatten(b)
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def scenario_equivalence(url: str, iters: int = 3) -> list[dict]:
    """Batched vs one-robot answers for identical observations."""
    rows: list[dict] = []
    with _client(url, "replay", "replay-equiv") as cli:
        cli.call(func_name="reset")
        for i in range(iters):
            batch = [("A", i), ("B", i), ("A", i + 100)]
            solo = [cli.call(func_name="infer_obs", obs=make_obs(p, seed=s))
                    for p, s in batch]
            res = cli.call(func_name="infer_obs_batch", obs={
                "obs_list": [make_obs(p, seed=s) for p, s in batch],
                "env_idx_list": None})
            again = cli.call(func_name="infer_obs", obs=make_obs(*batch[0]))
            for (p, s), one, many in zip(batch, solo, res):
                rows.append({"iter": i, "robot": f"{p}{s}",
                             "batch_vs_solo": max_abs_diff(one, many),
                             "solo": flatten(one)})
            rows.append({"iter": i, "robot": "repeat",
                         "batch_vs_solo": max_abs_diff(solo[0], again)})
            # Different robots must get different answers (else the check
            # above is vacuous: e.g. a model ignoring its input).
            rows.append({"iter": i, "robot": "A_vs_B",
                         "batch_vs_solo": max_abs_diff(solo[0], solo[1])})
    return rows


def scenario_fleet(url: str, n_clients: int, iters: int, think_s: float,
                   scheduler: str, out: str = "",
                   warmup: int = 2) -> list[RequestRecord]:
    """K independent clients, closed loop: infer_obs, then 'execute' chunk."""
    import random
    import threading

    recs: list[RequestRecord] = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_clients)

    def robot(k: int):
        persona = "AB"[k % 2]
        jitter = random.Random(k)
        with _client(url, "replay", f"fleet-{scheduler}-{k}") as cli:
            cli.call(func_name="reset")
            for i in range(warmup):
                cli.call(func_name="infer_obs", obs=make_obs(persona, k))
            barrier.wait()
            time.sleep(jitter.uniform(0, think_s))  # desync start phases
            for i in range(iters):
                obs = make_obs(persona, seed=1000 * k + i)
                t0 = time.perf_counter()
                res = cli.call(func_name="infer_obs", obs=obs)
                e2e = (time.perf_counter() - t0) * 1000.0
                with lock:
                    recs.append(to_record(f"{k}-{i}", f"robot{k}", scheduler,
                                          e2e, 1, chunk_len=len(res)))
                    recs[-1].extra = {"t_done": time.perf_counter()}
                time.sleep(think_s * jitter.uniform(0.8, 1.2))

    threads = [threading.Thread(target=robot, args=(k,))
               for k in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if out:
        from metrics import write_jsonl

        write_jsonl(out, recs)
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="ws://localhost:19732")
    ap.add_argument("--scenario", choices=("mixed_batch", "contention",
                                           "equivalence", "fleet"),
                    default="mixed_batch")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--gap-s", type=float, default=3.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--clients", type=int, default=2)
    ap.add_argument("--think-s", type=float, default=0.0)
    ap.add_argument("--label", default="fleet")
    args = ap.parse_args(argv)
    if args.scenario == "equivalence":
        rows = scenario_equivalence(args.url, args.iters)
        for row in rows:
            print(json.dumps(row))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        return 0
    if args.scenario == "fleet":
        recs = scenario_fleet(args.url, args.clients, args.iters,
                              args.think_s, args.label, args.out)
        from metrics import summarize

        span = (max(r.extra["t_done"] for r in recs)
                - min(r.extra["t_done"] - r.e2e_ms / 1000.0 for r in recs))
        print(json.dumps({"label": args.label, "clients": args.clients,
                          "think_s": args.think_s,
                          **summarize(recs, wall_s=span).to_dict()}))
        return 0
    if args.scenario == "mixed_batch":
        recs = scenario_mixed_batch(args.url, args.iters)
    else:
        recs = scenario_contention(args.url, args.iters, args.gap_s,
                                   args.out)
    if args.out and args.scenario == "mixed_batch":
        from metrics import write_jsonl

        write_jsonl(args.out, recs)
    for r in recs:
        print(json.dumps({"robot": r.robot_id, "e2e_ms": round(r.e2e_ms, 1),
                          "batch": r.batch_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
