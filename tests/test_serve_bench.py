"""Tests for scripts/serve_bench (stdlib unittest, Mac-safe, no GPU)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "serve_bench"))

from metrics import RequestRecord, read_jsonl, summarize, write_jsonl
from schedulers import PendingRequest, earliest_deadline, round_robin, sequential
from simulate import FleetSim, compare

try:
    from replay_client import PERSONAS, make_obs, to_record

    import numpy  # noqa: F401 (replay obs builder needs it)

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from equivalence_report import compare as equivalence_compare
from replay_client import flatten, max_abs_diff
from stats import (fisher_exact_p, mcnemar_exact_p, min_detectable_gap,
                   paired_episodes, wilson_ci)

try:
    from compare_runs import pairwise, run_stats, table

    HAS_COMPARE = True
except ImportError:
    HAS_COMPARE = False

# Synthetic steady-state pools (same shape as the box timing logs).
POOLS = {1: [120.0, 130.0], 2: [200.0, 220.0], 3: [280.0, 300.0]}


def _rec(robot="r0", e2e=100.0, infer=80.0, queue=20.0, **kw):
    d = dict(request_id=f"{robot}-{e2e}", robot_id=robot, e2e_ms=e2e,
             infer_ms=infer, queue_ms=queue, chunk_len=25, batch_size=1)
    d.update(kw)
    return RequestRecord(**d)


class TestMetrics(unittest.TestCase):
    def test_percentiles(self):
        recs = [_rec(e2e=float(i)) for i in range(1, 101)]
        s = summarize(recs, wall_s=10.0)
        self.assertEqual(s.n_requests, 100)
        self.assertEqual(s.e2e_p50_ms, 50.0)
        self.assertEqual(s.e2e_p99_ms, 99.0)
        self.assertAlmostEqual(s.req_per_s, 10.0)

    def test_empty(self):
        s = summarize([])
        self.assertEqual(s.n_requests, 0)
        self.assertIsNone(s.e2e_p99_ms)

    def test_starvation_and_success(self):
        recs = [_rec(robot="r0", starved=True, success=True),
                _rec(robot="r1", starved=False, success=False)]
        s = summarize(recs, wall_s=2.0)
        self.assertAlmostEqual(s.starvation_rate, 0.5)
        self.assertAlmostEqual(s.success_rate, 0.5)
        self.assertEqual(s.n_robots, 2)

    def test_jsonl_roundtrip(self):
        recs = [_rec(), _rec(robot="r1", batch_size=2)]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            write_jsonl(path, recs)
            back = read_jsonl(path)
            self.assertEqual(len(back), 2)
            self.assertEqual(back[1].batch_size, 2)
        finally:
            os.unlink(path)


class TestSchedulers(unittest.TestCase):
    def test_sequential(self):
        p = [PendingRequest("a"), PendingRequest("b")]
        self.assertEqual(len(sequential(p)), 2)

    def test_round_robin_cap(self):
        p = [PendingRequest(str(i)) for i in range(5)]
        batches = round_robin(p, max_batch=2)
        self.assertEqual([len(b) for b in batches], [2, 2, 1])

    def test_edf_starving_first(self):
        p = [PendingRequest("full", actions_left=10),
             PendingRequest("starving", actions_left=0)]
        batches = earliest_deadline(p, max_batch=2)
        self.assertEqual(batches[0][0].robot_id, "starving")


class TestSimulate(unittest.TestCase):
    def test_deterministic(self):
        a = FleetSim(n_robots=3, infer_pools=POOLS, seed=1).run(500)
        b = FleetSim(n_robots=3, infer_pools=POOLS, seed=1).run(500)
        self.assertEqual(a, b)

    def test_no_starvation_when_fast(self):
        for name in ("sequential", "round_robin", "earliest_deadline"):
            s = FleetSim(n_robots=3, infer_pools=POOLS, scheduler=name, seed=0)
            # startup transient only (<1% of robot-steps)
            self.assertLess(s.run(1000)["starvation_rate"], 0.01)

    def test_batching_wins_when_stressed(self):
        res = compare(["sequential", "earliest_deadline"], POOLS,
                      n_robots=10, drop_rate=0.0, seed=0)
        # with chunk_len=25 default this is mild; force stress via short chunks
        stressed = {}
        for name in ("sequential", "earliest_deadline"):
            s = FleetSim(n_robots=10, chunk_len=8, infer_pools=POOLS,
                         scheduler=name, seed=0)
            stressed[name] = s.run(3000)["starvation_rate"]
        self.assertLess(stressed["earliest_deadline"], stressed["sequential"])

    def test_drops_hurt(self):
        clean = FleetSim(n_robots=6, chunk_len=8, infer_pools=POOLS,
                         drop_rate=0.0, seed=0).run(2000)["starvation_rate"]
        lossy = FleetSim(n_robots=6, chunk_len=8, infer_pools=POOLS,
                         drop_rate=0.2, seed=0).run(2000)["starvation_rate"]
        self.assertGreaterEqual(lossy, clean)


@unittest.skipUnless(HAS_NUMPY, "numpy missing (replay obs builder)")
class TestReplayClient(unittest.TestCase):
    def test_personas_differ(self):
        self.assertNotEqual(PERSONAS["A"]["instruction"],
                            PERSONAS["B"]["instruction"])
        self.assertNotEqual(PERSONAS["A"]["state"], PERSONAS["B"]["state"])

    def test_make_obs_shape(self):
        obs = make_obs("A", seed=0)
        self.assertEqual(list(obs["state"].shape), [14])
        self.assertEqual(set(obs["images"]), {"cam_high", "cam_left_wrist",
                                              "cam_right_wrist"})
        self.assertEqual(obs["images"]["cam_high"].shape, (224, 224, 3))

    def test_to_record_queue_math(self):
        r = to_record("id1", "A", "batched", e2e_ms=200.0, batch_size=2,
                      infer_ms=150.0, chunk_len=50)
        self.assertAlmostEqual(r.queue_ms, 50.0)
        self.assertAlmostEqual(r.ttfac_ms, 200.0)
        s = summarize([r], wall_s=1.0)
        self.assertEqual(s.n_requests, 1)
        self.assertAlmostEqual(s.actions_per_s, 50.0)


@unittest.skipUnless(HAS_COMPARE, "compare_runs missing")
class TestCompareRuns(unittest.TestCase):
    RUN = {"name": "r", "strategy": "S2", "n_envs": 2, "eval_time": 4,
           "success_rate": 0.5, "score": 50.0, "wall_s": 100,
           "timing_log": "synth.jsonl"}

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "synth.jsonl"), "w") as fh:
            # two calls: one solo (120 ms), one pair (220 ms) -> 3 robot chunks
            fh.write(json.dumps({"infer_ms": 120.0, "batch": 1}) + "\n")
            fh.write(json.dumps({"infer_ms": 220.0, "batch": 2}) + "\n")
            fh.write("not json\n")  # must be skipped, not crash

    def test_robot_chunks_counts_batch_sizes(self):
        s = run_stats(self.RUN, self.dir)
        self.assertEqual(s["calls"], 2)
        self.assertEqual(s["robot_chunks"], 3)
        self.assertAlmostEqual(s["gpu_s"], 0.34)
        self.assertAlmostEqual(s["ms_per_robot_chunk"], 340.0 / 3, places=1)

    def test_gpu_busy_pct(self):
        s = run_stats(self.RUN, self.dir)
        self.assertAlmostEqual(s["gpu_busy_pct"], 0.34)

    def test_missing_log_is_not_fatal(self):
        run = {**self.RUN, "timing_log": "nope.jsonl"}
        s = run_stats(run, self.dir)
        self.assertEqual(s["calls"], 0)
        self.assertIsNone(s["ms_per_robot_chunk"])

    def test_table_has_every_run(self):
        md = table([self.RUN, {"name": "noLog", "n_envs": 1}], self.dir)
        self.assertTrue(md.startswith("| run |"))
        self.assertIn("|---|", md)
        self.assertIn("| r |", md)
        self.assertIn("| noLog |", md)
        # header is line 0 (no leading newline), so only data rows match
        self.assertEqual(md.count("\n| "), 2)

    def test_measured_manifest_is_internally_consistent(self):
        """Real captured runs: batching must never cost more per robot chunk."""
        here = os.path.dirname(os.path.abspath(__file__))
        bench = os.path.join(here, os.pardir, "scripts", "serve_bench")
        with open(os.path.join(bench, "runs_2026_09_30.json")) as fh:
            runs = json.load(fh)
        stats = {r["name"]: run_stats(r, os.path.join(bench, "data"))
                 for r in runs}
        for name, s in stats.items():
            self.assertGreater(s["calls"], 0, name)
            self.assertGreaterEqual(s["robot_chunks"], s["calls"], name)
        base = {2: stats["long_S0_N2"], 3: stats["long_S0_N3"]}
        for n in (2, 3):
            for name in (f"long_S1_N{n}", f"long_S2_N{n}"):
                self.assertLess(stats[name]["ms_per_robot_chunk"],
                                base[n]["ms_per_robot_chunk"], name)
        # the brain is barely used: sim is the bottleneck at these sizes
        for name, s in stats.items():
            self.assertLess(s["gpu_busy_pct"], 5.0, name)


class TestStats(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = wilson_ci(0, 4)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.3)  # 0/4 does not mean "never"
        lo, hi = wilson_ci(8, 12)
        self.assertLess(lo, 8 / 12)
        self.assertGreater(hi, 8 / 12)

    def test_fisher_known_value(self):
        # 8/12 vs 5/12: the S0 vs S1 N=2 gap that looked like a regression
        self.assertAlmostEqual(fisher_exact_p(8, 12, 5, 12), 0.414, places=3)
        self.assertAlmostEqual(fisher_exact_p(8, 12, 8, 12), 1.0, places=6)
        self.assertLess(fisher_exact_p(20, 20, 0, 20), 1e-6)

    def test_fisher_symmetric(self):
        self.assertAlmostEqual(fisher_exact_p(3, 10, 7, 10),
                               fisher_exact_p(7, 10, 3, 10))

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(0, 6), 2 / 64)
        self.assertAlmostEqual(mcnemar_exact_p(3, 3), 1.0)

    def test_paired_matches_by_layout(self):
        a = [[0, True, 1.0], [1, False, 0.2], [2, True, 1.0]]
        b = [[2, False, 0.5], [1, True, 1.0], [0, True, 1.0], [9, True, 1.0]]
        p = paired_episodes(a, b)
        self.assertEqual(p["pairs"], 3)
        self.assertEqual((p["only_a"], p["only_b"]), (1, 1))
        self.assertAlmostEqual(p["score_diff"], (0.0 + 0.8 - 0.5) / 3)

    def test_detectable_gap_shrinks_with_n(self):
        self.assertGreater(min_detectable_gap(12), 0.5)
        self.assertLess(min_detectable_gap(100), 0.2)


class TestEquivalenceHelpers(unittest.TestCase):
    def test_flatten_nested_actions(self):
        self.assertEqual(flatten({"b": [3, 4], "a": [[1], [2]]}),
                         [1.0, 2.0, 3.0, 4.0])

    def test_max_abs_diff(self):
        self.assertAlmostEqual(max_abs_diff([1, 2], [1.5, 1]), 1.0)
        self.assertEqual(max_abs_diff([1], [1, 2]), float("inf"))

    def test_ratio_against_seed_spread(self):
        a = {(0, "A0"): {"solo": [0.0, 1.0], "batch_vs_solo": 0.01}}
        b = {(0, "A0"): {"solo": [0.0, 1.5], "batch_vs_solo": 0.02}}
        (row,) = equivalence_compare(a, b)
        self.assertAlmostEqual(row["spread"], 0.5)
        self.assertAlmostEqual(row["drift"], 0.02)
        self.assertAlmostEqual(row["ratio"], 0.04)


@unittest.skipUnless(HAS_COMPARE, "compare_runs missing")
class TestCompareRunsStats(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "c.jsonl"), "w") as fh:
            for row in ({"mode": "batched", "batch": 2, "infer_ms": 10000.0},
                        {"mode": "warmup", "batch": 3, "infer_ms": 290.0},
                        {"mode": "batched", "batch": 2, "infer_ms": 210.0},
                        {"mode": "batched", "batch": 2, "infer_ms": 214.0},
                        {"mode": "fallback", "error": "RESOURCE_EXHAUSTED"}):
                fh.write(json.dumps(row) + "\n")

    def test_compiles_and_warmup_excluded(self):
        s = run_stats({"name": "c", "timing_log": "c.jsonl", "wall_s": 10},
                      self.dir)
        self.assertEqual(s["calls"], 2)
        self.assertEqual(s["compiles"], 1)
        self.assertEqual(s["fallbacks"], 1)
        self.assertEqual(s["median_ms_by_batch"], {2: 212.0})

    def test_pairwise_matches_same_task_and_n(self):
        runs = [{"name": "base", "strategy": "S0", "task": "t", "n_envs": 2,
                 "eval_time": 12, "success_rate": 8 / 12},
                {"name": "x", "strategy": "S1", "task": "t", "n_envs": 2,
                 "eval_time": 12, "success_rate": 5 / 12},
                {"name": "other", "strategy": "S1", "task": "t", "n_envs": 3,
                 "eval_time": 12, "success_rate": 0.5}]
        md = pairwise(runs)
        self.assertIn("| x | base | 5/12 | 8/12 | 0.41 |", md)
        self.assertNotIn("| other |", md)

    def test_episodes_override_requested_count(self):
        from compare_runs import _successes

        run = {"eval_time": 54, "success_rate": 0.4,
               "episodes": [[i, i < 10, 1.0] for i in range(25)]}
        self.assertEqual(_successes(run), (10, 25))


def _patched_server_module():
    """model_server with the batch-window patch applied, else None."""
    root = os.path.join(os.path.dirname(__file__), os.pardir, "RoboDojo")
    for path in (root, os.path.join(root, "XPolicyLab")):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from client_server.ws import model_server
    except Exception:
        return None
    return model_server if hasattr(model_server, "_SERVER_BATCH") else None


@unittest.skipUnless(_patched_server_module(), "patched model_server not importable")
class TestServerBatchWindow(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.batches = []

        def infer_obs(self, obs):
            return [obs]

        def infer_obs_batch(self, payload):
            import time

            time.sleep(0.05)  # a forward: lets later arrivals queue up
            self.batches.append(len(payload["obs_list"]))
            return [[o] for o in payload["obs_list"]]

    def _run(self, n, window_s=0.0, cap=0, stagger_s=0.0):
        import asyncio

        ms = _patched_server_module()
        ms._BATCH_WINDOW_S, ms._BATCH_MAX = window_s, cap
        model = self.FakeModel()
        server = ms.PolicyServer(model)

        async def go():
            async def one(i):
                await asyncio.sleep(i * stagger_s)
                return await server._batched_infer_obs(i)
            return await asyncio.gather(*(one(i) for i in range(n)))

        return asyncio.run(go()), model.batches

    def test_each_client_gets_its_own_answer(self):
        results, _ = self._run(5)
        self.assertEqual([r for r, _ in results], [[i] for i in range(5)])

    def test_concurrent_arrivals_share_a_forward(self):
        _, batches = self._run(4)
        self.assertEqual(batches, [4])

    def test_cap_splits_batches(self):
        _, batches = self._run(5, cap=2)
        self.assertEqual(batches, [2, 2, 1])

    def test_continuous_batching_collects_during_forward(self):
        # arrivals 20 ms apart: first goes alone, the rest queue behind it
        _, batches = self._run(3, stagger_s=0.02)
        self.assertEqual(batches, [1, 2])

    def test_window_waits_for_stragglers(self):
        _, batches = self._run(3, window_s=0.08, stagger_s=0.02)
        self.assertEqual(batches, [3])

    def test_errors_reach_every_waiter(self):
        import asyncio

        ms = _patched_server_module()
        ms._BATCH_WINDOW_S, ms._BATCH_MAX = 0.0, 0

        class Broken(self.FakeModel):
            def infer_obs_batch(self, payload):
                raise RuntimeError("boom")

        server = ms.PolicyServer(Broken())

        async def go():
            return await asyncio.gather(
                *(server._batched_infer_obs(i) for i in range(3)),
                return_exceptions=True)

        out = asyncio.run(go())
        self.assertTrue(all(isinstance(e, RuntimeError) for e in out))


if __name__ == "__main__":
    unittest.main()
