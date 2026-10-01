"""Tiny closed-loop serving simulator (stdlib only, Mac-safe).

Replays what we measured on the GPU box: robots eat one action per control
step, the server answers in batches that take as long as our real timing
logs say. Fault knobs (dropped messages, extra delay) are the DRTC ideas:
newest-plan-wins merging and cooldowns only help if we can show the harm.

Robot timing follows the box: 25 Hz control steps (collect_freq in
env_cfg/arx_x5.yml), chunk_len actions per answer, infer_ms sampled from
measured pools keyed by batch size.
"""
from __future__ import annotations

import random

from schedulers import PendingRequest, SCHEDULERS


class FleetSim:
    # chunk_len default = pi0.5 action_horizon (openpi pi0.py Pi0Config,
    # action_horizon=50): one answer = 50 control steps of motion.
    def __init__(self, n_robots=3, chunk_len=50, step_ms=40.0,
                 infer_pools=None, max_batch=4, scheduler="sequential",
                 drop_rate=0.0, seed=0):
        self.n_robots = n_robots
        # chunk_len may be one number (same for all robots) or a list with
        # one entry per robot (mixed fleet: slow-deliberate + reactive robots,
        # the Armory heterogeneity setup).
        if isinstance(chunk_len, (list, tuple)):
            assert len(chunk_len) == n_robots
            self.chunk_lens = list(chunk_len)
        else:
            self.chunk_lens = [chunk_len] * n_robots
        self.chunk_len = self.chunk_lens[0]
        self.step_ms = step_ms
        # batch size -> list of measured infer_ms (first-call warmup excluded
        # by callers; steady-state serving is what we compare)
        self.infer_pools = infer_pools or {1: [125.0]}
        self.max_batch = max_batch
        self.scheduler = scheduler
        self.drop_rate = drop_rate
        self.rng = random.Random(seed)
        self.actions_left = list(self.chunk_lens)
        self.arrived = [0.0] * n_robots
        self.starved_steps = 0
        self.total_steps = 0
        self.completed_chunks = 0
        self.gpu_free_at = 0.0
        self.now = 0.0

    def _sample_infer_ms(self, batch):
        pool = self.infer_pools.get(batch) or self.infer_pools[max(self.infer_pools)]
        return self.rng.choice(pool)

    def _serve_once(self):
        # Robots asking for a new plan: fewest actions left first (all
        # schedulers see the same queue; they differ in grouping).
        needy = [i for i in range(self.n_robots) if self.actions_left[i] <= 1]
        if not needy:
            return False
        pending = [PendingRequest(str(i), self.actions_left[i], self.arrived[i])
                   for i in needy]
        fn = SCHEDULERS[self.scheduler]
        try:
            batches = fn(pending, self.max_batch)
        except TypeError:
            batches = fn(pending)
        batch = batches[0]
        infer_ms = self._sample_infer_ms(len(batch))
        # GPU is busy until now + infer_ms; the clock (run loop) advances on
        # its own, so a slow answer blocks the next batch. Do NOT fast-forward
        # self.now here (that bug made speed irrelevant).
        self.gpu_free_at = self.now + infer_ms
        for r in batch:
            i = int(r.robot_id)
            if self.rng.random() < self.drop_rate:
                continue  # lost answer (DRTC fault); robot keeps starving
            self.actions_left[i] = self.chunk_lens[i]
            self.arrived[i] = self.gpu_free_at
            self.completed_chunks += 1
        return True

    def run(self, steps=2000):
        for _ in range(steps):
            self.total_steps += 1
            for i in range(self.n_robots):
                if self.actions_left[i] > 0:
                    self.actions_left[i] -= 1
                else:
                    self.starved_steps += 1
            if self.now >= self.gpu_free_at:
                self._serve_once()
            self.now += self.step_ms
        return {
            "starvation_rate": self.starved_steps / max(1, self.total_steps * self.n_robots),
            "completed_chunks": self.completed_chunks,
        }


def compare(schedulers, infer_pools, n_robots=3, drop_rate=0.0, seed=0, steps=2000):
    out = {}
    for name in schedulers:
        sim = FleetSim(n_robots=n_robots, infer_pools=infer_pools,
                       scheduler=name, drop_rate=drop_rate, seed=seed)
        out[name] = sim.run(steps)
    return out
