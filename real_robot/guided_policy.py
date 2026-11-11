"""Verifier-guided pi0.5: propose many action chunks, run the one the verifier prefers.

Each replan: for every instruction in `instructions` the VLM prefix runs once and N
noises (scaled by each of `scales`) share its KV cache; the DINOv2 features of the
current frame are computed once; the verifier scores all candidates (~ms). If the best
score is below `min_score` (no candidate looks demo-like) the caller can fall back to a
guarded recovery. Drop-in for LerobotPolicy.call("get_action").
"""
from __future__ import annotations

import time

import numpy as np

from .pi05_samples import Sampler
from .verifier import chunk_feat, load


class GuidedPolicy:
    def __init__(self, ckpt: str, verifier: str, instructions: list[str] | None = None,
                 scales=(1.0, 1.5), n: int = 8, device: str = "cuda"):
        import torch

        from .jevlite import Encoder

        self.s = Sampler(ckpt, device=device)
        self.h = load(verifier, device)
        self.enc = Encoder("dinos", grid=4)
        self.instructions = instructions
        self.scales, self.n = list(scales), n
        self.torch = torch
        self.last: dict = {}

    def act(self, images: dict, state: np.ndarray, instruction: str, seed: int = 0) -> np.ndarray:
        torch = self.torch
        t0 = time.perf_counter()
        item = {"images": images, "state": np.asarray(state, np.float32)}
        cands, tags = [], []
        for ins in [instruction] + [i for i in (self.instructions or []) if i != instruction]:
            for sc in self.scales:
                c = self.s.sample(item, ins, self.n, sc, seed=seed, shared=True)
                cands.append(c)
                tags += [(ins, sc)] * len(c)
        C = np.concatenate(cands)
        t1 = time.perf_counter()
        x = np.concatenate([self.enc([images["wrist"]]).reshape(-1), self.enc([images["base"]]).reshape(-1)])
        cf = chunk_feat(C, item["state"][None].repeat(len(C), 0)).astype(np.float32)
        with torch.no_grad():
            sc = self.h(torch.tensor(x[None], dtype=torch.float32, device=self.s.p.device),
                        torch.tensor(item["state"][None], device=self.s.p.device),
                        torch.tensor(cf[None], device=self.s.p.device))[0].float().cpu().numpy()
        b = int(sc.argmax())
        p = np.exp(sc - sc.max()); p /= p.sum()
        self.last = {"instruction": tags[b][0], "scale": tags[b][1], "p_best": float(p[b]),
                     "n": len(C), "sample_ms": 1000 * (t1 - t0), "verify_ms": 1000 * (time.perf_counter() - t1)}
        return C[b]
