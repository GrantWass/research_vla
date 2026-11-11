"""Sample many pi0.5 action chunks per observation: noise scale x instruction x N.

Offline study for verifier-guided sampling: does the weak checkpoint ever propose
the demonstrator's action, and how does that depend on the noise scale and the
instruction? Frames: held-out demo episodes around the grasp (where 004033 hovers
live), the carry and the release.

python -m real_robot.pi05_samples --ckpt <merged ckpt> --dataset demos/cube_grid_all_pi05 \
    --out outputs/verifier/samples_004033.npz --n 16 --scales 0.5,1,1.5,2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# episodes the 004033 checkpoint never trained on (its train_config lists the rest)
UNSEEN = (7, 25, 35, 50, 70, 90)
INSTR = ["pick up the Rubik's cube and put it in the bowl",
         "move down to the Rubik's cube",
         "grasp the Rubik's cube",
         "put the Rubik's cube in the bowl",
         "open the gripper"]
# frames relative to the demo's segments (open o, close start i, stall j, release k)
POINTS = {"approach_45": lambda s: s[1] - 45, "approach_30": lambda s: s[1] - 30,
          "pregrasp_15": lambda s: s[1] - 15, "pregrasp_5": lambda s: s[1] - 5,
          "carry": lambda s: s[2] + 30, "prerelease": lambda s: s[3] - 10}


def frames(ds, H: int, max_eps: int, episodes=UNSEEN, every: int = 30):
    from .jevlite_data import segments

    eps = ds.meta.episodes
    out = []
    for e in range(ds.meta.total_episodes):
        if e not in episodes:
            continue
        lo, hi = int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])
        sub = ds.hf_dataset.select(range(lo, hi))
        st = np.asarray(sub["observation.state"], np.float32)
        act = np.asarray(sub["action"], np.float32)
        seg = segments(st[:, 6])
        if seg is None:
            continue
        for name, f in POINTS.items():
            f = f(seg)
            if 0 <= f and f + H <= len(st):
                out.append({"episode": e, "frame": f, "point": name, "idx": lo + f,
                            "state": st[f], "demo": act[f:f + H]})
        for f in range(0, len(st) - H, every):
            out.append({"episode": e, "frame": f, "point": "every", "idx": lo + f,
                        "state": st[f], "demo": act[f:f + H]})
        if len({o["episode"] for o in out}) >= max_eps:
            break
    return out


class Sampler:
    """pi0.5 with the VLM prefix computed once per (image, instruction) and shared by N noises."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        from .lerobot_policy import LerobotPolicy

        self.p = LerobotPolicy(ckpt, camera_map={"base": "base", "wrist": "wrist"}, device=device)
        self.torch = self.p.torch
        self.model = self.p.policy.model
        self.cfg = self.p.policy.config

    def batch(self, item: dict, instruction: str):
        from lerobot.policies.utils import prepare_observation_for_inference

        raw = {"observation.state": item["state"]}
        for cam in ("base", "wrist"):
            raw[f"observation.images.{cam}"] = item["images"][cam]
        b = prepare_observation_for_inference(raw, self.p.device, instruction, self.p.robot_type)
        return self.p.pre(b)

    def sample(self, item: dict, instruction: str, n: int, scale: float, seed: int = 0,
               shared: bool = False) -> np.ndarray:
        """(n, chunk, 7) absolute joint chunks."""
        torch = self.torch
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        b = self.batch(item, instruction)
        pol = self.p.policy
        g = torch.Generator(device=self.p.device).manual_seed(seed)
        noise = scale * torch.randn((n, self.cfg.chunk_size, self.cfg.max_action_dim), generator=g,
                                    device=self.p.device, dtype=torch.float32)
        with torch.inference_mode():
            images, masks = pol._preprocess_images(b)
            tok, tm = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
            rep = lambda x: x.expand(n, *x.shape[1:]) if x.shape[0] == 1 else x
            if shared:
                a = sample_shared(self.model, images, masks, tok, tm, noise)
            else:
                a = self.model.sample_actions([rep(i) for i in images], [rep(m) for m in masks],
                                              rep(tok), rep(tm), noise=noise)
            a = a[:, :, :self.cfg.output_features["action"].shape[0]]
            a = self.p.post(a)
        a = a.float().cpu().numpy()
        st = b["observation.state"] if False else item["state"]
        return np.stack([self.p.absolute(c, st) for c in a])


def sample_shared(model, images, masks, tok, tm, noise, num_steps=None):
    """sample_actions with the VLM prefix run once (batch 1) and its KV cache
    repeated for the N noises: the prefix is most of pi0.5's compute."""
    import torch

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    n = noise.shape[0]
    num_steps = num_steps or model.config.num_inference_steps
    pe, pm, pa = model.embed_prefix(images, masks, tok, tm)
    a2 = make_att_2d_masks(pm, pa)
    pos = torch.cumsum(pm, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, kv = model.paligemma_with_expert.forward(attention_mask=model._prepare_attention_masks_4d(a2),
                                                position_ids=pos, past_key_values=None,
                                                inputs_embeds=[pe, None], use_cache=True)
    kv.batch_repeat_interleave(n)
    pm = pm.expand(n, -1)
    dt = -1.0 / num_steps
    x = noise
    for k in range(num_steps):
        t = torch.tensor(1.0 + k * dt, dtype=torch.float32, device=noise.device).expand(n)
        x = x + dt * model.denoise_step(prefix_pad_masks=pm, past_key_values=kv, x_t=x, timestep=t)
    return x


def _kv_tensors(kv):
    if hasattr(kv, "layers"):
        return [t for L in kv.layers for t in (L.keys, L.values)]
    return list(kv.key_cache) + list(kv.value_cache)


def run_prefix(model, images, masks, tok, tm):
    import torch

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    pe, pm, pa = model.embed_prefix(images, masks, tok, tm)
    a2 = make_att_2d_masks(pm, pa)
    pos = torch.cumsum(pm, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, kv = model.paligemma_with_expert.forward(attention_mask=model._prepare_attention_masks_4d(a2),
                                                position_ids=pos, past_key_values=None,
                                                inputs_embeds=[pe, None], use_cache=True)
    return pm, kv


class GraphDenoiser:
    """The N-sample denoise loop captured as one CUDA graph. Eager pi0.5 issues ~1,000
    kernel launches per step (31 ms wall, 5.5 ms GPU); replaying a graph removes the
    launch cost. Prefix (KV cache) and noise are copied into static buffers per call.
    Valid while the prefix length stays fixed (same cameras, padded instruction)."""

    def __init__(self, model, images, masks, tok, tm, n: int, num_steps: int = 10):
        import torch

        self.torch, self.model, self.n, self.steps = torch, model, n, num_steps
        pm, kv = run_prefix(model, images, masks, tok, tm)
        kv.batch_repeat_interleave(n)
        self.kv, self.pm = kv, pm.expand(n, -1).clone()
        self.static = _kv_tensors(kv)
        cfg = model.config
        self.x = torch.zeros(n, cfg.chunk_size, cfg.max_action_dim, device=pm.device)
        dt = -1.0 / num_steps
        self.ts = [torch.full((n,), 1.0 + k * dt, dtype=torch.float32, device=pm.device) for k in range(num_steps)]
        self.dt = dt

        def loop():
            x = self.x
            for t in self.ts:
                x = x + self.dt * model.denoise_step(prefix_pad_masks=self.pm, past_key_values=self.kv, x_t=x, timestep=t)
            return x

        # pi0.5's embed_suffix rebuilds a constant mask with torch.tensor(list, device=cuda) each
        # step: a host->device copy, illegal while capturing. Memoise such constants (filled by
        # the warm-up runs, reused during capture); only list/tuple constants are cached.
        real_tensor, cache = torch.tensor, {}

        def const_tensor(data, *a, **kw):
            if isinstance(data, (list, tuple)):
                key = (tuple(data), str(kw.get("dtype")), str(kw.get("device")))
                if key not in cache:
                    cache[key] = real_tensor(data, *a, **kw)
                return cache[key]
            return real_tensor(data, *a, **kw)

        torch.tensor = const_tensor
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    loop()
            torch.cuda.current_stream().wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = loop()
        finally:
            torch.tensor = real_tensor

    def __call__(self, images, masks, tok, tm, noise):
        pm, kv = run_prefix(self.model, images, masks, tok, tm)
        if pm.shape[1] != self.pm.shape[1]:
            raise ValueError("prefix length changed; rebuild the graph")
        for dst, src in zip(self.static, _kv_tensors(kv)):
            dst.copy_(src.expand_as(dst))
        self.pm.copy_(pm.expand_as(self.pm))
        self.x.copy_(noise)
        self.graph.replay()
        return self.out.clone()


def main(argv=None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="demos/cube_grid_all_pi05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--scales", default="0.5,1,1.5,2")
    ap.add_argument("--instr", default="all", help="'all' or 'main'")
    ap.add_argument("--max-eps", type=int, default=12)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--episodes", default="unseen", help="unseen, or comma-separated indices")
    ap.add_argument("--shared", action="store_true", help="share the VLM prefix across the N noises")
    a = ap.parse_args(argv)
    scales = [float(s) for s in a.scales.split(",")]
    instr = INSTR if a.instr == "all" else INSTR[:1]
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    eps = UNSEEN if a.episodes == "unseen" else tuple(int(e) for e in a.episodes.split(","))
    items = frames(ds, a.H, a.max_eps, episodes=eps)
    print(f"{len(items)} frames from {len({i['episode'] for i in items})} episodes", flush=True)
    s = Sampler(a.ckpt)
    C = np.zeros((len(items), len(instr), len(scales), a.n, a.H, 7), np.float32)
    lat = []
    for k, it in enumerate(items):
        x = ds[it["idx"]]
        it["images"] = {cam: (x[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        for cam in ("base", "wrist")}
        for u, ins in enumerate(instr):
            for v, sc in enumerate(scales):
                t0 = time.perf_counter()
                C[k, u, v] = s.sample(it, ins, a.n, sc, seed=k, shared=a.shared)[:, :a.H]
                s.torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
        print(k, it["point"], f"{1000 * np.median(lat):.0f} ms per {a.n} samples", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, cands=C, demo=np.stack([i["demo"] for i in items]),
                        state=np.stack([i["state"] for i in items]),
                        meta=json.dumps({"instr": instr, "scales": scales, "n": a.n,
                                         "items": [{k: i[k] for k in ("episode", "frame", "point", "idx")}
                                                   for i in items],
                                         "latency_ms_median": 1000 * float(np.median(lat))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
