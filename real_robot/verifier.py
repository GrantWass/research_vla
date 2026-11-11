"""Action verifier: score(camera features, candidate action chunk) -> how demo-like.

Trained only from demos (no policy samples), so it is policy-agnostic: for each
training frame the candidates are the demo's own chunk, time-shifted chunks of the
same episode (too early / too late / stalled), a "hold still" chunk, noisy copies
and chunks from other episodes. Target: softmax(-d / tau) over candidates, d = joint
RMSE to the demo chunk. Listwise cross-entropy.

python -m real_robot.verifier train --feats outputs/jevlite/feats_dinos_g4.npz --dataset demos/cube_grid_all_pi05 --out outputs/verifier/v1
python -m real_robot.verifier eval --model outputs/verifier/v1 --samples outputs/verifier/samples_004033.npz --dataset demos/cube_grid_all_pi05
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

K, STRIDE = 15, 3
GRIP_W, GRIP_SCALE = 1.0, 10.0  # v2: the grasp decision is the gripper; v1 used 0.3, 3  # score the first 0.5 s of a chunk, every 3rd step -> 5 x 7
J = 7


def chunk_feat(chunks: np.ndarray, state: np.ndarray) -> np.ndarray:
    """(..., >=K, 7) absolute chunks, (..., 13) state -> (..., 35) relative, scaled."""
    c = chunks[..., :K:STRIDE, :J] - state[..., None, :J]
    return (c * np.array([10, 10, 10, 10, 10, 10, GRIP_SCALE], np.float32)).reshape(*c.shape[:-2], -1)


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """joint RMSE (rad, gripper down-weighted) over the first K steps."""
    w = np.array([1, 1, 1, 1, 1, 1, GRIP_W], np.float32)
    return np.sqrt((((a[..., :K, :J] - b[..., :K, :J]) * w) ** 2).mean((-1, -2)))


def episode_actions(dataset: str):
    import pandas as pd
    import glob

    df = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "observation.state"])
                    for p in sorted(glob.glob(os.path.join(dataset, "data", "*", "*.parquet")))])
    out = {}
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(e)] = (np.stack(g["action"].values).astype(np.float32), np.stack(g["observation.state"].values).astype(np.float32))
    return out


def window(act: np.ndarray, f: int) -> np.ndarray:
    idx = np.minimum(np.arange(f, f + K), len(act) - 1)
    return act[idx]


def candidates(rng, acts, e: int, f: int, state: np.ndarray, n_noise: int = 4):
    a = acts[e][0]
    c = [window(a, f)]
    for s in (-60, -30, -15, -6, 6, 15, 30, 60, 90):
        if 0 <= f + s < len(a):
            c.append(window(a, f + s))
    c.append(np.repeat(state[:7][None], K, 0))  # hold still: joints and gripper where they are (the hover)
    for sc in (0.02, 0.05, 0.1, 0.2)[:n_noise]:
        c.append(c[0] + rng.normal(0, sc, c[0].shape).astype(np.float32).cumsum(0) / np.sqrt(K))
    for _ in range(3):
        e2 = int(rng.choice(list(acts)))
        a2, s2 = acts[e2]
        f2 = int(rng.integers(0, len(a2)))
        c.append(window(a2, f2) - s2[f2, None, :7] * np.r_[np.ones(6), 0] + np.r_[state[:6], 0] * np.ones((K, 1)))
    return np.stack(c)


class Head:
    def __init__(self, in_img: int, torch, hidden: int = 512):
        nn = torch.nn
        self.img = nn.Sequential(nn.LayerNorm(in_img), nn.Linear(in_img, hidden), nn.GELU())
        self.act = nn.Sequential(nn.Linear(35 + 13, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mods = nn.ModuleList([self.img, self.act, self.out])

    def __call__(self, x, st, c):
        """x (B, D), st (B, 13), c (B, M, 35) -> (B, M)"""
        h = self.img(x)[:, None] * 1.0
        a = self.act(self.torch_cat(c, st))
        return self.out(h * a).squeeze(-1)

    def torch_cat(self, c, st):
        import torch
        return torch.cat([c, st[:, None].expand(-1, c.shape[1], -1)], -1)


def build(in_img):
    import torch
    h = Head(in_img, torch)
    return h


def train(a):
    import torch

    from .pi05_samples import UNSEEN
    z = np.load(a.feats, allow_pickle=True)
    X = np.concatenate([z["wrist"], z["base"]], 1).astype(np.float16)
    E, F, S = z["episode"], z["frame"], z["state"].astype(np.float32)
    acts = episode_actions(a.dataset)
    tr = np.array([int(e) not in UNSEEN for e in E])
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    ph = z["phase"]
    grasp = np.isin(np.asarray(ph).astype(str), ["2", "grasp", "1", "approach"])
    idx = np.concatenate([idx] + [idx[grasp[idx]]] * a.oversample)
    print(f"train frames {len(idx)}", flush=True)
    C, D = [], []
    for i in idx:
        c = candidates(rng, acts, int(E[i]), int(F[i]), S[i])
        M = 17
        if len(c) < M:
            c = np.concatenate([c, c[-1:].repeat(M - len(c), 0)])
        C.append(chunk_feat(c[:M], S[i][None].repeat(M, 0)))
        D.append(dist(c[:M], c[0][None]))
    C, D = np.stack(C).astype(np.float32), np.stack(D).astype(np.float32)
    dev = "cuda"
    h = build(X.shape[1])
    h.mods.to(dev)
    opt = torch.optim.AdamW(h.mods.parameters(), lr=3e-4, weight_decay=1e-2)
    Xt = torch.tensor(X[idx]); St = torch.tensor(S[idx]); Ct = torch.tensor(C); Dt = torch.tensor(D)
    tau = a.tau
    for ep in range(a.epochs):
        perm = torch.randperm(len(idx))
        tot = 0
        for b in range(0, len(idx), 256):
            j = perm[b:b + 256]
            x = Xt[j].to(dev).float() + 0.05 * torch.randn(len(j), X.shape[1], device=dev)
            s = h(x, St[j].to(dev), Ct[j].to(dev))
            tgt = torch.softmax(-Dt[j].to(dev) / tau, -1)
            loss = -(tgt * torch.log_softmax(s, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        print(f"epoch {ep} loss {tot / len(idx):.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(h.mods.state_dict(), os.path.join(a.out, "head.pt"))
    json.dump({"in_img": int(X.shape[1]), "tau": tau, "K": K, "stride": STRIDE}, open(os.path.join(a.out, "config.json"), "w"))


def load(model_dir, dev="cuda"):
    import torch
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    h = build(cfg["in_img"])
    h.mods.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
    h.mods.to(dev).eval()
    return h


def evaluate(a):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .jevlite import Encoder
    z = np.load(a.samples, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    C, demo, state = z["cands"], z["demo"], z["state"]  # C: (F, I, S, N, H, 7)
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    enc = Encoder("dinos", grid=4)
    feats = []
    for it in meta["items"]:
        x = ds[it["idx"]]
        im = {c: (x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in ("wrist", "base")}
        feats.append(np.concatenate([enc([im["wrist"]]).reshape(-1), enc([im["base"]]).reshape(-1)]))
    X = torch.tensor(np.stack(feats), dtype=torch.float32, device="cuda")
    h = load(a.model)
    Fn, I, Sn, N = C.shape[:4]
    d = dist(C, demo[:, None, None, None])  # (F, I, S, N)
    with torch.no_grad():
        flat = C.reshape(Fn, -1, C.shape[-2], 7)
        cf = chunk_feat(flat, state[:, None].repeat(flat.shape[1], 1))
        sc = h(X, torch.tensor(state, device="cuda"), torch.tensor(cf, device="cuda")).cpu().numpy().reshape(Fn, I, Sn, N)
    rep = {"frames": Fn, "instr": meta["instr"], "scales": meta["scales"], "n": N,
           "sampler_latency_ms": meta.get("latency_ms_median")}

    def summarize(mask_i, mask_s, n):
        dd = d[:, mask_i][:, :, mask_s][..., :n].reshape(Fn, -1)
        ss = sc[:, mask_i][:, :, mask_s][..., :n].reshape(Fn, -1)
        pick = dd[np.arange(Fn), ss.argmax(1)]
        return {"single": round(float(d[:, 0, meta["scales"].index(1.0), 0].mean()), 4),
                "random": round(float(dd.mean()), 4), "oracle": round(float(dd.min(1).mean()), 4),
                "verifier": round(float(pick.mean()), 4),
                "verifier_beats_single": round(float((pick < d[:, 0, meta["scales"].index(1.0), 0]).mean()), 3)}
    allI, allS = np.ones(I, bool), np.ones(Sn, bool)
    main = np.zeros(I, bool); main[0] = True
    s1 = np.array([s == 1.0 for s in meta["scales"]])
    for n in (1, 2, 4, 8, 16):
        if n <= N:
            rep[f"main_s1_n{n}"] = summarize(main, s1, n)
    for v, sc_ in enumerate(meta["scales"]):
        m = np.zeros(Sn, bool); m[v] = True
        rep[f"main_scale{sc_}_n{N}"] = summarize(main, m, N)
    rep[f"main_allscales_n{N}"] = summarize(main, allS, N)
    rep[f"allinstr_s1_n{N}"] = summarize(allI, s1, N)
    rep[f"allinstr_allscales_n{N}"] = summarize(allI, allS, N)
    # does the chunk start closing on the cube when the demo does (the live hover failure)?
    g0 = state[:, 6]
    closes = C[..., 6].min(-1) < 0.6 * g0[:, None, None, None]          # (F, I, S, N)
    demo_closes = demo[..., 6].min(-1) < 0.6 * g0
    flat_sc = sc.reshape(Fn, -1); flat_cl = closes.reshape(Fn, -1)
    pick_cl = flat_cl[np.arange(Fn), flat_sc.argmax(1)]
    k = demo_closes
    if k.any():
        rep["closing_frames"] = {"n": int(k.sum()),
                                 "single_closes": round(float(closes[k, 0, s1.argmax(), 0].mean()), 3),
                                 "any_candidate_closes": round(float(flat_cl[k].any(1).mean()), 3),
                                 "main_instr_any_closes": round(float(closes[k, 0].reshape(k.sum(), -1).any(1).mean()), 3),
                                 "verifier_pick_closes": round(float(pick_cl[k].mean()), 3),
                                 "per_instr_close_rate": closes[k].mean((0, 2, 3)).round(3).tolist(),
                                 "per_scale_close_rate": closes[k].mean((0, 1, 3)).round(3).tolist()}
    rep["false_close_rate_on_other_frames"] = {"single": round(float(closes[~k, 0, s1.argmax(), 0].mean()), 3),
                                               "verifier_pick": round(float(pick_cl[~k].mean()), 3)}
    pts = sorted({it["point"] for it in meta["items"]})
    P = np.array([it["point"] for it in meta["items"]])
    rep["by_point_allinstr_allscales"] = {}
    for p in pts:
        k = P == p
        dd = d[k].reshape(k.sum(), -1); ss = sc[k].reshape(k.sum(), -1)
        rep["by_point_allinstr_allscales"][p] = {
            "single": round(float(d[k][:, 0, s1.argmax(), 0].mean()), 4), "oracle": round(float(dd.min(1).mean()), 4),
            "verifier": round(float(dd[np.arange(k.sum()), ss.argmax(1)].mean()), 4),
            "picked_instr": np.bincount(ss.argmax(1) // (Sn * N), minlength=I).tolist()}
    # with timing: verifier cost on all candidates
    import time
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(20):
            h(X[:1], torch.tensor(state[:1], device="cuda"), torch.tensor(cf[:1], device="cuda")); torch.cuda.synchronize()
        rep["verifier_ms_all_candidates"] = round(1000 * (time.perf_counter() - t0) / 20, 2)
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


def train_on_samples(a):
    """v3: candidates are the policy's own samples (+ the demo chunk and hold-still), target softmax(-d/tau)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .jevlite import Encoder
    z = np.load(a.samples, allow_pickle=True)
    meta = json.loads(str(z["meta"])); Cs, demo, state = z["cands"], z["demo"], z["state"]
    Fn = Cs.shape[0]
    flat = Cs.reshape(Fn, -1, Cs.shape[-2], 7)[:, :, :K]
    hold = np.repeat(state[:, None, :7], K, 1)[:, None]
    cands = np.concatenate([demo[:, None, :K], hold, flat], 1)
    D = dist(cands, demo[:, None])
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    enc = Encoder("dinos", grid=4)
    X = []
    for it in meta["items"]:
        x = ds[it["idx"]]
        im = {c: (x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in ("wrist", "base")}
        X.append(np.concatenate([enc([im["wrist"]]).reshape(-1), enc([im["base"]]).reshape(-1)]))
    X = np.stack(X).astype(np.float32)
    CF = chunk_feat(cands, state[:, None].repeat(cands.shape[1], 1)).astype(np.float32)
    dev = "cuda"
    h = build(X.shape[1])
    if a.init:
        h.mods.load_state_dict(torch.load(os.path.join(a.init, "head.pt"), map_location="cpu"))
    h.mods.to(dev)
    opt = torch.optim.AdamW(h.mods.parameters(), lr=a.lr, weight_decay=1e-2)
    Xt, St, Ct, Dt = (torch.tensor(v, device=dev) for v in (X, state.astype(np.float32), CF, D.astype(np.float32)))
    M = Ct.shape[1]
    for ep in range(a.epochs):
        perm = torch.randperm(Fn, device=dev); tot = 0
        for b in range(0, Fn, 32):
            j = perm[b:b + 32]
            sub = torch.randperm(M - 2, device=dev)[:62] + 2  # demo, hold + 62 random samples
            cols = torch.cat([torch.tensor([0, 1], device=dev), sub])
            s = h(Xt[j] + 0.05 * torch.randn_like(Xt[j]), St[j], Ct[j][:, cols])
            tgt = torch.softmax(-Dt[j][:, cols] / a.tau, -1)
            loss = -(tgt * torch.log_softmax(s, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        if ep % 10 == 9:
            print(f"epoch {ep} loss {tot / Fn:.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(h.mods.state_dict(), os.path.join(a.out, "head.pt"))
    json.dump({"in_img": int(X.shape[1]), "tau": a.tau, "K": K, "stride": STRIDE, "from_samples": a.samples},
              open(os.path.join(a.out, "config.json"), "w"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--feats", required=True); t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True); t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--tau", type=float, default=0.02)
    t.add_argument("--oversample", type=int, default=4, help="extra copies of approach/grasp frames")
    t3 = sub.add_parser("train-samples")
    t3.add_argument("--samples", required=True); t3.add_argument("--dataset", required=True)
    t3.add_argument("--out", required=True); t3.add_argument("--epochs", type=int, default=60)
    t3.add_argument("--tau", type=float, default=0.02); t3.add_argument("--lr", type=float, default=3e-4)
    t3.add_argument("--init", default="")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True); e.add_argument("--samples", required=True)
    e.add_argument("--dataset", required=True); e.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"train": train, "eval": evaluate, "train-samples": train_on_samples}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
