"""Progress scorer: how far along the task would the arm be if it reached this state?

P(image features now, future joint+gripper state) -> task progress in [0, 1].
Trained from demos only: for frame t and horizon d (0..1.5 s), the state the demo
reached at t+d has progress (t+d)/n. Corrupted futures (noise on the demo's future)
get progress(t) minus a penalty, so leaving the demo manifold scores worse than
staying put. To pick among pi0.5 candidates, score each chunk's state after `--at`
steps and take the max: hovering keeps progress(t); descending and closing on the
cube raise it. The gripper is part of the state, so the grasp decision is scored too.

python -m real_robot.progress train --feats outputs/jevlite/feats_dinos_g4.npz --dataset demos/cube_grid_all_pi05 --out outputs/progress/p1
python -m real_robot.progress eval --model outputs/progress/p1 --samples outputs/verifier/samples_004033.npz --dataset demos/cube_grid_all_pi05
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .verifier import dist, episode_actions

DELTAS = (0, 3, 6, 9, 15, 21, 30, 45)
SCALE = np.array([1, 1, 1, 1, 1, 1, 1], np.float32)


class Head:
    def __init__(self, in_img: int, hidden: int = 512):
        import torch.nn as nn

        self.img = nn.Sequential(nn.LayerNorm(in_img), nn.Linear(in_img, hidden), nn.GELU())
        self.st = nn.Sequential(nn.Linear(14, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mods = nn.ModuleList([self.img, self.st, self.out])

    def __call__(self, x, now, fut):
        """x (B, D), now (B, 7), fut (B, M, 7) -> (B, M) progress."""
        import torch

        h = self.img(x)[:, None]
        s = self.st(torch.cat([fut - now[:, None], fut], -1))
        return torch.sigmoid(self.out(h * s).squeeze(-1))


def load(model_dir, dev="cuda"):
    import torch

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    h = Head(cfg["in_img"])
    h.mods.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
    h.mods.to(dev).eval()
    return h, cfg


def train(a):
    import torch

    from .pi05_samples import UNSEEN

    z = np.load(a.feats, allow_pickle=True)
    E, F = z["episode"].astype(int), z["frame"].astype(int)
    tr = np.flatnonzero(~np.isin(E, UNSEEN))
    X = torch.tensor(np.concatenate([z["wrist"][tr], z["base"][tr]], 1).astype(np.float16))
    acts = episode_actions(a.dataset)
    rng = np.random.default_rng(0)
    now, fut, tgt = [], [], []
    for i in tr:
        st = acts[E[i]][1]
        n = len(st)
        f = F[i]
        p0 = f / n
        row_f, row_t = [], []
        for d in DELTAS:
            g = min(f + d, n - 1)
            row_f.append(st[g, :7]); row_t.append(g / n)
        for sc in (0.03, 0.08, 0.2, 0.4):  # off-manifold futures: worse than standing still
            g = min(f + int(rng.choice(DELTAS[1:])), n - 1)
            noise = rng.normal(0, sc, 7).astype(np.float32) * np.r_[np.ones(6), 0.5]
            row_f.append(st[g, :7] + noise); row_t.append(max(0.0, p0 - a.penalty * min(1.0, sc / 0.1)))
        now.append(st[f, :7]); fut.append(row_f); tgt.append(row_t)
    now, fut, tgt = (torch.tensor(np.asarray(v, np.float32)) for v in (now, fut, tgt))
    dev = "cuda"
    h = Head(X.shape[1]); h.mods.to(dev)
    opt = torch.optim.AdamW(h.mods.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(a.epochs):
        perm = torch.randperm(len(tr)); tot = 0
        for b in range(0, len(tr), 256):
            j = perm[b:b + 256]
            x = X[j].to(dev).float() + 0.05 * torch.randn(len(j), X.shape[1], device=dev)
            p = h(x, now[j].to(dev), fut[j].to(dev))
            t = tgt[j].to(dev)
            # absolute progress + pairwise order within a row (what selection uses)
            loss = (p - t).abs().mean()
            dp, dt = p[:, :, None] - p[:, None], t[:, :, None] - t[:, None]
            m = dt.abs() > 0.01
            loss = loss + torch.relu(0.02 - dp * torch.sign(dt))[m].mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        if ep % 5 == 4:
            print(f"epoch {ep} loss {tot / len(tr):.4f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(h.mods.state_dict(), os.path.join(a.out, "head.pt"))
    json.dump({"in_img": int(X.shape[1]), "penalty": a.penalty, "deltas": DELTAS},
              open(os.path.join(a.out, "config.json"), "w"))


def check(a):
    """Unseen episodes: does the demo's future beat standing still and corrupted futures?"""
    import torch

    from .pi05_samples import UNSEEN

    h, _ = load(a.model)
    z = np.load(a.feats, allow_pickle=True)
    E, F, P = z["episode"].astype(int), z["frame"].astype(int), z["phase"].astype(int)
    te = np.flatnonzero(np.isin(E, UNSEEN))
    X = torch.tensor(np.concatenate([z["wrist"][te], z["base"][te]], 1).astype(np.float32), device="cuda")
    acts = episode_actions(a.dataset)
    rng = np.random.default_rng(1)
    now, fut = [], []
    for i in te:
        st = acts[E[i]][1]; f = F[i]; g = min(f + 15, len(st) - 1)
        now.append(st[f, :7])
        fut.append([st[g, :7], st[f, :7], st[g, :7] + rng.normal(0, 0.1, 7) * np.r_[np.ones(6), 0.5],
                    st[max(f - 15, 0), :7]])
    with torch.no_grad():
        p = h(X, torch.tensor(np.asarray(now, np.float32), device="cuda"),
              torch.tensor(np.asarray(fut, np.float32), device="cuda")).cpu().numpy()
    rows = {"all": np.ones(len(te), bool)}
    rows.update({f"phase{k}": P[te] == k for k in range(6)})
    for k, m in rows.items():
        print(k, int(m.sum()), "demo>hold", round(float((p[m, 0] > p[m, 1]).mean()), 3),
              "demo>noisy", round(float((p[m, 0] > p[m, 2]).mean()), 3),
              "demo>backward", round(float((p[m, 0] > p[m, 3]).mean()), 3))


def evaluate(a):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .jevlite import Encoder

    h, _ = load(a.model)
    z = np.load(a.samples, allow_pickle=True)
    meta = json.loads(str(z["meta"])); C, demo, state = z["cands"], z["demo"], z["state"]
    Fn, I, Sn, N = C.shape[:4]
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    enc = Encoder("dinos", grid=4)
    X = []
    for it in meta["items"]:
        x = ds[it["idx"]]
        im = {c: (x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in ("wrist", "base")}
        X.append(np.concatenate([enc([im["wrist"]]).reshape(-1), enc([im["base"]]).reshape(-1)]))
    X = torch.tensor(np.stack(X).astype(np.float32), device="cuda")
    now = torch.tensor(state[:, :7].astype(np.float32), device="cuda")
    with torch.no_grad():
        sc = h(X, now, torch.tensor(C[..., a.at, :].reshape(Fn, -1, 7), device="cuda")).cpu().numpy().reshape(Fn, I, Sn, N)
    d = dist(C, demo[:, None, None, None])
    closes = C[..., 6].min(-1) < 0.6 * state[:, 6, None, None, None]
    dc = demo[..., 6].min(-1) < 0.6 * state[:, 6]
    s1 = meta["scales"].index(1.0)
    rep = {"single": {"dist": round(float(d[:, 0, s1, 0].mean()), 4),
                      "close_when_demo_closes": round(float(closes[dc, 0, s1, 0].mean()), 3),
                      "close_when_not": round(float(closes[~dc, 0, s1, 0].mean()), 3)}}

    def pick(mi, ms, consensus=0.0):
        out_d, out_c = [], []
        for k in range(Fn):
            c = C[k][mi][:, ms].reshape(-1, C.shape[-2], 7)
            s = sc[k][mi][:, ms].reshape(-1)
            dd = d[k][mi][:, ms].reshape(-1); cl = closes[k][mi][:, ms].reshape(-1)
            pool = np.arange(len(c))
            if consensus:  # keep the most typical fraction, then maximise progress
                D = dist(c[:, None], c[None]).mean(1)
                pool = np.argsort(D)[:max(1, int(len(c) * consensus))]
            b = pool[s[pool].argmax()]
            out_d.append(dd[b]); out_c.append(cl[b])
        od, oc = np.array(out_d), np.array(out_c)
        return {"dist": round(float(od.mean()), 4), "close_when_demo_closes": round(float(oc[dc].mean()), 3),
                "close_when_not": round(float(oc[~dc].mean()), 3)}

    main = np.zeros(I, bool); main[0] = True
    for name, ss in {"s0.5": [0], "s0.5+1": [0, 1], "all": list(range(Sn))}.items():
        ms = np.zeros(Sn, bool); ms[ss] = True
        for iname, mi in {"main": main, "allinstr": np.ones(I, bool)}.items():
            rep[f"progress|{iname}|{name}"] = pick(mi, ms)
            rep[f"progress+consensus50|{iname}|{name}"] = pick(mi, ms, 0.5)
    for k, v in rep.items():
        print(k, v)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--feats", required=True); t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True); t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--penalty", type=float, default=0.05)
    c = sub.add_parser("check")
    c.add_argument("--model", required=True); c.add_argument("--feats", required=True)
    c.add_argument("--dataset", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True); e.add_argument("--samples", required=True)
    e.add_argument("--dataset", required=True); e.add_argument("--at", type=int, default=15)
    e.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"train": train, "check": check, "eval": evaluate}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
