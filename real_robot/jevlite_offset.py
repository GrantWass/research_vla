"""Where is the grasp? Regress the offset from the gripper to the demo's grasp point.

For every approach frame of a demo, the end-effector pose at the moment the
gripper starts closing is known, so the offset to it is an exact, real label
(no synthetic image shifts). It is expressed in the arm's own frame:
  radial      along the base->gripper direction (reach), m
  tangential  sideways (what J1 changes), m
  dz          up/down, m
A small MLP on the JEV-lite features (wrist + base, DINOv2-small, 4x4 grid)
predicts it. The guard uses it for --recover align: when pi0.5 hovers, move by
the predicted (radial, tangential), then descend.

  python -m real_robot.jevlite_offset train --dataset demos/cube_grid_all_pi05 \
      --frames outputs/jevlite/frames --feats outputs/jevlite/feats_dinos_g4.npz \
      --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz \
      --out outputs/jevlite/offset_dinos_g4
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .jevlite import build_head, inputs, test_episode

MAX_DIST = 0.15  # m: only frames this close to the grasp (the hover regime)
# Categorical mode (JEV-style answers, per axis): bucket edges in m, symmetric.
EDGES = np.array([-0.06, -0.03, -0.01, 0.01, 0.03, 0.06])
BUCKETS = ["far -", "medium -", "small -", "centred", "small +", "medium +", "far +"]
AXES = ("radial", "tangential", "dz")


def load_collected(dirs, grid: int = 4, backbone: str = "dinos", repeat: int = 3):
    """Arm-collected frames (real_robot/jevlite_collect.py) -> (X, Y): all for training."""
    if not dirs:
        return None
    from PIL import Image

    from .jevlite import Encoder

    enc = Encoder(backbone, grid=grid)
    X, Y = [], []
    for d in dirs:
        for r in (json.loads(l) for l in open(os.path.join(d, "labels.jsonl"))):
            w = enc([np.asarray(Image.open(r["cam_wrist"]).convert("RGB"))])[0]
            b = enc([np.asarray(Image.open(r["cam_high"]).convert("RGB"))])[0]
            X.append(np.concatenate([w, b]))
            Y.append(r["offset_rtz"])
    X, Y = np.array(X, np.float32), np.array(Y, np.float32)
    print(f"collected: {len(X)} frames from {len(dirs)} sessions (x{repeat})")
    return np.concatenate([X] * repeat), np.concatenate([Y] * repeat)


def bucketize(y: np.ndarray) -> np.ndarray:
    """(n, 3) offsets in m -> (n, 3) bucket indices 0..6."""
    return np.searchsorted(EDGES, y, side="right")


def to_arm_frame(p: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Base-frame delta (..., 3) at gripper position p (..., 3) -> (radial, tangential, dz)."""
    xy = p[..., :2]
    u = xy / np.maximum(np.linalg.norm(xy, axis=-1, keepdims=True), 1e-6)
    t = np.stack([-u[..., 1], u[..., 0]], axis=-1)
    return np.stack([(delta[..., :2] * u).sum(-1), (delta[..., :2] * t).sum(-1), delta[..., 2]], axis=-1)


def from_arm_frame(p: np.ndarray, rtz: np.ndarray) -> np.ndarray:
    """Inverse of to_arm_frame: (radial, tangential, dz) at p -> base-frame delta."""
    xy = p[:2]
    u = xy / max(float(np.linalg.norm(xy)), 1e-6)
    t = np.array([-u[1], u[0]])
    return np.r_[rtz[0] * u + rtz[1] * t, rtz[2]]


def grasp_poses(dataset: str) -> dict[int, np.ndarray]:
    """Episode -> end-effector xyz when the gripper starts closing (exact frame)."""
    import pandas as pd

    from .jevlite_data import segments

    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(dataset, "data/*/*.parquet")))])
    out = {}
    for e in d["episode_index"].unique():
        s = np.stack(d[d.episode_index == e]["observation.state"].values)
        seg = segments(s[:, 6])
        if seg is not None:
            out[int(e)] = s[seg[1], 7:10].astype(np.float64)
    return out


def labels(rows: list[dict], gp: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(mask over rows, targets (n, 3) in the arm frame)."""
    y = np.zeros((len(rows), 3), np.float32)
    m = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        if r["episode"] not in gp or r["phase"] not in ("approach", "grasp") or r["frame"] > r["seg"][1]:
            continue
        p = np.asarray(r["state"][7:10], np.float64)
        d = gp[r["episode"]] - p
        if np.linalg.norm(d) <= MAX_DIST:
            y[i] = to_arm_frame(p, d)
            m[i] = True
    return m, y


def train(dataset, frames, feats, out, extra=(), epochs=150, seed=0, members: int = 1,
          collected=()) -> dict:
    """members > 1: an ensemble (seeds seed..seed+members-1, head_<k>.pt); its spread
    gates --recover align (act only when the members agree)."""
    if members > 1:
        return train_ensemble(dataset, frames, feats, out, extra, epochs, seed, members, collected)
    import torch

    torch.manual_seed(seed)
    rows = [json.loads(l) for l in open(os.path.join(frames, "labels.jsonl"))]
    m, Y = labels(rows, grasp_poses(dataset))
    d = dict(np.load(feats))
    X = inputs(d)
    te = np.array([test_episode(int(e)) for e in d["episode"]])
    tr = m & ~te
    xs, ys = [X[tr]], [Y[tr]]
    for f in extra:
        xs.append(inputs(dict(np.load(f)))[tr])
        ys.append(Y[tr])
    c = load_collected(collected)
    if c is not None:
        xs.append(c[0])
        ys.append(c[1])
    Xtr, Ytr = torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ys) * 100.0)  # cm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(X.shape[1], n_out=3).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 128):
            b = perm[s:s + 128]
            xb = Xtr[b].to(dev)
            loss = torch.nn.functional.smooth_l1_loss(head(xb + 0.05 * torch.randn_like(xb)), Ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    tm = m & te
    with torch.no_grad():
        P = head(torch.tensor(X[tm], device=dev)).cpu().numpy() / 100.0
    T = Y[tm]
    dist = np.linalg.norm(T, axis=1)
    res = {"n_train": int(tr.sum()), "n_test": int(tm.sum())}
    base = np.abs(T - Y[tr].mean(0)).mean(0)  # predict the mean offset
    res["mae_cm"] = dict(zip(("radial", "tangential", "dz"), (100 * np.abs(P - T).mean(0)).round(2).tolist()))
    res["baseline_mae_cm"] = dict(zip(("radial", "tangential", "dz"), (100 * base).round(2).tolist()))
    for lo, hi in ((0, 0.03), (0.03, 0.06), (0.06, 0.15)):
        k = (dist >= lo) & (dist < hi)
        if k.any():
            e_xy = np.linalg.norm(P[k, :2] - T[k, :2], axis=1)
            res[f"{int(lo*100)}-{int(hi*100)}cm"] = {
                "n": int(k.sum()), "xy_err_cm_median": round(100 * float(np.median(e_xy)), 2),
                "xy_true_cm_median": round(100 * float(np.median(np.linalg.norm(T[k, :2], axis=1))), 2),
                # does applying the prediction bring the gripper closer in xy?
                "closer_frac": round(float((np.linalg.norm(T[k, :2] - P[k, :2], axis=1)
                                            < np.linalg.norm(T[k, :2], axis=1)).mean()), 3)}
    os.makedirs(out, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(out, "head.pt"))
    np.save(os.path.join(out, "test_pred.npy"), P)
    np.save(os.path.join(out, "test_true.npy"), T)
    json.dump({"in_dim": int(X.shape[1]), "units": "cm", "backbone": "dinos", "grid": 4,
               "cams": ["wrist", "base"], "metrics": res}, open(os.path.join(out, "config.json"), "w"), indent=1)
    return res


def train_buckets(dataset, frames, feats, out, extra=(), epochs=150, seed=0, collected=()) -> dict:
    """Per-axis 7-way classifier (BUCKETS). Each bucket moves by its training median
    offset; the step is the probability-weighted move, so a confident "far" means a
    big step and an unsure answer a small one."""
    import torch

    torch.manual_seed(seed)
    rows = [json.loads(l) for l in open(os.path.join(frames, "labels.jsonl"))]
    m, Y = labels(rows, grasp_poses(dataset))
    B = bucketize(Y)
    d = dict(np.load(feats))
    X = inputs(d)
    te = np.array([test_episode(int(e)) for e in d["episode"]])
    tr = m & ~te
    nb = len(BUCKETS)
    centres = np.zeros((3, nb))
    for a in range(3):
        for b in range(nb):
            v = Y[tr & (B[:, a] == b), a]
            lo = EDGES[b - 1] if b > 0 else EDGES[0] - 0.03
            hi = EDGES[b] if b < nb - 1 else EDGES[-1] + 0.03
            centres[a, b] = float(np.median(v)) if len(v) >= 5 else (lo + hi) / 2
    xs, bs = [X[tr]], [B[tr]]
    for f in extra:
        xs.append(inputs(dict(np.load(f)))[tr])
        bs.append(B[tr])
    c = load_collected(collected)
    if c is not None:
        xs.append(c[0])
        bs.append(bucketize(c[1]))
    Xtr, Btr = torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(bs), dtype=torch.long)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(X.shape[1], n_out=3 * nb).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for s0 in range(0, len(perm), 128):
            b = perm[s0:s0 + 128]
            xb = Xtr[b].to(dev)
            logits = head(xb + 0.05 * torch.randn_like(xb)).view(-1, 3, nb)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, nb), Btr[b].to(dev).reshape(-1),
                                                     label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    tm = m & te
    with torch.no_grad():
        Pb = torch.softmax(head(torch.tensor(X[tm], device=dev)).view(-1, 3, nb), -1).cpu().numpy()
    T, Bt = Y[tm], B[tm]
    step = (Pb * centres[None]).sum(-1)  # expected move per axis, m
    pred = Pb.argmax(-1)
    res = {"n_train": int(tr.sum()), "n_test": int(tm.sum()), "buckets": BUCKETS,
           "edges_cm": (100 * EDGES).tolist(),
           "bucket_centres_cm": {ax: (100 * centres[a]).round(2).tolist() for a, ax in enumerate(AXES)},
           "test_bucket_counts": {ax: np.bincount(Bt[:, a], minlength=nb).tolist() for a, ax in enumerate(AXES)}}
    for a, ax in enumerate(AXES):
        res[ax] = {"bucket_acc": round(float((pred[:, a] == Bt[:, a]).mean()), 3),
                   "within_one": round(float((np.abs(pred[:, a] - Bt[:, a]) <= 1).mean()), 3),
                   "sign_acc_when_off": round(float((np.sign(step[Bt[:, a] != 3, a]) ==
                                                     np.sign(T[Bt[:, a] != 3, a])).mean()), 3),
                   "step_mae_cm": round(100 * float(np.abs(step[:, a] - T[:, a]).mean()), 2)}
    dist = np.linalg.norm(T[:, :2], axis=1)
    after = np.linalg.norm(T[:, :2] - step[:, :2], axis=1)
    far = dist >= 0.03
    res["xy_after_one_step_cm_median"] = round(100 * float(np.median(after[far])), 2)
    res["xy_before_cm_median"] = round(100 * float(np.median(dist[far])), 2)
    res["closer_frac"] = round(float((after[far] < dist[far]).mean()), 3)
    os.makedirs(out, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(out, "head.pt"))
    json.dump({"in_dim": int(X.shape[1]), "units": "buckets", "backbone": "dinos", "grid": 4,
               "cams": ["wrist", "base"], "centres_m": centres.tolist(), "metrics": res},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    return res


def train_ensemble(dataset, frames, feats, out, extra, epochs, seed, members, collected=()) -> dict:
    import shutil

    preds = []
    for k in range(members):
        sub = os.path.join(out, f"m{k}")
        train(dataset, frames, feats, sub, extra, epochs, seed + k, collected=collected)
        preds.append(np.load(os.path.join(sub, "test_pred.npy")))
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(sub, "head.pt"), os.path.join(out, f"head_{k}.pt"))
    T = np.load(os.path.join(out, "m0", "test_true.npy"))
    P = np.stack(preds)  # (members, n, 3)
    mean, spread = P.mean(0), np.linalg.norm(P[:, :, :2].std(0), axis=1)
    dist = np.linalg.norm(T[:, :2], axis=1)
    after = np.linalg.norm(T[:, :2] - mean[:, :2], axis=1)
    res = {"members": members, "n_test": int(len(T)),
           "xy_err_cm_median": round(100 * float(np.median(after)), 2),
           "spread_err_corr": round(float(np.corrcoef(spread, after)[0, 1]), 3)}
    far = dist >= 0.03
    for q in (50, 75, 90, 100):
        thr = float(np.percentile(spread[far], q)) if q < 100 else np.inf
        act = far & (spread <= thr)
        res[f"gate_p{q}"] = {"spread_thr_cm": None if q == 100 else round(100 * thr, 2),
                             "acted_frac": round(float(act.sum() / far.sum()), 3),
                             "closer_frac_when_acted": round(float((after[act] < dist[act]).mean()), 3),
                             "worse_by_1cm_frac": round(float((after[act] > dist[act] + 0.01).mean()), 3)}
    cfg = json.load(open(os.path.join(out, "m0", "config.json")))
    cfg.update(members=members, metrics=res)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    return res


class GraspOffset:
    """Runtime: GraspOffset(model_dir, encoder)(images, end_pose_xyz) -> base-frame delta (m)."""

    def __init__(self, model_dir: str, encoder=None):
        import torch

        from .jevlite import Encoder

        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.enc = encoder or Encoder(self.cfg["backbone"], grid=self.cfg["grid"])
        self.buckets = self.cfg.get("units") == "buckets"
        self.centres = np.asarray(self.cfg["centres_m"]) if self.buckets else None
        self.answer: dict | None = None  # bucket answers at the last call (bucket models)
        n = int(self.cfg.get("members", 1))
        names = [f"head_{k}.pt" for k in range(n)] if n > 1 else ["head.pt"]
        self.heads = []
        for name in names:
            h = build_head(self.cfg["in_dim"], n_out=3 * len(BUCKETS) if self.buckets else 3)
            h.load_state_dict(torch.load(os.path.join(model_dir, name), map_location="cpu"))
            self.heads.append(h.eval().to(self.enc.device))
        self.torch = torch
        self.spread = 0.0  # m: xy disagreement between members at the last call

    def rtz(self, images: dict) -> np.ndarray:
        key = {"wrist": "cam_wrist", "base": "cam_high"}
        f = self.enc([images[key[c]] for c in self.cfg["cams"]]).reshape(1, -1)
        x = self.torch.tensor(f, device=self.enc.device)
        with self.torch.no_grad():
            outs = [h(x).float().cpu().numpy()[0] for h in self.heads]
        if self.buckets:
            P = np.exp(outs[0].reshape(3, -1))
            P /= P.sum(1, keepdims=True)
            self.answer = {ax: {"bucket": BUCKETS[int(P[a].argmax())], "p": round(float(P[a].max()), 3)}
                           for a, ax in enumerate(AXES)}
            self.spread = 0.0
            return (P * self.centres).sum(1)
        ys = np.stack(outs) / 100.0
        self.spread = float(np.linalg.norm(ys[:, :2].std(0))) if len(ys) > 1 else 0.0
        return ys.mean(0)

    def __call__(self, images: dict, p_xyz: np.ndarray) -> np.ndarray:
        return from_arm_frame(np.asarray(p_xyz, float), self.rtz(images))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--frames", required=True)
    t.add_argument("--feats", required=True)
    t.add_argument("--extra", default="")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--members", type=int, default=1)
    t.add_argument("--buckets", action="store_true", help="per-axis 7-bucket classifier instead of regression")
    t.add_argument("--collected", default="", help="comma-separated jevlite_collect output dirs")
    a = ap.parse_args(argv)
    if a.buckets:
        print(json.dumps(train_buckets(a.dataset, a.frames, a.feats, a.out,
                                       tuple(x for x in a.extra.split(",") if x), seed=a.seed,
                                       collected=tuple(x for x in a.collected.split(",") if x)), indent=1))
        return 0
    res = train(a.dataset, a.frames, a.feats, a.out, tuple(x for x in a.extra.split(",") if x), seed=a.seed,
                members=a.members, collected=tuple(x for x in a.collected.split(",") if x))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
