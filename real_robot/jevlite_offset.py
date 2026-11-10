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


def train(dataset, frames, feats, out, extra=(), epochs=150, seed=0) -> dict:
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
    json.dump({"in_dim": int(X.shape[1]), "units": "cm", "backbone": "dinos", "grid": 4,
               "cams": ["wrist", "base"], "metrics": res}, open(os.path.join(out, "config.json"), "w"), indent=1)
    return res


class GraspOffset:
    """Runtime: GraspOffset(model_dir, encoder)(images, end_pose_xyz) -> base-frame delta (m)."""

    def __init__(self, model_dir: str, encoder=None):
        import torch

        from .jevlite import Encoder

        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.enc = encoder or Encoder(self.cfg["backbone"], grid=self.cfg["grid"])
        self.head = build_head(self.cfg["in_dim"], n_out=3)
        self.head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
        self.head.eval().to(self.enc.device)
        self.torch = torch

    def rtz(self, images: dict) -> np.ndarray:
        key = {"wrist": "cam_wrist", "base": "cam_high"}
        f = self.enc([images[key[c]] for c in self.cfg["cams"]]).reshape(1, -1)
        with self.torch.no_grad():
            y = self.head(self.torch.tensor(f, device=self.enc.device)).float().cpu().numpy()[0]
        return y / 100.0

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
    a = ap.parse_args(argv)
    res = train(a.dataset, a.frames, a.feats, a.out, tuple(x for x in a.extra.split(",") if x), seed=a.seed)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
