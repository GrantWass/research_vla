"""How well do JEV's direction buckets track the real correction? And a
calibrated map from them to joint space.

Reads real_robot.jev_dir_eval output.
  shift: zero-shot image-direction accuracy (sign of the bucket value vs the
         known image offset, offsets >= 0.04 of the frame) and Spearman rho
  real:  JEV bucket probabilities -> dJ1 (deg) / reach (m) still to go, by
         ridge regression, scored leave-episodes-out (5 folds): Spearman rho,
         sign accuracy where |dJ1| > 3 deg / |dr| > 3 cm, MAE vs predicting 0.
         The fitted map is saved for real_robot.jev_servo (--dir-map).

  python -m real_robot.jev_dir_fit --eval outputs/jev_dir_eval
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

VALUES = np.array([-2, -1, 0, 1, 2], float)


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def feats(rows) -> np.ndarray:
    P = np.array([r["p_lr"] + r["p_ud"] for r in rows], float)
    v = np.stack([P[:, :5] @ VALUES, P[:, 5:] @ VALUES], 1)
    return np.hstack([P, v, np.ones((len(P), 1))])


def ridge(X, y, lam=1.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    a = ap.parse_args(argv)
    lines = []
    sp = os.path.join(a.eval, "shift.jsonl")
    if os.path.exists(sp):
        S = [json.loads(x) for x in open(sp)]
        for axis, key, pk in (("x", "img_dx", "p_lr"), ("y", "img_dy", "p_ud")):
            rs = [r for r in S if r["axis"] == axis]
            off = np.array([r[key] for r in rs])
            v = np.array([np.dot(r[pk], VALUES) for r in rs])
            big = np.abs(off) >= 0.04
            acc = float(np.mean(np.sign(v[big]) == np.sign(off[big])))
            rho = spearman(v, off)
            cen = v[np.abs(off) < 1e-9]
            lines.append(f"shift {axis}: n={len(rs)}, direction acc {acc:.0%} (|offset| >= 4% of frame), "
                         f"Spearman {rho:+.2f}" + (f", |v| at centre {np.abs(cen).mean():.2f}" if len(cen) else ""))
            for o in sorted(set(off.round(3))):
                lines.append(f"  offset {o:+.2f}: mean bucket value {v[off.round(3) == o].mean():+.2f}")
    R = [json.loads(x) for x in open(os.path.join(a.eval, "real.jsonl"))]
    X = feats(R)
    eps = np.array([r["episode"] for r in R])
    uniq = np.unique(eps)
    folds = np.array_split(np.random.default_rng(0).permutation(uniq), 5)
    maps = {}
    for name, key, thr in (("dJ1 deg", "dj1_deg", 3.0), ("reach m", "dr_m", 0.03)):
        y = np.array([r[key] for r in R])
        pred = np.zeros_like(y)
        for f in folds:
            te = np.isin(eps, f)
            pred[te] = X[te] @ ridge(X[~te], y[~te])
        big = np.abs(y) > thr
        rho0 = spearman(X[:, 10 if key == "dj1_deg" else 11], y)
        lines.append(f"real {name}: n={len(y)}, zero-shot rho(bucket value, target) {rho0:+.2f}; "
                     f"fitted (held-out episodes) rho {spearman(pred, y):+.2f}, sign acc {np.mean(np.sign(pred[big]) == np.sign(y[big])):.0%} "
                     f"on |target|>{thr} (n={big.sum()}), MAE {np.abs(pred - y).mean():.3f} vs {np.abs(y).mean():.3f} for 0")
        maps[key] = ridge(X, y).tolist()
    json.dump({"features": "p_lr(5) + p_ud(5) + [v_lr, v_ud, 1]", **maps},
              open(os.path.join(a.eval, "dir_map.json"), "w"), indent=1)
    open(os.path.join(a.eval, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
