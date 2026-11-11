"""Offline: JEV-9B 3-D (served) vs JEV-lite offset models on the same held-out frames.

python -m real_robot.offset3d_eval --frames outputs/jev_lora_3d/frames.jsonl --n 300 \
    --lite outputs/jevlite/offset_ens5,outputs/jevlite/offset_buckets --out outputs/jev_lora_3d/offline_eval.json
Per axis: bucket accuracy, direction when off-centre, MAE; one correction: median xy error before/after,
fraction moved closer (from >= 3 cm). Uses the server's exact path (JPEG, fp8, calibration).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
from PIL import Image

from .jev_offset3d import JevGraspOffset
from .jevlite import test_episode
from .jevlite_offset import AXES, GraspOffset, bucketize


def score(pred: np.ndarray, true: np.ndarray, lat: list) -> dict:
    res = {}
    pb, tb = bucketize(pred), bucketize(true)
    for a, ax in enumerate(AXES):
        off = tb[:, a] != 3
        res[ax] = {"bucket_acc": round(float((pb[:, a] == tb[:, a]).mean()), 3),
                   "direction_when_off": round(float((np.sign(pred[off, a]) == np.sign(true[off, a])).mean()), 3),
                   "mae_cm": round(100 * float(np.abs(pred[:, a] - true[:, a]).mean()), 2)}
    before = np.linalg.norm(true[:, :2], axis=1)
    after = np.linalg.norm(true[:, :2] - pred[:, :2], axis=1)
    far = before >= 0.03
    res["one_correction"] = {"n_far": int(far.sum()),
                             "median_xy_before_cm": round(100 * float(np.median(before[far])), 2),
                             "median_xy_after_cm": round(100 * float(np.median(after[far])), 2),
                             "closer": round(float((after[far] < before[far]).mean()), 3)}
    res["latency_ms"] = round(1000 * float(np.median(lat)), 1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="outputs/jev_lora_3d/frames.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--lite", default="outputs/jevlite/offset_ens5")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ex = [json.loads(l) for l in open(a.frames)]
    ex = [e for e in ex if test_episode(e["episode"])]
    random.Random(0).shuffle(ex)
    ex = ex[:a.n]
    true = np.array([e["offset"] for e in ex])
    models = {} if a.no_jev else {"jev9b_3d": JevGraspOffset(a.url)}
    for d in filter(None, a.lite.split(",")):
        models[d.rstrip("/").split("/")[-1]] = GraspOffset(d)
    report = {"n": len(ex), "episodes": len({e["episode"] for e in ex})}
    for name, m in models.items():
        preds, lat = [], []
        for i, e in enumerate(ex):
            imgs = {"cam_wrist": np.asarray(Image.open(e["wrist"]).convert("RGB")),
                    "cam_high": np.asarray(Image.open(e["side"]).convert("RGB"))}
            t0 = time.perf_counter()
            preds.append(m.rtz(imgs))
            lat.append(time.perf_counter() - t0)
            if i % 50 == 0:
                print(name, i, flush=True)
        report[name] = score(np.array(preds), true, lat)
        print(name, json.dumps(report[name]), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
