"""JEV-lite vs JEV-9B on the same held-out demo frames, and their latency.

Cases (episodes held out of JEV-lite training, full-resolution frames):
  late_carry   held over the bowl, not released  -> holding yes, placed NO (the live false positive)
  placed       1-4 s after release                -> holding no,  placed yes
  holding      early in the carry                  -> holding yes, placed no
  approach     open, moving to the cube            -> holding no,  placed no

  python -m real_robot.jevlite_compare --dataset demos/cube_grid_all_pi05 \
      --frames outputs/jevlite/frames --lite outputs/jevlite/model_dinos_g4 --out outputs/jevlite/compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from .jevlite import JevLite, auroc, test_episode

CASES = {"late_carry": (1, 0), "placed": (0, 1), "holding": (1, 0), "approach": (0, 0)}


def pick(rows: list[dict], per_case: int, seed: int = 0) -> list[tuple[str, dict]]:
    rnd = random.Random(seed)
    test = [r for r in rows if test_episode(r["episode"])]
    pools = {
        "late_carry": [r for r in test if r["late_carry"]],
        "placed": [r for r in test if r["phase"] == "placed" and 30 <= r["frame"] - r["seg"][3] < 120],
        "holding": [r for r in test if r["phase"] == "holding" and not r["late_carry"]],
        "approach": [r for r in test if r["phase"] == "approach"],
    }
    return [(c, r) for c, pool in pools.items() for r in rnd.sample(pool, min(per_case, len(pool)))]


def main(argv=None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .jev import JevClient, questions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--lite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-case", type=int, default=60)
    ap.add_argument("--jev-url", default="http://localhost:8000")
    ap.add_argument("--no-jev", action="store_true")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(os.path.join(a.frames, "labels.jsonl"))]
    sel = pick(rows, a.per_case)
    ds = LeRobotDataset(repo_id=os.path.basename(a.dataset), root=a.dataset)
    eps = ds.meta.episodes
    lite = JevLite(a.lite)
    cli = None if a.no_jev else JevClient(a.jev_url)
    q = questions("Rubik's cube", "bowl")
    task = "pick up the Rubik's cube and put it in the bowl"
    out, t_lite, t_jev = [], [], []
    for n, (case, r) in enumerate(sel):
        item = ds[int(eps["dataset_from_index"][r["episode"]]) + r["frame"]]
        ims = {k: (item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
               for k, c in (("cam_high", "base"), ("cam_wrist", "wrist"))}
        g_m = float(r["state"][6]) * 0.07  # sim units -> m, as the live monitor sees it
        t0 = time.perf_counter()
        p = lite(ims, gripper=r["state"][6])
        t_lite.append(time.perf_counter() - t0)
        rec = {"case": case, "episode": r["episode"], "frame": r["frame"],
               "lite_holding": p["holding"], "lite_placed": p["placed"]}
        if cli is not None:
            for name in ("holding", "placed"):
                py, dt = cli.yes(ims, q[name], task, g_m)
                rec[f"jev_{name}"] = py
                t_jev.append(dt)
        out.append(rec)
        if n % 40 == 0:
            print(f"{n}/{len(sel)}", flush=True)

    def summary(model: str) -> dict:
        s = {}
        for qn, idx in (("holding", 0), ("placed", 1)):
            key = f"{model}_{qn}"
            if key not in out[0]:
                continue
            score = np.array([o[key] for o in out])
            truth = np.array([CASES[o["case"]][idx] for o in out])
            s[f"{qn}_auroc"] = round(auroc(score, truth), 4)
            s[f"{qn}_acc@0.5"] = round(float(((score >= 0.5) == truth).mean()), 4)
        lc = [o[f"{model}_placed"] for o in out if o["case"] == "late_carry"]
        s["late_carry_says_placed"] = round(float(np.mean(np.array(lc) >= 0.5)), 4)
        s["late_carry_mean_p_placed"] = round(float(np.mean(lc)), 4)
        return s

    res = {"n": len(out), "lite": summary("lite"),
           "lite_ms_median": round(1000 * float(np.median(t_lite[3:])), 1)}
    if cli is not None:
        res["jev9b"] = summary("jev")
        res["jev9b_ms_median_per_question"] = round(1000 * float(np.median(t_jev)), 1)
    json.dump({"summary": res, "rows": out}, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
