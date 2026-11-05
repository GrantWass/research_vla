"""Score JEV's grasp/place judgement on labelled frames (real_robot.jev_frames).

For every frame asks both questions real_robot.jev asks live and compares
with the label: holding <-> "holding" phase, placed <-> "end" phase. Reports
accuracy at the guard's thresholds, AUROC, and latency, per variant
(cameras / gripper text), so the live defaults are picked from data.

  python -m real_robot.jev_judge_eval --frames outputs/jev_frames --out outputs/jev_judge
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .jev import JevClient, questions

VARIANTS = {  # name: (cameras, tell gripper width)
    "wrist+side+grip": (("cam_wrist", "cam_high"), True),
    "wrist+side": (("cam_wrist", "cam_high"), False),
    "side": (("cam_high",), False),
    "wrist": (("cam_wrist",), False),
}
GRIP_M_PER_UNIT = 0.07  # dataset gripper units: 1.0 = 70 mm


def auroc(p, y) -> float:
    p, y = np.asarray(p), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def main(argv=None) -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=os.environ.get("JEV_URL", "http://localhost:8000"))
    ap.add_argument("--object", default="Rubik's cube")
    ap.add_argument("--container", default="bowl")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--grasp-thr", type=float, default=0.35)
    ap.add_argument("--place-thr", type=float, default=0.6)
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(os.path.join(a.frames, "labels.jsonl"))]
    q = questions(a.object, a.container)
    os.makedirs(a.out, exist_ok=True)
    results, lines = {}, []
    for v in a.variants.split(","):
        cams, grip = VARIANTS[v]
        cli = JevClient(a.url, cameras=cams)
        recs = []
        for r in rows:
            imgs = {"cam_wrist": np.asarray(Image.open(r["wrist"]).convert("RGB")),
                    "cam_high": np.asarray(Image.open(r["base"]).convert("RGB"))}
            g = r["gripper"] * GRIP_M_PER_UNIT if grip else None
            rec = {k: r[k] for k in ("episode", "frame", "phase", "holding", "in_bowl")}
            for name in ("holding", "placed"):
                p, dt = cli.yes(imgs, q[name], r["task"], g)
                rec[f"p_{name}"], rec[f"s_{name}"] = p, dt
            recs.append(rec)
        hp, hy = [x["p_holding"] for x in recs], [x["holding"] for x in recs]
        pp, py = [x["p_placed"] for x in recs], [x["in_bowl"] for x in recs]
        res = {"n": len(recs),
               "holding_acc": float(np.mean((np.array(hp) >= a.grasp_thr) == np.array(hy))),
               "holding_auroc": auroc(hp, hy),
               "placed_acc": float(np.mean((np.array(pp) >= a.place_thr) == np.array(py))),
               "placed_auroc": auroc(pp, py),
               "ms_median": 1000 * float(np.median([x["s_holding"] for x in recs] +
                                                  [x["s_placed"] for x in recs])),
               "by_phase": {ph: {"p_holding": float(np.mean([x["p_holding"] for x in recs if x["phase"] == ph])),
                                 "p_placed": float(np.mean([x["p_placed"] for x in recs if x["phase"] == ph]))}
                            for ph in ("start", "holding", "end")}}
        results[v] = res
        with open(os.path.join(a.out, f"{v}.jsonl"), "w") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        print(v, json.dumps(res))
    lines += [f"JEV judge on {len(rows)} labelled demo frames ({a.frames}); "
              f"thresholds grasp {a.grasp_thr}, place {a.place_thr}.", "",
              "| variant | holding acc | holding AUROC | in-bowl acc | in-bowl AUROC | ms |",
              "|---|---|---|---|---|---|"]
    for v, r in results.items():
        lines.append(f"| {v} | {r['holding_acc']:.0%} | {r['holding_auroc']:.2f} | {r['placed_acc']:.0%} "
                     f"| {r['placed_auroc']:.2f} | {r['ms_median']:.0f} |")
    lines += ["", "Mean P(yes) by phase (start / holding / end):", ""]
    for v, r in results.items():
        b = r["by_phase"]
        lines.append(f"- {v}: holding {b['start']['p_holding']:.2f} / {b['holding']['p_holding']:.2f} / "
                     f"{b['end']['p_holding']:.2f}; in bowl {b['start']['p_placed']:.2f} / "
                     f"{b['holding']['p_placed']:.2f} / {b['end']['p_placed']:.2f}")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    json.dump(results, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
