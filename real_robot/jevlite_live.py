"""Turn live runs into JEV-lite training data (the data flywheel).

Every judgement during a live run saves its frames to
real_robot_logs/<run>/attemptN/judged/ (real_robot/jev.py). You say what really
happened per attempt; the frames get labels; JEV-lite retrains with them.

  1. python -m real_robot.jevlite_live template --logs real_robot_logs --out real_robot_logs/live_labels.csv
     fill in grasped / placed (y/n) for each attempt you watched; leave blank to skip
  2. python -m real_robot.jevlite_live build --csv real_robot_logs/live_labels.csv --out outputs/jevlite/live_feats.npz
  3. python -m real_robot.jevlite train --feats outputs/jevlite/feats_dinos_g4.npz --grid 4 \
         --extra outputs/jevlite/feats_dinos_g4_aug1.npz,outputs/jevlite/feats_dinos_g4_aug2.npz \
         --live outputs/jevlite/live_feats.npz --out outputs/jevlite/model_dinos_g4_live

Labels per saved frame (only the two questions the judge answers matter):
  question holding: grasped=y -> holding, grasped=n -> approach (not holding)
  question placed:  placed=y  -> placed,  placed=n  -> holding if grasped and never released, else approach
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from .jevlite_data import PHASES


def attempts(logs: str) -> list[str]:
    return sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(logs, "*", "attempt*", "judged", "frames.jsonl")))


def template(logs: str, out: str) -> int:
    rows = []
    for d in attempts(logs):
        recs = [json.loads(l) for l in open(os.path.join(d, "judged", "frames.jsonl"))]
        rows.append({"attempt_dir": d, "n_frames": len(recs),
                     "jev_said": " ".join(f"{r['question']}={r['p_yes']:.2f}" for r in recs),
                     "grasped": "", "placed": "", "released": ""})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["attempt_dir", "n_frames", "jev_said", "grasped", "placed", "released"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} attempts -> {out}; fill grasped/placed/released with y or n")
    return len(rows)


def label(question: str, grasped: bool, placed: bool, released: bool) -> str:
    q = question.split("_")[0]
    if q == "holding":
        return "holding" if grasped else "approach"
    if placed:
        return "placed"
    return "holding" if grasped and not released else "approach"


def build(csv_path: str, out: str, backbone: str = "dinos", grid: int = 4) -> int:
    from PIL import Image

    from .jevlite import Encoder

    yes = lambda v: str(v).strip().lower() in ("y", "yes", "1", "true")
    items = []
    for r in csv.DictReader(open(csv_path)):
        if not r["grasped"].strip() or not r["placed"].strip():
            continue
        released = yes(r["released"]) if r.get("released", "").strip() else yes(r["placed"])
        for f in (json.loads(l) for l in open(os.path.join(r["attempt_dir"], "judged", "frames.jsonl"))):
            if "cam_wrist" not in f or "cam_high" not in f:
                continue
            items.append((f, label(f["question"], yes(r["grasped"]), yes(r["placed"]), released)))
    if not items:
        print("no labelled attempts")
        return 0
    enc = Encoder(backbone, grid=grid)
    W, B = [], []
    for f, _ in items:
        W.append(enc([np.asarray(Image.open(f["cam_wrist"]).convert("RGB"))])[0])
        B.append(enc([np.asarray(Image.open(f["cam_high"]).convert("RGB"))])[0])
    phase = np.array([PHASES.index(p) for _, p in items])
    np.savez(out, wrist=np.array(W, np.float16), base=np.array(B, np.float16), phase=phase,
             late=np.array([p == "holding" and f["question"].startswith("placed") for f, p in items]),
             episode=np.full(len(items), -1), frame=np.array([f["step"] for f, _ in items]),
             state=np.zeros((len(items), 13), np.float32))
    print(f"{len(items)} live frames -> {out}: " + str({p: int((phase == PHASES.index(p)).sum()) for p in PHASES}))
    return len(items)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--logs", default="real_robot_logs")
    t.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--csv", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "template":
        template(a.logs, a.out)
    else:
        build(a.csv, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
