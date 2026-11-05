"""Can JEV tell which way the arm must move? Ask direction buckets, compare.

Two sets:
  real  approach frames from real_robot.jev_dirs, labelled with the joint /
        end-pose delta still to go to the demo's grasp pose (dj1_deg, dr_m)
  shift frames 12 steps before a grasp (cube centred between the fingers),
        wrist image cropped off-centre by a known fraction of its width or
        height: the cube then sits at a known image offset (img_dx, img_dy;
        + = right / down)
For each frame JEV answers two 5-way questions about the wrist image (where
is the cube left/right, above/below the fingers' centre). Every probability
is saved, so calibrate/train on them afterwards (real_robot.jev_dir_fit).

  python -m real_robot.jev_dir_eval --dirs outputs/jev_dirs --out outputs/jev_dir_eval
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .jev import JevClient, data_url

LR = ["far to the left", "slightly to the left", "centred between the fingers",
      "slightly to the right", "far to the right"]
UD = ["far above the fingertips (farther from the gripper)", "slightly above the fingertips",
      "level with the fingertips", "slightly below the fingertips (closer to the gripper)",
      "far below the fingertips"]
VALUES = np.array([-2, -1, 0, 1, 2], float)
SHIFTS = (-0.16, -0.08, -0.04, 0.04, 0.08, 0.16)  # crop offset, fraction of the image


def ask(cli: JevClient, wrist: np.ndarray, obj: str, which: str) -> list[float]:
    q = (f"In the wrist camera image, where is the {obj} relative to the centre between the gripper fingers, "
         + ("left to right?" if which == "lr" else "up and down in the image?"))
    state = ["Wrist camera (mounted on the gripper; the two black fingers are at the bottom of the image, "
             "the point between them is where the gripper would grasp):", {"image": data_url(wrist)}]
    r = cli.http.post(cli.url, json={"kind": "choice", "state": state, "question": q,
                                     "options": LR if which == "lr" else UD, "thinking": "off"},
                      timeout=cli.timeout_s)
    r.raise_for_status()
    return r.json()["probabilities"]


def shifted(im: np.ndarray, dx: float, dy: float, keep: float = 0.68) -> np.ndarray:
    """Crop a keep-sized window moved by (dx, dy) of the image; the content
    then appears moved by (-dx, -dy) relative to the window centre."""
    from PIL import Image

    h, w = im.shape[:2]
    cw, ch = int(w * keep), int(h * keep)
    x0 = int((w - cw) / 2 + dx * w)
    y0 = int((h - ch) / 2 + dy * h)
    x0, y0 = max(0, min(w - cw, x0)), max(0, min(h - ch, y0))
    return np.asarray(Image.fromarray(im[y0:y0 + ch, x0:x0 + cw]).resize((w, h)))


def main(argv=None) -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=os.environ.get("JEV_URL", "http://localhost:8000"))
    ap.add_argument("--object", default="Rubik's cube")
    ap.add_argument("--sets", default="real,shift")
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(os.path.join(a.dirs, "labels.jsonl"))]
    cli = JevClient(a.url)
    os.makedirs(a.out, exist_ok=True)
    load = lambda p: np.asarray(Image.open(p).convert("RGB"))
    if "real" in a.sets:
        with open(os.path.join(a.out, "real.jsonl"), "w") as f:
            for i, r in enumerate(rows):
                w = load(r["wrist"])
                rec = {k: r[k] for k in ("episode", "frame", "before_grasp", "dj1_deg", "dr_m", "dz_m")}
                rec["p_lr"], rec["p_ud"] = ask(cli, w, a.object, "lr"), ask(cli, w, a.object, "ud")
                f.write(json.dumps(rec) + "\n")
                if i % 100 == 0:
                    print(f"real {i}/{len(rows)}", flush=True)
    if "shift" in a.sets:
        base = [r for r in rows if r["before_grasp"] == 12]
        with open(os.path.join(a.out, "shift.jsonl"), "w") as f:
            for i, r in enumerate(base):
                w = load(r["wrist"])
                for axis in ("x", "y"):
                    for s in (0.0,) + SHIFTS:
                        if axis == "y" and s == 0.0:
                            continue
                        dx, dy = (s, 0.0) if axis == "x" else (0.0, s)
                        im = shifted(w, dx, dy)
                        rec = {"episode": r["episode"], "axis": axis, "img_dx": -dx, "img_dy": -dy}
                        rec["p_lr" if axis == "x" else "p_ud"] = ask(cli, im, a.object, "lr" if axis == "x" else "ud")
                        f.write(json.dumps(rec) + "\n")
                if i % 10 == 0:
                    print(f"shift {i}/{len(base)}", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
