"""JEV-lite: a ~25M-parameter task-phase monitor distilled from the demos.

JEV-9B answers yes/no questions in ~170 ms, so the guard can only ask it at
gripper events. JEV-lite runs every control step (a few ms): a frozen DINOv2
encodes the wrist and base frames, and a small MLP predicts the task phase
(real_robot/jevlite_data.PHASES):
  idle / approach / grasp / holding / release / placed
from which the guard reads P(holding), P(placed) and P(ready to grasp).
"Held over the bowl, not released" is labelled holding, never placed: the
case JEV-9B got wrong live on 2026-11-05.

  python -m real_robot.jevlite features --frames outputs/jevlite/frames --out outputs/jevlite/feats_dinos.npz
  python -m real_robot.jevlite train    --feats outputs/jevlite/feats_dinos.npz --out outputs/jevlite/model_dinos [--gripper]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .jevlite_data import PHASES

BACKBONES = {"dinos": "facebook/dinov2-small", "dinob": "facebook/dinov2-base"}
CAMS = ("wrist", "base")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def test_episode(e: int) -> bool:
    return e % 5 == 0  # held-out episodes: 0, 5, 10, ...


class Encoder:
    """Frozen backbone: image (H, W, 3) uint8 -> [CLS, mean patch] feature."""

    def __init__(self, name: str = "dinos", device: str | None = None, size: int = 224,
                 grid: int = 0):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(BACKBONES.get(name, name), torch_dtype=self.dtype)
        self.model.eval().to(self.device)
        self.size, self.grid = size, grid
        self.dim = (2 + grid * grid) * self.model.config.hidden_size

    def _prep(self, ims: list[np.ndarray]):
        from PIL import Image

        arr = []
        for im in ims:
            if im.shape[0] != self.size or im.shape[1] != self.size:
                im = np.asarray(Image.fromarray(im).resize((self.size, self.size), Image.BILINEAR))
            arr.append((im.astype(np.float32) / 255.0 - MEAN) / STD)
        x = self.torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
        return x.to(self.device, self.dtype)

    def __call__(self, ims: list[np.ndarray]) -> np.ndarray:
        with self.torch.inference_mode():
            h = self.model(pixel_values=self._prep(ims)).last_hidden_state
            parts = [h[:, 0], h[:, 1:].mean(1)]
            if self.grid:  # where things are: patch tokens average-pooled to grid x grid
                side = int(round((h.shape[1] - 1) ** 0.5))
                t = h[:, 1:1 + side * side].transpose(1, 2).reshape(h.shape[0], -1, side, side)
                g = self.torch.nn.functional.adaptive_avg_pool2d(t.float(), self.grid)
                parts.append(g.flatten(2).transpose(1, 2).flatten(1).to(h.dtype))
            f = self.torch.cat(parts, dim=1)
        return f.float().cpu().numpy()


def augment(im: np.ndarray, rnd: np.random.Generator) -> np.ndarray:
    """Lighting and framing drift: colour jitter, random crop (85-100%) and shift."""
    from PIL import Image, ImageEnhance

    h, w = im.shape[:2]
    s = rnd.uniform(0.85, 1.0)
    ch, cw = int(h * s), int(w * s)
    y, x = rnd.integers(0, h - ch + 1), rnd.integers(0, w - cw + 1)
    p = Image.fromarray(im[y:y + ch, x:x + cw]).resize((w, h), Image.BILINEAR)
    for enh, lo, hi in ((ImageEnhance.Brightness, 0.7, 1.3), (ImageEnhance.Contrast, 0.7, 1.3),
                        (ImageEnhance.Color, 0.7, 1.3)):
        p = enh(p).enhance(rnd.uniform(lo, hi))
    return np.asarray(p)


def extract(frames: str, out: str, backbone: str, batch: int = 64, grid: int = 0,
            aug_seed: int | None = None) -> None:
    from PIL import Image

    rnd = np.random.default_rng(aug_seed) if aug_seed is not None else None

    def load(path: str) -> np.ndarray:
        im = np.asarray(Image.open(path).convert("RGB"))
        return augment(im, rnd) if rnd is not None else im

    rows = [json.loads(l) for l in open(os.path.join(frames, "labels.jsonl"))]
    enc = Encoder(backbone, grid=grid)
    feats = {c: np.zeros((len(rows), enc.dim), np.float16) for c in CAMS}
    t0 = time.time()
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        for c in CAMS:
            feats[c][s:s + len(chunk)] = enc([load(r[c]) for r in chunk])
        if s % (batch * 50) == 0:
            print(f"{s}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, wrist=feats["wrist"], base=feats["base"],
             phase=np.array([PHASES.index(r["phase"]) for r in rows]),
             late=np.array([r["late_carry"] for r in rows]),
             episode=np.array([r["episode"] for r in rows]),
             frame=np.array([r["frame"] for r in rows]),
             state=np.array([r["state"] for r in rows], np.float32))
    print(f"wrote {out}: {len(rows)} frames, {enc.dim}-d per camera, {time.time() - t0:.0f}s")


# -- head ---------------------------------------------------------------------------
def build_head(in_dim: int, hidden: int = 256, n_out: int = len(PHASES)):
    import torch.nn as nn

    return nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(0.2), nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Dropout(0.2), nn.Linear(hidden, n_out))


def inputs(d, cams=CAMS, gripper: bool = False) -> np.ndarray:
    x = [d[c].astype(np.float32) for c in cams]
    if gripper:
        x.append(d["state"][:, 6:7] * 4.0)  # sim units (0..1.4), scaled up to matter next to features
    return np.concatenate(x, axis=1)


def train(feats: str, out: str, cams=CAMS, gripper: bool = False, epochs: int = 60,
          seed: int = 0, extra: tuple[str, ...] = (), live: tuple[str, ...] = (),
          live_repeat: int = 5) -> dict:
    import torch

    torch.manual_seed(seed)
    d = dict(np.load(feats))
    X, y, ep = inputs(d, cams, gripper), d["phase"], d["episode"]
    te = np.array([test_episode(int(e)) for e in ep])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
    w = torch.tensor(1.0 / np.maximum(np.bincount(y[~te], minlength=len(PHASES)), 1), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=(w / w.mean()).to(dev), label_smoothing=0.05)
    xs, ys = [X[~te]], [y[~te]]
    for f in extra:  # augmented copies of the same frames: training episodes only
        da = dict(np.load(f))
        xs.append(inputs(da, cams, gripper)[~te])
        ys.append(da["phase"][~te])
    for f in live:  # live-run frames (real_robot/jevlite_live.py): all training, oversampled
        dl = dict(np.load(f))
        xs += [inputs(dl, cams, gripper)] * live_repeat
        ys += [dl["phase"]] * live_repeat
    Xtr, ytr = torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ys))  # CPU; batches go to dev
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            xb = Xtr[b].to(dev)
            xb = xb + 0.05 * torch.randn_like(xb)  # feature noise: cheap augmentation
            loss = lossf(head(xb), ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    with torch.no_grad():
        P = np.concatenate([torch.softmax(head(torch.tensor(X[s:s + 2048], device=dev)), 1).cpu().numpy()
                            for s in range(0, len(X), 2048)])
    m = metrics(P[te], y[te], d["late"][te])
    os.makedirs(out, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(out, "head.pt"))
    cfg = {"cams": list(cams), "gripper": gripper, "in_dim": int(X.shape[1]), "phases": PHASES,
           "feats": feats, "metrics": m}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    np.save(os.path.join(out, "test_probs.npy"), P[te])
    return m


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    pos = pos.astype(bool)
    if pos.all() or (~pos).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def metrics(P: np.ndarray, y: np.ndarray, late: np.ndarray) -> dict:
    H, PL, G = PHASES.index("holding"), PHASES.index("placed"), PHASES.index("grasp")
    pred = P.argmax(1)
    m = {"phase_acc": float((pred == y).mean()), "n": int(len(y)),
         "holding_auroc": auroc(P[:, H], y == H), "placed_auroc": auroc(P[:, PL], y == PL),
         "grasp_auroc": auroc(P[:, G], y == G)}
    # the live failure: held over the bowl, not released -> must not say placed
    late = late.astype(bool)
    m["late_carry_placed_fp"] = float((P[late, PL] >= 0.5).mean()) if late.any() else float("nan")
    m["placed_recall"] = float((P[y == PL, PL] >= 0.5).mean())
    m["confusion"] = [[int(((y == a) & (pred == b)).sum()) for b in range(len(PHASES))]
                      for a in range(len(PHASES))]
    return m


class JevLite:
    """Runtime: JevLite(model_dir)(images, gripper) -> {phase: prob}."""

    def __init__(self, model_dir: str, encoder: Encoder | None = None):
        import torch

        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.enc = encoder or Encoder(self.cfg.get("backbone", "dinos"), grid=self.cfg.get("grid", 0))
        self.head = build_head(self.cfg["in_dim"])
        self.head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
        self.head.eval().to(self.enc.device)
        self.torch = torch

    def __call__(self, images: dict, gripper: float | None = None) -> dict:
        key = {"wrist": "cam_wrist", "base": "cam_high"}
        f = self.enc([images[key[c]] for c in self.cfg["cams"]])  # one batch, all cameras
        x = [f.reshape(1, -1)]
        if self.cfg["gripper"]:
            x.append(np.array([[float(gripper) * 4.0]], np.float32))
        with self.torch.no_grad():
            p = self.torch.softmax(self.head(self.torch.tensor(np.concatenate(x, 1),
                                                                device=self.enc.device)), 1)
        return dict(zip(PHASES, p[0].float().cpu().numpy().tolist()))


class LiteJudge:
    """JevClient-compatible judge (.yes) backed by JEV-lite, for JevMonitor and the
    guard. mode "lite": JEV-lite alone (~15 ms, no server). mode "both": the place
    check needs JEV-lite AND JEV-9B to agree (min of the two); grasp checks stay lite.
    Direction questions (the servo) still go to JEV-9B."""

    def __init__(self, model_dir: str, jev=None, mode: str = "lite"):
        self.lite, self.jev, self.mode = JevLite(model_dir), jev, mode
        self.last: dict | None = None

    @staticmethod
    def kind(question: str) -> str:
        q = question.lower()
        if " held " in q or "holding" in q:
            return "holding"
        if "inside" in q or " in the " in q:
            return "placed"
        raise ValueError(f"JEV-lite has no answer for: {question!r}")

    def yes(self, images: dict, question: str, task: str = "", gripper_m: float | None = None):
        t0 = time.perf_counter()
        p = self.lite(images, gripper=None if gripper_m is None else gripper_m / 0.07)
        self.last = p
        k = self.kind(question)
        y = p["holding"] if k == "holding" else p["placed"] + p["release"]
        if self.mode == "both" and k == "placed" and self.jev is not None:
            y = min(y, self.jev.yes(images, question, task, gripper_m)[0])
        return float(y), time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--frames", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--backbone", default="dinos")
    f.add_argument("--grid", type=int, default=0)
    f.add_argument("--aug-seed", type=int, default=None, help="extract an augmented copy")
    t = sub.add_parser("train")
    t.add_argument("--feats", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--cams", default="wrist,base")
    t.add_argument("--gripper", action="store_true")
    t.add_argument("--backbone", default="dinos")
    t.add_argument("--grid", type=int, default=0, help="must match the features")
    t.add_argument("--extra", default="", help="comma-separated augmented feature files")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--live", default="", help="comma-separated live-frame features (jevlite_live build)")
    a = ap.parse_args(argv)
    if a.cmd == "features":
        extract(a.frames, a.out, a.backbone, grid=a.grid, aug_seed=a.aug_seed)
    else:
        m = train(a.feats, a.out, tuple(a.cams.split(",")), a.gripper, seed=a.seed,
                  extra=tuple(x for x in a.extra.split(",") if x),
                  live=tuple(x for x in a.live.split(",") if x))
        cfgp = os.path.join(a.out, "config.json")
        c = json.load(open(cfgp))
        c["backbone"], c["grid"] = a.backbone, a.grid
        json.dump(c, open(cfgp, "w"), indent=1)
        print(json.dumps({k: v for k, v in m.items() if k != "confusion"}))
        print("confusion (rows true, cols pred):", PHASES)
        for row in m["confusion"]:
            print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
