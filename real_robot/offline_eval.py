"""Offline check of pi0.5 checkpoints on frames of a converted dataset; no robot.

Feeds recorded frames (13-D state, both cameras, task text) to each checkpoint
and compares the predicted 50-step chunk with the recorded actions, next to a
"hold still" baseline. Frames are taken at fixed points of every episode
(FRACS). On a training dataset this shows the fit, not generalization.

Also runs the first checkpoint through LerobotPolicy the way `make policy`
does (raw robot frame, --base-offset-deg 90, uint8 camera images) and checks it
gives the same chunk as the dataset path, so the base offset and camera map
match training.

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.offline_eval \\
      --dataset demos/rubiks_bowl_v5_straight_pi05 \\
      --ckpt final=outputs/<run>/checkpoints/last/pretrained_model \\
      --ckpt base=richardshkim/pi05_piper_banana_v2_full --out outputs/offline_eval

Writes report.md, summary.json, preds.npz, camera snapshots and (with
matplotlib) chunks.png to --out. ~1 min to load each checkpoint, ~0.4 s/chunk
eager on the 3090.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

FRACS = (0.02, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (0, 9, 24, 49)
CAMERA_MAP = {"cam_high": "base", "cam_wrist": "wrist"}  # make policy's --camera-map


def sample_indices(ep_from, ep_to, fracs=FRACS) -> list[tuple[int, float, int]]:
    """(episode, fraction, global frame index) at each fraction of every episode."""
    return [(e, f, int(ep_from[e] + round(f * (ep_to[e] - ep_from[e] - 1))))
            for e in range(len(ep_from)) for f in fracs]


def chunk_metrics(pred, gt, pad, state, move_deg: float = 3.0) -> dict:
    """Errors of predicted chunks pred (N, T, 7) against recorded gt (N, T, 7).

    pad (N, T) marks steps past the episode end (ignored). Joints are rad,
    reported in deg; the gripper is 0..1. direction: share of joints that
    really moved more than move_deg over the chunk that the prediction moved
    the same way.
    """
    valid = ~pad
    err = np.abs(pred - gt)
    mae = (err * valid[..., None]).sum((0, 1)) / valid.sum()
    joint_err = np.degrees(err[..., :6].mean(-1))
    last = valid[:, -1]
    moved = np.abs(gt[last, -1, :6] - state[last, :6]) > np.radians(move_deg)
    same = np.sign(gt[last, -1, :6] - state[last, :6]) == np.sign(pred[last, -1, :6] - state[last, :6])
    return {
        "joint_mae_deg": np.degrees(mae[:6]).tolist(),
        "joints_deg": float(np.degrees(mae[:6]).mean()),
        "gripper": float(mae[6]),
        "by_horizon_deg": {f"t+{h + 1}": float(joint_err[:, h][valid[:, h]].mean())
                           if valid[:, h].any() else float("nan") for h in HORIZONS},
        "direction": float(same[moved].mean()) if moved.any() else float("nan"),
    }


def predict(policy, item, seed: int) -> np.ndarray:
    """Chunk (T, 7) for one dataset item, fed straight through pre/post-processors."""
    torch = policy.torch
    batch = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
    batch["task"] = [item["task"]]
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = policy.post(policy.policy.predict_action_chunk(policy.pre(batch)))
    return out[0].float().cpu().numpy()


def predict_live_path(policy, item, seed: int) -> np.ndarray:
    """Same item through LerobotPolicy.call as the runner sends it: raw robot frame."""
    from .lerobot_policy import rotate_end_pose

    st = item["observation.state"].numpy().astype(np.float64)
    state7 = st[:7].copy()
    state7[0] -= policy.base_offset
    images = {ours: (item[f"observation.images.{theirs}"].permute(1, 2, 0).numpy() * 255)
              .round().astype(np.uint8) for ours, theirs in policy.camera_map.items()}
    policy.noise_seed = seed
    policy.call("update_obs", obs={"state": state7, "images": images, "instruction": item["task"],
                                   "extra_state": rotate_end_pose(st[7:], -policy.base_offset)})
    out = policy.call("get_action")
    out[:, 0] += policy.base_offset
    return out


def report_md(summary: dict, spread: dict, wrapper_diff: float | None, n: int, n_ep: int,
              fracs=FRACS) -> str:
    joints = ["J1", "J2", "J3", "J4", "J5", "J6"]
    hs = [f"t+{h + 1}" for h in HORIZONS]
    lines = [f"Offline eval: {n} frames from {n_ep} episodes (fractions {list(fracs)}).", "",
             "| model | " + " | ".join(joints) + " | joints (deg) | gripper | "
             + " | ".join(hs) + " | direction |",
             "|---" * (len(joints) + len(hs) + 4) + "|"]
    for name, m in summary.items():
        lines.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in m["joint_mae_deg"])
                     + f" | **{m['joints_deg']:.2f}** | {m['gripper']:.3f} | "
                     + " | ".join(f"{m['by_horizon_deg'][h]:.2f}" for h in hs)
                     + f" | {m['direction']:.0%} |")
    lines.append("")
    for name, s in spread.items():
        lines.append(f"- {name}: seed-to-seed spread {s:.2f} deg (mean |seed0 - seed1|)")
    if wrapper_diff is not None:
        lines.append(f"- make policy path (LerobotPolicy, base offset) vs dataset path: "
                     f"max |diff| {wrapper_diff:.2e}")
    return "\n".join(lines) + "\n"


def plot(path: str, gt, preds: dict, idx, ks) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]
    fig, ax = plt.subplots(len(ks), 7, figsize=(22, 2.7 * len(ks)), squeeze=False)
    t = np.arange(gt.shape[1]) / 30
    for r, k in enumerate(ks):
        for j in range(7):
            sc = np.degrees if j < 6 else (lambda x: x)
            a = ax[r, j]
            a.plot(t, sc(gt[k, :, j]), "k-", lw=2.5, label="recorded")
            for c, (name, p) in enumerate(preds.items()):
                a.plot(t, sc(p[0, k, :, j]), f"C{c}--", lw=1.3, label=name)
            a.set_title(f"ep{idx[k][0]} @{int(idx[k][1] * 100)}%  {names[j]}", fontsize=9)
            a.tick_params(labelsize=7)
    ax[0, 0].legend(fontsize=7)
    fig.supxlabel("seconds ahead")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    return True


def run(dataset: str, ckpts: list[tuple[str, str]], out: str, seeds: int = 2,
        base_offset_deg: float = 90.0, max_episodes: int | None = None, fracs=FRACS,
        log=print) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .lerobot_policy import LerobotPolicy

    os.makedirs(out, exist_ok=True)
    meta = LeRobotDataset(repo_id="local/eval", root=dataset, episodes=[0]).meta
    ds = LeRobotDataset(repo_id="local/eval", root=dataset,
                        delta_timestamps={"action": [i / meta.fps for i in range(50)]})
    ep_from = np.asarray(ds.meta.episodes["dataset_from_index"])[:max_episodes]
    ep_to = np.asarray(ds.meta.episodes["dataset_to_index"])[:max_episodes]
    idx = sample_indices(ep_from, ep_to, fracs)
    items = [ds[i] for _, _, i in idx]
    log(f"{len(items)} frames from {len(ep_from)} episodes")
    gt = np.stack([it["action"].numpy() for it in items])
    pad = np.stack([it["action_is_pad"].numpy() for it in items])
    state = np.stack([it["observation.state"].numpy() for it in items])

    preds, wrapper_diff = {}, None
    for c, (name, path) in enumerate(ckpts):
        t0 = time.time()
        pol = LerobotPolicy(path, CAMERA_MAP, base_offset=np.radians(base_offset_deg))
        log(f"[{name}] loaded in {time.time() - t0:.0f} s")
        p = np.stack([[predict(pol, it, s) for it in items] for s in range(seeds)])
        preds[name] = p
        log(f"[{name}] {seeds * len(items)} chunks in {time.time() - t0:.0f} s")
        if c == 0:
            k = min(20, len(items))
            wrapper_diff = float(max(np.abs(predict_live_path(pol, items[i], 0) - p[0, i]).max()
                                     for i in range(k)))
            log(f"[{name}] make policy path vs dataset path over {k} frames: max |diff| {wrapper_diff:.2e}")
        del pol
        torch.cuda.empty_cache()

    hold = np.repeat(state[None, :, None, :7], gt.shape[1], axis=2)
    summary = {"hold still": chunk_metrics(hold[0], gt, pad, state)}
    summary.update({n: chunk_metrics(p[0], gt, pad, state) for n, p in preds.items()})
    spread = {n: float(np.degrees(np.abs(p[0, ..., :6] - p[1, ..., :6]).mean()))
              for n, p in preds.items() if len(p) > 1}
    md = report_md(summary, spread, wrapper_diff, len(items), len(ep_from), fracs)
    open(os.path.join(out, "report.md"), "w").write(md)
    json.dump({"summary": summary, "seed_spread_deg": spread, "wrapper_max_abs_diff": wrapper_diff},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(out, "preds.npz"), gt=gt, pad=pad, state=state, idx=np.array(idx), **preds)

    from PIL import Image
    ks = sorted({len(fracs) // 2 + len(fracs) * e for e in (0, len(ep_from) // 3, 2 * len(ep_from) // 3)})
    for k in ks:
        e, f, _ = idx[k]
        im = np.concatenate([items[k][f"observation.images.{v}"].permute(1, 2, 0).numpy()
                             for v in CAMERA_MAP.values()], 1)
        Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(out, f"snap_ep{e}_{int(f * 100)}.jpg"))
    if not plot(os.path.join(out, "chunks.png"), gt, preds, idx, ks):
        log("matplotlib not installed: no chunks.png")
    log(md)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline check of pi0.5 checkpoints on dataset frames (no robot)")
    ap.add_argument("--dataset", required=True, help="converted dataset dir (make convert)")
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                    help="checkpoint dir or Hub id; repeat to compare (the first is also run through the make policy path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2, help="noise seeds per frame (2+ also reports seed spread)")
    ap.add_argument("--base-offset-deg", type=float, default=90.0, help="as make policy's")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--fracs", default=",".join(map(str, FRACS)),
                    help="points of each episode to sample, comma-separated in [0, 1]")
    args = ap.parse_args(argv)
    ckpts = [tuple(c.split("=", 1)) if "=" in c else (f"ckpt{i}", c) for i, c in enumerate(args.ckpt)]
    fracs = tuple(float(f) for f in args.fracs.split(","))
    run(args.dataset, ckpts, args.out, args.seeds, args.base_offset_deg, args.max_episodes, fracs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
