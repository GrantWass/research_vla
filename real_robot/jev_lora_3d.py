"""Teach JEV-9B to answer "where is the grasp, and how far?" in 3-D buckets.

JEV's System 1 answers multiple-choice questions with calibrated letter
probabilities. Here each question is one axis of the arm's frame (radial =
reach, tangential = sideways, height), and the options are the 7 buckets of
real_robot/jevlite_offset.BUCKETS (edges ±1, ±3, ±6 cm). Labels are exact:
the offset from the gripper to where the demo actually grasped
(jevlite_offset.labels), not synthetic image shifts. Both cameras go in, in
the two-image state format JevClient sends, so vLLM serves the adapter as is
(--lora-modules jev-decision=<out>) and real_robot.jev_3d.ask() queries it.

Same frames and held-out episodes as JEV-lite (jevlite.test_episode), so the
two are directly comparable. QLoRA (4-bit base) on the 3090.

  python -m real_robot.jev_lora_3d --dataset demos/cube_grid_all_pi05 --frames outputs/jevlite/frames \
      --adapter outputs/jev_lora_dir/adapter_vllm --base ~/jev/Qwen3.5-9B --out outputs/jev_lora_3d
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import numpy as np

from .jevlite import test_episode
from .jevlite_offset import AXES, BUCKETS, bucketize, grasp_poses, labels

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
LETTERS = "ABCDEFG"
WRIST_TXT = "Wrist camera (mounted on the gripper, looking past the fingers):"
SIDE_TXT = "Side camera (shows the whole table and the arm):"
AXIS_Q = {
    "radial": "Along the arm's reach (toward or away from the robot base), how far must the gripper move "
              "to reach the grasp position on the {obj}?",
    "tangential": "Sideways (rotating the robot base left or right), how far must the gripper move to reach "
                  "the grasp position on the {obj}?",
    "dz": "Up or down, how far must the gripper move to reach the grasp height on the {obj}?",
}
OPTIONS = {
    "radial": ["far back toward the base (over 6 cm)", "back toward the base 3-6 cm", "back toward the base 1-3 cm",
               "already there (within 1 cm)", "farther out 1-3 cm", "farther out 3-6 cm", "far out (over 6 cm)"],
    "tangential": ["far to one side, base turning negative (over 6 cm)", "3-6 cm, base turning negative",
                   "1-3 cm, base turning negative", "already lined up (within 1 cm)", "1-3 cm, base turning positive",
                   "3-6 cm, base turning positive", "far, base turning positive (over 6 cm)"],
    "dz": ["far down (over 6 cm)", "down 3-6 cm", "down 1-3 cm", "already at grasp height (within 1 cm)",
           "up 1-3 cm", "up 3-6 cm", "far up (over 6 cm)"],
}


def question(obj: str, axis: str) -> str:
    return AXIS_Q[axis].format(obj=obj)


def prompt(obj: str, axis: str) -> str:
    lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(OPTIONS[axis]))
    return (f"[kind] choice\n[state] {WRIST_TXT}{IMAGE}{SIDE_TXT}{IMAGE}\n[question] {question(obj, axis)}\n"
            f"[options]\n{lines}\n[decision]:")


def prepare(dataset: str, frames: str, out: str, width: int = 448) -> list[dict]:
    """Offset-labelled frames, decoded at full resolution once (cached as JPEG)."""
    cache = os.path.join(out, "frames.jsonl")
    if os.path.exists(cache):
        return [json.loads(l) for l in open(cache)]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    rows = [json.loads(l) for l in open(os.path.join(frames, "labels.jsonl"))]
    m, Y = labels(rows, grasp_poses(dataset))
    ds = LeRobotDataset(repo_id=os.path.basename(dataset), root=dataset)
    eps = ds.meta.episodes
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    keep = []
    for i in np.flatnonzero(m):
        r = rows[i]
        item = ds[int(eps["dataset_from_index"][r["episode"]]) + r["frame"]]
        rec = {"episode": r["episode"], "frame": r["frame"], "offset": Y[i].tolist(),
               "bucket": bucketize(Y[i:i + 1])[0].tolist()}
        for cam, key in (("wrist", "wrist"), ("side", "base")):
            im = Image.fromarray((item[f"observation.images.{key}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            im = im.resize((width, round(im.height * width / im.width)))
            rec[cam] = os.path.join(out, "img", f"e{r['episode']:03d}_f{r['frame']:04d}_{cam}.jpg")
            im.save(rec[cam], quality=90)
        keep.append(rec)
    with open(cache, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    print(f"prepared {len(keep)} frames", flush=True)
    return keep


def main(argv=None) -> int:

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object", default="Rubik's cube")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--test-max", type=int, default=600, help="held-out questions evaluated")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="decode frames (lerobot venv) and stop")
    a = ap.parse_args(argv)

    recs = prepare(a.dataset, a.frames, a.out)
    if a.prepare_only:
        return 0
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    rng = random.Random(0)
    ex = [{**r, "axis": ax, "label": int(r["bucket"][k])} for r in recs for k, ax in enumerate(AXES)]
    train = [e for e in ex if not test_episode(e["episode"])]
    test = [e for e in ex if test_episode(e["episode"])]
    random.Random(1).shuffle(test)
    test = test[:a.test_max]
    print(f"train {len(train)} questions, test {len(test)}", flush=True)

    proc = AutoProcessor.from_pretrained(a.base)
    tok = proc.tokenizer
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               llm_int8_skip_modules=["lm_head", "visual"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    head = json.load(open(os.path.join(a.adapter, "decision_head.json")))
    lo = head["slots"]["ranges"]["choice"][0]
    bias = torch.tensor(head["bias"][lo:lo + len(LETTERS)], device="cuda")

    cache: dict[str, Image.Image] = {}

    def img(p):
        if p not in cache:
            cache[p] = Image.open(p).convert("RGB")
        return cache[p]

    def logits_for(e):
        batch = proc(text=[prompt(a.object, e["axis"])], images=[img(e["wrist"]), img(e["side"])],
                     return_tensors="pt").to(0)
        return model(**batch).logits[0, -1, letter_ids].float() + bias

    centres = np.array([-0.075, -0.045, -0.02, 0.0, 0.02, 0.045, 0.075])

    def evaluate(tag):
        model.eval()
        res = {}
        with torch.no_grad():
            P = np.array([torch.softmax(logits_for(e), -1).cpu().numpy() for e in test])
        Y = np.array([e["label"] for e in test])
        T = np.array([e["offset"][AXES.index(e["axis"])] for e in test])
        step = P @ centres
        for ax in AXES:
            k = np.array([e["axis"] == ax for e in test])
            off = k & (Y != 3)
            res[ax] = {"bucket_acc": round(float((P[k].argmax(1) == Y[k]).mean()), 3),
                       "within_one": round(float((np.abs(P[k].argmax(1) - Y[k]) <= 1).mean()), 3),
                       "sign_acc_when_off": round(float((np.sign(step[off]) == np.sign(T[off])).mean()), 3)
                       if off.any() else None,
                       "step_mae_cm": round(100 * float(np.abs(step[k] - T[k]).mean()), 2)}
        res["nll"] = round(float(-np.log(P[np.arange(len(Y)), Y] + 1e-9).mean()), 3)
        print(tag, json.dumps(res), flush=True)
        model.train()
        return res

    os.makedirs(a.out, exist_ok=True)
    report = {"zero_shot": evaluate("zero-shot (held-out episodes)")}
    if not a.eval_only:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        steps = int(len(train) * a.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) *
                                                  max(0.05, 1 - s * a.accum / max(steps, 1)))
        model.train()
        order = list(range(len(train)))
        run = 0.0
        for i in range(steps):
            if i % len(train) == 0:
                rng.shuffle(order)
            e = train[order[i % len(train)]]
            loss = torch.nn.functional.cross_entropy(logits_for(e)[None], torch.tensor([e["label"]], device="cuda"))
            (loss / a.accum).backward()
            run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (i + 1) % 200 == 0:
                print(f"step {i + 1}/{steps} loss {run:.3f}", flush=True)
            if (i + 1) % 3000 == 0 or i + 1 == steps:
                report[f"step_{i + 1}"] = evaluate(f"step {i + 1}")
                model.save_pretrained(a.out)
        shutil.copy(os.path.join(a.adapter, "decision_head.json"), os.path.join(a.out, "decision_head.json"))
        cal = os.path.join(os.path.dirname(os.path.abspath(a.adapter)), "calibration.json")
        if os.path.exists(cal):
            shutil.copy(cal, os.path.join(os.path.dirname(os.path.abspath(a.out)), "calibration.json"))
    json.dump(report, open(os.path.join(a.out, "train_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
