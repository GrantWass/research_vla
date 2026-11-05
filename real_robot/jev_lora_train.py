"""Fine-tune JEV's own System 1 LoRA on our wrist-camera direction questions.

JEV System 1 = Qwen3.5 + LoRA (incl. lm_head): one forward pass over
  [kind] choice\\n[state] <wrist image>\\n[question] ...\\n[options]\\nA) ..\\nE) ..\\n[decision]:
and the next-token logits of the option letters are the answer
(autotrust JEV-9B vl/serve_decide.py, protocol bare-v1). This continues that
adapter with cross-entropy on the right letter, so vLLM serves the result
unchanged (--lora-modules jev-decision=<out>).

Data: wrist frames 12/25 steps before a demo grasp (cube between the fingers),
cropped off-centre by a random fraction of the frame, so the cube's image
offset - and its bucket - is known exactly (real_robot.jev_dir_eval.shifted).
Held out by episode. Base in 4-bit (QLoRA) so it trains beside the pi0.5
server on the 24 GB 3090.

  python -m real_robot.jev_lora_train --dirs outputs/jev_dirs --adapter ~/jev/JEV-9B/vl/adapter_vllm \\
      --base ~/jev/Qwen3.5-9B --out outputs/jev_lora_dir
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

import numpy as np

from .jev_dir_eval import LR, UD, shifted

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
LETTERS = "ABCDE"
CENTRE_FRAC, FAR_FRAC = 0.03, 0.10  # |image offset| (frame fraction) bucket edges
PRE = ("Wrist camera (mounted on the gripper; the two black fingers are at the bottom of the image, "
       "the point between them is where the gripper would grasp):")


def question(obj: str, which: str) -> str:  # must match real_robot.jev_dir_eval.ask
    return (f"In the wrist camera image, where is the {obj} relative to the centre between the gripper fingers, "
            + ("left to right?" if which == "lr" else "up and down in the image?"))


def prompt(obj: str, which: str) -> str:
    opts = LR if which == "lr" else UD
    lines = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))
    return (f"[kind] choice\n[state] {PRE}{IMAGE}\n[question] {question(obj, which)}\n[options]\n{lines}\n[decision]:")


def bucket(off: float) -> int:
    """Image offset (+ = right / down) -> 0..4 (far left/up .. far right/down)."""
    if abs(off) < CENTRE_FRAC:
        return 2
    return (3 if abs(off) < FAR_FRAC else 4) if off > 0 else (1 if abs(off) < FAR_FRAC else 0)


def make_examples(rows, n_per_frame: int, rng: random.Random):
    ex = []
    for r in rows:
        for _ in range(n_per_frame):
            which = rng.choice(["lr", "ud"])
            # a third near centre so "centred" is learned, the rest spread out
            s = rng.uniform(-0.03, 0.03) if rng.random() < 0.3 else rng.uniform(-0.2, 0.2)
            other = rng.uniform(-0.05, 0.05)  # small offset on the other axis too
            dx, dy = (s, other) if which == "lr" else (other, s)
            ex.append({"wrist": r["wrist"], "episode": r["episode"], "which": which,
                       "dx": dx, "dy": dy, "label": bucket(-s)})
    return ex


def main(argv=None) -> int:
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object", default="Rubik's cube")
    ap.add_argument("--per-frame", type=int, default=12)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--frames-before", default="12,25", help="demo frames this many steps before the grasp")
    a = ap.parse_args(argv)
    a.frames_before = [int(x) for x in a.frames_before.split(",")]

    rng = random.Random(0)
    rows = [json.loads(x) for x in open(os.path.join(a.dirs, "labels.jsonl"))]
    rows = [r for r in rows if r["before_grasp"] in a.frames_before]
    eps = sorted({r["episode"] for r in rows})
    rng.shuffle(eps)
    test_eps = set(eps[: max(1, len(eps) // 5)])
    train = make_examples([r for r in rows if r["episode"] not in test_eps], a.per_frame, rng)
    test = make_examples([r for r in rows if r["episode"] in test_eps], 6, random.Random(1))
    print(f"train {len(train)} examples ({len(eps) - len(test_eps)} episodes), test {len(test)} "
          f"({len(test_eps)} episodes)", flush=True)

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
    bias = torch.tensor(head["bias"][lo:lo + 5], device="cuda")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {n_train / 1e6:.1f} M", flush=True)

    cache = {}

    def load(p):
        if p not in cache:
            im = Image.open(p).convert("RGB")
            im = im.resize((a.width, round(im.height * a.width / im.width)))
            cache[p] = np.asarray(im)
        return cache[p]

    def logits_for(e):
        img = shifted(load(e["wrist"]), e["dx"], e["dy"])
        batch = proc(text=[prompt(a.object, e["which"])], images=[Image.fromarray(img)], return_tensors="pt").to(0)
        out = model(**batch)
        return out.logits[0, -1, letter_ids].float() + bias

    def evaluate(tag):
        model.eval()
        P, Y, W = [], [], []
        with torch.no_grad():
            for e in test:
                P.append(torch.softmax(logits_for(e), -1).cpu().numpy())
                Y.append(e["label"])
                W.append(e["which"])
        P, Y = np.array(P), np.array(Y)
        v = P @ np.array([-2, -1, 0, 1, 2.0])
        sgn = np.sign(Y - 2)
        res = {"bucket_acc": float((P.argmax(1) == Y).mean()),
               "direction_acc": float((np.sign(v[sgn != 0]) == sgn[sgn != 0]).mean()),
               "centred_acc": float((np.abs(v[Y == 2]) < 0.3).mean()),
               "nll": float(-np.log(P[np.arange(len(Y)), Y] + 1e-9).mean())}
        for w in ("lr", "ud"):
            m = np.array(W) == w
            res[f"{w}_direction_acc"] = float((np.sign(v[m & (sgn != 0)]) == sgn[m & (sgn != 0)]).mean())
        print(tag, json.dumps({k: round(x, 3) for k, x in res.items()}), flush=True)
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
            if (i + 1) % 1000 == 0 or i + 1 == steps:
                report[f"step_{i + 1}"] = evaluate(f"step {i + 1}")
        model.save_pretrained(a.out)
        for f in ("decision_head.json",):
            shutil.copy(os.path.join(a.adapter, f), os.path.join(a.out, f))
        cal = os.path.join(os.path.dirname(os.path.abspath(a.adapter)), "calibration.json")
        if os.path.exists(cal):  # serve_decide reads calibration.json from the adapter's parent
            shutil.copy(cal, os.path.join(os.path.dirname(os.path.abspath(a.out)), "calibration.json"))
    json.dump(report, open(os.path.join(a.out, "train_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
