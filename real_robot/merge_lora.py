"""Turn a LoRA (PEFT) pi0.5 checkpoint into a full one that make policy,
the policy server and offline_eval load like any other checkpoint.

A LoRA run (LORA=16 bash scripts/finetune_pi05.sh) saves only the adapter
weights; this loads the base model they were trained on, merges the adapters
into its weights and writes a normal checkpoint next to the processors (with
the run's normalization stats):

  ~/lerobot_robot_piper/.venv/bin/python -m real_robot.merge_lora \\
      outputs/<run>/checkpoints/002000/pretrained_model      # -> .../pretrained_model_merged
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil


def merge(src: str, dst: str | None = None, log=print) -> str:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from peft import PeftConfig, PeftModel

    src = src.rstrip("/")
    dst = dst or src + "_merged"
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    base = PeftConfig.from_pretrained(src).base_model_name_or_path
    cfg = PreTrainedConfig.from_pretrained(src)
    cfg.use_peft = False
    log(f"base {base}; merging adapters from {src}")
    policy = get_policy_class(cfg.type).from_pretrained(base, config=cfg)
    merged = PeftModel.from_pretrained(policy, src).merge_and_unload()
    merged.save_pretrained(dst)
    # Processors carry the run's normalization stats; the base's would be wrong.
    for f in glob.glob(os.path.join(src, "policy_*")) + glob.glob(os.path.join(src, "train_config.json")):
        shutil.copy2(f, dst)
    log(f"wrote {dst}")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="LoRA checkpoint folder (.../checkpoints/<step>/pretrained_model)")
    ap.add_argument("dst", nargs="?", help="output folder (default: <src>_merged)")
    args = ap.parse_args(argv)
    merge(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
