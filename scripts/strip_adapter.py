"""Copy a PEFT adapter for vLLM: drop lm_head.base_layer (vLLM rejects it).
python scripts/strip_adapter.py <src_dir> <dst_dir>"""
import json, os, shutil, sys
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
w = load_file(os.path.join(src, "adapter_model.safetensors"))
drop = [k for k in w if "base_layer" in k]
save_file({k: v for k, v in w.items() if k not in drop}, os.path.join(dst, "adapter_model.safetensors"))
for f in os.listdir(src):
    if f.endswith((".json", ".md")):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print(json.dumps({"dropped": drop, "kept": len(w) - len(drop)}))
