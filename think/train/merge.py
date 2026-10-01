"""Merge a train_lora.py adapter into the base and write a full HF checkpoint that vLLM / mlx_vlm.convert can load.

python merge.py --base Qwen/Qwen3.6-35B-A3B --adapter runs/x/step-001000 --out merged/x

CPU only (the 35B needs ~75 GB RAM: the merged model plus one shard). PEFT does the merge math (also for --train-experts
adapters); the output is the base checkpoint shard by shard with only the LoRA-touched tensors replaced, so names,
shards, dtypes (fp32 A_log / norms), the `mtp.*` head and the vision tower stay byte-identical to the base.
Tokenizer, chat template, processor, config and index files are copied; if training ran with
--image-max-pixels/--image-min-pixels, preprocessor_config.json gets the same values so serving sees the same images.
"""
import argparse, glob, json, os, shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.base if os.path.isdir(a.base) else snapshot_download(a.base, local_files_only=True)

    model = PeftModel.from_pretrained(
        AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu"), a.adapter)
    n_lora = sum(1 for m in model.modules() if hasattr(m, "lora_A"))
    merged = model.merge_and_unload().state_dict()

    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(base, "*")):
        if os.path.isfile(f) and not f.endswith(".safetensors"):
            shutil.copy(f, a.out)
    shards = sorted(glob.glob(os.path.join(base, "*.safetensors")))
    changed = []
    for shard in shards:
        t = load_file(shard)
        for k, v in t.items():
            m = merged.get(k)
            if m is not None and m.dtype == v.dtype and m.shape == v.shape and not torch.equal(m, v):
                t[k] = m.contiguous()
                changed.append(k)
        save_file(t, os.path.join(a.out, os.path.basename(shard)), metadata={"format": "pt"})
    assert len(changed) == n_lora, f"{len(changed)} tensors changed but the adapter wraps {n_lora} layers"

    targs = os.path.join(os.path.dirname(os.path.abspath(a.adapter)), "train_args.json")
    targs = json.load(open(targs)) if os.path.exists(targs) else {}
    if targs.get("image_max_pixels") or targs.get("image_min_pixels"):
        pp = os.path.join(a.out, "preprocessor_config.json")
        cfg = json.load(open(pp))
        cfg["size"]["longest_edge"] = targs.get("image_max_pixels") or cfg["size"]["longest_edge"]
        cfg["size"]["shortest_edge"] = targs.get("image_min_pixels") or cfg["size"]["shortest_edge"]
        json.dump(cfg, open(pp, "w"), indent=2)
    print(json.dumps({"out": a.out, "shards": len(shards), "tensors_changed": len(changed), "lora_layers": n_lora,
                      "example": changed[:3]}))


if __name__ == "__main__":
    main()
