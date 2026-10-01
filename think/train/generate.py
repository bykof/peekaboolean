"""Load a model (merged, or base + --adapter) with transformers: teacher-forced loss on the rows' responses, then greedy
generation, and whether the generated final letter equals the row's. Same loss for merged and base+adapter = merge OK.
python generate.py --model merged/x --data smoke/eval.jsonl --n 8 [--adapter runs/x/step-N] [--template-kwargs '{}']
"""
import argparse, json, re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_lora import Encoder, collate, loss_sum

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter")
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max-new-tokens", type=int, default=2048)
ap.add_argument("--template-kwargs", default="{}")
a = ap.parse_args()

proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).eval()
enc = Encoder(proc, json.loads(a.template_kwargs))
rows = [json.loads(l) for l in open(a.data)][:a.n]
letter = lambda s: (re.findall(r"Answer\s*:\s*\**\s*([A-Z])", s) or [None])[-1]
tot = n = hits = 0
with torch.no_grad():
    for row in rows:
        batch = {k: v.cuda() for k, v in collate([enc(row)], proc.tokenizer.pad_token_id).items()}
        n += int((batch["labels"] != -100).sum())
        tot += loss_sum(model, dict(batch)).item()
        x = proc(text=[enc.prompt_text(row)], images=[Image.open(p).convert("RGB") for p in row["images"]] or None,
                 return_tensors="pt").to("cuda")
        out = model.generate(**x, max_new_tokens=a.max_new_tokens, do_sample=False)
        text = proc.tokenizer.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=False)
        hits += letter(text) == letter(row["response"])
        print(json.dumps({"gold": letter(row["response"]), "got": letter(text), "tail": text[-90:]}))
print(json.dumps({"model": a.model, "adapter": a.adapter, "loss": round(tot / n, 5), "letter_acc": f"{hits}/{len(rows)}"}))
