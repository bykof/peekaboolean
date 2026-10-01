"""Check that train_lora.py's ids are token-identical to what vLLM feeds and samples at inference.

Sends one jevsrv-style request (image first, then text) to the running vLLM with return_token_ids, then compares
  prompt:   vLLM prompt_token_ids  vs  Encoder prompt ids (chat template + processor image expansion)
  response: vLLM sampled token_ids vs  Encoder.response_ids(message.content)  (retokenized text + <|im_end|>)
python check_template.py --model Qwen/Qwen3.6-35B-A3B --image /root/big/imajev-bench/assets/f5c651efe765736d.jpg
"""
import argparse, base64, json, string, sys, urllib.request

sys.path.insert(0, "/root/big")
import jevsrv
from transformers import AutoProcessor
from PIL import Image

from train_lora import Encoder

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
ap.add_argument("--vllm", default="http://127.0.0.1:8000/v1")
ap.add_argument("--served", default="q36")
ap.add_argument("--image", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--question", default="Is there a person visible in the image?")
ap.add_argument("--save", default="check_template.vllm.json")
a = ap.parse_args()

q = {"type": "noul", "instructions": a.question, "criteria": {}}
labels, shown = jevsrv.options_for(q)
text = jevsrv.PROMPT.format(state="{}", question=q["instructions"], unk=string.ascii_uppercase[len(labels) - 1],
                            options="\n".join(f"{string.ascii_uppercase[i]}. {s}" for i, s in enumerate(shown)))
url = "data:image/jpeg;base64," + base64.b64encode(open(a.image, "rb").read()).decode()
body = {"model": a.served, "max_tokens": 8000, "temperature": 0.6, "top_p": 0.95, "n": a.n, "return_token_ids": True,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                  {"type": "text", "text": text}]}]}
req = urllib.request.Request(a.vllm + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
json.dump(out, open(a.save, "w"))

enc = Encoder(AutoProcessor.from_pretrained(a.model), {})
row = {"images": [a.image], "prompt": text}
p = enc.p(text=[enc.prompt_text(row)], images=[Image.open(a.image).convert("RGB")])["input_ids"][0]
p = list(p)
print("prompt tail:", repr(enc.prompt_text(row)[-60:]))
print(f"prompt: vllm {len(out['prompt_token_ids'])} ids, ours {len(p)}, identical={out['prompt_token_ids'] == p}")
ok = out["prompt_token_ids"] == p
for c in out["choices"]:
    content, gen = c["message"]["content"], c["token_ids"]
    ours = enc.response_ids({"response": content})
    same = gen == ours or gen + [enc.end] == ours  # vLLM may or may not list the stop token
    ok &= same
    first = next((i for i, (x, y) in enumerate(zip(gen, ours)) if x != y), None)
    print(f"sample {c['index']}: finish={c['finish_reason']} gen {len(gen)} ids, ours {len(ours)}, identical={same}, "
          f"</think> in content={'</think>' in content}, starts={content[:40]!r}, ends={content[-40:]!r}"
          + ("" if same else f", first diff at {first}: vllm {enc.p.tokenizer.decode(gen[first:first+5])!r} "
                             f"ours {enc.p.tokenizer.decode(ours[first:first+5])!r}"))
print("ALL IDENTICAL" if ok else "MISMATCH")
