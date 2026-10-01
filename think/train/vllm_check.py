"""Send JSONL rows to a vLLM OpenAI server the way jevsrv does (images first, then text) and compare final letters.
python vllm_check.py --url http://127.0.0.1:8011/v1 --model NAME --data smoke/eval.jsonl --n 8 [--template-kwargs '{}']
"""
import argparse, base64, json, mimetypes, re, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--template-kwargs", default="{}")
a = ap.parse_args()

letter = lambda s: (re.findall(r"Answer\s*:\s*\**\s*([A-Z])", s or "") or [None])[-1]
hits = 0
rows = [json.loads(l) for l in open(a.data)][:a.n]
for row in rows:
    imgs = [{"type": "image_url", "image_url": {"url": f"data:{mimetypes.guess_type(p)[0]};base64,"
                                                       + base64.b64encode(open(p, "rb").read()).decode()}}
            for p in row["images"]]
    body = {"model": a.model, "max_tokens": a.max_tokens, "temperature": 0,
            "chat_template_kwargs": json.loads(a.template_kwargs),
            "messages": [{"role": "user", "content": imgs + [{"type": "text", "text": row["prompt"]}]}]}
    req = urllib.request.Request(a.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]
    hits += letter(text) == letter(row["response"])
    print(json.dumps({"gold": letter(row["response"]), "got": letter(text), "head": (text or "")[:70], "tail": (text or "")[-70:]}))
print(json.dumps({"model": a.model, "letter_acc": f"{hits}/{len(rows)}"}))
