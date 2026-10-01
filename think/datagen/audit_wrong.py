"""25 random all-wrong items (no correct sample): question, options, gold, the model's answers, image paths.
python audit_wrong.py > audit/wrong.txt
"""
import glob, json, random, string
from collections import Counter, defaultdict
from rft import build

D = "/root/big/data"
items = {it["id"]: it for f in glob.glob(f"{D}/items/*.jsonl") for it in map(json.loads, open(f))}
S = defaultdict(list)
for f in glob.glob(f"{D}/rft/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        S[r["item_id"]] += r.get("samples", [])
wrong = sorted(i for i, ss in S.items() if i in items and ss and not any(s["correct"] for s in ss))
print("all-wrong items:", len(wrong), dict(Counter(items[i]["source"] for i in wrong).most_common()))
for i in random.Random(5).sample(wrong, 25):
    it = items[i]
    labels = build(it, urls=False)[1]
    got = Counter(labels[string.ascii_uppercase.index(s["letter"])] if s["letter"] and string.ascii_uppercase.index(s["letter"]) < len(labels)
                  else s["letter"] for s in S[i])
    q = it["question"]
    print(f"\n== {i} [{it['dataset']}] gold={it['gold']!r} model={dict(got)}")
    print("   state:", json.dumps(it["state"], ensure_ascii=False)[:400])
    print("   Q:", q["instructions"][:300])
    print("   options:", json.dumps(q["criteria"], ensure_ascii=False)[:300])
    print("   images:", " ".join(it["images"]))
