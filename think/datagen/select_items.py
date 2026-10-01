"""Pick the ~6,000 items for the full RFT run and split them by samples per item.

Pilot items that still exist are kept. Unknown sources ~16% of items, with their controls: every ctrl-cover and
ctrl-field item, and the vqav2 item asking an unk-premise question where the premise holds.
n=4 for sources whose pilot accuracy is < 0.9 (plus those named by the coordinator and the rebuilt aokvqa), else n=2.
python select_items.py  -> rft/full_n2.txt, rft/full_n4.txt (random order)
"""
import glob, json, random
from collections import Counter, defaultdict

D = "/root/big/data"
R = random.Random(31)
QUOTA = {"vqav2": 600, "unk-premise": 300, "unk-cover": 220, "unk-field": 200, "unk-nocorrect": 150, "unk-domain": 130,
         "ctrl-cover": 10 ** 6, "ctrl-field": 10 ** 6}
DEFAULT, HARD = 190, 280
N4_ALWAYS = {"aokvqa", "rule-claim", "rule-count", "nlvr2", "ctrl-field"}


def main():
    items = {it["id"]: it for f in sorted(glob.glob(f"{D}/items/*.jsonl")) for it in map(json.loads, open(f))}
    acc, pilot = defaultdict(lambda: [0, 0]), set()
    for f in glob.glob(f"{D}/rft/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["item_id"] in items and "samples" in r:
                pilot.add(r["item_id"])
                a = acc[items[r["item_id"]]["source"]]
                a[0] += sum(s["correct"] for s in r["samples"])
                a[1] += len(r["samples"])
    pacc = {k: v[0] / v[1] for k, v in acc.items()}
    n4 = {s for s, a in pacc.items() if a < 0.9} | N4_ALWAYS
    by = defaultdict(list)
    for i, it in items.items():
        by[it["source"]].append(i)
    chosen = set(pilot)
    for src, ids in by.items():
        q = QUOTA.get(src, HARD if src in n4 else DEFAULT)
        rest = [i for i in ids if i not in chosen]
        R.shuffle(rest)
        have = sum(i in chosen for i in ids)
        chosen |= set(rest[:max(0, q - have)])
    # the premise-holds twin of every chosen unk-premise question
    twins = defaultdict(list)
    for i in by["vqav2"]:
        twins[items[i]["question"]["instructions"]].append(i)
    for i in [i for i in chosen if items[i]["source"] == "unk-premise"]:
        chosen |= set(twins.get(items[i]["question"]["instructions"], [])[:1])
    two = [i for i in chosen if items[i]["source"] not in n4]
    four = [i for i in chosen if items[i]["source"] in n4]
    R.shuffle(two)
    R.shuffle(four)
    open(f"{D}/rft/full_n2.txt", "w").write("\n".join(two) + "\n")
    open(f"{D}/rft/full_n4.txt", "w").write("\n".join(four) + "\n")
    c = Counter(items[i]["source"] for i in chosen)
    print("pilot accuracy:", {k: round(v, 3) for k, v in sorted(pacc.items(), key=lambda x: x[1])})
    print("n=4 sources:", sorted(n4))
    print("chosen", len(chosen), "n2", len(two), "n4", len(four), "unknown share",
          round(sum(items[i]["gold"] == "__unknown__" for i in chosen) / len(chosen), 3))
    print(dict(sorted(c.items())))


if __name__ == "__main__":
    main()
