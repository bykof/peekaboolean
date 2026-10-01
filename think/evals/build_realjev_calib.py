"""A calibration set from the unused remainder of RealJev's sources: same conversion, another seed, and no item,
image or cluster that is in RealJev. Only for fitting calib.py's 2-parameter map.

venv/bin/python build_realjev_calib.py  ->  realjev-calib/records.jsonl, realjev-calib/images/
"""
import json, random
from collections import Counter
from pathlib import Path

import build_realjev as b

PER_SOURCE = 80
taken_ids = {json.loads(l)["id"] for l in open("realjev/records.jsonl")}
taken_groups = {json.loads(l)["group"] for l in open("realjev/records.jsonl")}
b.OUT = Path("/root/big/evals/realjev-calib")


def sample(cands, n, source, strata=None):
    rng = random.Random(f"calib:{b.SEED}:{source}")
    cands = [c for c in cands if c]
    rng.shuffle(cands)
    quota = {k: max(1, round(v * PER_SOURCE / n)) for k, v in strata.items()} if strata else None
    out, taken = [], Counter()
    for c in cands:
        key = c["task"] if strata else None
        if taken[key] >= (quota.get(key, 0) if quota else PER_SOURCE):
            continue
        r = b.finalize(c)
        if r and r["id"] not in taken_ids and r["group"] not in taken_groups:
            out.append(r)
            taken[key] += 1
    b.stats[source]["kept"] = len(out)
    return out


b.sample = sample
if __name__ == "__main__":
    b.OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for fn in [b.mmstar, b.realworldqa, b.hallusionbench, b.pope, b.cvbench, b.blink, b.mme_realworld_lite]:
        recs += fn()
        print(fn.__name__, dict(b.stats[recs[-1]["source"]]), flush=True)
    with open(b.OUT / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(recs), "records")
