"""imajev's v1 held-out image exam (decision-v1 records from its own scripts/v1/convert_heldout_*.py) -> jrun records.

cd imajev-src && PYTHONPATH=src ../venv/bin/python scripts/v1/convert_heldout_<x>.py   (x = abstention countqa fashionpedia livewild pairs)
venv/bin/python convert_heldout.py   ->  heldout/records.jsonl (all), records-sub.jsonl + subsample-ids.txt, images/

Wire mapping is imajev_bench.runner.jev_payload's: boolean -> noul, choice -> choice {value: description},
ordinal -> score [level descriptions] (gold = zero-based level position). target None -> gold "__unknown__".
"""
import hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("/root/big/evals/imajev-src/data/decision-v1")
OUT = Path("/root/big/evals/heldout")
PANELS = ["heldout_abstention", "heldout_countqa", "heldout_fashionpedia", "heldout_livewild", "heldout_pairs"]


def convert(r):
    req, (f,) = r["request"], r["request"]["fields"]
    t, target = f["type"], r["target"]
    if t == "boolean":
        q = {"type": "noul", "criteria": {"true": f.get("yes_description"), "false": f.get("no_description")}}
        gold = "__unknown__" if target is None else ("true" if target else "false")
    elif t == "choice":
        q = {"type": "choice", "criteria": {o["value"]: o.get("description") for o in f["options"]}}
        gold = "__unknown__" if target is None else target
    else:
        q = {"type": "score", "criteria": [l["description"] for l in f["levels"]]}
        gold = "__unknown__" if target is None else str([l["value"] for l in f["levels"]].index(target))
    q = {"type": q["type"], "instructions": f["question"], "criteria": q["criteria"]}
    images = []
    for im in r["images"]:
        rel = Path("images") / r["source"] / Path(im["image"]).name
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC.parent.parent / im["image"], OUT / rel)
        images.append(str(rel))
    return {"id": r["id"], "source": r["source"], "group": f"{r['source']}:{r['source_group']}",
            "wire": {"state": req["state"], "questions": {"decision": q}, "images": images}, "gold": gold,
            "task": r["family"], "cause": r["abstention_cause"] or "answerable"}


def subsample(recs, per_panel=400, seed=20261001):
    """<= per_panel per panel, stratified by cause (proportional, largest remainder), sha256-ranked; pairs kept whole."""
    out = []
    for panel in PANELS:
        rs = [r for r in recs if r["source"] == panel]
        if panel == "heldout_pairs" or len(rs) <= per_panel:
            out += rs
            continue
        by = defaultdict(list)
        for r in rs:
            by[r["cause"]].append(r)
        quota = {c: per_panel * len(v) // len(rs) for c, v in by.items()}
        for c in sorted(by, key=lambda c: -(per_panel * len(by[c]) % len(rs)))[:per_panel - sum(quota.values())]:
            quota[c] += 1
        for c, v in by.items():
            out += sorted(v, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())[:quota[c]]
    return out


if __name__ == "__main__":
    recs = [convert(json.loads(l)) for p in PANELS for l in (SRC / p / "records.jsonl").open()]
    assert len({r["id"] for r in recs}) == len(recs)
    with (OUT / "records.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    sub = subsample(recs)
    with (OUT / "records-sub.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in sub)
    (OUT / "subsample-ids.txt").write_text("".join(r["id"] + "\n" for r in sub))
    print(len(sub), Counter((r["source"], r["cause"]) for r in sub))
    print(len(recs), Counter(r["source"] for r in recs), Counter(r["gold"] == "__unknown__" for r in recs))
