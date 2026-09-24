"""Training mixture (v6, v7): v5 public rows + teacher-written requests + count scores.

v5 already fixed the image splits; everything here inherits them, so no image
crosses train/val/calib/test. Count scores turn "how many" answers from photo
sources into ordinal rubrics, the one general-purpose Score supervision that has an
exact label rather than a teacher's belief. The count stays a Choice row too: the
same fact asked as a rubric and as options is two primitives, not a duplicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "calib", "test")
NOUNS = re.compile(r"how many (.+?)(?: are| is| can| do| does| in| on| there|\?|$)", re.I)

# Bucketings as (upper bound inclusive, label template). Several so the model reads
# the rubric rather than memorizing one ladder. "{n}" is the counted noun phrase.
BUCKETS = [
    [(0, "no {n}"), (1, "one"), (2, "two"), (3, "three"), (10**9, "four or more")],
    [(0, "none"), (1, "exactly one"), (3, "two or three"), (6, "four to six"), (10**9, "more than six")],
    [(0, "zero"), (2, "one or two"), (5, "three to five"), (10**9, "six or more")],
    [(1, "at most one"), (10**9, "several")],
    [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (10**9, "7+")],
    [(0, "empty of {n}"), (2, "a few {n}"), (6, "a handful of {n}"), (10**9, "many {n}")],
    [(2, "fewer than three"), (4, "three or four"), (10**9, "five or more")],
]


def count_score(row, rng):
    answer = row["criteria"][row["label"]]
    if not answer.isdigit() or int(answer) > 30:
        return None
    m = NOUNS.match(row["instructions"].strip())
    noun = m.group(1).strip() if m else "of them"
    value = int(answer)
    buckets = rng.choice(BUCKETS)
    levels = [label.format(n=noun) for _, label in buckets]
    hist = [0.0] * len(levels)
    hist[next(i for i, (hi, _) in enumerate(buckets) if value <= hi)] = 1.0
    return {"image": row["image"], "type": "score", "state": "", "instructions": row["instructions"],
            "criteria": levels, "hist": hist, "source": f"{row['source']}-count",
            "supervision": "upstream_count_bucketed"}


def temper(view, t):
    w = [max(x, 1e-9) ** (1 / t) for x in view]
    return [x / sum(w) for x in w]


def soften(views, t):
    """Teacher belief at temperature t: each option-order view tempered, then averaged."""
    a, b = temper(views[0], t), temper(views[1], t)
    return [(x + y) / 2 for x, y in zip(a, b)]


def fit_temperatures(path):
    """One temperature per question type from teacher answers to public rows with known
    answers. Score uses count rubrics (exact one-hot labels) when the calibration file has
    them, and otherwise borrows the choice temperature."""
    import math
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    grid = [0.5 + 0.1 * i for i in range(46)]
    fitted = {}
    for kind in ("choice", "noul", "score"):
        part = [r for r in rows if r["type"] == kind]
        if len(part) < 50: continue
        nll = lambda t: sum(-math.log(max(soften(r["views"], t)[r["truth"]], 1e-9)) for r in part) / len(part)
        fitted[kind] = min(grid, key=nll)
        fitted[f"{kind}_nll_raw"], fitted[f"{kind}_nll_fitted"] = nll(1.0), nll(fitted[kind])
        argmax = lambda p: max(range(len(p)), key=p.__getitem__)
        fitted[f"{kind}_accuracy"] = sum(argmax(soften(r["views"], 1.0)) == r["truth"] for r in part) / len(part)
        fitted[f"{kind}_n"] = len(part)
    fitted.setdefault("choice", 1.0); fitted.setdefault("noul", fitted["choice"])
    fitted.setdefault("score", fitted["choice"])
    return fitted


def apply_temperature(row, temperatures):
    if "teacher_views" not in row: return row
    p = soften(row["teacher_views"], temperatures[row["type"]])
    row = dict(row)
    if row["type"] == "choice":
        keys = list(row["criteria"]); row["target_probs"] = dict(zip(keys, p))
    elif row["type"] == "score":
        row["hist"] = p
    else:
        row["value"] = p[1]
    return row


def balance_noul(rows, cap, rng):
    """Drop majority-answer nouls until that answer is at most `cap` of all nouls."""
    yes = [r for r in rows if r["type"] == "noul" and r["value"] > 0.5]
    no = [r for r in rows if r["type"] == "noul" and r["value"] <= 0.5]
    major, minor = (yes, no) if len(yes) > len(no) else (no, yes)
    keep = min(len(major), round(cap / (1 - cap) * len(minor)))
    dropped = {id(r) for r in rng.sample(major, len(major) - keep)}
    return [r for r in rows if id(r) not in dropped]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default="data/general-v5")
    ap.add_argument("--teacher", default="data/teacher-v6")
    ap.add_argument("--out", default="data/general-v6")
    ap.add_argument("--count-sources", default="vqav2,clevr")
    ap.add_argument("--teacher-repeat", type=int, default=1, help="repeat teacher train rows to raise their share")
    ap.add_argument("--drop-train-sources", default="", help="comma list of sources left out of train only (kept in eval splits)")
    ap.add_argument("--balance-teacher-noul", type=float, default=0,
                    help="cap the majority answer (value above/below 0.5) of teacher nouls in train at this share; 0 = off")
    ap.add_argument("--extra", nargs="*", default=[], help="further prepared datasets (same split files) to append")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--teacher-calibration", help="teacher-calibration.jsonl (default: <teacher>/teacher-calibration.jsonl)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    count_sources = set(args.count_sources.split(","))
    calib = Path(args.teacher_calibration or Path(args.teacher) / "teacher-calibration.jsonl")
    temperatures = fit_temperatures(calib) if calib.exists() else {"choice": 1.0, "noul": 1.0, "score": 1.0}
    print(f"[v6] teacher temperatures {temperatures}", flush=True)
    manifest = {"public": args.public, "teacher": args.teacher, "seed": args.seed,
                "teacher_temperatures": temperatures, "splits": {}}
    split_images = {}
    for split in SPLITS:
        rows = [json.loads(l) for l in (Path(args.public) / f"{split}.jsonl").read_text().splitlines()]
        if split == "train" and args.drop_train_sources:
            drop = set(args.drop_train_sources.split(","))
            rows = [r for r in rows if r.get("source") not in drop]
        rng = random.Random(f"{args.seed}:count:{split}")
        counts = [s for r in rows if r["type"] == "choice" and r["source"] in count_sources
                  and r["instructions"].lower().startswith("how many") and (s := count_score(r, rng))]
        teacher_path = Path(args.teacher) / f"{split}.jsonl"
        teacher = [apply_temperature(json.loads(l), temperatures)
                   for l in teacher_path.read_text().splitlines()] if teacher_path.exists() else []
        if split == "train" and args.balance_teacher_noul:
            teacher = balance_noul(teacher, args.balance_teacher_noul, random.Random(f"{args.seed}:balance"))
        extra = [json.loads(l) for d in args.extra if (Path(d) / f"{split}.jsonl").exists()
                 for l in (Path(d) / f"{split}.jsonl").read_text().splitlines()]
        mix = rows + counts + teacher * (args.teacher_repeat if split == "train" else 1) + extra
        random.Random(f"{args.seed}:{split}").shuffle(mix)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mix))
        split_images[split] = {str(Path(r["image"]).resolve()) for r in mix}
        manifest["splits"][split] = {"rows": len(mix), "images": len(split_images[split]),
            "types": dict(Counter(r["type"] for r in mix)),
            "sources": dict(Counter(r["source"] for r in mix)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and split_images[a] & split_images[b]:
                raise RuntimeError(f"image leakage between {a} and {b}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
