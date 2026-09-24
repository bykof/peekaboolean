"""Age and person questions with real labels, from FairFace (CC BY 4.0).

FairFace labels each face crop with an age group (0-2, 3-9, 10-19, ..., 70+) and a
binary gender annotation. That gives exact supervision the teacher cannot: Score rows
whose levels are age ranges (merged and reworded several ways, so the model reads the
rubric rather than memorizing one ladder), and Noul rows for "is there a child / a
woman / a man". Ambiguous cases (teenagers for child/adult questions) are left out.
The race annotation is not used.

Crops show one person with some context (the 1.25 padding variant), not whole scenes;
scene-level person questions come from the teacher data.

  python -m peekaboolean.prepare_age --out data/age
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

GROUPS = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 99)]

# Level sets as lists of (first group, last group) index ranges, youngest first.
BUCKETINGS = [
    [(i, i) for i in range(9)],                                           # all nine groups
    [(0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8)],
    [(0, 1), (2, 2), (3, 4), (5, 6), (7, 8)],                             # child/teen/young/middle/senior
    [(0, 2), (3, 4), (5, 6), (7, 8)],
    [(0, 2), (3, 5), (6, 8)],
    [(0, 3), (4, 8)],
]
WORDED = {(0, 1): "a child", (2, 2): "a teenager", (3, 4): "a young adult", (5, 6): "middle-aged",
          (7, 8): "a senior", (0, 2): "under 20", (3, 5): "an adult under 50", (6, 8): "50 or older"}


def level_text(lo, hi, style):
    a, b = GROUPS[lo][0], GROUPS[hi][1]
    if style == "words" and (lo, hi) in WORDED:
        return WORDED[(lo, hi)]
    if b >= 99:
        return f"{a}+" if style == "short" else f"{a} years or older"
    return f"{a} to {b} years" if style != "short" else f"{a}-{b}"


def age_row(image, group, female, rng):
    buckets = rng.choice(BUCKETINGS)
    style = rng.choice(["years", "years", "short", "words"])
    levels = [level_text(lo, hi, style) for lo, hi in buckets]
    if len(set(levels)) < len(levels):
        levels = [level_text(lo, hi, "years") for lo, hi in buckets]
    target = next(i for i, (lo, hi) in enumerate(buckets) if lo <= group <= hi)
    # Perceived age is noisy: keep most mass on the labelled level, a little on neighbours.
    hist = [0.0] * len(levels)
    hist[target] = 1.0
    for n in (target - 1, target + 1):
        if 0 <= n < len(levels):
            hist[n] += 0.1
    total = sum(hist); hist = [h / total for h in hist]
    who = "person"
    if group >= 3 and rng.random() < 0.5:
        who = "woman" if female else "man"
    elif group <= 1 and rng.random() < 0.5:
        who = "child"
    question = rng.choice([f"How old is the {who} in the picture?", f"How old is this {who}?",
                           f"What is the approximate age of the {who} shown?", f"Estimate the age of the {who}."])
    return {"image": image, "type": "score", "state": "", "instructions": question,
            "criteria": levels, "hist": hist, "source": "fairface-age", "supervision": "fairface_age_group"}


def person_rows(image, group, female, rng):
    rows = []
    base = {"image": image, "type": "noul", "state": "", "supervision": "fairface_labels"}
    # Teenagers are neither clearly child nor clearly adult. Children are ~16% of FairFace,
    # so most adult "no" cases are dropped to keep about two negatives per positive.
    if group <= 1 or (group >= 3 and rng.random() < 0.4):
        # Person-level wording: a crop shows one person, so "is there a child in the picture"
        # would teach "is the main person a child" -- wrong for scenes with an adult and a
        # child (v8 said 0.44 for the child on a mother-and-child photo). Scene-level
        # presence questions come from the teacher data.
        rows.append({**base, "instructions": rng.choice(["Is this person a child?", "Is the person shown a child?",
                                                          "Is the person in the picture a kid?"]),
                     "value": float(group <= 1), "source": "fairface-child"})
    if group >= 3:
        asked_female = rng.random() < 0.5
        noun = "woman" if asked_female else "man"
        rows.append({**base, "instructions": rng.choice([f"Is this person a {noun}?", f"Is the person shown a {noun}?",
                                                          f"Is the person in the picture a {noun}?"]),
                     "value": float(asked_female == female), "source": "fairface-gender"})
    return rows


def split_for(digest):
    n = int(digest[:8], 16) % 100
    return "test" if n < 5 else "calib" if n < 10 else "val" if n < 15 else "train"


def main():
    from datasets import load_dataset
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/age")
    ap.add_argument("--images", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    out = Path(args.out).resolve(); (out / "images").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25")
    records = [r for split in ("train", "validation") for r in ds[split].select_columns(["image", "age", "gender"])]
    rng = random.Random(args.seed); rng.shuffle(records)
    rows = {s: [] for s in ("train", "val", "calib", "test")}
    for r in records[:args.images]:
        im = r["image"].convert("RGB")
        digest = hashlib.sha256(im.tobytes()).hexdigest()
        path = out / "images" / f"{digest}.jpg"
        if not path.exists():
            im.save(path, quality=95)
        split = split_for(digest)
        female = r["gender"] == 1
        rrng = random.Random(f"{args.seed}:{digest}")
        rows[split].append(age_row(str(path), r["age"], female, rrng))
        rows[split].extend(person_rows(str(path), r["age"], female, rrng))
    manifest = {"source": "HuggingFaceM4/FairFace 1.25 (CC BY 4.0)", "images": args.images, "splits": {}}
    for split, rr in rows.items():
        random.Random(f"{args.seed}:{split}").shuffle(rr)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rr))
        manifest["splits"][split] = {"rows": len(rr), "sources": dict(Counter(x["source"] for x in rr))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
