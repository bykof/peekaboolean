#!/usr/bin/env python3
"""Derive Choice and Noul rows from labels that already exist.

  python -m peekaboolean.prepare_derived --from ava  --part data/ava  --out data/ava-noul
  python -m peekaboolean.prepare_derived --from aadb --part data/aadb --out data/aadb-pick

Every real row in this project is a `score` row, so the choice and noul paths have
trained on nothing but the synthetic smoke set. Both sources already hold the labels
for the other two primitives; they are simply not written down as questions.

AVA keeps the whole vote histogram over 50+ raters, so the share of raters placing a
picture above a threshold is a *measured* probability with its dispersion intact. That
is exactly what a calibrated noul wants, and it is rare: most vision datasets give one
hard label and so teach accuracy without teaching uncertainty.

AADB rates 11 named attributes per image on a signed scale where negative means the
attribute actively detracts. Which attribute a picture is strongest in is then a real
choice label, and whether an attribute helps or hurts is a real noul. Two caveats, both
worth carrying into any metric computed on these rows:

  * AADB records a mean with no dispersion, so its noul value is a monotone
    re-expression of that mean, not a measured rate the way AVA's is. A model can be
    perfectly calibrated against it and still be wrong about how often raters agree.
  * Where two attributes are rated within noise of each other, "which is strongest" has
    no true answer. `--min-margin` drops those instead of teaching a coin flip. Noul
    rows get no such filter: a neutral attribute genuinely is a 0.5, and dropping the
    neutral cases would leave a model that has never seen one and cannot say so.

Output is a part directory in the shape prepare_mix expects, with `images` symlinked to
the source's, so nothing is copied and `--image-root data` still resolves.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "calib", "testnew")

# AADB's overall score shares the file but not the scale: it is [0, 1] where 0.5 is
# mediocre, while the 11 attributes are signed and 0.5 means "makes no difference".
# Comparing them would be comparing two different questions.
OVERALL = "aesthetic"

AVA_NOULS = [
    ("Would a randomly chosen viewer rate this picture above the middle of a 1-10 scale?",
     {"true": "they would give it 6 or more", "false": "they would give it 5 or less"},
     lambda h: _tail(h, 5)),
    ("Would a randomly chosen viewer call this picture outstanding?",
     {"true": "they would give it 8 or more out of 10", "false": "they would give it less"},
     lambda h: _tail(h, 7)),
    ("Would a randomly chosen viewer rate this picture poorly?",
     {"true": "they would give it 3 or less out of 10", "false": "they would give it more"},
     lambda h: 1.0 - _tail(h, 3)),
]


def _tail(hist: list[float], first_bin: int) -> float:
    """Share of the vote at or above a bin. AVA bin i is the count of rating i+1."""
    total = sum(hist)
    return sum(hist[first_bin:]) / total if total else 0.5


def implied_mean(hist: list[float]) -> float:
    """The value a spike histogram encodes, on [0, 1]. Exact for prepare_aadb's spikes."""
    total = sum(hist)
    if total <= 0:
        return 0.5
    k = len(hist)
    return sum(h * (i + 0.5) / k for i, h in enumerate(hist)) / total


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if path.exists() else []


def from_ava(rows: list[dict], rng: random.Random, per_image: int, kinds: set, **_) -> list[dict]:
    out = []
    if "noul" not in kinds:       # AVA's histogram supports no choice label
        return out
    for r in rows:
        for question, criteria, fn in rng.sample(AVA_NOULS, min(per_image, len(AVA_NOULS))):
            out.append({"image": r["image"], "type": "noul", "state": "",
                        "instructions": question, "criteria": criteria,
                        "value": round(float(fn(r["hist"])), 5)})
    return out


def from_aadb(rows: list[dict], rng: random.Random, per_image: int,
              min_margin: float, options: int, stats: dict, kinds: set) -> list[dict]:
    by_image: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["attribute"] != OVERALL:
            by_image[r["image"]][r["attribute"]] = implied_mean(r["hist"])

    out = []
    for image, attrs in by_image.items():
        if len(attrs) < options:
            continue
        # "Does this attribute help the picture?" -- the unit value IS P(helps), since
        # the signed scale maps 0 (no effect) onto 0.5 and +1 (helps most) onto 1.
        for name in (rng.sample(list(attrs), min(per_image, len(attrs)))
                     if "noul" in kinds else []):
            out.append({"image": image, "type": "noul", "state": "",
                        "instructions": f"Does the picture being {name} help it?",
                        "criteria": {"true": f"being {name} adds to the picture",
                                     "false": f"it does not, or it detracts"},
                        "value": round(attrs[name], 5)})
            stats["noul_values"].append(attrs[name])

        for label_name, pick in ((("strength", max), ("weakness", min)) if "choice" in kinds else ()):
            chosen = rng.sample(list(attrs), options)
            ranked = sorted(chosen, key=lambda a: attrs[a], reverse=(pick is max))
            margin = abs(attrs[ranked[0]] - attrs[ranked[1]])
            stats["margins"].append(margin)
            if margin < min_margin:
                stats["dropped"] += 1
                continue
            question = ("Which of these is this photograph's greatest strength?"
                        if label_name == "strength" else
                        "Which of these does this photograph handle worst?")
            out.append({"image": image, "type": "choice", "state": "",
                        "instructions": question,
                        "criteria": {a: f"the picture is {a}" for a in chosen},
                        "label": ranked[0]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True, choices=["ava", "aadb"])
    ap.add_argument("--part", required=True, help="a directory a prepare_* script wrote")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=2,
                    help="noul questions drawn per image (AVA has 3 to draw from)")
    ap.add_argument("--options", type=int, default=4, help="attributes offered per choice")
    ap.add_argument("--min-margin", type=float, default=0.08,
                    help="drop a choice whose top two are rated closer than this")
    ap.add_argument("--kinds", default="choice,noul",
                    help="which primitives to derive; AADB's nouls sit near neutral for "
                         "44%% of images, so they are worth keeping in their own part")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    part, out = Path(args.part), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if not link.exists():
        os.symlink(os.path.relpath(part / "images", out), link)

    build = from_ava if args.source == "ava" else from_aadb
    stats = {"margins": [], "noul_values": [], "dropped": 0}
    for split in SPLITS:
        rows = load(part / f"{split}.jsonl")
        if not rows:
            continue
        derived = build(rows, random.Random(f"{args.seed}:{split}"), args.per_image,
                        min_margin=args.min_margin, options=args.options, stats=stats,
                        kinds={k.strip() for k in args.kinds.split(",")})
        kinds: dict[str, int] = {}
        for r in derived:
            kinds[r["type"]] = kinds.get(r["type"], 0) + 1
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in derived), encoding="utf-8")
        print(f"  {split:8} {len(rows):7} source rows -> {len(derived):7} derived {kinds}")

    if stats["margins"]:
        m = sorted(stats["margins"])
        print(f"\n  choice margins: median {m[len(m)//2]:.3f}, "
              f"{stats['dropped']}/{len(m)} dropped below {args.min_margin}")
    if stats["noul_values"]:
        v = sorted(stats["noul_values"])
        near = sum(1 for x in v if abs(x - 0.5) < 0.05) / len(v)
        print(f"  noul values: median {v[len(v)//2]:.3f}, "
              f"10th {v[len(v)//10]:.3f}, 90th {v[9*len(v)//10]:.3f}, "
              f"{near:.0%} within 0.05 of neutral")
    print(f"[derived] -> {out}   (images -> {os.readlink(link)})")


if __name__ == "__main__":
    main()
