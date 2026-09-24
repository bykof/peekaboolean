#!/usr/bin/env python3
"""Merge prepared datasets into one training mixture.

  python -m peekaboolean.prepare_mix --root data --part ava --part aadb:19437 --out data/mix

Each part is a directory produced by one of the prepare_* scripts, holding
{train,val,calib}.jsonl and images/. Rows are rewritten so `image` is relative to
--root ("aadb/images/x.jpg"), which is what you then pass as --image-root, so no
copying or symlinking is needed.

`--part name:N` caps that part's *train* rows at N. The same keep fraction is applied
to its val and calib rows, so every split holds the same mixture as training — a val
set with a different balance than train measures the wrong thing.

Why cap at all: AADB emits 12 questions per image and AVA one, so a naive
concatenation is ~84% attribute questions and the model drifts toward those at the
expense of the overall aesthetic judgement. Equal row counts is a starting point, not
a tuned ratio.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SPLITS = ("train", "val", "calib")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="parent of the part directories")
    ap.add_argument("--part", action="append", default=[], metavar="NAME[:MAX_TRAIN_ROWS]")
    ap.add_argument("--out", default="data/mix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.part:
        raise SystemExit("need at least one --part")

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for spec in args.part:
        name, _, cap = spec.partition(":")
        part_dir = root / name
        if not (part_dir / "train.jsonl").exists():
            raise SystemExit(f"no train.jsonl under {part_dir}")

        train = load(part_dir / "train.jsonl")
        keep = min(int(cap), len(train)) if cap else len(train)
        frac = keep / len(train) if train else 0.0

        for split in SPLITS:
            rows = load(part_dir / f"{split}.jsonl")
            if not rows:
                continue
            n = keep if split == "train" else round(len(rows) * frac)
            rng = random.Random(f"{args.seed}:{name}:{split}")
            picked = rows if n >= len(rows) else rng.sample(rows, n)
            for r in picked:
                r = dict(r)
                r["image"] = f"{name}/images/{r['image']}"
                r["source"] = name
                merged[split].append(r)
            print(f"  {name:6} {split:6} {len(picked):7} / {len(rows):7} rows")

    for split in SPLITS:
        rows = merged[split]
        if not rows:
            continue
        # Shuffle so the loop alternates sources instead of training one then the other.
        random.Random(f"{args.seed}:{split}").shuffle(rows)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        by_src: dict[str, int] = {}
        for r in rows:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        print(f"[mix] {split}.jsonl {len(rows)} rows {by_src}")

    print(f"[mix] done -> {out}   (use --image-root {root})")


if __name__ == "__main__":
    main()
