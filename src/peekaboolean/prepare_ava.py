#!/usr/bin/env python3
"""Convert the AVA aesthetics subset into the row schema in data.py.

  python -m peekaboolean.prepare_ava --out data/ava

Source: trojblue/AVA-aesthetics-10pct-min50-10bins on the Hub — a 10% sample of AVA
restricted to images with at least 50 votes. It carries `rating_counts`, the native
10-bin DPChallenge vote histogram, so `hist` is copied across untouched and the
loader rebins it per question. Nothing is reconstructed from mean scores.

Image bytes are written out verbatim rather than re-encoded, so the JPEGs on disk are
the originals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

INSTRUCTIONS = "How aesthetic is the picture?"
ATTRIBUTE = "aesthetic"
REPO = "trojblue/AVA-aesthetics-10pct-min50-10bins"


def write_split(ds, out_dir: Path, image_dir: Path, name: str, indices=None) -> int:
    """Write one JSONL split, saving each image under image_dir. Returns row count."""
    rows = 0
    skipped = 0
    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for i in indices if indices is not None else range(len(ds)):
            ex = ds[int(i)]
            hist = [int(c) for c in ex["rating_counts"]]
            if sum(hist) <= 0:
                skipped += 1
                continue
            img = ex["image"]
            data = img["bytes"] if isinstance(img, dict) else None
            if data is None:
                skipped += 1
                continue
            stem = str(ex["image_id"])
            (image_dir / f"{stem}.jpg").write_bytes(data)
            f.write(json.dumps({
                "image": f"{stem}.jpg",
                "type": "score",
                "state": "",
                "attribute": ATTRIBUTE,
                "instructions": INSTRUCTIONS,
                "hist": hist,
            }) + "\n")
            rows += 1
    print(f"  {name}.jsonl  {rows} rows" + (f"  ({skipped} skipped)" if skipped else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ava")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--val-rows", type=int, default=1000, help="held out of train for val")
    ap.add_argument("--max-rows", type=int, default=0, help="cap train rows, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Image as HFImage
    from datasets import load_dataset

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] loading {args.repo}")
    ds = load_dataset(args.repo)
    # decode=False hands back the stored bytes, so the original JPEG is written as-is.
    ds = ds.cast_column("image", HFImage(decode=False))
    print(f"[prepare] splits: { {k: len(v) for k, v in ds.items()} }")

    train = ds["train"]
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    val_idx = order[: args.val_rows]
    train_idx = order[args.val_rows:]
    if args.max_rows:
        train_idx = train_idx[: args.max_rows]

    write_split(train, out, images, "train", train_idx)
    write_split(train, out, images, "val", val_idx)
    # Whichever held-out split the Hub repo ships becomes the calibration set: in-domain
    # and never trained on, which is what calibrate.py needs. The repo names it
    # "validation"; other mirrors use "test".
    held = next((s for s in ("test", "validation") if s in ds), None)
    if held:
        print(f"[prepare] calibration set from the '{held}' split")
        write_split(ds[held], out, images, "calib")
    else:
        print("[prepare] no held-out split found; calibrate.py will need its own rows")

    print(f"[prepare] done -> {out}")


if __name__ == "__main__":
    main()
