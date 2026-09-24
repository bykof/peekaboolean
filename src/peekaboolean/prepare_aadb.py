#!/usr/bin/env python3
"""Convert AADB into the row schema in data.py, one row per (image, attribute).

  python -m peekaboolean.prepare_aadb --zip data/raw/AADB.zip --out data/aadb

AADB rates 11 named attributes plus an overall score for ~10k Flickr photos. Each
attribute becomes its own question, with the attribute named in `instructions`, so the
model has to condition on what is being asked instead of emitting one global quality
number. That is the whole reason this dataset is here alongside AVA.

Two scale conventions in the source, and they are not the same:
  * `score` is already [0, 1].
  * the 11 attributes are signed [-1, 1], where negative means the attribute actively
    detracts from the picture and 0 means it neither helps nor hurts. They are mapped
    to [0, 1] by (v + 1) / 2, so a neutral rating lands mid-rubric. Symmetry and
    Repetition happen never to be rated negative, but they share the signed scale and
    are mapped the same way — treating their empirical [0, 1] range as if it were the
    full scale would silently stretch them against the others.

AADB records a mean, not a vote histogram, so there is no dispersion to preserve. The
target is a spike at the rated value, split linearly between the two nearest bin
centres. Under EMD that is ordinary ordinal regression. `--smooth` widens it with a
Gaussian if you would rather not train AVA's real vote spread against AADB's certainty.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

# attribute -> (question, adjective used to phrase generated rubric levels)
# The adjective is suffixed onto adverb-style ladders ("Very {adjective}"), so it has
# to read as one. Reword freely; nothing downstream depends on the exact text.
ATTRIBUTES: dict[str, tuple[str, str]] = {
    "score": ("How aesthetic is the picture?", "aesthetic"),
    "BalacingElements": ("How well balanced are the elements in the picture?", "well balanced"),
    "ColorHarmony": ("How harmonious are the colours?", "harmonious in colour"),
    "Content": ("How interesting is the content?", "interesting in content"),
    "DoF": ("How effective is the depth of field?", "effective in depth of field"),
    "Light": ("How good is the lighting?", "well lit"),
    "MotionBlur": ("How effective is the motion blur?", "effective in its motion blur"),
    "Object": ("How clearly does the main subject stand out?", "clear in its main subject"),
    "Repetition": ("How effective is the repetition in the picture?", "effective in its repetition"),
    "RuleOfThirds": ("How well does the composition follow the rule of thirds?", "well composed on the thirds"),
    "Symmetry": ("How effective is the symmetry?", "effective in its symmetry"),
    "VividColor": ("How vivid are the colours?", "vivid in colour"),
}

# source split -> output file. The Test split becomes the calibration set: held out,
# in-domain, never trained on, which is what calibrate.py needs.
SPLITS = {
    "imgListTrain": "train",
    "imgListValidation": "val",
    "imgListTest": "calib",
    "imgListTestNew": "testnew",
}

LABEL_DIR = "AADB/imgListFiles_label"


def to_unit(attr: str, v: float) -> float:
    """Put a raw AADB value on [0, 1]. See the module docstring on the two scales."""
    return v if attr == "score" else (v + 1.0) / 2.0


def spike(v: float, bins: int, smooth: float) -> list[float]:
    """A distribution over `bins` ordered levels peaking at v in [0, 1]."""
    centres = [(i + 0.5) / bins for i in range(bins)]
    if smooth > 0:
        sigma = smooth / bins
        h = [math.exp(-0.5 * ((c - v) / sigma) ** 2) for c in centres]
    else:
        # linear split between the two nearest centres
        h = [0.0] * bins
        pos = v * bins - 0.5
        lo = max(0, min(bins - 1, math.floor(pos)))
        hi = max(0, min(bins - 1, lo + 1))
        frac = max(0.0, min(1.0, pos - lo))
        h[lo] += 1.0 - frac
        h[hi] += frac
    total = sum(h)
    return [round(x / total, 6) for x in h] if total > 0 else [1.0 / bins] * bins


def read_labels(z: zipfile.ZipFile, split: str, attr: str) -> dict[str, float]:
    name = f"{LABEL_DIR}/{split}Regression_{attr}.txt"
    try:
        txt = z.read(name).decode()
    except KeyError:
        return {}
    out = {}
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/AADB.zip")
    ap.add_argument("--out", default="data/aadb")
    ap.add_argument("--bins", type=int, default=10, help="target histogram length")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian width in bins; 0 = a hard spike at the rated value")
    ap.add_argument("--attributes", default="", help="comma-separated subset, default all")
    args = ap.parse_args()

    attrs = [a.strip() for a in args.attributes.split(",") if a.strip()] or list(ATTRIBUTES)
    unknown = [a for a in attrs if a not in ATTRIBUTES]
    if unknown:
        raise SystemExit(f"unknown attributes: {unknown}\nknown: {list(ATTRIBUTES)}")

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    # filename -> path inside the zip, preferring the original-size copies
    by_name: dict[str, str] = {}
    for n in z.namelist():
        if n.lower().endswith((".jpg", ".jpeg", ".png")) and "warp256" not in n:
            by_name[n.rsplit("/", 1)[-1]] = n

    written: set[str] = set()
    for split, out_name in SPLITS.items():
        labels = {a: read_labels(z, split, a) for a in attrs}
        files = sorted({f for a in attrs for f in labels[a]})
        if not files:
            print(f"  {out_name}: no labels, skipped")
            continue
        rows = missing = 0
        with (out / f"{out_name}.jsonl").open("w", encoding="utf-8") as fh:
            for fname in files:
                src = by_name.get(fname)
                if src is None:
                    missing += 1
                    continue
                if fname not in written:
                    (images / fname).write_bytes(z.read(src))
                    written.add(fname)
                for a in attrs:
                    if fname not in labels[a]:
                        continue
                    question, adjective = ATTRIBUTES[a]
                    fh.write(json.dumps({
                        "image": fname,
                        "type": "score",
                        "state": "",
                        "attribute": adjective,
                        "instructions": question,
                        "hist": spike(to_unit(a, labels[a][fname]), args.bins, args.smooth),
                    }) + "\n")
                    rows += 1
        note = f"  ({missing} images missing from the archive)" if missing else ""
        print(f"  {out_name}.jsonl  {rows} rows over {len(files)} images{note}")

    print(f"[prepare] {len(written)} images -> {out}")


if __name__ == "__main__":
    main()
