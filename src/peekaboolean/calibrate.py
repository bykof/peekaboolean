#!/usr/bin/env python3
"""Fit the calibration temperatures, and report whether they worked.

  python -m peekaboolean.calibrate --adapter runs/v1/final --val data/calib.jsonl \
      --image-root data --model Qwen/Qwen3-VL-8B-Instruct

Run this on data from the SAME distribution you will serve. A temperature fitted on
synthetic or out-of-domain rows produces confidence numbers that are precisely
calibrated for data you do not have. Since confidence is the product, that failure is
worse than shipping no confidence at all.

One temperature is not enough, for two reasons.

The first is arithmetic: the softmax runs over K candidates, and K is whatever the
request asked for. The scale that makes a 2-level rubric honest is not the scale that
makes a 10-level one honest, so a single number is fitted to a mixture and is right for
none of it. Temperatures are therefore fitted per (question type, K) and shrunk toward
the global fit, so a bucket with forty rows borrows from the rest instead of chasing its
own noise.

The second is what "calibrated" should mean here. Top-1 accuracy is the familiar check
and it is the wrong one for a score: the API publishes the whole distribution, and a
model that puts 0.45 on the right level and 0.45 on its neighbour is doing well, not
failing. So the reliability table below is over every (row, outcome) pair -- when the
model says a tenth of the vote lands on a level, does a tenth of it land there? -- which
is the honest statement of what `probabilities` promises, and reads the same way for all
three types. Top-1 ECE is still reported for choice, where hard labels make it mean
something.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel

import random

from .data import DecisionDataset, Example, build_inputs, rebin, sample_levels
from .model import CandidateScorer


def reliability(pred: torch.Tensor, actual: torch.Tensor, bins: int = 10):
    """Bin by predicted value, compare to what actually happened in that bin.

    Works for a probability against a hard hit, and equally for a probability against
    the share of the vote that landed there -- the question is the same either way.
    """
    edges = torch.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = ((pred >= edges[i]) if i == 0 else (pred > edges[i])) & (pred <= edges[i + 1])
        if m.sum() == 0:
            continue
        stated, observed = float(pred[m].mean()), float(actual[m].mean())
        rows.append((float(edges[i]), float(edges[i + 1]), int(m.sum()), stated, observed))
        ece += float(m.float().mean()) * abs(stated - observed)
    return ece, rows


def table(rows) -> str:
    out = ["    bin          n   stated   actual"]
    for lo, hi, n, stated, observed in rows:
        out.append(f"    {lo:.1f}-{hi:.1f} {n:8d}   {stated:.3f}    {observed:.3f}")
    return "\n".join(out)


def variants(ds, i: int, levels: list[int]) -> list[Example]:
    """A score row evaluated at each level count we intend to serve.

    Without this the whole calibration set arrives at one K -- `augment=False` fixes it
    at five -- and per-K temperatures would have exactly one populated bucket, which is
    a single temperature wearing a hat. A row whose rubric came with the request keeps
    the K it was sent with; that one is not ours to vary.
    """
    ex = ds[i]
    row = ds.rows[i]
    if ex.qtype != "score" or not levels or row.get("criteria") or row.get("candidates"):
        return [ex]
    out = []
    for k in levels:
        rng = random.Random(f"calib:{i}:{k}")
        out.append(Example(image=ex.image, state=ex.state, instructions=ex.instructions,
                           candidates=sample_levels(k, row.get("attribute", ""), rng),
                           target=rebin(row["hist"], k), qtype="score",
                           names=[str(j) for j in range(k)]))
    return out


@torch.no_grad()
def collect(model, processor, ds, device, levels: list[int]):
    """Cache raw logits once so the temperature search is instant."""
    rows = []
    for i in range(len(ds)):
        for ex in variants(ds, i, levels):
            logits = model(build_inputs(ex, processor).to(device)).float().cpu()
            rows.append((logits, ex.target, ex.qtype))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} source rows, {len(rows)} questions", flush=True)
    return rows


def nll_at(rows, t: float) -> float:
    """Mean cross-entropy against the target distribution: a proper scoring rule for
    every type here, including the soft targets a vote histogram produces."""
    total = 0.0
    for logits, target, _ in rows:
        total += -(target * torch.log_softmax(logits / t, dim=-1)).sum().item()
    return total / max(1, len(rows))


GRID_MAX = 10.0   # top of the coarse sweep; the refinement can reach 1.4x it


def fit_temperature(rows) -> float:
    if not rows:
        return 1.0
    # Coarse sweep then refine. The objective is convex in log T, so this is plenty.
    grid = torch.logspace(-1, 1, 41).tolist()
    best = min(grid, key=lambda t: nll_at(rows, t))
    fine = torch.linspace(best * 0.7, best * 1.4, 29).tolist()
    return min(fine, key=lambda t: nll_at(rows, t))


def saturated(t: float) -> bool:
    """The search wanted a hotter temperature than the grid offers.

    Temperature scaling cannot fix a bucket like that. It means the model is confident
    in a shape it was never trained on -- the usual cause is a K the training
    augmentation never sampled -- and flattening the distribution afterwards only hides
    it. Retrain covering that K instead of trusting the number.
    """
    return t >= GRID_MAX


def bucket_of(logits, qtype) -> str:
    return f"{qtype}:{logits.numel()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="0 uses every row")
    ap.add_argument("--levels", default="2,3,5,8,10",
                    help="level counts each score row is calibrated at; empty keeps one")
    ap.add_argument("--shrink", type=float, default=200.0,
                    help="rows a bucket needs before it outweighs the global fit")
    ap.add_argument("--out", default=None, help="defaults to <adapter>/calibration.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CandidateScorer(args.model, gradient_checkpointing=False)
    model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, args.adapter)
    model.head.load_state_dict(torch.load(f"{args.adapter}/head.pt", map_location="cpu"))
    model = model.to(device).eval()
    processor = CandidateScorer.load_processor(args.model)

    ds = DecisionDataset(args.val, args.image_root, augment=False)
    if args.limit:
        ds.rows = ds.rows[: args.limit]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    print(f"[calibrate] {len(ds)} rows, score rows at levels {levels or '[as written]'}")
    rows = collect(model, processor, ds, device, levels)

    glob = fit_temperature(rows)
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[bucket_of(r[0], r[2])].append(r)

    # Shrink each bucket toward the global fit in log space: n rows of evidence against
    # `--shrink` rows' worth of prior. A bucket seen twice does not get its own scale.
    import math
    temps = {}
    for name, part in sorted(buckets.items()):
        raw = fit_temperature(part)
        w = len(part) / (len(part) + args.shrink)
        temps[name] = math.exp(w * math.log(raw) + (1 - w) * math.log(glob))

    print(f"\nglobal temperature {glob:.3f}   NLL {nll_at(rows, 1.0):.4f} -> {nll_at(rows, glob):.4f}")
    print(f"per-bucket NLL {sum(nll_at(p, temps[b]) * len(p) for b, p in buckets.items()) / len(rows):.4f}"
          "   (the number the served model will actually achieve)")
    print("\n  bucket          n   raw T   used T")
    warned = []
    for name, part in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        raw = fit_temperature(part)
        note = ""
        if saturated(raw):
            note = "   <- hit the search ceiling"
            warned.append(name)
        print(f"  {name:<12} {len(part):6d}  {raw:6.3f}   {temps[name]:6.3f}{note}")
    if warned:
        print(f"\n  {', '.join(warned)} wanted a temperature past the grid. Temperature\n"
              "  scaling does not rescue a bucket like that -- the model is confident about a\n"
              "  shape it never trained on, usually a K the augmentation never sampled, and\n"
              "  flattening it afterwards only hides that. Cover the K in training instead.")

    # ---- reliability, per type, over every (row, outcome) pair ----
    report = {"temperature": glob, "temperatures": temps, "shrink": args.shrink,
              "saturated": [b for b, part in buckets.items() if saturated(fit_temperature(part))],
              "n_calibration_rows": len(rows), "nll_before": nll_at(rows, 1.0),
              "nll_global": nll_at(rows, glob), "types": {}}
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r[2]].append(r)

    for qtype, part in sorted(by_type.items()):
        pred, actual = [], []
        top_conf, top_hit = [], []
        for logits, target, _ in part:
            p = torch.softmax(logits / temps[bucket_of(logits, qtype)], dim=-1)
            pred.append(p)
            actual.append(target)
            top_conf.append(p.max())
            top_hit.append(torch.tensor(float(p.argmax() == target.argmax())))
        ece, rel = reliability(torch.cat(pred), torch.cat(actual))
        entry = {"rows": len(part), "probability_ece": ece}
        print(f"\n{qtype}: {len(part)} rows, probability ECE {ece:.4f}")
        print(table(rel))
        if qtype == "choice":
            top_ece, _ = reliability(torch.stack(top_conf), torch.stack(top_hit))
            acc = float(torch.stack(top_hit).mean())
            entry.update(top1_ece=top_ece, accuracy=acc)
            print(f"    top-1 accuracy {acc:.4f}, top-1 ECE {top_ece:.4f}")
        report["types"][qtype] = entry

    out = Path(args.out or f"{args.adapter}/calibration.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")
    print("Read the table, not the single number: stated 0.9 against actual 0.6 in a\n"
          "well-populated bin is a model no temperature can rescue, and needs either\n"
          "more calibration data or a bucket of its own.")


if __name__ == "__main__":
    main()
