#!/usr/bin/env python3
"""Does the model give the same answer when the rubric changes shape?

  python -m peekaboolean.robustness --adapter runs/v1/final --val data/ava/val.jsonl \
      --image-root data/ava/images --model Qwen/Qwen3-VL-8B-Instruct --levels 3,5,8

Nothing in the training loop measures this. A served model is asked to score against
whatever rubric arrives in the request, so scoring one image on 3 levels and on 8
should imply the same underlying judgement. If the implied means disagree, the model
is reading the rubric's shape rather than the picture, and no aggregate accuracy
number on a fixed level count would reveal it.

For each score row and each level count K, the rubric is generated K levels wide, the
candidates are scored, and the predicted distribution is reduced to one number: the
implied mean, sum_i p_i * (i + 0.5) / K, on [0, 1] regardless of K. Those are directly
comparable across K, and against the same reduction of the ground-truth histogram.

Rubric wording is sampled, so `--repeats` averages several samplings per (row, K) to
keep wording noise out of the level-count comparison.

The check ends in a verdict and an exit code, so it can gate a release. The bar is the
one the paragraph above states: the spread a rubric's shape induces has to be small
against the model's own error, not merely small in the abstract.

`--consistency N` measures three invariants the primitives ought to obey and that no
loss term enforces:

  * a two-level score and the noul asking the same thing should agree -- in this
    architecture they are the same computation, so a gap is the primitive talking;
  * swapping which of "yes" and "no" comes first must not move the answer. This one
    holds by construction rather than by training -- candidates are scored in separate
    sequences, so there is no position for a prior to attach to -- and it reads 0.0000
    exactly. It stays because that isolation is what the whole design rests on and what
    the API sells, and it is the first thing a single-pass forward over shared
    candidates would quietly break;
  * asking the negated question should return one minus the answer. TypeSafe's own
    documentation lists this as not guaranteed by the request format; here it is at least measurable,
    and a consistency term in training could enforce it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from .data import Example, build_inputs, rebin, sample_levels
from .model import CandidateScorer


def implied_mean(p: torch.Tensor) -> float:
    """Reduce a distribution over K ordered levels to one number on [0, 1]."""
    k = p.numel()
    centres = torch.tensor([(i + 0.5) / k for i in range(k)], dtype=p.dtype)
    return float((p * centres).sum())


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


@torch.no_grad()
def consistency(model, processor, rows, root: Path, max_edge: int, seed: int) -> dict:
    """Ask the same judgement three ways round and see whether the answers line up."""
    def p_yes(image, instructions, true_text, false_text, swap=False):
        cands = [true_text, false_text] if swap else [false_text, true_text]
        ex = Example(image=image, state="", instructions=instructions, candidates=cands,
                     target=torch.tensor([0.5, 0.5]), qtype="noul",
                     names=["true", "false"] if swap else ["false", "true"])
        p = torch.softmax(model(build_inputs(ex, processor, max_edge)
                                .to(next(model.head.parameters()).device)).float(), dim=-1).cpu()
        return float(p[0] if swap else p[1])     # P of the "it is X" candidate either way

    gaps: dict[str, list[float]] = {"score2_vs_noul": [], "position": [], "negation": []}
    pairs: list[tuple[float, float]] = []
    for n, row in enumerate(rows):
        image = root / row["image"]
        attr = row.get("attribute", "good")
        rng = random.Random(f"{seed}:consistency:{n}")
        levels = sample_levels(2, attr, rng)
        two = Example(image=image, state="", instructions=row["instructions"],
                      candidates=levels, target=rebin(row["hist"], 2), qtype="score",
                      names=["0", "1"])
        p_high = float(torch.softmax(
            model(build_inputs(two, processor, max_edge)
                  .to(next(model.head.parameters()).device)).float(), dim=-1).cpu()[1])

        yes = f"the picture is {attr}"
        no = f"the picture is not {attr}"
        p = p_yes(image, f"Is the picture {attr}?", yes, no)
        p_swapped = p_yes(image, f"Is the picture {attr}?", yes, no, swap=True)
        p_negated = p_yes(image, f"Is the picture not {attr}?", no, yes)

        gaps["score2_vs_noul"].append(abs(p_high - p))
        gaps["position"].append(abs(p - p_swapped))
        gaps["negation"].append(abs(p_negated - (1.0 - p)))
        pairs.append((p_high, p))

    out = {k: {"mean_abs_gap": sum(v) / len(v), "max_abs_gap": max(v)} for k, v in gaps.items()}
    out["score2_vs_noul"]["corr"] = pearson([a for a, _ in pairs], [b for _, b in pairs])
    print("\ncross-primitive invariants (nothing in the loss enforces these)")
    print(f"  two-level score vs the same noul   mean |gap| {out['score2_vs_noul']['mean_abs_gap']:.4f}"
          f"   corr {out['score2_vs_noul']['corr']:.4f}")
    print(f"  yes/no order swapped               mean |gap| {out['position']['mean_abs_gap']:.4f}"
          "   (structural: expect exactly 0)")
    print(f"  negated question vs 1 - answer     mean |gap| {out['negation']['mean_abs_gap']:.4f}"
          "   (an invariant the request format does not promise)")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--levels", default="3,5,8")
    ap.add_argument("--limit", type=int, default=200, help="score rows to evaluate")
    ap.add_argument("--repeats", type=int, default=3, help="rubric samplings per row and K")
    ap.add_argument("--max-edge", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="defaults to <adapter>/robustness.json")
    ap.add_argument("--gate-ratio", type=float, default=0.5,
                    help="pass if mean rubric spread stays under this fraction of the MAE")
    ap.add_argument("--consistency", type=int, default=0,
                    help="rows to run the cross-primitive invariants on (0 skips)")
    args = ap.parse_args()

    from peft import PeftModel

    ks = [int(x) for x in args.levels.split(",") if x.strip()]
    root = Path(args.image_root)

    rows = [json.loads(l) for l in Path(args.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("type") == "score"][: args.limit]
    if not rows:
        raise SystemExit("no score rows found; this check only applies to score questions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CandidateScorer(args.model, gradient_checkpointing=False)
    model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, args.adapter)
    model.head.load_state_dict(torch.load(f"{args.adapter}/head.pt", map_location="cpu"))
    model = model.to(device).eval()
    processor = CandidateScorer.load_processor(args.model)

    print(f"[robustness] {len(rows)} rows x levels {ks} x {args.repeats} samplings")
    # per K: one averaged implied mean per row, aligned by row index
    pred: dict[int, list[float]] = {k: [] for k in ks}
    truth: dict[int, list[float]] = {k: [] for k in ks}

    for n, row in enumerate(rows):
        for k in ks:
            got = []
            for rep in range(args.repeats):
                rng = random.Random(f"{args.seed}:{n}:{k}:{rep}")
                levels = sample_levels(k, row.get("attribute", ""), rng)
                ex = Example(
                    image=root / row["image"],
                    state=row.get("state", ""),
                    instructions=row["instructions"],
                    candidates=levels,
                    target=rebin(row["hist"], k),
                    qtype="score",
                )
                logits = model(build_inputs(ex, processor, args.max_edge).to(device))
                got.append(implied_mean(torch.softmax(logits, dim=-1).float().cpu()))
            pred[k].append(sum(got) / len(got))
            truth[k].append(implied_mean(rebin(row["hist"], k)))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)

    print("\nper level count (implied mean on [0,1], vs the same reduction of the histogram)")
    print("  K    pred    truth     MAE     corr")
    report = {"levels": {}, "pairs": {}, "rows": len(rows), "repeats": args.repeats}
    for k in ks:
        p, t = pred[k], truth[k]
        mae = sum(abs(a - b) for a, b in zip(p, t)) / len(p)
        r = pearson(p, t)
        print(f"  {k:<3} {sum(p)/len(p):7.4f} {sum(t)/len(t):7.4f} {mae:7.4f} {r:8.4f}")
        report["levels"][k] = {"pred_mean": sum(p) / len(p), "truth_mean": sum(t) / len(t),
                               "mae": mae, "corr": r}

    print("\ncross-rubric agreement (the number this check exists for)")
    print("  K vs K    mean|diff|   max|diff|     corr")
    for a, b in itertools.combinations(ks, 2):
        diffs = [abs(x - y) for x, y in zip(pred[a], pred[b])]
        r = pearson(pred[a], pred[b])
        print(f"  {a} vs {b}   {sum(diffs)/len(diffs):9.4f} {max(diffs):11.4f} {r:9.4f}")
        report["pairs"][f"{a}v{b}"] = {"mean_abs_diff": sum(diffs) / len(diffs),
                                       "max_abs_diff": max(diffs), "corr": r}

    spread = [max(pred[k][i] for k in ks) - min(pred[k][i] for k in ks) for i in range(len(rows))]
    report["mean_spread"] = sum(spread) / len(spread)
    report["max_spread"] = max(spread)
    print(f"\nper-row spread across all level counts: mean {report['mean_spread']:.4f}, "
          f"worst {report['max_spread']:.4f}")
    print("A model reading the picture rather than the rubric keeps this small. Judge it "
          "against the MAE above:\nspread comparable to MAE means rubric shape is as large "
          "a source of error as the model's own inaccuracy.")

    mean_mae = sum(report["levels"][k]["mae"] for k in ks) / len(ks)
    ratio = report["mean_spread"] / mean_mae if mean_mae else float("inf")
    passed = ratio < args.gate_ratio
    report["gate"] = {"spread_over_mae": ratio, "threshold": args.gate_ratio, "passed": passed}
    print(f"\nspread / MAE = {ratio:.2f} against a bar of {args.gate_ratio}: "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("The rubric's shape moves the answer about as much as the model's own error.\n"
              "Whatever the aggregate metrics say, this model cannot be sold on taking any\n"
              "rubric the caller sends.")

    if args.consistency:
        report["consistency"] = consistency(model, processor, rows[: args.consistency],
                                            root, args.max_edge, args.seed)

    dest = Path(args.out) if args.out else Path(args.adapter) / "robustness.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {dest}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
