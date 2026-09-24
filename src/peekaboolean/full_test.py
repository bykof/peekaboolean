"""Score an adapter on every row of a test split (no sampling), for tight per-source numbers.

  python -m peekaboolean.full_test --adapter runs/v6/step-033500 --data data/general-v6 --out runs/v6/full-test-512.json
"""
import argparse
import json
import torch
from .data import DecisionDataset
from .serve import load
from .train_general import evaluate
from .baselines import QuestionPriors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-edge", type=int, default=512)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ablation", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(4)
    test = DecisionDataset(f"{args.data}/{args.split}.jsonl", augment=False, seed=31)
    priors = QuestionPriors(DecisionDataset(f"{args.data}/train.jsonl", augment=False).rows)
    model, proc, _ = load(args.adapter)
    device = next(model.head.parameters()).device
    report = evaluate(model, proc, test, device, limit=0, max_edge=args.max_edge, ablation=args.ablation,
                      priors=priors, workers=args.workers)
    with open(args.out, "w") as fh: json.dump(report, fh, indent=1)
    print(f"[full-test] {args.adapter}: macro {report['macro_nll']:.4f} selection {report['selection_nll']:.4f}", flush=True)


if __name__ == "__main__":
    main()
