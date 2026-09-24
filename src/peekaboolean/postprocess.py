"""Calibrate the selected checkpoint, then evaluate on a separate test split."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from .data import DecisionDataset, build_inputs
from .model import configure_image_size, select_device
from .serve import load
from .calibrate import fit_temperature, reliability, variants
from .train_general import eval_indices, evaluate
from .baselines import QuestionPriors


@torch.no_grad()
def collect(model, proc, ds, device, size, limit, sweep=False):
    result = []
    for i in eval_indices(ds, limit):
        for ex in variants(ds, i, [2, 3, 5, 8, 10]) if sweep else [ds[i]]:
            logits = model(build_inputs(ex, proc, size).to(device)).float().cpu()
            result.append((logits, ex.target, ex.qtype))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--sizes", default="256,384,512")
    args = ap.parse_args()
    torch.set_num_threads(4)
    run, data = Path(args.run), Path(args.data)
    adapter = run / (run / "best.txt").read_text().strip()
    device = select_device()
    model, proc, _ = load(str(adapter), device=device)
    calibration = DecisionDataset(data / "calib.jsonl", augment=False, seed=31)
    test = DecisionDataset(data / "test.jsonl", augment=False, seed=31)
    priors = QuestionPriors(DecisionDataset(data / "train.jsonl", augment=False).rows)
    reports = {}
    for size in map(int, args.sizes.split(",")):
        configure_image_size(proc, size)
        rows = collect(model, proc, calibration, device, size, args.limit, sweep=True)
        global_t = fit_temperature(rows)
        buckets = defaultdict(list)
        for row in rows: buckets[f"{row[2]}:{len(row[0])}"].append(row)
        temperatures = {key: fit_temperature(part) if len(part) >= 50 else global_t for key, part in buckets.items()}
        result = {"temperature": global_t, "temperatures": temperatures, "image_size": size,
                  "n_calibration_rows": len(rows), "dataset": str(data), "adapter": str(adapter)}
        (adapter / f"calibration-{size}.json").write_text(json.dumps(result, indent=2))
        held = collect(model, proc, test, device, size, args.limit)
        ps, ts = [], []
        for logits, target, kind in held:
            p = (logits / temperatures.get(f"{kind}:{len(logits)}", global_t)).softmax(-1)
            ps.append(p); ts.append(target)
        ece, table = reliability(torch.cat(ps), torch.cat(ts))
        report = evaluate(model, proc, test, device, args.limit, size, ablation=120, priors=priors)
        report["calibrated_test"] = {"probability_ece": ece, "reliability": table,
            "nll": sum(float(-(t * p.clamp_min(1e-9).log()).sum()) for p, t in zip(ps, ts)) / len(ps)}
        reports[size] = report
        print(f"[postprocess] size={size} test={json.dumps(report)}", flush=True)
    (run / "test-report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__": main()
