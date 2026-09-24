"""Warm end-to-end latency on CUDA, CPU or Apple MPS. No network/model-load timing."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import torch
from .model import select_device
from .serve import load, evaluate, synchronize


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", default="256,384,512")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["auto", "shared", "single", "naive"], default="auto")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto",
                    help="auto = bf16 on CUDA, fp32 elsewhere; float16 is the one to try on MPS")
    ap.add_argument("--breakdown", action="store_true", help="also report median ms per serving stage")
    ap.add_argument("--target-ms", type=float, default=500, help="p95 budget per request")
    args = ap.parse_args()
    if args.repeats < 2: ap.error("at least two repetitions required")
    if not Path(args.image).is_file(): ap.error("image path does not exist")
    torch.set_num_threads(4)
    device = select_device(args.device)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model, proc, calib = load(args.adapter, device=device, merge=True, dtype=dtype)
    reference = None
    if dtype is not None and dtype != torch.float32:
        # A faster dtype is only usable if it answers the same; keep an fp32 copy to compare.
        reference = load(args.adapter, device=device, merge=True, dtype=torch.float32)[0]
    questions = {
        "noul": {"q": {"type": "noul", "instructions": "Is there a person visible?"}},
        "choice4": {"q": {"type": "choice", "instructions": "What does the image primarily show?",
                           "criteria": {"photo": "a photograph", "screen": "an application screenshot",
                                        "document": "a document page", "chart": "a chart or diagram"}}},
        # A typical served request: several named questions of every type in one call.
        "request6": {
            "kind": {"type": "choice", "instructions": "What kind of image is this?",
                     "criteria": {"photo": "a photograph of a real scene", "screenshot": "an app or website screenshot",
                                  "document": "a scanned or photographed page", "chart": "a chart or diagram"}},
            "setting": {"type": "choice", "instructions": "Where does the scene take place?",
                        "criteria": {k: f"the setting is {k.replace('_', ' ')}" for k in
                                     ("indoors_home", "office", "street", "nature", "vehicle", "unclear")}},
            "clutter": {"type": "score", "instructions": "How cluttered is the image?",
                        "criteria": ["empty", "sparse", "moderate", "busy", "chaotic"]},
            "legibility": {"type": "score", "instructions": "How legible is any text in the image?",
                           "criteria": [f"{i} out of 10" for i in range(1, 11)]},
            "person": {"type": "noul", "instructions": "Is a person visible?"},
            "damage": {"type": "noul", "instructions": "Does anything shown look damaged?",
                       "criteria": {"true": "visible damage", "false": "no visible damage"}},
        },
    }
    report = {"platform": platform.platform(), "machine": platform.machine(), "device": device,
              "torch": torch.__version__, "model": model.model_id, "adapter": args.adapter,
              "mode": args.mode, "dtype": str(next(model.backbone.parameters()).dtype), "includes": "image file decode, resize, tokenization, inference, typed response; excludes model loading and network",
              "results": []}
    for size in map(int, args.sizes.split(",")):
        for name, qq in questions.items():
            for _ in range(3): evaluate(model, proc, "", qq, args.image, calib, args.mode, size)
            times, stages = [], {}
            for _ in range(args.repeats):
                timing = {} if args.breakdown else None
                synchronize(device); start = time.perf_counter()
                answer = evaluate(model, proc, "", qq, args.image, calib, args.mode, size, report=timing)
                json.dumps(answer); synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
                for stage, value in (timing or {}).get("ms", {}).items(): stages.setdefault(stage, []).append(value)
            result = {"size": size, "task": name, "repeats": len(times),
                      "p50_ms": statistics.median(times), "p95_ms": percentile(times, .95),
                      "meets_target_p95": percentile(times, .95) < args.target_ms, "target_ms": args.target_ms}
            if stages:
                result["stage_p50_ms"] = {k: round(statistics.median(v), 2) for k, v in stages.items()}
            if reference is not None:
                ref = evaluate(reference, proc, "", qq, args.image, calib, args.mode, size)["answers"]
                values = lambda ans: [ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                result["max_prob_diff_vs_fp32"] = round(max(
                    abs(a - b) for q in answer["answers"]
                    for a, b in zip(values(answer["answers"][q]), values(ref[q]))), 6)
            report["results"].append(result); print(json.dumps(result), flush=True)
    if args.out: Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__": main()
