"""Detached supervisor: validated data -> train -> calibrate/test -> benchmark."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/general-v4")
    ap.add_argument("--run", default="runs/v4")
    ap.add_argument("--prepare-run", help="data preparation job to wait for (default: <run>-prepare)")
    ap.add_argument("--eval-limit", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", default="5")
    ap.add_argument("--max-hours", default="72")
    ap.add_argument("--eval-every", default="250")
    ap.add_argument("--patience", default="8")
    ap.add_argument("--workers", default="8")
    ap.add_argument("--micro", default="1")
    ap.add_argument("--max-edge", default="384", help="validation/selection image size")
    ap.add_argument("--image-sizes", default="256,384,512", help="training sizes; repeat one to weight it")
    ap.add_argument("--ablation", default="60")
    ap.add_argument("--eval-teacher-weight", default="1")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--head", default="mlp")
    ap.add_argument("--init-adapter")
    ap.add_argument("--lr")
    args = ap.parse_args()
    data, run = Path(args.data), Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    stopped = {"value": False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(value=True))
    def stage(name, command):
        if stopped["value"]: raise SystemExit("pipeline stopped")
        (run / "stage.json").write_text(json.dumps({"stage": name, "started": time.time(), "command": command}))
        print(f"[pipeline] {name}: {command}", flush=True)
        code = subprocess.call(command)
        if code: raise SystemExit(f"{name} failed with exit code {code}")
        if stopped["value"]: raise SystemExit("pipeline stopped")
    while True:
        prep = Path(args.prepare_run or f"{run}-prepare")
        info = json.loads((prep / "process.json").read_text()) if (prep / "process.json").exists() else None
        status = json.loads((prep / "exit.json").read_text()) if (prep / "exit.json").exists() else None
        finished = status and (not info or status["finished"] >= info["started"])
        if finished and status["exit_code"] != 0: raise SystemExit("data preparation failed; inspect its job.log")
        if (data / "manifest.json").exists() and (info is None or finished): break
        if stopped["value"]: return
        print("[pipeline] waiting for data preparation", flush=True)
        time.sleep(30)
    command = [sys.executable, "-u", "-m", "peekaboolean.train_general", "--train", str(data / "train.jsonl"),
               "--val", str(data / "val.jsonl"), "--out", str(run), "--epochs", args.epochs,
               "--max-hours", args.max_hours, "--eval-every", args.eval_every, "--log-every", "25",
               "--eval-limit", str(args.eval_limit), "--patience", args.patience,
               "--workers", args.workers, "--micro", args.micro, "--max-edge", args.max_edge,
               "--image-sizes", args.image_sizes, "--ablation", args.ablation,
               "--eval-teacher-weight", args.eval_teacher_weight, "--model", args.model,
               "--head", args.head]
    if args.init_adapter: command += ["--init-adapter", args.init_adapter]
    if args.lr: command += ["--lr", args.lr]
    if args.resume: command += ["--resume", "latest"]
    stage("training", command)
    if not (run / "complete.json").exists():
        print("[pipeline] training paused; resume required, no final evaluation yet", flush=True); return
    stage("calibration-and-test", [sys.executable, "-u", "-m", "peekaboolean.postprocess", "--run", str(run), "--data", str(data)])
    adapter = run / (run / "best.txt").read_text().strip()
    image = json.loads((data / "test.jsonl").read_text().splitlines()[0])["image"]
    stage("host-benchmark", [sys.executable, "-u", "-m", "peekaboolean.benchmark", "--adapter", str(adapter),
          "--image", image, "--out", str(run / "host-latency.json")])
    (run / "stage.json").write_text(json.dumps({"stage": "complete", "adapter": str(adapter)}))
    print(f"[pipeline] complete: {adapter}. Mac latency still requires testing on the Mac.", flush=True)


if __name__ == "__main__": main()
