"""imajev_bench's `run` for the imajev-http adapter with parallel requests; same artifacts, so `imajev_bench score` accepts them.

python prun.py --records R.jsonl --root DIR --split dev --output OUT --endpoint URL --workers 16
"""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from imajev_bench import runner
from imajev_bench.cli import read_jsonl
from imajev_bench.schema import model_payload


def one(record, root, endpoint, timeout):
    payload = model_payload(record)
    start, raw, wire_hash = time.perf_counter(), None, None
    try:
        wire = runner.jev_payload(payload, root)
        wire_hash = runner.digest(wire)
        req = urllib.request.Request(endpoint, data=runner.canonical_bytes(wire),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        result = runner.decode_jev(json.loads(raw), payload)
    except Exception as exc:
        result = {"status": "error", "value": None, "error_type": type(exc).__name__, "error": str(exc)}
    result.update(id=record["id"], latency_ms=(time.perf_counter() - start) * 1000)
    return result, {"id": record["id"], "payload_sha256": runner.digest(payload), "wire_sha256": wire_hash,
                    "response": raw.decode("utf-8", errors="replace") if raw else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = a.root or a.records.parent.parent
    records = [r for r in read_jsonl(a.records) if r["split"] in a.split][:a.limit]
    a.output.mkdir(parents=True, exist_ok=False)
    config = {"format_version": "0.0.1", "adapter": "imajev-http", "endpoint": a.endpoint, "timeout_seconds": a.timeout,
              "seed": 0, **runner.run_hashes(records), "record_count": len(records),
              "reviewed": all(x["annotation_status"] == "reviewed" for x in records),
              "started_at": datetime.now(timezone.utc).isoformat(), "concurrency": a.workers,
              "cost_basis": "not_measured", "retries": 0, "splits": a.split,
              "decision_rule": "server abstention flag; substantive argmax; ordinal indices restored"}
    (a.output / "manifest.json").write_bytes(runner.canonical_bytes(config) + b"\n")
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda r: one(r, root, a.endpoint, a.timeout), records))
    with (a.output / "predictions.jsonl").open("x") as p, (a.output / "raw.jsonl").open("x") as w:
        for result, raw in rows:
            p.write(json.dumps(result, allow_nan=False) + "\n")
            w.write(json.dumps(raw) + "\n")
    completion = {"status": "complete", "completed_count": len(records),
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": runner.file_digest(a.output / "manifest.json"),
                  "predictions_sha256": runner.file_digest(a.output / "predictions.jsonl"),
                  "raw_sha256": runner.file_digest(a.output / "raw.jsonl")}
    (a.output / "completion.json").write_bytes(runner.canonical_bytes(completion) + b"\n")
    ok = sum(r["status"] != "error" for r, _ in rows)
    print(f"{ok}/{len(rows)} answered without error -> {a.output}")


if __name__ == "__main__":
    main()
