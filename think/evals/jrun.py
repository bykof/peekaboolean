"""Post RealJev-style records to any Jev /v1/systemone endpoint, in parallel and resumable.

python jrun.py --records realjev/records.jsonl --endpoint http://127.0.0.1:8765/v1/systemone --out runs/NAME --workers 32

Writes OUT/predictions.jsonl ({id, source, status answered|abstained|error, value, probabilities incl. __unknown__,
latency_s}) and OUT/raw.jsonl ({id, response}). Rerunning skips ids already answered/abstained and retries errors;
when an id appears twice, the last line wins.
"""
import argparse, base64, json, math, mimetypes, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UNK = "__unknown__"


def inline(wire, root):
    imgs = []
    for p in wire.get("images", []):
        mime = mimetypes.guess_type(p)[0]
        imgs.append(f"data:{mime};base64," + base64.b64encode((root / p).read_bytes()).decode())
    return {**wire, "images": imgs}


def decode(resp, q):
    """Full distribution over the option keys + __unknown__ (imajev extension; a plain Jev answer has unknown 0)."""
    a = resp["answers"]["decision"]
    u = float(a.get("unknown_probability") or 0.0)
    if q["type"] == "noul":
        pt = float(a["noul"]) - u / 2
        probs = {"true": pt, "false": 1 - u - pt}
    else:
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        raw = a["probabilities"]
        if set(raw) != set(keys):
            raise ValueError(f"probability keys {sorted(raw)} != options {sorted(keys)}")
        probs = {k: float(raw[k]) * (1 - u) for k in keys}
    probs[UNK] = u
    if any(not math.isfinite(p) or p < -1e-6 or p > 1 + 1e-6 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-4:
        raise ValueError(f"invalid distribution {probs}")
    probs = {k: min(1.0, max(0.0, p)) for k, p in probs.items()}
    abstained = bool(a.get("abstained", False))
    value = None if abstained else max((k for k in probs if k != UNK), key=probs.get)
    return {"status": "abstained" if abstained else "answered", "value": value, "probabilities": probs}


def one(rec, root, endpoint, timeout):
    t0, raw = time.time(), None
    try:
        body = json.dumps(inline(rec["wire"], root)).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        pred = decode(json.loads(raw), rec["wire"]["questions"]["decision"])
    except Exception as e:
        if raw is None and hasattr(e, "read"):
            raw = e.read().decode(errors="replace")
        pred = {"status": "error", "value": None, "probabilities": None, "error": f"{type(e).__name__}: {e}"[:500]}
    return {"id": rec["id"], "source": rec["source"], **pred, "latency_s": round(time.time() - t0, 3)}, {"id": rec["id"], "response": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--source", nargs="*", help="only these sources")
    a = ap.parse_args()
    recs = [json.loads(l) for l in a.records.open()]
    if a.source:
        recs = [r for r in recs if r["source"] in a.source]
    recs = recs[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    pp = a.out / "predictions.jsonl"
    done = set()
    if pp.exists():
        done = {p["id"] for p in map(json.loads, pp.open()) if p["status"] != "error"}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} records, {len(done & {r['id'] for r in recs})} done, {len(todo)} to run", flush=True)
    lock, n, err = threading.Lock(), 0, 0
    with pp.open("a") as pf, (a.out / "raw.jsonl").open("a") as rf, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, r, a.records.parent, a.endpoint, a.timeout) for r in todo]):
            pred, raw = fut.result()
            with lock:
                pf.write(json.dumps(pred) + "\n"); pf.flush()
                rf.write(json.dumps(raw) + "\n"); rf.flush()
                n += 1; err += pred["status"] == "error"
                if n % 50 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)} done, {err} errors", flush=True)


if __name__ == "__main__":
    main()
