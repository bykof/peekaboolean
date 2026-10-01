"""RFT sampler: n thinking traces per item through jevsrv's exact prompt and message layout; resumable.

python rft.py --items 'items/*.jsonl' --out rft/samples.jsonl --n 2 [--n-source '{"docvqa": 4}'] [--ids rft/pilot_ids.txt]
Each output line is one request: {item_id, source, gold, gold_letter, prompt, prompt_tokens, samples: [...]}.
A sample is {text, letter, correct, probs, n_tokens, finish}: letter is the sampled letter after the last "Answer:"
(the token jevsrv.letter_probs reads), probs that position's letter distribution.
"""
import argparse, base64, glob, json, mimetypes, os, string, sys, threading, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/big")
import jevsrv  # noqa: E402


class Capture(jevsrv.Model):
    """jevsrv.Model whose chat records the content it would send, so prompts are built by jevsrv itself."""
    def chat(self, content):
        self.content = content
        return {"choices": []}


def data_url(p):
    return f"data:{mimetypes.guess_type(p)[0]};base64," + base64.b64encode(open(p, "rb").read()).decode()


def build(item, urls=True):
    m = Capture(None)
    labels = m.decide(item["state"], item["question"], [data_url(p) if urls else p for p in item["images"]])[0]
    return m.content, labels


def sampled_letter(choice):
    """The letter token jevsrv.letter_probs reads: first uppercase token right after the last 'Answer:'."""
    toks = (choice.get("logprobs") or {}).get("content") or []
    text, hit = "", None
    for t in toks:
        if jevsrv.ANSWER.search(text) and t["token"].strip()[:1] in string.ascii_uppercase:
            hit = t["token"].strip()[:1]  # "" when jevsrv's check passes on a whitespace token
        text += t["token"]
    return hit or None


def run_one(item, n, a):
    content, labels = build(item)
    gl = string.ascii_uppercase[labels.index(item["gold"])]
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens,
            "temperature": a.temperature, "top_p": 0.95, "n": n, "logprobs": True, "top_logprobs": a.top_logprobs}
    for attempt in range(6):
        try:
            req = urllib.request.Request(a.vllm + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=a.timeout) as r:  # socket timeout: also catches dead connections
                out = json.loads(r.read())
            break
        except Exception as e:  # server restart or overload: back off (up to ~7 min in total) and retry
            err = f"{type(e).__name__}: {e}"
            time.sleep(30 * (attempt + 1))
    else:
        return {"item_id": item["id"], "source": item["source"], "error": err}
    samples = []
    for c in out["choices"]:
        letter = sampled_letter(c)
        p = jevsrv.letter_probs(c, len(labels))
        samples.append({"text": c["message"]["content"], "letter": letter, "correct": letter == gl,
                        "probs": [round(x, 5) for x in p] if p else None,
                        "n_tokens": len((c.get("logprobs") or {}).get("content") or []), "finish": c["finish_reason"]})
    return {"item_id": item["id"], "source": item["source"], "gold": item["gold"], "gold_letter": gl,
            "prompt": content[-1]["text"], "prompt_tokens": out["usage"]["prompt_tokens"], "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="/root/big/data/items/*.jsonl")
    ap.add_argument("--out", default="/root/big/data/rft/samples.jsonl")
    ap.add_argument("--ids", help="file with item ids to run (default: all)")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--n-source", default="{}", help='JSON {source: n} overrides')
    ap.add_argument("--concurrency", type=int, default=32, help="requests in flight; keep requests*n modest")
    ap.add_argument("--vllm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="q36")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--timeout", type=float, default=1200, help="seconds without a response before a retry")
    ap.add_argument("--top-logprobs", type=int, default=5,
                    help="jevsrv uses 20; building them dominates the vLLM API server's CPU, and the letter mass beyond 5 is negligible")
    a = ap.parse_args()
    items = [json.loads(l) for f in sorted(glob.glob(a.items)) for l in open(f)]
    if a.ids:  # run in the ids file's order
        order = {i: k for k, i in enumerate(open(a.ids).read().split())}
        items = sorted((it for it in items if it["id"] in order), key=lambda it: order[it["id"]])
    nsrc = json.loads(a.n_source)
    have = Counter()
    for f in glob.glob(os.path.join(os.path.dirname(a.out), "*.jsonl")):  # resume across every sample file in the out dir
        for l in open(f):
            r = json.loads(l)
            if "samples" in r:
                have[r["item_id"]] += len(r["samples"])
    todo = [(it, nsrc.get(it["source"], a.n) - have[it["id"]]) for it in items]
    todo = [(it, k) for it, k in todo if k > 0]
    print(f"{len(items)} items, {len(todo)} to sample, {sum(k for _, k in todo)} samples", flush=True)
    stop = os.path.join(os.path.dirname(a.out), "STOP")
    lock, done, t0 = threading.Lock(), Counter(), time.time()
    with open(a.out, "a") as fh, ThreadPoolExecutor(a.concurrency) as ex:
        def job(x):
            if os.path.exists(stop):  # stop launching new items; in-flight requests finish
                return
            try:
                r = run_one(*x, a)
            except Exception as e:  # never let one bad response stop the run; resume retries it
                r = {"item_id": x[0]["id"], "source": x[0]["source"], "error": f"{type(e).__name__}: {e}"}
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done["req"] += 1
                for s in r.get("samples", []):
                    done["s"] += 1
                    done["ok"] += s["correct"]
                    done["tok"] += s["n_tokens"]
                done["err"] += "error" in r
                if done["req"] % 50 == 0:
                    el = time.time() - t0
                    print(f"{done['req']}/{len(todo)} req  acc {done['ok'] / max(1, done['s']):.3f}  "
                          f"tok/sample {done['tok'] / max(1, done['s']):.0f}  {done['tok'] / el:.0f} tok/s  err {done['err']}  "
                          f"{el / 60:.1f} min", flush=True)
        list(ex.map(job, todo))
    print("done", dict(done), f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
