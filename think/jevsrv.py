"""Jev /v1/systemone over a vLLM OpenAI server: the model reasons, then the option letter's logprobs are the answer.

python jevsrv.py --vllm http://127.0.0.1:8000/v1 --model q36 --port 8765 --samples 4
"""
import argparse, json, math, re, string, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

UNKNOWN = "Unknown: the evidence does not determine the answer, the question's premise is false, or none of the other options is correct."
PROMPT = """You are a decision model. Decide the question below from the evidence only: the image(s) shown, if any, and the state.

State:
{state}

Question: {question}

Options:
{options}

Pick exactly one option. Choose {unk} only when the evidence does not settle the answer (the needed fact is not visible, unreadable, cut off or missing, or the rule cannot be decided from what is given), when the question assumes something the evidence contradicts, or when none of the other options is correct. Otherwise give the answer the evidence supports.
Reason step by step, then end with a final line of the form "Answer: <letter>"."""
ANSWER = re.compile(r"Answer\s*:\s*\**\s*$")


def options_for(q):
    """Return (labels, display lines); the last option is always Unknown."""
    t, crit = q["type"], q.get("criteria")
    if t == "noul":
        crit = crit or {}
        labels = ["true", "false"]
        shown = ["Yes" + (f": {crit['true']}" if crit.get("true") else ""),
                 "No" + (f": {crit['false']}" if crit.get("false") else "")]
    elif t == "choice":
        labels = list(crit)
        shown = [k + (f": {v}" if v else "") for k, v in crit.items()]
    elif t == "score":
        labels = [str(i) for i in range(len(crit))]
        shown = [f"Level {i}: {d}" for i, d in enumerate(crit)]
    else:
        raise ValueError(f"unsupported question type {t}")
    return labels + ["__unknown__"], shown + [UNKNOWN]


def letter_probs(choice, n):
    """Distribution over the n option letters at the token after the last 'Answer:'; None if absent."""
    toks = (choice.get("logprobs") or {}).get("content") or []
    text, hit = "", None
    for i, t in enumerate(toks):
        if ANSWER.search(text) and t["token"].strip()[:1] in string.ascii_uppercase:
            hit = i
        text += t["token"]
    if hit is None:
        return None
    letters = string.ascii_uppercase[:n]
    mass = [0.0] * n
    for alt in toks[hit]["top_logprobs"]:
        s = alt["token"].strip()
        if s and s[0] in letters and (len(s) == 1 or not s[1].isalnum()):
            mass[letters.index(s[0])] += math.exp(alt["logprob"])
    z = sum(mass)
    return [m / z for m in mass] if z > 0 else None


class Model:
    def __init__(self, args):
        self.a = args
        self.cal = json.load(open(args.calibration)) if getattr(args, "calibration", None) else None

    def chat(self, content):
        body = {"model": self.a.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.a.max_tokens, "temperature": self.a.temperature, "top_p": 0.95,
                "n": self.a.samples, "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(self.a.vllm + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def decide(self, state, q, images):
        labels, shown = options_for(q)
        n = len(labels)
        lines = "\n".join(f"{string.ascii_uppercase[i]}. {s}" for i, s in enumerate(shown))
        state_text = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
        text = PROMPT.format(state=state_text, question=q["instructions"], options=lines,
                             unk=string.ascii_uppercase[n - 1])
        content = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": text}]
        dists = []
        for _ in range(3):  # retry when no sample ends with a parsable answer
            out = self.chat(content)
            dists = [d for c in out["choices"] if (d := letter_probs(c, n))]
            if dists:
                break
        p = [sum(d[i] for d in dists) / len(dists) for i in range(n)] if dists else [1 / n] * n
        if self.cal:  # calib.py's map: p' ∝ (p + eps) ** (1 / T)
            w = [(x + self.cal["eps"]) ** (1 / self.cal["T"]) for x in p]
            p = [x / sum(w) for x in w]
        return labels, p, len(dists), out.get("usage", {})


def answer(q, labels, p):
    u = p[-1]
    sub = p[:-1]
    z = sum(sub)
    cond = [x / z for x in sub] if z > 1e-12 else [1 / len(sub)] * len(sub)
    abstained = u >= max(sub)
    best = labels[max(range(len(sub)), key=lambda i: sub[i])]
    a = {"type": q["type"], "unknown_probability": u, "abstained": abstained}
    if q["type"] == "noul":
        a["noul"] = sub[0] + u / 2
    elif q["type"] == "choice":
        a.update(choice=best, probabilities=dict(zip(labels[:-1], cond)), confidence=max(cond))
    else:
        a.update(probabilities=dict(zip(labels[:-1], cond)), score=sum(i * c for i, c in enumerate(cond)),
                 legend={str(i): d for i, d in enumerate(q["criteria"])})
    return a


def make_handler(model):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                qs = req["questions"]
                with ThreadPoolExecutor(len(qs)) as ex:
                    res = dict(zip(qs, ex.map(lambda q: model.decide(req.get("state", ""), q, req.get("images", [])), qs.values())))
                body = {"model": model.a.name, "answers": {k: answer(qs[k], *res[k][:2]) for k in qs},
                        "parsed_samples": {k: res[k][2] for k in qs}}
                code = 200
            except Exception as e:  # report, don't crash the server
                body, code = {"error": f"{type(e).__name__}: {e}"}, 500
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="q36")
    ap.add_argument("--name", default="q36-think")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--calibration", help="calib.py fit output")
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Model(args))).serve_forever()
