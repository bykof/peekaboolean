"""Score jrun.py runs against records.

python jscore.py --records realjev/records.jsonl runs/A [runs/B ...]      # per run: overall + per source
python jscore.py --records realjev/records.jsonl --compare runs/A runs/B  # paired A-B accuracy, group bootstrap 95% CI
  --by task   break down by record["task"] instead of source
  --done-only score only ids the (first) run answered or abstained on, for a partial run

Correct = answered with the gold key, or abstained when gold is __unknown__; errors and missing ids count as wrong.
ECE: 10 fixed bins, top label of the full distribution (incl. __unknown__) vs gold, as in imajev_bench.
Brier: sum over all keys incl. __unknown__. Both over valid predictions only (n_valid shown).
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

UNK = "__unknown__"


def load(run):
    preds = {}
    for line in (Path(run) / "predictions.jsonl").open():
        p = json.loads(line)
        if p["id"] not in preds or p["status"] != "error":  # a later success replaces an error, never the reverse
            preds[p["id"]] = p
    return preds


def correct(rec, p):
    if not p or p["status"] == "error":
        return False
    if rec["gold"] == UNK:
        return p["status"] == "abstained"
    return p["status"] == "answered" and p["value"] == rec["gold"]


def metrics(recs, preds):
    n = len(recs)
    c = sum(correct(r, preds.get(r["id"])) for r in recs)
    ps = [(r, preds[r["id"]]) for r in recs if preds.get(r["id"]) and preds[r["id"]]["probabilities"]]
    bins, brier = [[] for _ in range(10)], 0.0
    for r, p in ps:
        v = p["probabilities"]
        top = max(v, key=v.get)
        bins[min(int(v[top] * 10), 9)].append((v[top], top == r["gold"]))
        brier += sum((q - (k == r["gold"])) ** 2 for k, q in v.items())
    ece = sum(len(b) / len(ps) * abs(sum(x[1] for x in b) / len(b) - sum(x[0] for x in b) / len(b)) for b in bins if b) if ps else None
    unk = [r for r in recs if r["gold"] == UNK]
    sub = [r for r in recs if r["gold"] != UNK]
    abst = lambda rs: sum(bool(preds.get(r["id"])) and preds[r["id"]]["status"] == "abstained" for r in rs)
    out = {"n": n, "acc": c / n, "abstain": abst(recs) / n, "errors": sum(not preds.get(r["id"]) or preds[r["id"]]["status"] == "error" for r in recs),
           "n_valid": len(ps), "ece": ece, "brier": brier / len(ps) if ps else None}
    if unk:
        out["unk_n"] = len(unk)
        out["unk_recall"] = abst(unk) / len(unk)  # correct-Unknown rate
    out["false_abstain"] = abst(sub) / len(sub) if sub else None
    return out


def boot_ci(recs, delta, samples=2000, seed=0):
    """Percentile group bootstrap of mean(delta) where delta[id] is per-item; groups = record['group']."""
    g = defaultdict(lambda: [0.0, 0])
    for r in recs:
        g[r["group"]][0] += delta[r["id"]]
        g[r["group"]][1] += 1
    groups, rng, est = list(g.values()), random.Random(seed), []
    for _ in range(samples):
        draw = rng.choices(groups, k=len(groups))
        est.append(sum(x[0] for x in draw) / sum(x[1] for x in draw))
    est.sort()
    return est[int(0.025 * samples)], est[int(0.975 * samples) - 1]


def fmt(m):
    s = f"n={m['n']:5d} acc={m['acc']*100:5.1f} abst={m['abstain']*100:4.1f} err={m['errors']:3d}"
    s += f" ece={m['ece']:.3f} brier={m['brier']:.3f}" if m["ece"] is not None else " ece=  -   brier=  -  "
    if "unk_recall" in m:
        s += f" unkRecall={m['unk_recall']*100:5.1f}(n={m['unk_n']}) falseAbst={(m['false_abstain'] or 0)*100:4.1f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--by", default="source")
    ap.add_argument("--json", type=Path, help="also write the report here")
    ap.add_argument("--done-only", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    recs = [json.loads(l) for l in a.records.open()]
    if a.done_only:
        first = load(a.runs[0])
        recs = [r for r in recs if r["id"] in first and first[r["id"]]["status"] != "error"]
    keys = sorted({r.get(a.by, "-") for r in recs})
    report = {}
    if a.compare:
        A, B = a.runs
        pa, pb = load(A), load(B)
        print(f"A={A}\nB={B}\n{'':24s} {'n':>5s} {'A':>6s} {'B':>6s} {'A-B':>6s}  95% CI (group bootstrap)")
        for k in ["ALL"] + keys:
            rs = recs if k == "ALL" else [r for r in recs if r.get(a.by, "-") == k]
            d = {r["id"]: float(correct(r, pa.get(r["id"]))) - float(correct(r, pb.get(r["id"]))) for r in rs}
            ma, mb = metrics(rs, pa)["acc"], metrics(rs, pb)["acc"]
            lo, hi = boot_ci(rs, d)
            report[k] = {"n": len(rs), "acc_a": ma, "acc_b": mb, "diff": ma - mb, "ci95": [lo, hi]}
            print(f"{k:24s} {len(rs):5d} {ma*100:6.1f} {mb*100:6.1f} {(ma-mb)*100:+6.1f}  [{lo*100:+.1f}, {hi*100:+.1f}]")
    else:
        for run in a.runs:
            preds = load(run)
            print(f"== {run}")
            report[run] = {}
            for k in ["ALL"] + keys:
                rs = recs if k == "ALL" else [r for r in recs if r.get(a.by, "-") == k]
                m = metrics(rs, preds)
                if k == "ALL":
                    m["acc_ci95"] = boot_ci(rs, {r["id"]: float(correct(r, preds.get(r["id"]))) for r in rs})
                report[run][k] = m
                print(f"{k:24s} {fmt(m)}" + (f" ci95=[{m['acc_ci95'][0]*100:.1f}, {m['acc_ci95'][1]*100:.1f}]" if k == "ALL" else ""))
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
