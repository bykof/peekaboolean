"""Post-hoc calibration of jevsrv's averaged distributions: p' ∝ (p + eps) ** (1 / T), over the options and Unknown.

Monotone and the same for every option, so the argmax and the abstention never change; only the probabilities do.

fit:   python calib.py fit --pred runs/X/predictions.jsonl --records items.jsonl --out calibration.json
apply: python calib.py apply --pred runs/X/predictions.jsonl --calibration calibration.json --out runs/X-cal/predictions.jsonl

--pred rows: {"id", "probabilities": {key: p, ..., "__unknown__": p}} (jrun.py and prun.py both write this);
--records rows: {"id", "gold"} with gold a key of probabilities ("__unknown__" or None for Unknown).
"""
import argparse, json, math

UNK = "__unknown__"


def calibrate(p, eps, t):
    """p: dict key -> prob (sums to 1). Returns the calibrated dict."""
    w = {k: (v + eps) ** (1 / t) for k, v in p.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def gold_key(g):
    return UNK if g is None else (("true" if g else "false") if isinstance(g, bool) else str(g))


def nll(rows, eps, t):
    return -sum(math.log(max(calibrate(p, eps, t)[g], 1e-12)) for p, g in rows) / len(rows)


def fit(rows):
    grid_e = [10 ** (x / 4) for x in range(-40, 1)]  # 1e-10 .. 1
    grid_t = [0.5 + 0.25 * i for i in range(31)]     # 0.5 .. 8
    return min(((nll(rows, e, t), e, t) for e in grid_e for t in grid_t))


def load(pred, records):
    gold = {}
    for l in open(records):
        r = json.loads(l)
        if "gold" in r and not r.get("gold_withheld"):
            gold[r["id"]] = gold_key(r["gold"])
    rows = []
    for l in open(pred):
        r = json.loads(l)
        p = r.get("probabilities")
        if p and r["id"] in gold and gold[r["id"]] in p:
            rows.append((p, gold[r["id"]]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "apply"))
    ap.add_argument("--pred", required=True, nargs="+")
    ap.add_argument("--records", nargs="+")
    ap.add_argument("--calibration")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "fit":
        rows = [x for pr, rc in zip(a.pred, a.records) for x in load(pr, rc)]
        best, e, t = fit(rows)
        res = {"eps": e, "T": t, "nll": best, "nll_raw": nll(rows, 0.0, 1.0) if all(min(p.values()) > 0 for p, _ in rows) else None,
               "n": len(rows), "fitted_on": a.pred}
        json.dump(res, open(a.out, "w"), indent=2)
        print(json.dumps(res))
    else:
        c = json.load(open(a.calibration))
        with open(a.out, "w") as f:
            for l in open(a.pred[0]):
                r = json.loads(l)
                if r.get("probabilities"):
                    r["probabilities"] = calibrate(r["probabilities"], c["eps"], c["T"])
                f.write(json.dumps(r) + "\n")
