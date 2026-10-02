"""Every number in docs/think-35b.md, from the run outputs on the training server.

venv/bin/python collect.py [--cal /root/big/calibration-q36-v1.json] > results.md
"""
import argparse, json, subprocess, sys
from pathlib import Path

B = Path("/root/big")
E = B / "evals"
UNK = "__unknown__"
sys.path.insert(0, str(B))
import calib  # noqa: E402

IJB = {  # name -> per-split score.json and predictions
    "imajev-4b (its HTTP server, as shipped)": "baselines/imajev4b-http-{s}",
    "imajev-4b (its leaderboard protocol)": "baselines/imajev4b-direct-{s}",
    "Qwen3.6-35B-A3B zero-shot": "runs/q36p2-{s}",
    "Qwen3.8-27B zero-shot": "runs/q38zs-{s}",
    "**peekaboolean-think-35b**": "runs/q36-v1-{s}",
}


def ijb_paths(t, s):
    d = B / t.format(s=s)
    score = d / "score.json" if (d / "score.json").exists() else Path(str(d) + "-score.json")
    return score, d / "predictions.jsonl"


def gold_key(g):
    return UNK if g is None else (("true" if g else "false") if isinstance(g, bool) else str(g))


def ece_brier(pairs):
    """pairs: (probabilities incl. __unknown__, gold key). 10-bin top-label ECE and multi-class Brier, as jscore."""
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    brier = 0.0
    for p, g in pairs:
        k = max(p, key=p.get)
        b = min(int(p[k] * 10), 9)
        bins[b][0] += 1; bins[b][1] += p[k]; bins[b][2] += k == g
        brier += sum((v - (key == g)) ** 2 for key, v in p.items())
    n = len(pairs)
    return sum(abs(c - a) for m, c, a in bins if m) / n, brier / n


def imajevbench(cal):
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(B / "imajev-bench/records/records-public.jsonl")}
    print("### ImajevBench v2.0-lite, dev + calibration (254 items with gold; the benchmark's own harness and scorer)\n")
    print("| System | dev (173) | calibration (81) | all (254) | ECE | ECE, calibrated | contrast sets all right (45) "
          "| Unknown items abstained (24) | false abstentions (230) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, t in IJB.items():
        row, tot, cs, pairs = [], 0, 0, []
        unk_ok = false_abs = 0
        for s in ("dev", "calibration"):
            score, pred = ijb_paths(t, s)
            if not score.exists():
                break
            j = json.load(open(score))
            a = j["capability"]["all_records"]
            row.append(a["correct"]); tot += a["correct"]; cs += j["contrast_sets"]["all_correct"]
            for l in open(pred):
                p = json.loads(l)
                g = gold_key(recs[p["id"]]["gold"])
                if p.get("probabilities"):
                    pairs.append((p["probabilities"], g))
                unk_ok += g == UNK and p["status"] == "abstained"
                false_abs += g != UNK and p["status"] == "abstained"
        if len(row) < 2:
            continue
        e, _ = ece_brier(pairs)
        ec = ece_brier([(calib.calibrate(p, cal["eps"], cal["T"]), g) for p, g in pairs])[0] if "think" in name else None
        print(f"| {name} | {row[0]} | {row[1]} | {tot} ({100 * tot / 254:.1f}%) | {e:.3f} | "
              f"{'' if ec is None else f'{ec:.3f}'} | {cs} | {unk_ok} | {false_abs} |")
    print()


def jscore(records, runs, done_only=False):
    out = Path("/tmp/collect-jscore.json")
    cmd = [str(E / "venv/bin/python"), "jscore.py", "--records", records, "--json", str(out)] + (["--done-only"] if done_only else []) + runs
    subprocess.run(cmd, cwd=E, check=True, capture_output=True)
    return json.load(open(out))


def recalibrated(run, cal_path, tag):
    """Write run/predictions.jsonl through calib.py into runs/<run>-<tag>/ and return that run dir."""
    out = E / f"{run}-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    c = json.load(open(cal_path))
    with open(out / "predictions.jsonl", "w") as f:
        for l in open(E / run / "predictions.jsonl"):
            r = json.loads(l)
            if r.get("probabilities"):
                r["probabilities"] = calib.calibrate(r["probabilities"], c["eps"], c["T"])
            f.write(json.dumps(r) + "\n")
    return str(out.relative_to(E))


def table(title, records, systems, cal_path, partial=()):
    """systems: [(label, run)]; a run in partial is scored on its own answered ids only."""
    runs = [r for _, r in systems]
    cal_run = recalibrated(dict(systems)["**peekaboolean-think-35b**"], cal_path, "CAL")
    full = jscore(records, [r for r in runs if r not in partial] + [cal_run])
    keys = [k for k in full[runs[0]] if k != "ALL"]
    print(f"### {title}\n")
    print("| System | all | " + " | ".join(keys) + " | ECE | Brier |")
    print("|---|---|" + "---|" * len(keys) + "---|---|")
    for label, r in systems + [("peekaboolean-think-35b, calibrated", cal_run)]:
        j = full[r] if r in full else jscore(records, [r], done_only=True)[r]
        a = j["ALL"]
        n = f" (n={a['n']})" if r in partial else ""
        print(f"| {label}{n} | {100 * a['acc']:.1f} | " + " | ".join(f"{100 * j[k]['acc']:.1f}" if k in j else "–" for k in keys)
              + f" | {a['ece']:.3f} | {a['brier']:.3f} |")
    extra = {k: v for k, v in full[runs[-1]]["ALL"].items() if k in ("unk_recall", "false_abstain")}
    if extra:
        print(f"\nOurs: {', '.join(f'{k} {100 * v:.1f}%' for k, v in extra.items())}; "
              f"imajev-4b: {', '.join(f'{k} {100 * full[runs[0]]['ALL'][k]:.1f}%' for k in extra)}.")
    print()
    r = subprocess.run([str(E / "venv/bin/python"), "jscore.py", "--records", records, "--compare", runs[-1], runs[0]],
                       cwd=E, capture_output=True, text=True).stdout.strip().splitlines()
    print("Paired, ours − imajev-4b (group bootstrap 95% CI):\n\n```\n" + "\n".join(r[-12:]) + "\n```\n")


def jevbench():
    print("### JevBench public splits (text only; the benchmark's own harness and scorer)\n")
    print("| System | hard (111) | original (72) | easy (48) |")
    print("|---|---|---|---|")
    for label, d in (("imajev-4b", "imajev-4b"), ("Qwen3.6-35B-A3B zero-shot", "q36-zeroshot"),
                     ("**peekaboolean-think-35b**", "q36-v1")):
        f = E / "jevbench-runs" / d / "scores.txt"
        if not f.exists():
            continue
        s = {l.split()[0]: l.split() for l in open(f) if l.strip()}
        print(f"| {label} | " + " | ".join(f"{s[t][1]} (ECE {float(s[t][5]):.3f})" if t in s else "–"
                                           for t in ("hard", "original", "easy")) + " |")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default=str(B / "calibration-q36-v1.json"))
    a = ap.parse_args()
    cal = json.load(open(a.cal))
    print(f"Calibration map: eps {cal['eps']:g}, T {cal['T']:g}, fitted on {cal['n']} held-out items ({', '.join(cal['fitted_on'])}).\n")
    imajevbench(cal)
    table("RealJev (2,100 human-labelled real-image items from 8 public sets)", "realjev/records.jsonl",
          [("imajev-4b", "runs/realjev-imajev4b-best"), ("Qwen3.6-35B-A3B zero-shot", "runs/realjev-q36-zs"),
           ("Qwen3.8-27B zero-shot", "runs/realjev-q38zs"), ("**peekaboolean-think-35b**", "runs/realjev-q36-v1")],
          a.cal, partial=("runs/realjev-q36-zs", "runs/realjev-q38zs"))
    table("imajev's held-out real-photo exam (its own converters; 1,779-item subsample)", "heldout/records-sub.jsonl",
          [("imajev-4b", "runs/heldout-imajev4b"), ("Qwen3.6-35B-A3B zero-shot", "runs/heldout-q36-zs"),
           ("**peekaboolean-think-35b**", "runs/heldout-q36-v1")],
          a.cal, partial=("runs/heldout-q36-zs",))
    jevbench()
