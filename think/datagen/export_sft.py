"""SFT rows from RFT samples: correct traces only, prompts re-checked against the current jevsrv, decontaminated.

val = image groups with sha1(group) % 20 == 0 (~5% of every source, no image shared with train).
Kept: every mixed-outcome item (<= 2 correct traces); always-correct items (1 trace each, matched controls first) at
~40% of rows; Unknown-gold rows steered into 12-15%; samples that hit max_tokens are never kept.
python export_sft.py  -> sft/train.jsonl, sft/val.jsonl, sft/stats.md
"""
import glob, hashlib, json, os, random, statistics, sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/big/data")
from rft import build  # noqa: E402

D = "/root/big/data"
R = random.Random(7)
RFT = os.environ.get("RFT_DIR", f"{D}/rft")  # traces of the model being trained
SFT = os.environ.get("SFT_DIR", f"{D}/sft")
ALWAYS_SHARE = 0.40
UNK_SHARE = (0.12, 0.15)


def is_val(it):
    return int(hashlib.sha1(it["group"].encode()).hexdigest(), 16) % 20 == 0


def main():
    items = {it["id"]: it for f in sorted(glob.glob(f"{D}/items/*.jsonl")) for it in map(json.loads, open(f))}
    drop_path = f"{D}/decontam/drop_ids.txt"
    drop = set(open(drop_path).read().split()) if os.path.exists(drop_path) else set()
    cur = {}
    samples, stale, errors = defaultdict(list), Counter(), Counter()
    for l in (l for f in sorted(glob.glob(f"{RFT}/*.jsonl")) for l in open(f)):
        r = json.loads(l)
        i = r["item_id"]
        if i not in items:
            continue
        if "samples" not in r:
            errors[r["source"]] += 1
            continue
        if i not in cur:
            cur[i] = build(items[i], urls=False)[0][-1]["text"]
        if r["prompt"] != cur[i]:  # sampled with an older jevsrv prompt
            stale[r["source"]] += len(r["samples"])
            continue
        samples[i] += r["samples"]
    rows = {"train": [], "val": [], "train_all": [], "val_all": []}
    twins = {it["question"]["instructions"] for it in items.values() if it["source"] == "unk-premise"}
    st = defaultdict(Counter)
    toks = defaultdict(list)
    for split in ("train", "val"):
        mixed, always = [], []
        for i, ss in samples.items():
            it = items[i]
            if i in drop or is_val(it) != (split == "val"):
                continue
            src = it["source"]
            ok = [s for s in ss if s["correct"] and s["finish"] == "stop"]
            if split == "train":
                st[src]["items"] += 1
                st[src]["samples"] += len(ss)
                st[src]["correct"] += sum(s["correct"] for s in ss)
                st[src]["no_answer"] += sum(s["letter"] is None for s in ss)
                st[src]["abstain"] += sum(s["letter"] == chr(64 + len(s["probs"] or [0] * 9)) for s in ss if s["probs"])
                toks[src] += [s["n_tokens"] for s in ss]
            if not ok:
                st[src][f"{split}_none"] += split == "train"
                continue
            if len(ok) == len(ss):
                always.append((i, ok))
            else:
                mixed.append((i, ok))
        outs = {split: fill(mixed, always, items, twins), split + "_all": fill_all(mixed, always, items)}
        if split == "train":
            for i, _ in mixed:
                st[items[i]["source"]]["mixed"] += 1
            for i, _ in always:
                st[items[i]["source"]]["always"] += 1
        for name, out in outs.items():
            R.shuffle(out)
            for i, s in out:
                it = items[i]
                rows[name].append({"images": it["images"], "prompt": cur[i], "response": s["text"], "source": it["source"],
                                   "item_id": i})
                st[it["source"]][f"{name}_rows"] += 1
    os.makedirs(f"{D}/sft", exist_ok=True)
    for split, rs in rows.items():
        with open(f"{SFT}/{split}.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_stats(items, st, toks, rows, stale, errors, drop)


def fill(mixed, always, items, twins):
    """All mixed-outcome traces (<= 2 per item), then always-correct items (1 trace each, controls first) at ~40% of rows;
    Unknown-gold rows steered into 12-15% by how many always-correct Unknown items are added."""
    def unk(i):
        return items[i]["gold"] == "__unknown__"

    def ctrl(i):
        it = items[i]
        return it["source"].startswith("ctrl-") or (it["source"] == "vqav2" and it["question"]["instructions"] in twins)
    M = [(i, s) for i, ok in mixed for s in R.sample(ok, min(2, len(ok)))]
    total = round(len(M) / (1 - ALWAYS_SHARE))
    mu = sum(unk(i) for i, _ in M)
    if mu > UNK_SHARE[1] * total:  # too many Unknown mixed rows: drop their second traces first
        extra, seen, keep = mu - round(UNK_SHARE[1] * total), set(), []
        for i, s in M:
            if unk(i) and i in seen and extra > 0:
                extra -= 1
                continue
            seen.add(i)
            keep.append((i, s))
        M = keep
        mu = sum(unk(i) for i, _ in M)
    R.shuffle(always)
    always.sort(key=lambda x: not ctrl(x[0]))
    au = [(i, ok[0]) for i, ok in always if unk(i)]
    ak = [(i, ok[0]) for i, ok in always if not unk(i)]
    n_au = max(0, min(len(au), round(sum(UNK_SHARE) / 2 * total) - mu))
    n_ak = max(0, total - len(M) - n_au)
    return M + au[:n_au] + ak[:n_ak]


def fill_all(mixed, always, items):
    """Every item with a correct trace: mixed <= 2 traces, always-correct 1 trace; Unknown-gold rows steered to ~13.5%
    (subsampling Unknown always-correct items, or adding their second traces)."""
    def unk(i):
        return items[i]["gold"] == "__unknown__"
    M = [(i, s) for i, ok in mixed for s in R.sample(ok, min(2, len(ok)))]
    known = [(i, ok[0]) for i, ok in always if not unk(i)]
    u1 = [(i, ok[0]) for i, ok in always if unk(i)]
    u2 = [(i, ok[1]) for i, ok in always if unk(i) and len(ok) > 1]
    R.shuffle(u1)
    R.shuffle(u2)
    n_known = len(known) + sum(not unk(i) for i, _ in M)
    want = max(0, round(sum(UNK_SHARE) / 2 / (1 - sum(UNK_SHARE) / 2) * n_known) - sum(unk(i) for i, _ in M))
    return M + known + (u1 + u2)[:want]


def write_stats(items, st, toks, rows, stale, errors, drop):
    lic = json.load(open(f"{D}/items/LICENSES.json"))
    built = Counter(it["source"] for it in items.values())
    ds = defaultdict(Counter)
    for it in items.values():
        ds[it["source"]][it["dataset"]] += 1
    L = ["# SFT data from rejection-sampled self-distillation", "",
         "Base: Qwen3.6-35B-A3B-FP8 (`q36` on vLLM), prompts built by `/root/big/jevsrv.py` itself (PROMPT/options_for, "
         "Unknown last), T=0.6, top_p=0.95, max_tokens=12000, logprobs on (top-20 for the first pilot requests, then top-5: building top-20 logprobs saturated the vLLM API server; sampling is unaffected). A sample is correct when the letter after its last "
         "`Answer:` (the token `jevsrv.letter_probs` reads) is the gold letter and it finished with `stop`.", "",
         "`response` is the assistant text exactly as generated: the reasoning, `</think>`, then the answer ending in `Answer: <letter>`. "
         "Qwen3.6's chat template already ends the generation prompt with `<think>\\n`, so the response has no opening `<think>`; "
         "train with the same template and `add_generation_prompt=True`.", "",
         f"Rows: train {len(rows['train'])}, val {len(rows['val'])}; alternative train_all {len(rows['train_all'])}, "
         f"val_all {len(rows['val_all'])} (val = image groups with sha1(group) % 20 == 0, about 5% of every source, "
         "no image shared with train).", "",
         "train_all/val_all: every item with a correct trace (mixed <= 2 traces, always-correct 1 trace), Unknown-gold rows "
         f"steered to ~13.5%: train_all {sum(items[r['item_id']]['gold'] == '__unknown__' for r in rows['train_all'])} Unknown rows.", "",
         "Selection: all mixed-outcome items (at most 2 correct traces each), then always-correct items (1 trace each, "
         f"matched controls first) at ~{ALWAYS_SHARE:.0%} of rows; Unknown-gold rows steered into 12-15%; samples that hit "
         "max_tokens and items with no correct sample are dropped. "
         f"Unknown-gold rows: train {sum(items[r['item_id']]['gold'] == '__unknown__' for r in rows['train'])}, "
         f"val {sum(items[r['item_id']]['gold'] == '__unknown__' for r in rows['val'])}.", "",
         f"Decontamination: {len(drop)} items dropped (image sha256 or 8x8 DCT phash equal to an eval image); see decontam/report.json.",
         f"Stale samples (older jevsrv prompt) skipped: {sum(stale.values())}. Request errors: {sum(errors.values())}.", "",
         "Per source, sorted by the share of mixed-outcome items (where the base is unsure); counts are train-split items.", "",
         "| source | mixed items | mixed % | always | none | sampled items | samples | acc | abstain rate | mean tok | "
         "train rows | val rows | train_all rows | val_all rows | built | datasets |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for src in sorted(built, key=lambda s: -st[s]["mixed"] / max(1, st[s]["items"])):
        c = st[src]
        acc = c["correct"] / c["samples"] if c["samples"] else float("nan")
        L.append(f"| {src} | {c['mixed']} | {100 * c['mixed'] / max(1, c['items']):.1f} | {c['always']} | {c['train_none']} | "
                 f"{c['items']} | {c['samples']} | {acc:.3f} | {c['abstain'] / max(1, c['samples']):.3f} | "
                 f"{statistics.mean(toks[src]) if toks[src] else 0:.0f} | {c['train_rows']} | {c['val_rows']} | "
                 f"{c['train_all_rows']} | {c['val_all_rows']} | {built[src]} | {', '.join(f'{k} {v}' for k, v in ds[src].most_common())} |")
    allt = [t for v in toks.values() for t in v]
    tot = Counter()
    for c in st.values():
        tot.update(c)
    L += ["", f"Overall: {tot['samples']} train-split samples, accuracy {tot['correct'] / max(1, tot['samples']):.3f}, "
          f"mean completion tokens {statistics.mean(allt) if allt else 0:.0f}, unparsed {tot['no_answer']}.", "",
          "## Licences (per dataset)", ""] + [f"- {k}: {v}" for k, v in sorted(lic.items())]
    open(f"{SFT}/stats.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
