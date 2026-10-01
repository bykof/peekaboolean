"""Build RealJev: human-labelled real-image eval sets, converted mechanically to typed Jev requests.

venv/bin/python build_realjev.py   ->  realjev/records.jsonl, realjev/images/, realjev/stats.json
"""
import ast, base64, hashlib, io, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

OUT = Path("/root/big/evals/realjev")
SEED = 20261001
STATE = "Answer from the image."
MAX_WIRE = 2 * 1024 * 1024
MIME = {"JPEG": ("image/jpeg", "jpg"), "PNG": ("image/png", "png"), "WEBP": ("image/webp", "webp")}
stats = defaultdict(Counter)


def table(repo, fn):
    return pq.read_table(hf_hub_download(repo, fn, repo_type="dataset")).to_pylist()


def img(b):
    """Original bytes for JPEG/PNG/WEBP; anything else (GIF, MPO, BMP, ...) re-encoded once as PNG."""
    im = Image.open(io.BytesIO(b))
    if im.format in MIME:
        return b, im.format
    out = io.BytesIO()
    im.convert("RGBA" if "A" in im.getbands() else "RGB").save(out, "PNG")
    return out.getvalue(), "PNG"


def choice_q(question, options):
    return {"type": "choice", "instructions": question, "criteria": {o: None for o in options}}


def noul_q(question):
    return {"type": "noul", "instructions": question, "criteria": {"true": None, "false": None}}


def item(source, sid, q, gold, images, group=None, state=STATE, task=None):
    """images: bytes or HF image structs. Returns a candidate dict or None (with the reason counted)."""
    images = [x["bytes"] if isinstance(x, dict) else x for x in images]
    if q["type"] == "choice":
        keys = list(q["criteria"])
        if len(set(keys)) != len(keys) or any(not k.strip() for k in keys):
            stats[source]["skip_duplicate_or_empty_option"] += 1
            return None
        assert gold in q["criteria"], (source, sid, gold)
    if len(images) > 2:
        stats[source]["skip_more_than_2_images"] += 1
        return None
    return {"source": source, "sid": str(sid), "q": q, "gold": gold, "images": images, "state": state,
            "group": group, "task": task}


def finalize(c):
    """Write images, check the inlined wire size; None when over 2 MB."""
    imgs = [img(b) for b in c["images"]]
    size = len(json.dumps({"state": c["state"], "questions": {"decision": c["q"]}}))
    size += sum(len(MIME[f][0]) + 15 + 4 * ((len(b) + 2) // 3) for b, f in imgs)
    if size > MAX_WIRE:
        stats[c["source"]]["skip_request_over_2MB"] += 1
        return None
    rid = f"{c['source']}-{c['sid']}"
    paths = []
    for k, (b, f) in enumerate(imgs):
        p = Path("images") / c["source"] / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', c['sid'])}-{k}.{MIME[f][1]}"
        (OUT / p).parent.mkdir(parents=True, exist_ok=True)
        (OUT / p).write_bytes(b)
        paths.append(str(p))
    group = c["group"] or hashlib.sha1(c["images"][0]).hexdigest()[:16]
    rec = {"id": rid, "source": c["source"], "group": f"{c['source']}:{group}",
           "wire": {"state": c["state"], "questions": {"decision": c["q"]}, "images": paths}, "gold": c["gold"]}
    if c["task"]:
        rec["task"] = c["task"]
    return rec


def sample(cands, n, source, strata=None):
    """Seeded shuffle; take the first n (per stratum when strata=quota per key) that fit under 2 MB."""
    rng = random.Random(f"{SEED}:{source}")
    cands = [c for c in cands if c]
    rng.shuffle(cands)
    out, taken = [], Counter()
    for c in cands:
        key = c["task"] if strata else None
        if taken[key] >= (strata.get(key, 0) if strata else n):
            continue
        if (r := finalize(c)):
            out.append(r)
            taken[key] += 1
    stats[source]["eligible"] = len(cands)
    stats[source]["kept"] = len(out)
    return out


# ---- sources -------------------------------------------------------------------------------------------------

def split_letters(parts):
    """re.split output ['', 'A', 'x', 'B', 'y', ...] -> {'A': 'x', ...} if letters are A, B, C, ... in order."""
    letters, texts = parts[1::2], [t.strip() for t in parts[2::2]]
    if parts[0].strip() or letters != list("ABCDEFGH"[:len(letters)]) or len(letters) < 2:
        return None
    return dict(zip(letters, texts))


def mmstar():
    out = []
    for r in table("Lin-Chen/MMStar", "mmstar.parquet"):
        q = r["question"]
        if "\nOptions: " in q:
            stem, opts = q.rsplit("\nOptions: ", 1)
            opts = split_letters(re.split(r"(?:^|, )([A-H]): ", opts))
        elif "\nChoices:\n" in q:  # MathVista style: "Hint: <answer-format instruction>\nQuestion: ...\nChoices:\n(A) .."
            stem, opts = q.split("\nChoices:\n", 1)
            stem = stem.split("Question: ", 1)[-1]
            opts = split_letters(re.split(r"(?:^|\n)\(([A-H])\) ", opts))
        else:
            opts = None
        if not opts or r["answer"] not in opts:
            stats["mmstar"]["skip_unparsed"] += 1
            continue
        out.append(item("mmstar", r["index"], choice_q(stem.strip(), opts.values()), opts[r["answer"]],
                        [r["image"]], task=r["category"]))
    return sample(out, 300, "mmstar")


def realworldqa():
    out = []
    rows = table("xai-org/RealworldQA", "data/test-00000-of-00002.parquet") + table("xai-org/RealworldQA", "data/test-00001-of-00002.parquet")
    for sid, r in enumerate(rows):
        lines = r["question"].strip().split("\n")
        if lines[-1].startswith("Please answer"):  # answer-format instruction
            lines = lines[:-1]
        first = next((i for i, l in enumerate(lines) if re.match(r"^[A-H]\. ", l)), None)
        if first is None:
            if r["answer"].strip().lower() in ("yes", "no"):
                out.append(item("realworldqa", sid, noul_q("\n".join(lines).strip()),
                                "true" if r["answer"].strip().lower() == "yes" else "false", [r["image"]]))
            else:
                stats["realworldqa"]["skip_open_answer"] += 1
            continue
        opts = split_letters(re.split(r"(?:^|\n)([A-H])\. ", "\n".join(lines[first:])))
        if not opts or r["answer"].strip() not in opts:
            stats["realworldqa"]["skip_unparsed"] += 1
            continue
        out.append(item("realworldqa", sid, choice_q("\n".join(lines[:first]).strip(), opts.values()),
                        opts[r["answer"].strip()], [r["image"]]))
    return sample(out, 300, "realworldqa")


def hallusionbench():
    out = []
    for r in table("lmms-lab/HallusionBench", "data/image-00000-of-00001.parquet"):
        sid = f"{r['category']}_{r['subcategory']}_{r['set_id']}_{r['figure_id']}_{r['question_id']}"
        out.append(item("hallusionbench", sid, noul_q(r["question"].strip()), {"1": "true", "0": "false"}[r["gt_answer"]],
                        [r["image"]], group=f"{r['category']}/{r['subcategory']}/{r['set_id']}",
                        task=f"{r['category']}/{r['subcategory']}"))
    return sample(out, 300, "hallusionbench")


def pope():
    out = []
    for r in table("lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet"):
        out.append(item("pope_adversarial", r["id"], noul_q(r["question"].strip()),
                        {"yes": "true", "no": "false"}[r["answer"].strip().lower()], [r["image"]],
                        group=r["image_source"]))
    return sample(out, 300, "pope_adversarial")


def letter_gold(choices, answer):
    i = "ABCDEFGH".index(answer.strip().strip("()"))
    return choices[i]


def cvbench():
    out = []
    for fn in ["test_2d.parquet", "test_3d.parquet"]:
        for r in table("nyu-visionx/CV-Bench", fn):
            ch = [c.strip() for c in r["choices"]]
            out.append(item("cvbench", r["idx"], choice_q(r["question"].strip(), ch), letter_gold(ch, r["answer"]),
                            [r["image"]], task=f"{r['type']}/{r['task']}"))
    return sample(out, 300, "cvbench", strata={"2D/Count": 75, "2D/Relation": 75, "3D/Depth": 75, "3D/Distance": 75})


BLINK_TASKS = ["Counting", "Relative_Depth", "Spatial_Relation", "Object_Localization", "Multi-view_Reasoning",
               "Visual_Correspondence"]


def blink():
    out = []
    for t in BLINK_TASKS:
        for r in table("BLINK-Benchmark/BLINK", f"{t}/val-00000-of-00001.parquet"):
            # The prompt carries the task context (what the drawn marks mean); drop its option block and the
            # "Select from the following choices/options." line, the options go into criteria.
            stem = re.split(r"\n\(A\) ", r["prompt"], maxsplit=1)[0]
            stem = re.sub(r"\s*Select from the following (choices|options)\.\s*$", "", stem).strip()
            ch = [c.strip() for c in r["choices"]]
            imgs = [r[k] for k in ("image_1", "image_2", "image_3", "image_4") if r[k]]
            out.append(item("blink", r["idx"], choice_q(stem, ch), letter_gold(ch, r["answer"]), imgs, task=t))
    return sample(out, 240, "blink", strata={t: 40 for t in BLINK_TASKS})


def mme_realworld_lite():
    out = []
    for k in range(4):
        for r in table("yifanzhang114/MME-RealWorld-lite-lmms-eval", f"data/train-0000{k}-of-00004.parquet"):
            raw = ast.literal_eval(r["multi-choice options"]) if isinstance(r["multi-choice options"], str) else r["multi-choice options"]
            opts = split_letters(re.split(r"(?:^|\n)\(([A-H])\) ", "\n".join(o.strip() for o in raw)))
            if not opts or r["answer"].strip() not in opts:
                stats["mme_realworld_lite"]["skip_unparsed"] += 1
                continue
            out.append(item("mme_realworld_lite", r["index"], choice_q(r["question"].strip(), opts.values()),
                            opts[r["answer"].strip()], [base64.b64decode(r["bytes"])], task=r["category"]))
    return sample(out, 300, "mme_realworld_lite")


def jevbench_preview():
    """Image JevBench public preview (128 real items): rubric, gold and state verbatim (minus the state's asset path);
    image fetched from the exact source row. Items whose image is not in the source row are skipped."""
    items = json.load(open("/root/big/evals/preview-items.json"))
    fs = HfFileSystem()
    clevr_cfgs = None
    out = []
    for it in items:
        ds, row, ins = it["dataset"], it["source_row"], it["rubric"]["instructions"]
        b = None
        if ds == "CLEVR-HOPE":
            if clevr_cfgs is None:  # the item does not name its config: find the one whose rows match all 20 items
                clevr_cfgs = {}
                for cfg in [f"HOP{i:02d}" for i in range(29)]:
                    files = sorted(p for p in fs.ls(f"datasets/user9000/CLEVR-HOPE/{cfg}", detail=False)
                                   if p.split("/")[-1].startswith(it["split"] + "-"))
                    if files:
                        clevr_cfgs[cfg] = pq.ParquetFile(fs.open(files[0])).read_row_group(0).slice(0, 64).to_pylist()
                want = {i["source_row"]: (i["rubric"]["instructions"], i["gold"]) for i in items if i["dataset"] == ds}
                match = [c for c, rows in clevr_cfgs.items()
                         if all(k < len(rows) and rows[k]["query"] == q and rows[k]["answer"] == g for k, (q, g) in want.items())]
                stats["jevbench_preview"]["clevr_config"] = match[0] if len(match) == 1 else f"ambiguous:{match}"
                clevr_rows = clevr_cfgs[match[0]] if len(match) == 1 else None
            if clevr_rows and clevr_rows[row]["query"] == ins:
                b = clevr_rows[row]["image"]["bytes"]
        elif ds == "Geometry3K":
            r = globals().setdefault("_geo", table("hiyouga/geometry3k", "data/test-00000-of-00001.parquet"))[row]
            if r["problem"].replace("<image>", "").strip() == ins.strip() and len(r["images"]) == 1:
                b = r["images"][0]["bytes"]
        elif ds == "ArxivQA":
            r = globals().setdefault("_arx", pq.ParquetFile(fs.open("datasets/mm-eval/ArxivQA/data/train-00000-of-00036.parquet"))
                                     .read_row_group(0).slice(0, 64).to_pylist())[row]
            if json.loads(r["messages"])[0]["question"] == ins and r["answer"] == it["gold"] and len(r["media"]) == 1:
                b = r["media"][0]["bytes"]
        # FinQA (bevaya/FinQA) has no image column: the preview image is the builder's own table rendering.
        # ScreenSpot / Multimodal-Mind2Web: the question asks for "labelled marker A-E" drawn by the preview builder;
        # the source screenshot has no markers and the distractor positions are not published.
        if b is None:
            stats["jevbench_preview"][f"skip_not_locatable:{ds}"] += 1
            continue
        state = {k: v for k, v in it["state"].items() if k != "image"}
        out.append(item("jevbench_preview", it["id"], dict(it["rubric"]), it["gold"], [b], state=state, task=ds))
    return sample(out, len(out), "jevbench_preview")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for fn in [mmstar, realworldqa, hallusionbench, pope, cvbench, blink, mme_realworld_lite, jevbench_preview]:
        recs += fn()
        print(fn.__name__, dict(stats[recs[-1]["source"]]), flush=True)
    assert len({r["id"] for r in recs}) == len(recs)
    with open(OUT / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["total"]["kept"] = len(recs)
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(len(recs), "records")
