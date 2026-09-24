"""Build typed questions from public image QA and screenshot data.

Only upstream training splits are used. These internal splits measure adapter
generalization; the pretrained backbone may already have seen the source images.
Generated distractors are weak supervision, not human-verified alternatives.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

REPO = "HuggingFaceM4/the_cauldron"
DEFAULT_PARTS = "vqav2:18000,clevr:5000,textvqa:6000,docvqa:10000,chartqa:6000,screen2words:8000,ai2d:2500"
COLORS = "black white red green blue yellow orange brown gray pink purple beige silver gold".split()


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip()).rstrip(".").strip()


def clean_question(text):
    # Cauldron appends a short-answer instruction on its own final line.
    lines = text.strip().splitlines()
    if len(lines) > 1 and re.search(r"brief|concise|compact|terse|short|quick response|succinct|brevity", lines[-1], re.I):
        lines.pop()
    return "\n".join(lines).removeprefix("Question: ").strip()


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


@lru_cache(maxsize=None)
def canonical(answer):
    """Comparison form for synonyms: case, articles, punctuation, number words."""
    words = re.sub(r"[^\w\s%.]", " ", answer.casefold().replace(",", "")).split()
    words = [NUMBER_WORDS.get(w, w) for w in words if w not in ("a", "an", "the")]
    return " ".join(words).rstrip(".")


@lru_cache(maxsize=None)
def shape(answer):
    """Surface form of an answer. Distractors of another form are rejectable without the image."""
    s = canonical(answer)
    if re.fullmatch(r"[-+$€£]?\s*\d+(\.\d+)?\s*%?", s): return "number"
    if re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b.*\d", s):
        return "date"
    if re.search(r"\d", s): return "alnum"
    n = len(s.split())
    return "word" if n == 1 else "phrase" if n <= 4 else "sentence"


def parse_turn(turn):
    question = clean_question(turn["user"])
    answer = normalize(turn["assistant"])
    if "\nChoices:\n" in question:
        q, options = question.split("\nChoices:\n", 1)
        pairs = re.findall(r"^([A-Z])\. (.+)$", options, re.M)
        label = answer.removeprefix("Answer: ").strip()
        criteria = dict(pairs)
        if len(criteria) >= 2 and label in criteria:
            return q, answer, criteria, label
        return None
    if not question or not answer or len(question) > 800 or len(answer) > 220:
        return None
    return question, answer, None, None


def family(question, answer, source):
    q = question.lower()
    if "color" in q or "colour" in q:
        return source, "color"
    if "how many" in q or re.fullmatch(r"\d+(\.\d+)?", answer):
        return source, "number"
    return source, " ".join(q.split()[:3])


def split_for(image_digest, seed):
    n = int(hashlib.sha256(f"{seed}:{image_digest}".encode()).hexdigest()[:8], 16) % 100
    return "test" if n < 5 else "calib" if n < 10 else "val" if n < 15 else "train"


def pool_keys(question, answer, source):
    """Most specific first. Distractors from the same question and answer form cannot
    be told apart from the answer by a text-only prior; broad pools can."""
    form, q = shape(answer), " ".join(question.casefold().split())
    return [("question", source, q, form), ("prefix", source, " ".join(q.split()[:4]), form),
            (*family(question, answer, source), form), ("source", source, form)]


def make_rows(records, seed=0, min_pool=7):
    """Negatives come from the row's own split, so a held-out answer is no more novel
    than its distractors and no label crosses splits. All turns of an image share a split."""
    # Cached source records may predate a question-cleanup fix.
    records = [{**r, "question": clean_question(r["question"])} for r in records]
    pools = defaultdict(Counter)
    for r in records:
        if r["criteria"] is None and r["answer"].lower() not in ("yes", "no"):
            for key in pool_keys(r["question"], r["answer"], r["source"]):
                pools[r["split"], key][r["answer"]] += 1
    output = defaultdict(list)
    seen = set()
    for i, r in enumerate(records):
        identity = (r["image"], r["question"], r["answer"])
        if identity in seen:
            continue
        seen.add(identity)
        rng = random.Random(f"{seed}:{i}")
        base = {"image": r["image"], "state": "", "instructions": r["question"],
                "source": r["source"], "supervision": "upstream_answer"}
        if r["criteria"]:
            # Opaque keys prevent answer-letter priors; descriptions carry semantics.
            items = list(r["criteria"].items()); rng.shuffle(items)
            criteria = {f"option_{j}": desc for j, (_, desc) in enumerate(items)}
            label = next(f"option_{j}" for j, (name, _) in enumerate(items) if name == r["label"])
            row = {**base, "type": "choice", "criteria": criteria, "label": label}
        elif r["answer"].lower() in ("yes", "no"):
            row = {**base, "type": "noul", "value": float(r["answer"].lower() == "yes")}
        else:
            form, answer = shape(r["answer"]), canonical(r["answer"])
            keys = pool_keys(r["question"], r["answer"], r["source"])
            for key in keys:
                alternatives = pools[r["split"], key]
                if sum(canonical(a) != answer for a in alternatives) >= min_pool: break
            alternatives = alternatives.copy()
            if keys[2][1] == "color":
                for value in COLORS: alternatives.setdefault(value, 1)
            elif keys[2][1] == "number" and r["answer"].isdigit():
                value = int(r["answer"])
                for d in range(-4, 5): alternatives.setdefault(str(max(0, value + d)), 1)
            # Same surface form only, and never a synonym of the answer.
            valid = sorted(a for a in alternatives if shape(a) == form and canonical(a) != answer)
            if len(valid) < 1:
                continue
            count = min(rng.choice([2, 4, 4, 4, 8]), len(valid) + 1)
            # Match the training answer frequency. Uniform negatives make common
            # answers recognizable as positive without looking at the image.
            weights = [alternatives[a] for a in valid]
            negatives = []
            for _ in range(20):
                for value in rng.choices(valid, weights=weights, k=16):
                    if canonical(value) not in {canonical(v) for v in negatives}: negatives.append(value)
                    if len(negatives) == count - 1: break
                if len(negatives) == count - 1: break
            if len(negatives) < count - 1:
                for value in rng.sample(valid, len(valid)):
                    if len(negatives) == count - 1: break
                    if canonical(value) not in {canonical(v) for v in negatives}: negatives.append(value)
            if not negatives:
                continue
            options = [r["answer"], *negatives]
            rng.shuffle(options)
            criteria = {f"option_{j}": a for j, a in enumerate(options)}
            label = f"option_{options.index(r['answer'])}"
            row = {**base, "type": "choice", "criteria": criteria, "label": label,
                   "supervision": "upstream_answer_generated_distractors"}
        output[r["split"]].append(row)
    return output


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/general-v5")
    ap.add_argument("--parts", default=DEFAULT_PARTS)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--aesthetic-rows", type=int, default=6000)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--reuse-sources", help="reuse prepared source records and images without downloading")
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    reuse = Path(args.reuse_sources).resolve() if args.reuse_sources else None
    if reuse:
        previous = json.loads((reuse / "manifest.json").read_text())
        if previous["seed"] != args.seed: raise ValueError("reused source splits require the same seed")
        revision = previous["revision"]; files = []
    else:
        api = HfApi()
        revision = api.dataset_info(REPO, revision=args.revision).sha
        files = api.list_repo_files(REPO, repo_type="dataset", revision=revision)
    records, counts = [], {}
    for spec in args.parts.split(","):
        source, cap = spec.split(":"); cap = int(cap)
        cached = (reuse or out) / f"source-{source}.jsonl"
        if cached.exists():
            part = [json.loads(l) for l in cached.read_text().splitlines()]
            records.extend(part); counts[source] = len(part)
            print(f"[prepare] {source}: reused {len(part)} questions", flush=True)
            continue
        if reuse: raise ValueError(f"missing source cache {cached}")
        shards = sorted(f for f in files if f.startswith(source + "/") and f.endswith(".parquet"))
        if not shards:
            raise RuntimeError(f"no parquet shards for {source}")
        part = []
        visited = 0
        for shard in shards:
            print(f"[prepare] downloading {shard}", flush=True)
            local = hf_hub_download(REPO, shard, repo_type="dataset", revision=revision)
            for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
                for raw in batch.to_pylist():
                    if len(raw["images"]) != 1:
                        continue
                    turns = [p for t in raw["texts"] if (p := parse_turn(t))]
                    if not turns:
                        continue
                    blob = raw["images"][0]["bytes"]
                    try:
                        with Image.open(io.BytesIO(blob)) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                            image_path = out / "images" / f"{digest}.jpg"
                            if not image_path.exists():
                                im.thumbnail((1600, 1600))
                                im.save(image_path, quality=95)
                    except (OSError, ValueError):
                        continue
                    random.Random(f"{args.seed}:{digest}").shuffle(turns)
                    for q, a, criteria, label in turns[:args.max_turns]:
                        part.append({"image": str(image_path), "question": q, "answer": a,
                                     "criteria": criteria, "label": label, "source": source,
                                     "split": split_for(digest, args.seed)})
                    visited += 1
                    if visited % 500 == 0:
                        print(f"[prepare] {source}: {visited}/{cap} images, {len(part)} questions", flush=True)
                    if visited >= cap: break
                if visited >= cap: break
            if visited >= cap: break
        temp = cached.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(r) + "\n" for r in part)); temp.replace(cached)
        records.extend(part); counts[source] = len(part)
    rows = make_rows(records, args.seed)
    # Retain a small Score component. Original image partitions stay intact.
    for split in ("train", "val", "calib"):
        path = Path("data/mix2") / f"{split}.jsonl"
        if path.exists() and args.aesthetic_rows:
            score = [json.loads(l) for l in path.read_text().splitlines() if json.loads(l)["type"] == "score"]
            rng = random.Random(f"{args.seed}:score:{split}")
            for r in rng.sample(score, min(len(score), args.aesthetic_rows if split == "train" else 250)):
                r["image"] = str((Path("data") / r["image"]).resolve())
                rows[split].append(r)
    # Use unused calibration-source images for an independent Score test as well.
    old_calib = Path("data/mix2/calib.jsonl")
    if old_calib.exists() and args.aesthetic_rows:
        used = {str(Path(r["image"]).resolve()) for rr in rows.values() for r in rr}
        extra = []
        for line in old_calib.read_text().splitlines():
            r = json.loads(line)
            image = str((Path("data") / r["image"]).resolve())
            if r["type"] == "score" and image not in used:
                r["image"] = image; extra.append(r)
        random.Random(f"{args.seed}:score:test").shuffle(extra)
        rows["test"].extend(extra[:250])
    # Verify canonical paths cannot leak, including aliases in the old score mixture.
    sets = {s: {str(Path(r["image"]).resolve()) for r in rr} for s, rr in rows.items()}
    for s, paths in sets.items():
        for other in sets:
            if other != s and paths & sets[other]:
                raise RuntimeError(f"image leakage between {s} and {other}")
    manifest = {"repo": REPO, "revision": revision, "seed": args.seed, "source_questions": counts,
                "distractor_sampling": "training-answer-frequency", "source_cache": str(reuse or out),
                "note": "Internal adapter splits; backbone pretraining may overlap. Generated distractors require external validation.",
                "splits": {}}
    for split, rr in rows.items():
        random.Random(f"{args.seed}:{split}").shuffle(rr)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rr))
        manifest["splits"][split] = {"rows": len(rr), "images": len(sets[split]),
            "unique_questions": len({r["instructions"] for r in rr}),
            "types": dict(Counter(r["type"] for r in rr)),
            "sources": dict(Counter(r["source"] for r in rr)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    temp = out / "manifest.tmp"
    temp.write_text(json.dumps(manifest, indent=2)); temp.replace(out / "manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
