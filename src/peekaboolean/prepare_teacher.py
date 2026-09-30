"""Typed requests written and labelled by a large local VLM teacher.

The public QA sources teach the three primitives in a benchmark dialect: no state,
bare answer strings as options, Score only for photo aesthetics, Noul only as hard
0/1. Served requests look nothing like that. Here a teacher (Qwen3-VL-30B-A3B via
vLLM, run locally so no image leaves the machine) does two separate jobs per image:

1. author -- write one request in the served shape: a state (sentence, persona,
   structured object, or empty) and 3-6 named questions of a prescribed type mix,
   with descriptive choice options, ordinal score rubrics and noul wordings. Some
   nouls are steered toward a false answer so yes/no stays balanced.
2. label  -- answer every question as a lettered multiple choice and read the
   teacher's next-token probabilities over the letters. Each question is asked in
   two option orders; the distributions are mapped back and averaged, so a letter
   or position preference cancels instead of becoming a label. Questions whose two
   orders disagree, or where the letters carry little probability mass, are dropped.
3. blind    -- ask once more with no image. A question the teacher answers confidently
   and identically without looking is answerable from its wording and is dropped
   (--blind-threshold). Score questions are steered to a random target level, and some
   nouls get a "<name>_negated" twin, labelled independently (v7).

Labels are the teacher's beliefs, not ground truth. They are soft, which is what a
calibrated student wants, and the authoring and labelling prompts are separate so the
author's intent ("this should be false") never reaches the labeller.

Runs in the teacher environment (vLLM), not the project environment:
  VLLM_USE_FLASHINFER_SAMPLER=0 /path/to/vllm-env/bin/python src/peekaboolean/prepare_teacher.py --out data/teacher-v7
Resumable: images already present in the output are skipped.

Two teachers (docs/alternative-backbones.md §9b): the mean of Qwen3.6-35B-A3B and
Qwen3-VL-30B-A3B beat either alone on every question type. Author and label with the
first, then --relabel-from lets the second answer the same kept questions and appends its
two views; prepare_v6 averages all four. Run the calibration the same way, with identical
--splits-from / --seed / --calibrate, so its rows line up:
  prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 --out data/teacher-v9
  prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 --out data/teacher-v9 --splits-from data/general-v6 --calibrate 6000
  prepare_teacher.py --relabel-from data/teacher-v9 --out data/teacher-v9-2t
  prepare_teacher.py --relabel-from data/teacher-v9 --out data/teacher-v9-2t --splits-from data/general-v6 --calibrate 6000
FP8 checkpoints need VLLM_USE_DEEP_GEMM=0 on hosts without a CUDA toolkit.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

USE_CASES = [
    "an online marketplace checking seller listing photos",
    "a content moderation queue for a social network",
    "an accessibility tool describing images for blind users",
    "an insurance claim intake reviewing damage photos",
    "a real-estate site vetting property photos",
    "a document-processing pipeline triaging scanned pages",
    "a QA engineer checking app and website screenshots",
    "a retail audit of shelves, signage and packaging",
    "a photo library that auto-tags and sorts pictures",
    "a news desk verifying user-submitted pictures",
    "a workplace safety inspection",
    "a museum or archive cataloguing its collection",
    "a teacher grading diagrams and worksheets",
    "a financial analyst extracting facts from charts",
    "a travel site choosing hero images",
    "a food delivery app checking dish photos",
    "a fleet manager reviewing vehicle photos",
    "a stock-photo agency rating submissions",
    "a smart-home camera summarizing what it sees",
    "a recruiter screening portfolio images",
    "a customer-support bot reading user screenshots",
    "an e-commerce catalogue normalizing product attributes",
    "a city office triaging citizen reports with photos",
    "a researcher filtering a dataset for quality",
    "a marketing team checking brand guideline compliance",
    "a data-entry team reading forms, receipts and tables",
    "a sports analytics team reviewing match pictures",
    "a wildlife and outdoor app identifying scenes",
    "a UX researcher auditing interface clarity",
    "a print shop checking artwork before printing",
]

STATE_STYLES = {
    "empty": "The state is the empty string \"\".",
    "sentence": "The state is one or two plain sentences of context or instructions for the whole request.",
    "persona": "The state gives the evaluator a role and a goal, e.g. \"You are ... Your job is to ...\".",
    "object": "The state is a JSON object with 2-4 fields of context (e.g. task, audience, policy, "
              "a claim the uploader made about the image). A claim may be false; a question may test it.",
}

# v0.4 (--unknown): states the questions must be applied to, and a can't-tell option in every
# labelling view whose mass becomes the row's `unknown` target.
GROUNDED_STYLES = {
    "record": "The state is a JSON object: a record the caller already holds about what the image should show "
              "(a product listing, an order, an inspection form, an insurance claim, a catalogue entry) with 3-6 "
              "concrete fields such as colour, count, brand, size, condition, printed text or date. The fields are "
              "the caller's claims: make one or two of them wrong for THIS image and the rest right.",
    "rule": "The state is a rule or policy the caller applies, with concrete conditions: a numeric limit on a count, "
            "size, amount or price, a required or banned element, a list of allowed values, or an exception to one of "
            "these. Invent the rule for this use case; do not copy these words.",
}
UNKNOWN_OPTION = "Cannot be determined from the image and the context"

LETTERS = string.ascii_uppercase
BLIND_NOTE = "(No image is available. Give your best guess from the text alone.)\n\n"


def author_prompt(rng: random.Random, grounded: bool = False) -> tuple[str, dict]:
    use_case = rng.choice(USE_CASES)
    styles = {**STATE_STYLES, **(GROUNDED_STYLES if grounded else {})}
    state_style = rng.choices(list(styles), weights=[2, 3, 3, 2] + ([4, 4] if grounded else []))[0]
    n = rng.randint(3, 6)
    types = [rng.choices(["choice", "score", "noul"], weights=[4, 3, 3])[0] for _ in range(n)]
    # One question per request, sometimes, whose honest answer is "can't tell"; the labeller decides.
    cant_tell = rng.randrange(n) if grounded and rng.random() < 0.35 else None
    lines = []
    for i, t in enumerate(types, 1):
        if i - 1 == cant_tell:
            lines.append(f"{i}. {t} whose honest answer from this image and state CANNOT be determined: ask about "
                         "something hidden, cut off, too small to read, outside the frame, or not given in the state")
            continue
        if t == "choice":
            k = rng.choice([2, 3, 3, 4, 4, 5, 6])
            lines.append(f"{i}. choice with exactly {k} options")
        elif t == "score":
            k = rng.choice([2, 3, 3, 4, 5, 5, 5, 7, 10])
            # Without a target the teacher picks properties this image scores high on, so
            # the wording alone predicts the answer. A uniform target level breaks that.
            level = rng.randint(1, k)
            lines.append(f"{i}. score with exactly {k} levels, about a property on which THIS image sits at "
                         f"level {level} of {k} (1 = lowest)")
        else:
            # The blind filter mostly removes "is there X?" questions whose answer is no
            # (a prior can guess those), so author more of them to keep yes/no balanced.
            want = rng.choices(["true", "false"], weights=[35, 65])[0]
            wording = rng.random() < 0.4
            negated = rng.random() < 0.3
            lines.append(f"{i}. noul whose correct answer for THIS image is {want}"
                         + (" (give custom true/false wording)" if wording else "")
                         + (f'; then ALSO add a question named "<same name>_negated" asking the logical opposite'
                            if negated else ""))
    text = f"""You write evaluation requests for an image-understanding API. The caller is {use_case}.
Look carefully at the image, then write ONE request about it as JSON.

Format:
{{"state": <string or object>,
  "questions": {{
    "<snake_case_name>": {{"type": "choice", "instructions": "<question>", "criteria": {{"<option_key>": "<what this option means>", ...}}}},
    "<snake_case_name>": {{"type": "score", "instructions": "<question>", "criteria": ["<lowest level>", ..., "<highest level>"]}},
    "<snake_case_name>": {{"type": "noul", "instructions": "<yes/no question>", "criteria": {{"true": "<meaning of yes>", "false": "<meaning of no>"}}}}
  }}}}

Rules:
- {styles[state_style]}
- Write exactly these questions, in this order:
{chr(10).join('  ' + l for l in lines)}
- The state is what the caller knows BEFORE anyone looks at the image: who they are, what they need,
  their policy{", their records" if grounded else ""}. It must never describe, summarize or hint at what the image shows{
  " (a record's fields are claims to test, not a description)" if grounded else ""}.{
  chr(10) + "- At least two questions must apply the state to the image: for a record, whether a named field matches"
  " the image or which field it contradicts; for a rule, whether the image meets it, which clause or exception"
  " applies, or which listed value the image shows." if state_style in GROUNDED_STYLES else ""}
- Every question must be answerable only by looking at this image (plus the state). Ask about what is
  specific here: objects, people, text, numbers, layout, colours, condition, quality, style, intent.
- The answer must never be readable from the text alone. Option descriptions, levels and yes/no
  wordings are generic definitions that would read the same for any image: never mention what this
  image contains, never say which option applies, never write "none of the items" or quote the image.
- choice: option keys are short snake_case identifiers; each description says what that option means.
  Exactly one option should fit best, but make the others plausible, not absurd.
- score: levels are an ordered rubric from lowest to highest of one clear property (amount, quality,
  severity, legibility, clutter, size, confidence...). Each level is a short descriptive phrase.
- noul: a yes/no question. "criteria" is optional; omit it unless custom wording was requested.
  For an answer of false, ask about something plausible that is absent, wrong or different here.
- If people are visible, make at least one question about a specific person ("the woman", "the child",
  "the man on the left"): whether such a person is present, what they do or wear, or their approximate
  age (as a score whose levels are age ranges in years, youngest first).
- Vary phrasing; do not start every question the same way. Do not mention these rules.
Output only the JSON."""
    return text, {"use_case": use_case, "state_style": state_style, "types": types, "cant_tell": cant_tell}


def validate_request(req, types) -> dict | None:
    """Mirror serve.request_examples; anything the server would reject is dropped here."""
    if not isinstance(req, dict) or not isinstance(req.get("questions"), dict):
        return None
    state = req.get("state", "")
    if not isinstance(state, (str, dict)):
        return None
    out = {}
    for name, q in req["questions"].items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
            continue
        instr = q.get("instructions")
        if not isinstance(instr, str) or not 3 <= len(instr) <= 400:
            continue
        c = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= 8 or not all(isinstance(v, str) and v for v in c.values()):
                continue
            if len({v.strip().lower() for v in c.values()}) != len(c):
                continue
        elif q["type"] == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= 10 or not all(isinstance(v, str) and v for v in c):
                continue
        else:
            if c is not None and (not isinstance(c, dict) or set(c) - {"true", "false"}
                                  or not all(isinstance(v, str) for v in c.values())):
                continue
        out[str(name)] = {"type": q["type"], "instructions": instr, **({"criteria": c} if c is not None else {})}
    if len(out) < 2:
        return None
    return {"state": state, "questions": out}


def render(value) -> str:
    # Same contract as data.render; duplicated so this script runs outside the project env.
    if value is None: return ""
    if isinstance(value, str): return value
    return json.dumps(value, ensure_ascii=False)


def label_views(state, q, unknown: bool = False) -> list[tuple[str, list[int]]]:
    """Two lettered views of one question. Each returns (prompt, order) where letter j
    stands for canonical option order[j]. With `unknown`, the can't-tell option is the last
    canonical option; the reversed view moves it to the front."""
    if q["type"] == "choice":
        items = list(q["criteria"].items())
        texts = [f"{k}: {v}" for k, v in items]
    elif q["type"] == "score":
        texts = list(q["criteria"])
    else:
        c = q.get("criteria") or {}
        texts = [f"No{' - ' + c['false'] if c.get('false') else ''}",
                 f"Yes{' - ' + c['true'] if c.get('true') else ''}"]
    if unknown:
        texts = texts + [UNKNOWN_OPTION]
    n = len(texts)
    first = list(range(n))
    second = list(reversed(first))   # every option changes letter, except a middle one
    views = []
    s = render(state).strip()
    for order in (first, second):
        opts = "\n".join(f"{LETTERS[j]}. {texts[i]}" for j, i in enumerate(order))
        kind = {"choice": "Choose the option that best answers the question.",
                "score": "Choose the level of the scale that best fits.",
                "noul": "Answer the yes/no question."}[q["type"]]
        prompt = ((f"Context: {s}\n\n" if s else "")
                  + f"Question: {q['instructions']}\n{kind}\nOptions:\n{opts}\n\n"
                  + "Reply with the letter of your answer only.")
        views.append((prompt, order))
    return views


def letter_probs(logprobs: dict, n: int) -> tuple[list[float], float]:
    mass = [0.0] * n
    for lp in logprobs.values():
        tok = (lp.decoded_token or "").strip()
        if len(tok) == 1 and tok in LETTERS[:n]:
            mass[LETTERS.index(tok)] += math.exp(lp.logprob)
    total = sum(mass)
    return ([m / total for m in mass] if total > 0 else [1 / n] * n), total


def load_image(path: str, max_edge: int) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            return im
    except (OSError, ValueError):
        return None


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    # The teacher's commonest slip is dropping the final closing brace(s).
    for suffix in ("", "}", "}}"):
        try:
            return json.loads(m.group(0) + suffix)
        except json.JSONDecodeError:
            continue
    return None


def chat(llm, jobs, params):
    # Thinking models (Qwen3.5/3.6) would answer "<think>" first; others ignore the switch.
    return llm.chat(jobs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})


def belief(row) -> dict:
    """Label fields from the mean of all teacher views (two option orders per teacher)."""
    views = row["teacher_views"]
    p = [sum(x) / len(views) for x in zip(*views)]
    if row.get("with_unknown"):
        row["unknown"], p = split_unknown(p)
    if row["type"] == "choice":
        keys = list(row["criteria"])
        row.update(target_probs=dict(zip(keys, p)), label=keys[max(range(len(p)), key=p.__getitem__)])
    elif row["type"] == "score":
        row["hist"] = p
    else:
        row["value"] = p[1]
    return row


def split_unknown(p: list[float]) -> tuple[float, list[float]]:
    """(can't-tell mass, distribution over the caller's options given an answer)."""
    rest = p[:-1]
    total = sum(rest)
    return p[-1], ([x / total for x in rest] if total > 0 else [1 / len(rest)] * len(rest))


def relabel(llm, label_params, args, out):
    """Second teacher: ask every kept question of --relabel-from again, in both option
    orders, and append these views to the first teacher's. The mass and disagreement
    filters apply to the new views too; the blind check stays the first teacher's."""
    done_path = out / "done.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    stats = Counter()
    pool = ThreadPoolExecutor(16)
    for split in ("train", "val", "calib", "test"):
        source = Path(args.relabel_from) / f"{split}.jsonl"
        if not source.exists(): continue
        by_image = {}
        for line in source.read_text().splitlines():
            r = json.loads(line); by_image.setdefault(r["image"], []).append(r)
        todo = [p for p in by_image if p not in done]
        with (out / f"{split}.jsonl").open("a") as fh:
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start:start + args.chunk]
                pics = list(pool.map(lambda p: load_image(p, args.max_edge), chunk))
                jobs, index = [], []
                for image, pic in zip(chunk, pics):
                    if pic is None: stats["bad_image"] += 1; continue
                    for r in by_image[image]:
                        q = {"type": r["type"], "instructions": r["instructions"],
                             **({"criteria": r["criteria"]} if "criteria" in r else {})}
                        for prompt, order in label_views(r["state"], q, r.get("with_unknown", False)):
                            jobs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": pic},
                                                                      {"type": "text", "text": prompt}]}])
                            index.append((r, order))
                views = {}
                for (r, order), result in zip(index, chat(llm, jobs, label_params)):
                    probs, mass = letter_probs(result.outputs[0].logprobs[0], len(order))
                    canon = [0.0] * len(order)
                    for j, i in enumerate(order): canon[i] = probs[j]
                    views.setdefault(id(r), (r, []))[1].append((canon, mass))
                for r, ((a, ma), (b, mb)) in views.values():
                    if min(ma, mb) < args.min_mass: stats["low_mass"] += 1; continue
                    if 0.5 * sum(abs(x - y) for x, y in zip(a, b)) > args.max_disagreement:
                        stats["disagree"] += 1; continue
                    r["teacher_views"] += [[round(x, 6) for x in a], [round(x, 6) for x in b]]
                    r["relabelled_by"] = args.model
                    fh.write(json.dumps(belief(r), ensure_ascii=False) + "\n")
                    stats[f"kept_{r['type']}"] += 1
                fh.flush()
                with done_path.open("a") as d:
                    d.write("".join(p + "\n" for p in chunk))
                print(f"[teacher] relabel {split} {start + len(chunk)}/{len(todo)} images {dict(stats)}", flush=True)


def calibrate(llm, label_params, args, out):
    """Teacher letter probabilities on public rows whose answer is known (exact labels
    only), written with the truth so prepare_v6 can fit one temperature per type."""
    rows = [json.loads(l) for l in (Path(args.splits_from) / "calib.jsonl").read_text().splitlines()]
    # Exact labels only: public choice/noul and count rubrics (a one-hot hist). Teacher
    # rows in the same split are beliefs, not truth, and must not calibrate the teacher.
    rows = [r for r in rows if r.get("source") != "teacher" and (
            (r["type"] in ("choice", "noul") and r.get("value", 0) in (0, 1))
            or (r["type"] == "score" and r.get("criteria") and max(r["hist"]) == 1.0))]
    random.Random(args.seed).shuffle(rows); rows = rows[:args.calibrate]
    pool = ThreadPoolExecutor(16)
    pics = list(pool.map(lambda r: load_image(r["image"], args.max_edge), rows))
    jobs, index = [], []
    for n, (r, pic) in enumerate(zip(rows, pics)):
        if pic is None: continue
        if r["type"] == "choice":
            q = {"type": "choice", "instructions": r["instructions"], "criteria": r["criteria"]}
            truth = list(r["criteria"]).index(r["label"])
        elif r["type"] == "score":
            q = {"type": "score", "instructions": r["instructions"], "criteria": r["criteria"]}
            truth = r["hist"].index(1.0)
        else:
            q = {"type": "noul", "instructions": r["instructions"]}
            truth = int(r["value"])
        for prompt, order in label_views("", q, args.unknown):
            jobs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": pic},
                                                      {"type": "text", "text": prompt}]}])
            index.append((n, r["type"], r["source"], truth, order))
    results = chat(llm, jobs, label_params)
    merged = {}
    for (n, kind, source, truth, order), result in zip(index, results):
        probs, mass = letter_probs(result.outputs[0].logprobs[0], len(order))
        canon = [0.0] * len(order)
        for j, i in enumerate(order): canon[i] = probs[j]
        merged.setdefault(n, {"type": kind, "source": source, "truth": truth, "views": [], "mass": []})
        merged[n]["views"].append(canon); merged[n]["mass"].append(mass)
    rows = list(merged.values())
    if args.relabel_from:
        first = [json.loads(l) for l in (Path(args.relabel_from) / "teacher-calibration.jsonl").read_text().splitlines()]
        key = lambda r: (r["type"], r["source"], r["truth"])
        if len(first) != len(rows) or any(key(a) != key(b) for a, b in zip(first, rows)):
            raise ValueError("calibration rows differ from --relabel-from's: use the same --splits-from, --seed, --calibrate")
        if any(len(a["views"][0]) != len(b["views"][0]) for a, b in zip(first, rows)):
            raise ValueError("calibration views differ in length from --relabel-from's: use the same --unknown")
        for a, b in zip(first, rows):
            b["views"], b["mass"] = a["views"] + b["views"], a["mass"] + b["mass"]
    with (out / "teacher-calibration.jsonl").open("w") as fh:
        for row in rows: fh.write(json.dumps(row) + "\n")
    acc = sum(max(range(len(r["views"][0])), key=lambda i: sum(v[i] for v in r["views"])) == r["truth"]
              for r in rows) / max(1, len(rows))
    print(f"[teacher] calibration rows {len(rows)}, teacher accuracy {acc:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-from", default="data/general-v5", help="dataset whose image->split assignment is reused")
    ap.add_argument("--out", default="data/teacher-v6")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="images per split cap (0 = all)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-edge", type=int, default=896)
    ap.add_argument("--gpu-memory", type=float, default=0.80)
    ap.add_argument("--max-disagreement", type=float, default=0.5, help="drop if the two views' TV distance exceeds this")
    ap.add_argument("--min-mass", type=float, default=0.6)
    ap.add_argument("--blind-threshold", type=float, default=0.8,
                    help="drop questions the teacher answers this confidently (and the same way) without the image; 1 = off")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--calibrate", type=int, default=0,
                    help="instead: label N public calib rows with known answers, for fitting a teacher temperature")
    ap.add_argument("--relabel-from", help="instead: second teacher for this teacher output's kept questions "
                                           "(with --calibrate: for its teacher-calibration.jsonl)")
    ap.add_argument("--unknown", action="store_true",
                    help="v0.4: offer a can't-tell option when labelling (its mass becomes `unknown`) and author "
                         "record/rule states and some can't-tell questions")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    engine = lambda: LLM(args.model, gpu_memory_utilization=args.gpu_memory, max_model_len=8192,
                         limit_mm_per_prompt={"image": 1}, enable_prefix_caching=True, seed=args.seed,
                         max_num_seqs=256)
    label_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    if args.calibrate:
        return calibrate(engine(), label_params, args, out)
    if args.relabel_from:
        return relabel(engine(), label_params, args, out)
    images = {}
    for split in ("train", "val", "calib", "test"):
        for line in (Path(args.splits_from) / f"{split}.jsonl").read_text().splitlines():
            r = json.loads(line)
            images.setdefault(str(Path(r["image"]).resolve()), (split, r.get("source", "unknown")))
    by_split = {}
    for path, (split, source) in images.items():
        by_split.setdefault(split, []).append((path, source))
    todo = []
    for split, items in by_split.items():
        items.sort(); random.Random(f"{args.seed}:{split}").shuffle(items)
        todo += [(p, split, s) for p, s in (items[:args.limit] if args.limit else items)]
    random.Random(args.seed).shuffle(todo)   # mix sources inside each chunk
    done_path = out / "done.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    todo = [t for t in todo if t[0] not in done]
    print(f"[teacher] {len(todo)} images to process ({len(done)} already done)", flush=True)
    if not todo: return

    llm = engine()
    author_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=900, seed=args.seed)
    stats = Counter()
    pool = ThreadPoolExecutor(16)
    files = {s: (out / f"{s}.jsonl").open("a") for s in ("train", "val", "calib", "test")}
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        pics = list(pool.map(lambda t: load_image(t[0], args.max_edge), chunk))
        jobs, metas = [], []
        for (path, split, source), pic in zip(chunk, pics):
            if pic is None: stats["bad_image"] += 1; continue
            rng = random.Random(f"{args.seed}:author:{path}")
            text, meta = author_prompt(rng, args.unknown)
            jobs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": pic},
                                                      {"type": "text", "text": text}]}])
            metas.append((path, split, source, pic, meta))
        authored = chat(llm, jobs, author_params)
        requests = []
        for (path, split, source, pic, meta), result in zip(metas, authored):
            req = validate_request(parse_json(result.outputs[0].text), meta["types"])
            if req is None:
                stats["bad_request"] += 1
                with (out / "rejected.jsonl").open("a") as fh:
                    fh.write(json.dumps({"image": path, "meta": meta, "text": result.outputs[0].text}) + "\n")
                continue
            requests.append((path, split, source, pic, meta, req))
        # Label: two views per question, image first so the prefix cache shares it, plus
        # one view without the image. A question the teacher answers confidently blind is
        # answerable from its wording, and would teach the student to skip the picture.
        jobs, index = [], []
        for ri, (path, split, source, pic, meta, req) in enumerate(requests):
            for name, q in req["questions"].items():
                views_q = label_views(req["state"], q, args.unknown)
                for vi, (prompt, order) in enumerate(views_q):
                    jobs.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": pic},
                                                              {"type": "text", "text": prompt}]}])
                    index.append((ri, name, order, "seen"))
                if args.blind_threshold < 1:
                    # Blind, the can't-tell option would always win; without it the question is
                    # whether the wording alone gives the answer away, as before.
                    prompt, order = label_views(req["state"], q)[0]
                    jobs.append([{"role": "user", "content": [{"type": "text", "text": BLIND_NOTE + prompt}]}])
                    index.append((ri, name, order, "blind"))
        labelled = chat(llm, jobs, label_params)
        views, blind = {}, {}
        for (ri, name, order, kind), result in zip(index, labelled):
            probs, mass = letter_probs(result.outputs[0].logprobs[0], len(order))
            canon = [0.0] * len(order)
            for j, i in enumerate(order): canon[i] = probs[j]
            if kind == "blind": blind[(ri, name)] = canon
            else: views.setdefault((ri, name), []).append((canon, mass))
        for ri, (path, split, source, pic, meta, req) in enumerate(requests):
            for name, q in req["questions"].items():
                (a, ma), (b, mb) = views[(ri, name)]
                if min(ma, mb) < args.min_mass: stats["low_mass"] += 1; continue
                tv = 0.5 * sum(abs(x - y) for x, y in zip(a, b))
                if tv > args.max_disagreement: stats["disagree"] += 1; continue
                p = [(x + y) / 2 for x, y in zip(a, b)]
                seen_best = max(range(len(p)), key=p.__getitem__)
                sightless = blind.get((ri, name))
                cant_tell = args.unknown and seen_best == len(p) - 1
                if sightless is not None and not cant_tell and max(sightless) >= args.blind_threshold \
                        and max(range(len(sightless)), key=sightless.__getitem__) == seen_best:
                    stats["text_answerable"] += 1; continue
                stats["cant_tell"] += cant_tell
                row = {"image": path, "type": q["type"], "state": req["state"], "instructions": q["instructions"],
                       "source": "teacher", "image_source": source, "question_name": name,
                       "use_case": meta["use_case"], "supervision": "teacher_logprobs", "view_tv": round(tv, 4),
                       "teacher_views": [[round(x, 6) for x in a], [round(x, 6) for x in b]]}
                if sightless is not None: row["teacher_blind"] = [round(x, 6) for x in sightless]
                if args.unknown: row.update(with_unknown=True, state_style=meta["state_style"],
                                            steered_cant_tell=meta["cant_tell"] is not None
                                            and list(req["questions"]).index(name) == meta["cant_tell"])
                if name.endswith("_negated"): row["negation_of"] = name[:-len("_negated")]
                if "criteria" in q: row["criteria"] = q["criteria"]
                files[split].write(json.dumps(belief(row), ensure_ascii=False) + "\n")
                stats[f"kept_{q['type']}"] += 1
        for f in files.values(): f.flush()
        with done_path.open("a") as fh:
            fh.write("".join(t[0] + "\n" for t in chunk))
        print(f"[teacher] {start + len(chunk)}/{len(todo)} images {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
