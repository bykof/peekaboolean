#!/usr/bin/env python3
"""Answer a request: one state, many typed questions, in the shape the API returns.

  python -m peekaboolean.serve --adapter runs/probe-ava/final --model Qwen/Qwen3-VL-4B-Instruct \
      --image data/ava/images/1.jpg --demo
  python -m peekaboolean.serve --adapter ... --image ... --request req.json --mode shared
  python -m peekaboolean.serve --adapter ... --image ... --check --bench 8x5,1x64

A request carries one state and a map of named questions, each Choice, Score or Noul
with its own criteria. Every question is answered against that state and in isolation
from the others -- not as a promise to keep, but because each candidate is its own
sequence, so nothing another question said can reach this one.

Batching therefore changes cost, not answers. The naive path spends one forward per
question over K candidate sequences that each carry their own copy of the image, so a
request of N questions re-encodes the picture N*K times. `--mode shared` runs the state
once, keeps its KV cache, and scores every candidate as a short suffix against it: one
vision pass for the whole request, whatever N and K are. `--check` asserts the two
agree, because an optimisation that quietly moves the numbers is worse than none.

Questions are built through `data.to_example`, the same function the trainer uses, so a
rubric cannot reach the model in one shape during training and another when served.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import Example, build_inputs, candidate_prompts, candidate_signs, load_image, to_example
from .model import CandidateScorer, QWEN_FAMILIES, select_device, device_dtype, configure_image_size

# A served question carries no label. `to_example` computes a training target it will
# never use here, so hand it a neutral one rather than teaching the row schema to
# tolerate a missing `hist` -- from a training file that has to stay an error.
def _stub_label(qtype: str, n: int) -> dict:
    if qtype == "score":
        return {"hist": [1.0] * max(n, 2)}
    if qtype == "choice":
        return {"label": 0}
    return {"value": 0.5}


def request_examples(state, questions: dict, image: str | Path) -> list[tuple[str, Example]]:
    """One Example per question, in request order."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty map")
    out = []
    for qid, q in questions.items():
        criteria = q.get("criteria")
        n = len(criteria) if isinstance(criteria, (list, dict)) else 2
        if q["type"] == "choice" and (not isinstance(criteria, dict) or not 2 <= n <= 255):
            raise ValueError("choice requires 2..255 named options")
        if q["type"] == "score" and (not isinstance(criteria, list) or not 2 <= n <= 10):
            raise ValueError("score requires 2..10 ordered levels")
        if q["type"] == "noul" and criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            raise ValueError("noul criteria may contain only true and false")
        row = {"image": str(image), "type": q["type"], "state": state,
               "instructions": q["instructions"], **_stub_label(q["type"], n)}
        if criteria is not None:
            row["criteria"] = criteria
        if "attribute" in q:
            row["attribute"] = q["attribute"]
        out.append((qid, to_example(row, augment=False)))
    return out


# --------------------------------------------------------------------------- answers

def confidence(p: torch.Tensor) -> float:
    """How concentrated the distribution is, on 0..1: (K*pmax - 1) / (K - 1).

    1.0 is all the mass on one outcome, 0.0 is an even split. It is a reading of the
    shape, not a second opinion about the answer, so a caller who wants a different
    reading has `probabilities` and can compute their own.
    """
    k = p.numel()
    return 1.0 if k < 2 else float((k * p.max() - 1) / (k - 1))


def answer_for(ex: Example, probs: torch.Tensor) -> dict:
    names = ex.names or [str(i) for i in range(probs.numel())]
    table = {n: round(float(v), 6) for n, v in zip(names, probs)}
    if ex.qtype == "noul":
        # The docs return a bare probability here: no argmax, so no confidence either.
        return {"type": "noul", "noul": round(float(probs[1]), 6)}
    if ex.qtype == "choice":
        return {"type": "choice", "choice": names[int(probs.argmax())],
                "probabilities": table, "confidence": round(confidence(probs), 6)}
    # score: the level index weighted by its probability, which is why the levels have
    # to be ordered and why the loss that trained them was EMD and not cross-entropy.
    idx = torch.arange(probs.numel(), dtype=probs.dtype)
    return {"type": "score", "score": round(float((probs * idx).sum()), 6),
            "probabilities": table, "confidence": round(confidence(probs), 6),
            "legend": dict(zip(names, ex.candidates))}


# --------------------------------------------------------------------------- backends

def _inner(model: CandidateScorer):
    """The Qwen3VLModel under the PEFT wrapper. LoRA is injected in place, so reaching
    past the wrapper keeps the adapters and only skips the vocabulary projection --
    150k logits per token that a scoring head never looks at."""
    return model.inner


def _pool(model: CandidateScorer, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Head applied at each row's last real token -- the same pooling model.py trains."""
    lengths = mask.sum(dim=1) - 1
    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
    return model.head(pooled.float()).squeeze(-1)


@torch.no_grad()
def score_naive(model, processor, examples, image, max_edge=1024, chunk=32) -> list[torch.Tensor]:
    """One forward per question, K candidate sequences each carrying the image again."""
    device = next(model.head.parameters()).device
    out = []
    for _, ex in examples:
        parts = []
        for lo in range(0, len(ex.candidates), chunk):
            sub = Example(ex.image, ex.state, ex.instructions, ex.candidates[lo:lo + chunk],
                          ex.target, ex.qtype, ex.names)
            batch = build_inputs(sub, processor, max_edge, image=image).to(device)
            hidden = model.hidden(batch)
            parts.append(_pool(model, hidden, batch["attention_mask"]))
        out.append(torch.cat(parts) * _signs([(None, ex)], processor, parts[0].device))
    return out


def _split_point(texts: list[str]) -> int:
    """Where the candidates stop agreeing, backed up to a line break.

    Cutting on a newline keeps the split off a token boundary the tokenizer might merge
    across, so prefix ids + suffix ids are the ids the one-piece prompt would have had.
    """
    common = texts[0]
    for t in texts[1:]:
        i = 0
        while i < min(len(common), len(t)) and common[i] == t[i]:
            i += 1
        common = common[:i]
    cut = common.rfind("\n") + 1
    # Then back off the whole whitespace run. A prefix *ending* in "\n\n" tokenizes it as
    # one token, but inside the full prompt, followed by text, it is two -- so the model
    # would be served ids it never trained on. Starting the suffix at the run keeps
    # prefix ids + suffix ids identical to the one-piece prompt.
    while cut > 0 and common[cut - 1].isspace():
        cut -= 1
    return cut


def _signs(examples, processor, device) -> torch.Tensor:
    style = getattr(processor, "jev_prompt", "judge")
    return torch.tensor([s for _, ex in examples for s in candidate_signs(ex, style)], device=device)


def _expand_cache(cache, b: int):
    """The prefix cache, widened to b rows, as a fresh object the suffix pass may mutate.

    Attention layers get their keys/values expanded (the suffix concatenates onto them,
    which copies). Linear-attention layers (Qwen3.5) hold a conv window and a recurrent
    state that the suffix updates in place, so those are real per-row copies."""
    import copy
    fresh = copy.copy(cache)
    fresh.layers = []
    for layer in cache.layers:
        new = copy.copy(layer)
        if hasattr(layer, "recurrent_states"):
            new.conv_states = {i: t.repeat(b, *([1] * (t.dim() - 1))) for i, t in layer.conv_states.items()}
            new.recurrent_states = {i: t.repeat(b, *([1] * (t.dim() - 1))) for i, t in layer.recurrent_states.items()}
            for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state"):
                setattr(new, name, dict(getattr(layer, name)))
        else:
            new.keys = layer.keys.expand(b, -1, -1, -1)
            new.values = layer.values.expand(b, -1, -1, -1)
        fresh.layers.append(new)
    return fresh


@torch.no_grad()
def score_shared(model, processor, examples, image, max_edge=1024, chunk=32,
                 report: dict | None = None) -> list[torch.Tensor]:
    """Encode the state once; score every candidate of every question against its cache.

    With `report`, stage wall times (ms, device-synchronized) land in report["ms"]."""
    device = next(model.head.parameters()).device
    inner = _inner(model)
    ms = report.setdefault("ms", {}) if report is not None else None
    clock = [time.perf_counter()]
    def mark(stage):
        if ms is None: return
        synchronize(device); now = time.perf_counter()
        ms[stage] = ms.get(stage, 0.0) + (now - clock[0]) * 1000; clock[0] = now

    texts, spans = [], []
    for _, ex in examples:
        prompts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
            tokenize=False, add_generation_prompt=True)
            for t in candidate_prompts(ex, getattr(processor, "jev_prompt", "judge"))]
        spans.append((len(texts), len(texts) + len(prompts)))
        texts.extend(prompts)
    mark("chat_template")

    cut = _split_point(texts)
    img = image if image is not None else load_image(examples[0][1].image, max_edge)
    prefix = processor(text=[texts[0][:cut]], images=[img], return_tensors="pt").to(device)
    mark("image_preprocess")
    # Identical rows (the yes/no head's two noul rows) are scored once.
    unique = list(dict.fromkeys(t[cut:] for t in texts))
    where = torch.tensor([unique.index(t[cut:]) for t in texts])
    suffixes = processor.tokenizer(unique, return_tensors="pt",
                                   padding=True, add_special_tokens=False).to(device)
    mark("tokenize")

    # One pass over image + state. Everything after this is short text.
    if model.family in QWEN_FAMILIES:
        inner.rope_deltas = None
    else:
        prefix.pop("mm_token_type_ids", None)
    pref_out = inner(**prefix, use_cache=True)
    prefix_len = int(prefix["input_ids"].shape[1])
    mark("prefix_forward")

    # Text after the image continues the 3D positions sequentially from the highest one
    # the prefix reached, identically in all three rope sections. Passing them rather
    # than letting the model infer them keeps the suffix independent of any state the
    # model cached during an earlier call.
    if model.family in QWEN_FAMILIES:
        pos_prefix, _ = inner.get_rope_index(
            prefix["input_ids"], image_grid_thw=prefix.get("image_grid_thw"),
            attention_mask=prefix.get("attention_mask"),
            mm_token_type_ids=prefix.get("mm_token_type_ids"))
        start = int(pos_prefix.max()) + 1
    else:
        start = prefix_len
    if report is not None:
        report.update(prefix_tokens=prefix_len, suffix_tokens=int(suffixes["input_ids"].shape[1]),
                      sequences=len(texts))

    scores = []
    for lo in range(0, len(unique), chunk):
        ids = suffixes["input_ids"][lo:lo + chunk]
        mask = suffixes["attention_mask"][lo:lo + chunk]
        b, sl = ids.shape
        full_mask = torch.cat([torch.ones(b, prefix_len, dtype=mask.dtype, device=device), mask], 1)
        positions = start + torch.arange(sl, device=device)
        pos = positions.view(1, 1, -1).expand(3, b, -1) if model.family in QWEN_FAMILIES else positions.view(1, -1).expand(b, -1)
        out = inner(input_ids=ids, attention_mask=full_mask, position_ids=pos,
                    past_key_values=_expand_cache(pref_out.past_key_values, b),
                    cache_position=torch.arange(prefix_len, prefix_len + sl, device=device),
                    use_cache=True)
        scores.append(_pool(model, out.last_hidden_state, mask))
    flat = torch.cat(scores)[where.to(scores[0].device)] * _signs(examples, processor, scores[0].device)
    mark("suffix_forward")
    return [flat[a:b] for a, b in spans]


@torch.no_grad()
def score_single(model, processor, examples, image, max_edge=1024, chunk=None,
                 report: dict | None = None) -> list[torch.Tensor]:
    """Every candidate of every question in ONE forward, the image encoded once.

    Same token ids as score_shared (prefix ids + suffix ids, identical to the one-piece
    prompt), but the prefix is recomputed per row instead of cached. That costs more
    arithmetic and saves a whole second pass -- on Apple MPS a pass of this model is
    dominated by per-kernel launch cost, not by the arithmetic.
    """
    device = next(model.head.parameters()).device
    ms = report.setdefault("ms", {}) if report is not None else None
    clock = [time.perf_counter()]
    def mark(stage):
        if ms is None: return
        synchronize(device); now = time.perf_counter()
        ms[stage] = ms.get(stage, 0.0) + (now - clock[0]) * 1000; clock[0] = now

    texts, spans = [], []
    for _, ex in examples:
        prompts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
            tokenize=False, add_generation_prompt=True)
            for t in candidate_prompts(ex, getattr(processor, "jev_prompt", "judge"))]
        spans.append((len(texts), len(texts) + len(prompts)))
        texts.extend(prompts)
    mark("chat_template")
    cut = _split_point(texts)
    img = image if image is not None else load_image(examples[0][1].image, max_edge)
    prefix = processor(text=[texts[0][:cut]], images=[img], return_tensors="pt")
    mark("image_preprocess")
    unique = list(dict.fromkeys(t[cut:] for t in texts))
    where = torch.tensor([unique.index(t[cut:]) for t in texts])
    suffixes = processor.tokenizer(unique, return_tensors="pt",
                                   padding=True, add_special_tokens=False)
    n = len(unique)
    batch = {"input_ids": torch.cat([prefix["input_ids"].expand(n, -1), suffixes["input_ids"]], 1),
             "attention_mask": torch.cat([prefix["attention_mask"].expand(n, -1), suffixes["attention_mask"]], 1),
             "image_counts": torch.tensor([n])}
    for key in ("pixel_values", "pixel_attention_mask", "image_grid_thw"):
        if key in prefix: batch[key] = prefix[key]
    if "mm_token_type_ids" in prefix:   # Qwen: image positions for 3D rope; suffixes are text
        batch["mm_token_type_ids"] = torch.cat([prefix["mm_token_type_ids"].expand(n, -1),
                                                torch.zeros_like(suffixes["input_ids"])], 1)
    batch = {k: v.to(device) for k, v in batch.items()}
    mark("tokenize")
    flat = model(batch)[where.to(device)] * _signs(examples, processor, device)
    mark("forward")
    return [flat[a:b] for a, b in spans]


# ----------------------------------------------------------------------------- driver

# Up to this many candidate sequences, one pass (score_single) beats prefix-cache + suffix
# (score_shared): it saves a whole pass of launch overhead, and repeating the short prefix
# costs little. Measured on an M1 Pro (MPS, fp32): single wins at 2-4 candidates at every
# size, loses at 28 from 384 px up. Both modes return the same answers.
SINGLE_PASS_MAX = 8
NO_SHARED_CACHE = ()


def load(adapter: str, model_id: str | None = None, device: str = "auto", calibration: str | None = None,
         merge: bool = False, dtype: torch.dtype | None = None):
    device = select_device(device)
    if not Path(adapter).is_dir():
        # A Hub repo id such as "user/model"; huggingface_hub caches the download.
        from huggingface_hub import snapshot_download
        adapter = snapshot_download(adapter, allow_patterns=["*.json", "*.safetensors", "head.pt"])
    if model_id is None:
        model_id = json.loads((Path(adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    model = CandidateScorer(model_id, gradient_checkpointing=False, adapter=adapter,
                            dtype=dtype or device_dtype(device))
    if merge:
        model.backbone = model.backbone.merge_and_unload()
    model = model.to(device).eval()
    calib = Path(calibration) if calibration else Path(adapter) / "calibration.json"
    calibration_data = json.loads(calib.read_text()) if calib.exists() else {}
    if not calibration and not calibration_data:
        calibration_data = {"by_image_size": {str(size): json.loads(path.read_text())
                            for size in (256, 384, 512)
                            if (path := Path(adapter) / f"calibration-{size}.json").exists()}}
    prompt = json.loads(cfg.read_text()).get("prompt", "judge") if (cfg := Path(adapter) / "scorer_config.json").exists() else "judge"
    return (model, CandidateScorer.load_processor(model_id, prompt=prompt),
            calibration_data)


def temperature_for(calib: dict, qtype: str, k: int) -> float:
    """The scale fitted for this (type, K), or the global one where none was fitted.

    K is whatever the caller asked for, so a request can arrive in a bucket calibration
    never saw. Falling back to the global fit is right; silently using 1.0 is not.
    """
    return calib.get("temperatures", {}).get(f"{qtype}:{k}", calib.get("temperature", 1.0))


def evaluate(model, processor, state, questions, image_path, calib=None,
             mode="auto", max_edge=1024, chunk=32, report: dict | None = None) -> dict:
    started = time.perf_counter()
    configure_image_size(processor, max_edge)
    calib = (calib or {}).get("by_image_size", {}).get(str(max_edge), calib or {})
    if "image_size" in calib and calib["image_size"] != max_edge:
        raise ValueError("calibration was fitted at a different image resolution")
    examples = request_examples(state, questions, image_path)
    image = load_image(image_path, max_edge)
    if report is not None:
        report.setdefault("ms", {})["image_decode"] = (time.perf_counter() - started) * 1000
    if mode == "auto":
        # Hybrid linear-attention backbones (Qwen3.5) keep a recurrent state, not a KV
        # cache the suffixes can share, so they always take the one-pass path.
        mode = ("single" if model.family in NO_SHARED_CACHE
                or sum(len(ex.candidates) for _, ex in examples) <= SINGLE_PASS_MAX else "shared")
    if mode == "shared":
        logits = score_shared(model, processor, examples, image, max_edge, chunk, report=report)
    elif mode == "single":
        logits = score_single(model, processor, examples, image, max_edge, report=report)
    else:
        logits = score_naive(model, processor, examples, image, max_edge, chunk)
    answers = {}
    for (qid, ex), lg in zip(examples, logits):
        t = temperature_for(calib or {}, ex.qtype, lg.numel())
        probs = torch.softmax(lg.float() / t, dim=-1).cpu()
        answers[qid] = answer_for(ex, probs)
    return {"answers": answers}


DEMO_STATE = "You are selecting images for a magazine cover."
DEMO_QUESTIONS = {
    "aesthetic": {"type": "score", "attribute": "aesthetic",
                  "instructions": "How aesthetic is the picture?",
                  "criteria": ["Terrible", "Poor", "Average", "Good", "Outstanding"]},
    "lighting": {"type": "score", "instructions": "How well lit is the subject?",
                 "criteria": ["Badly lit", "Acceptable", "Beautifully lit"]},
    "cover_ready": {"type": "noul", "instructions": "Could this run as a cover as it is?",
                    "criteria": {"true": "it needs no further work",
                                 "false": "it needs retouching or a reshoot"}},
    "subject": {"type": "choice", "instructions": "What is the main subject?",
                "criteria": {"person": "a person or people fill the frame",
                             "landscape": "the scenery is the subject",
                             "object": "one object is the subject",
                             "abstract": "no identifiable subject"}},
    "crop": {"type": "choice", "instructions": "Which crop would help most?",
             "criteria": {"none": "it is already framed well",
                          "tighter": "move in on the subject",
                          "wider": "it needs more room around the subject"}},
    "colour": {"type": "score", "instructions": "How vivid is the colour?",
               "criteria": [{"what": "grey, almost no colour"}, {"what": "muted"},
                            {"what": "strong"}, {"what": "saturated to the point of glare"}]},
    "noise": {"type": "noul", "instructions": "Is there visible sensor noise?"},
    "focus": {"type": "noul", "instructions": "Is the main subject in focus?"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default=None, help="inferred from adapter when omitted")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--request", help="JSON file with {state, questions}; --demo if absent")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--mode", choices=["auto", "shared", "single", "naive"], default="auto")
    ap.add_argument("--chunk", type=int, default=32, help="candidate sequences per forward")
    ap.add_argument("--max-edge", type=int, default=384)
    ap.add_argument("--calibration", default=None,
                    help="temperature map to use; defaults to <adapter>/calibration.json")
    ap.add_argument("--check", action="store_true", help="assert shared == single == naive (in fp32)")
    ap.add_argument("--bench", default=None, help="e.g. 8x5,1x64: questions x candidates")
    args = ap.parse_args()

    device = select_device(args.device)
    # --check compares code paths, so it runs in fp32: in bf16 a larger batch alone moves
    # probabilities by ~0.02, which would hide or fake a real difference between paths.
    model, processor, calib = load(args.adapter, args.model, device, args.calibration, merge=True,
                                   dtype=torch.float32 if args.check else None)
    configure_image_size(processor, args.max_edge)
    # Calibration is stored per image size; report the one this request will use.
    active = calib.get("by_image_size", {}).get(str(args.max_edge), calib)
    fitted = len(active.get("temperatures", {}))
    print(f"[serve] {model.model_id} + {args.adapter}, {device}, {args.max_edge}px: "
          f"global T {active.get('temperature', 1.0):.3f} and {fitted} fitted buckets")

    if args.request:
        req = json.loads(Path(args.request).read_text())
        state, questions = req.get("state", ""), req["questions"]
    else:
        state, questions = DEMO_STATE, DEMO_QUESTIONS

    if args.check:
        check(model, processor, state, questions, args.image, args.max_edge, args.chunk)
    if args.bench:
        bench(model, processor, args.image, args.bench, args.max_edge, args.chunk)
    if not (args.check or args.bench):
        t0 = time.time()
        out = evaluate(model, processor, state, questions, args.image, calib,
                       args.mode, args.max_edge, args.chunk)
        out["usage"] = {"questions": len(questions),
                        "candidates": sum(len(q.get("criteria", [0, 0])) for q in questions.values()),
                        "seconds": round(time.time() - t0, 3)}
        print(json.dumps(out, indent=2))


@torch.no_grad()
def check(model, processor, state, questions, image_path, max_edge, chunk):
    """The shared prefix must not move the answer."""
    examples = request_examples(state, questions, image_path)
    image = load_image(image_path, max_edge)
    a = score_naive(model, processor, examples, image, max_edge, chunk)
    fast_paths = (("single", score_single),) if model.family in NO_SHARED_CACHE else \
        (("shared", score_shared), ("single", score_single))
    for label, fast in fast_paths:
        _compare(label, examples, a, fast(model, processor, examples, image, max_edge, chunk))


def _compare(label, examples, a, b):
    print(f"\n  [{label} vs naive]  question          K   max|dlogit|   max|dp|   argmax")
    worst_p = 0.0
    for (qid, ex), x, y in zip(examples, a, b):
        px, py = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
        dp = float((px - py).abs().max())
        worst_p = max(worst_p, dp)
        same = "same" if int(px.argmax()) == int(py.argmax()) else "DIFFERENT"
        print(f"  {qid:<16} {x.numel():>2}   {float((x-y).abs().max()):11.5f}   {dp:7.5f}   {same}")
    print(f"\n  worst probability difference across the request: {worst_p:.5f}")
    assert worst_p < 1e-3, f"{label} scoring changed the answers"
    print(f"  {label} reproduces the one-at-a-time path")


@torch.no_grad()
def bench(model, processor, image_path, spec, max_edge, chunk):
    image = load_image(image_path, max_edge)
    print(f"\n  shape      candidates   naive s   shared s   speedup   prefix tok")
    for part in spec.split(","):
        n, k = (int(x) for x in part.lower().split("x"))
        questions = {f"q{i}": {"type": "choice", "instructions": f"Question {i}?",
                               "criteria": {f"option_{j}": f"level {j}" for j in range(k)}} for i in range(n)}
        examples = request_examples("You are an art critic.", questions, image_path)
        report: dict = {}
        for fn in (score_naive, score_shared):   # warm the kernels once per path
            fn(model, processor, examples[:1], image, max_edge, chunk)
        synchronize(next(model.head.parameters()).device)
        try:
            t0 = time.time(); score_naive(model, processor, examples, image, max_edge, chunk)
            synchronize(next(model.head.parameters()).device); t_naive = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); t_naive = None
        t0 = time.time(); score_shared(model, processor, examples, image, max_edge, chunk, report)
        synchronize(next(model.head.parameters()).device); t_shared = time.time() - t0
        naive = f"{t_naive:7.2f}" if t_naive else "    OOM"
        gain = f"{t_naive/t_shared:6.1f}x" if t_naive else "     --"
        print(f"  {n:>3}x{k:<4}   {n*k:>10}   {naive}   {t_shared:8.2f}   "
              f"{gain}   {report.get('prefix_tokens','?'):>10}")


def synchronize(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize()
    elif str(device).startswith("mps"): torch.mps.synchronize()


if __name__ == "__main__":
    main()
