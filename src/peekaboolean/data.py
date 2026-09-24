"""Dataset and rubric augmentation.

Row schema (JSONL), one question per row:

  score:  {"image": "a.jpg", "type": "score", "state": "...",
           "attribute": "aesthetic",           # used to phrase generated levels
           "instructions": "How aesthetic is the picture?",
           "hist": [0.01, 0.03, ...]}          # native rating histogram, any length

  choice: {"image": "a.jpg", "type": "choice", "state": "...",
           "instructions": "What is the main subject?",
           "criteria": {"portrait": "one person fills the frame",
                        "landscape": "the subject is the scenery"},
           "label": "portrait"}                # option key, or a plain index

  noul:   {"image": "a.jpg", "type": "noul", "state": "...",
           "instructions": "Is the subject in focus?",
           "criteria": {"true": "...", "false": "..."},   # optional wording
           "value": 0.9}                       # P(yes), soft labels allowed

`criteria` is the served request's own field name: option -> description for choice,
an ordered list of level descriptions low end first for score, the true/false wording
for noul. Score rows may leave it out, which is the interesting case -- they keep
their native histogram and get rebinned to a randomly sampled level count at load
time, and that augmentation is the whole reason the served model can accept a rubric
it never saw in training.

state, instructions and every description may be a string, an object, an array or
null, because a request carries schemas, database rows and labelled bundles of fields
as often as it carries sentences. `render` decides how that structure reaches the
prompt, and the server has to call the same function or the model meets a format it
never trained on.

Rows written the older way -- choice options as `candidates` with an integer `label`,
noul wording as `true_text` / `false_text` -- still load unchanged.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# Ordered descriptor ladders, low to high. To build a K-level rubric we sample a
# style and take K evenly spaced entries, always keeping both endpoints. This
# gives wide wording variety from a compact table.
STYLES: dict[str, dict] = {
    "adverb": {
        "suffix_attribute": True,
        "scale": ["Not at all", "Barely", "Slightly", "Somewhat", "Moderately",
                  "Fairly", "Quite", "Very", "Highly", "Exceptionally"],
    },
    "quality": {
        "suffix_attribute": False,
        "scale": ["Terrible", "Very poor", "Poor", "Below average", "Average",
                  "Above average", "Good", "Very good", "Excellent", "Outstanding"],
    },
    "plain": {
        "suffix_attribute": False,
        "scale": ["Lowest", "Very low", "Low", "Slightly below middle", "Middling",
                  "Slightly above middle", "High", "Very high", "Highest"],
    },
    "casual": {
        "suffix_attribute": False,
        "scale": ["Not at all", "A little bit", "Medium", "Quite a lot", "Extremely"],
    },
    "stars": {
        "suffix_attribute": False,
        "scale": ["1 out of 10", "2 out of 10", "3 out of 10", "4 out of 10", "5 out of 10",
                  "6 out of 10", "7 out of 10", "8 out of 10", "9 out of 10", "10 out of 10"],
    },
}

# Personas and framings for the state field. A caller who sends
# "you are a professional photographer" expects it to matter, so state has to vary
# during training or the model learns to ignore it.
STATES = [
    "", "You are a professional photographer.", "You are an art critic.",
    "You are reviewing submissions for a photo contest.",
    "You are a casual viewer browsing a gallery.",
    "You are selecting images for a magazine cover.",
    "Judge this image on technical and artistic merit.",
]


# Answer-neutral framings for public QA rows, which arrive with no state at all. A
# served request nearly always carries one, and a model that only ever saw "" learns
# nothing about reading past it. None of these may change what the right answer is.
NEUTRAL_STATES = [
    "Answer using only what is visible in the image.",
    "Look at the image and answer each question.",
    "You are a careful visual inspector. Be precise.",
    "You help an operations team review incoming images.",
    "You are an assistant that answers questions about pictures, screenshots and documents.",
    "Judge only from the image; do not guess beyond what it shows.",
    {"task": "image review", "instructions": "Answer from the image content only."},
    {"role": "annotator", "guidelines": "Be literal and precise."},
    {"pipeline": "visual QA", "note": "Each question is independent."},
]
OPAQUE_KEY = re.compile(r"option_\d+")


def restyle_keys(names: list[str], descriptions: list, rng: random.Random) -> tuple[list[str], list]:
    """Rename opaque option_N keys the way real callers name options. Keys are what
    the answer reports back, so any style is valid as long as names stay unique."""
    style = rng.choice(["keep", "letter", "number", "slug", "bare", "bare"])
    if style == "keep":
        return names, descriptions
    if style == "letter":
        return [chr(65 + i) for i in range(len(names))], descriptions
    if style == "number":
        return [str(i + 1) for i in range(len(names))], descriptions
    texts = [render(d) for d in descriptions]
    if style == "slug":
        keys = [re.sub(r"[^0-9a-z]+", "_", t.lower()).strip("_")[:40] for t in texts]
        if all(keys) and len(set(keys)) == len(keys):
            return keys, descriptions
        return names, descriptions
    # bare: the option text itself is the key, with no separate description
    if len(set(texts)) == len(texts) and all(texts) and all(len(t) <= 80 for t in texts):
        return texts, [""] * len(texts)
    return names, descriptions


def render(value) -> str:
    """Flatten a request-shaped value -- string, object, array or null -- to prompt text.

    Strings pass through untouched, which keeps every string-only row byte-identical
    to what it was before structure was allowed. Anything else becomes compact JSON in
    its authored key order: `what` before `examples` is the author's emphasis, and
    sorting the keys would quietly throw that away.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def option_text(name: str, description) -> str:
    """A choice option as the model sees it: the name the answer reports back, plus
    whatever separates it from its neighbours."""
    rendered = render(description)
    return f"{name}: {rendered}" if rendered else name


def sample_levels(k: int, attribute: str, rng: random.Random) -> list[str]:
    style_name = rng.choice([s for s, v in STYLES.items() if len(v["scale"]) >= k])
    style = STYLES[style_name]
    scale = style["scale"]
    # evenly spaced indices across the ladder, endpoints included
    idx = [round(i * (len(scale) - 1) / (k - 1)) for i in range(k)]
    levels = [scale[i] for i in idx]
    if style["suffix_attribute"] and attribute:
        levels = [f"{lv} {attribute}" for lv in levels]
    return levels


def rebin(hist: list[float] | torch.Tensor, k: int) -> torch.Tensor:
    """Resample a rating histogram to k bins by exact interval overlap.

    Source bin i covers [i/S, (i+1)/S] of the rating range; target bin j covers
    [j/K, (j+1)/K]. Mass moves in proportion to overlap. Exact, not interpolated.
    """
    h = torch.as_tensor(hist, dtype=torch.float32)
    s = h.numel()
    src = torch.linspace(0, 1, s + 1)
    tgt = torch.linspace(0, 1, k + 1)
    out = torch.zeros(k)
    for j in range(k):
        lo, hi = tgt[j], tgt[j + 1]
        overlap = (torch.minimum(src[1:], hi) - torch.maximum(src[:-1], lo)).clamp(min=0)
        out[j] = (h * overlap * s).sum()
    total = out.sum()
    return out / total if total > 0 else torch.full((k,), 1.0 / k)


@dataclass
class Example:
    image: Path
    state: str
    instructions: str
    candidates: list[str]
    target: torch.Tensor   # score/choice: distribution over candidates. noul: P(yes) as 2-vector
    qtype: str
    # What each candidate is called in the answer: the option key for choice, the level
    # position for score, "false"/"true" for noul. The prompt shows `candidates`; a
    # caller reading probabilities back needs these.
    names: list[str] | None = None


class DecisionDataset(Dataset):
    def __init__(
        self,
        jsonl: str | Path,
        image_root: str | Path = ".",
        augment: bool = True,
        min_levels: int = 2,   # a served rubric may have as few as 2 levels and at most 10
        max_levels: int = 10,
        seed: int = 0,
    ):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.image_root = Path(image_root)
        self.augment = augment
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> Example:
        # Deterministic per-item RNG when not augmenting, so eval is reproducible.
        # (seed, i) tuples stopped being valid seeds in 3.11; a string is deterministic.
        rng = random.Random(f"{self.seed}:{i}:{self.epoch}" if self.augment else f"{self.seed}:{i}")
        return to_example(self.rows[i], self.image_root, rng,
                          augment=self.augment,
                          min_levels=self.min_levels, max_levels=self.max_levels)


def to_example(
    row: dict,
    image_root: str | Path = ".",
    rng: random.Random | None = None,
    augment: bool = False,
    min_levels: int = 2,
    max_levels: int = 10,
) -> Example:
    """Turn one row into the candidates the model scores.

    The server calls this too. A question that arrives over the wire is the same row
    minus its label, so serving and training cannot drift apart in how a rubric, an
    option map or a structured description reaches the prompt -- which is the one kind
    of bug that shows up as a quietly worse model rather than as an error.
    """
    rng = rng or random.Random()
    image_root = Path(image_root)
    qtype = row["type"]
    state = render(row.get("state", ""))
    instructions = render(row["instructions"])
    given = row.get("criteria")
    if given is None:
        given = row.get("candidates")

    if qtype == "score":
        if isinstance(given, dict):
            raise ValueError("score criteria is an ordered list of levels, low end first")
        if given:
            levels = [render(c) for c in given]
        else:
            if augment:
                k = rng.randint(min_levels, min(max_levels, 10))
            else:
                k = row.get("levels", 5)
            levels = sample_levels(k, row.get("attribute", ""), rng)
        target = rebin(row["hist"], len(levels))
        if augment and rng.random() < 0.5:
            # Reversed rubrics appear in real requests. Flip both together.
            levels = list(reversed(levels))
            target = torch.flip(target, dims=[0])
        if augment and row.get("augment_state", False):
            state = rng.choice(STATES) if rng.random() < 0.5 else state
        # Levels are addressed by position after any flip, which is how the answer
        # keys its probabilities: "0" is whatever the request listed first.
        candidates, tgt = levels, target
        names = [str(i) for i in range(len(levels))]

    elif qtype == "choice":
        if not given:
            raise ValueError("choice row needs criteria (option -> description) or candidates")
        if isinstance(given, dict):
            names = list(given)
            descriptions = [given[n] for n in names]
            label = row.get("label", 0)
            if isinstance(label, str) and label in names:
                label = names.index(label)
            weights = row.get("target_probs")
            if isinstance(weights, dict):
                weights = [weights.get(n, 0.0) for n in names]
            if augment and all(OPAQUE_KEY.fullmatch(n) for n in names):
                names, descriptions = restyle_keys(names, descriptions, rng)
            candidates = [option_text(n, d) for n, d in zip(names, descriptions)]
        else:
            names = [render(c) for c in given]
            candidates = list(names)
            label = row.get("label", 0)
            weights = row.get("target_probs")
            if isinstance(weights, dict):
                weights = [weights.get(n, 0.0) for n in names]
        if isinstance(label, str):
            if label not in names:
                raise ValueError(f"label {label!r} is not one of {names}")
            label = names.index(label)
        tgt = (torch.tensor(weights, dtype=torch.float32)
               if weights is not None else torch.zeros(len(candidates)))
        if weights is None:
            tgt[label] = 1.0
        if not torch.isfinite(tgt).all() or (tgt < 0).any() or tgt.sum() <= 0:
            raise ValueError("invalid choice target distribution")
        tgt = tgt / tgt.sum()
        if augment:
            # Shuffle options so the model cannot learn positional shortcuts.
            order = list(range(len(candidates)))
            rng.shuffle(order)
            candidates = [candidates[j] for j in order]
            names = [names[j] for j in order]
            label = order.index(label)
            tgt = tgt[order]

    elif qtype == "noul":
        wording = given if isinstance(given, dict) else {}
        candidates = [
            render(wording.get("false", row.get("false_text", "No"))) or "No",
            render(wording.get("true", row.get("true_text", "Yes"))) or "Yes",
        ]
        names = ["false", "true"]
        p = float(row["value"])
        if not 0 <= p <= 1:
            raise ValueError("noul value must be between 0 and 1")
        tgt = torch.tensor([1.0 - p, p])

    else:
        raise ValueError(f"unknown question type: {qtype}")

    if augment and not state and row.get("augment_neutral_state", True) and rng.random() < 0.5:
        state = render(rng.choice(NEUTRAL_STATES))

    return Example(
        image=image_root / row["image"],
        state=state,
        instructions=instructions,
        candidates=candidates,
        target=tgt,
        qtype=qtype,
        names=names,
    )


PROMPT = """{state}

Question: {instructions}
Proposed answer: {candidate}

Judge how well the proposed answer fits the image."""

# For the "yesno" head: a question the pretrained model already answers, so its own
# Yes/No logits carry meaning before any training.
PROMPT_YESNO = """{state}

Question: {instructions}
Proposed answer: {candidate}

Is the proposed answer correct? Answer yes or no."""

PROMPTS = {"judge": PROMPT, "yesno": PROMPT_YESNO}

# Noul under the "yesno" head: ask the question itself, the way the pretrained model
# answers it best. Both rows carry that same question and `candidate_signs` weights them
# -1/2 and +1/2, so the softmax over the two rows is exactly sigmoid(logit Yes - logit No).
# (A complementary "is the answer no?" row was tried; small models answer it "yes".)
NOUL_DIRECT = """{state}

Question: {instructions}{wording}
Answer yes or no."""


def candidate_prompts(example: Example, style: str = "judge") -> list[str]:
    """One prompt per candidate. Separate from the processor so a server, a test or a
    diff can see the exact text the model was trained on without loading a model."""
    if style == "yesno" and example.qtype == "noul":
        false_text, true_text = example.candidates
        wording = "" if (false_text, true_text) == ("No", "Yes") else \
            f"\n(Yes means: {true_text}. No means: {false_text}.)"
        prompt = NOUL_DIRECT.format(state=example.state.strip(), instructions=example.instructions.strip(),
                                    wording=wording)
        return [prompt, prompt]
    return [
        PROMPTS[style].format(
            state=example.state.strip(),
            instructions=example.instructions.strip(),
            candidate=cand,
        )
        for cand in example.candidates
    ]


def candidate_signs(example: Example, style: str = "judge") -> list[float]:
    """Per-row multiplier on the head's score. 1 everywhere except the yes/no-head noul,
    whose two identical rows read the same Yes-vs-No logit with opposite signs."""
    if style == "yesno" and example.qtype == "noul":
        return [-0.5, 0.5]
    return [1.0] * len(example.candidates)


def load_image(path, max_edge: int = 1024) -> Image.Image:
    from PIL import ImageOps
    with Image.open(path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return img


def collate(batches: list, pad_id: int) -> dict:
    """Join single-question inputs (from build_inputs, one shared image each) into one
    forward: candidate sequences padded right to a common length, one image per
    question, and `image_counts` telling the model how many candidates share each.
    Every question in a batch must use the same image size."""
    import torch.nn.functional as F
    length = max(b["input_ids"].shape[1] for b in batches)
    pad = lambda t, value: F.pad(t, (0, length - t.shape[1]), value=value)
    out = {"input_ids": torch.cat([pad(b["input_ids"], pad_id) for b in batches]),
           "attention_mask": torch.cat([pad(b["attention_mask"], 0) for b in batches]),
           "image_counts": torch.tensor([b["input_ids"].shape[0] for b in batches])}
    if any("row_sign" in b for b in batches):
        out["row_sign"] = torch.cat([b["row_sign"] if "row_sign" in b else torch.ones(b["input_ids"].shape[0])
                                     for b in batches])
    if "mm_token_type_ids" in batches[0]:
        out["mm_token_type_ids"] = torch.cat([pad(b["mm_token_type_ids"], 0) for b in batches])
    if "image_grid_thw" in batches[0]:
        # Qwen: pixel_values are flattened patches of every row's image; keep the first
        # image of each question (its patches, then its grid).
        out["pixel_values"] = torch.cat([b["pixel_values"][:int(b["image_grid_thw"][0].prod())] for b in batches])
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"][:1] for b in batches])
    else:
        for key in ("pixel_values", "pixel_attention_mask"):
            if key in batches[0]:
                out[key] = torch.cat([b[key][:1] for b in batches])
    from transformers import BatchFeature
    return BatchFeature(out)


def build_inputs(example: Example, processor, max_edge: int = 1024, image: Image.Image | None = None):
    """Turn one Example into K padded candidate sequences sharing the same image."""
    img = image if image is not None else load_image(example.image, max_edge)

    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]}],
            tokenize=False, add_generation_prompt=True)
        for text in candidate_prompts(example, getattr(processor, "jev_prompt", "judge"))
    ]

    batch = processor(
        text=texts,
        images=[img] * len(texts),   # same pixels K times; see the note in model.py
        return_tensors="pt",
        padding=True,
    )
    signs = candidate_signs(example, getattr(processor, "jev_prompt", "judge"))
    if any(s != 1.0 for s in signs):
        batch["row_sign"] = torch.tensor(signs)
    return batch
