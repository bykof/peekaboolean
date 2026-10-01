"""Typed-decision training items from human-labelled TRAIN splits; every gold is a human label or follows by construction.

Sources: the_cauldron subsets (train), VQAv2 train annotations (10 answers per question) and COCO train2014 instances.
Writes /root/big/data/items/<source>.jsonl rows {id, source, dataset, kind, images, state, question, gold}.
python build_items.py
"""
import glob, hashlib, io, json, math, os, random, re
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

CAUL = glob.glob("/root/.cache/huggingface/hub/datasets--HuggingFaceM4--the_cauldron/snapshots/*/")[0]
D = "/root/big/data"
MAXPIX = 3_200_000
INSTR = {"instruction": "Use only the supplied image."}
UNK = "__unknown__"
R = random.Random(20261001)


# ---------- generic helpers ----------
def load(sub, shards=None):
    out = []
    for f in sorted(glob.glob(CAUL + sub + "/*.parquet"))[:shards]:
        for r, texts in enumerate(pq.read_table(f, columns=["texts"]).column("texts").to_pylist()):
            out.append((f, r, texts))
    return out


def first_line(u):
    return u.split("\n")[0].strip()


def clean(a):
    a = a.strip()
    return a[:-1].strip() if a.endswith(".") else a


NUMW = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def vnorm(a):
    w = re.sub(r"\s+", " ", a.lower().strip().rstrip(".")).split()
    return " ".join(NUMW.get(x, x) for x in w if x not in ("a", "an", "the"))


def key(s):
    w = re.sub(r"[^\w\s]", " ", s.lower()).split()
    return " ".join(x[:-1] if len(x) > 3 and x.endswith("s") else x for x in w if x not in ("a", "an", "the"))


NUM = re.compile(r"([-+]?)([$€£]?\s*)(\d[\d,]*\.?\d*|\.\d+)(\s*(?:%|percent|million|billion|thousand|k|m|bn)?)", re.I)


def num(s):
    m = NUM.fullmatch(s.strip())
    if not m:
        return None
    try:
        v = float(m.group(3).replace(",", ""))
    except ValueError:
        return None
    return -v if m.group(1) == "-" else v


def decimals(s):
    d = NUM.fullmatch(s.strip()).group(3)
    return len(d.split(".")[1]) if "." in d else 0


def distinct(d, chosen, ban=()):
    kd = key(d)
    if not kd or kd in ban or len(d) > 80:
        return False
    nd = num(d)
    for c in chosen:
        kc = key(c)
        if kd == kc:
            return False
        nc = num(c)
        if nd is not None and nc is not None:
            if nd == nc:
                return False
        elif f" {kd} " in f" {kc} " or f" {kc} " in f" {kd} ":
            return False
    return True


def pick(pool, gold, k, ban=(), numeric=False):
    """k distractors from pool: distinct from gold and each other; numbers drawn from the closest magnitudes."""
    if numeric:
        g = num(gold)
        cands = sorted({p for p in pool if num(p) is not None}, key=lambda p: abs(math.log1p(abs(num(p))) - math.log1p(abs(g))))[:30]
        R.shuffle(cands)
    else:
        cands = R.sample(pool, min(len(pool), 300))
    out = []
    for d in cands:
        if len(out) == k:
            break
        if distinct(d, [gold] + out, ban):
            out.append(d)
    return out if len(out) == k else None


def choice_q(instr, gold, distractors, sort_numeric=False):
    opts = [gold] + distractors
    if sort_numeric:
        opts.sort(key=num)
    else:
        R.shuffle(opts)
    return {"type": "choice", "instructions": instr, "criteria": {o: None for o in opts}}, gold


def noul_q(instr, crit=None):
    return {"type": "noul", "instructions": instr, "criteria": crit}


def plain_state():
    return dict(INSTR) if R.random() < 0.6 else ""


def count_choice(instr, g):
    k = R.choice([3, 4, 5])
    near = [x for x in range(max(0, g - 3), g + 4) if x != g]
    near.sort(key=lambda x: (abs(x - g), R.random()))
    return choice_q(instr, str(g), [str(x) for x in near[:k - 1]], sort_numeric=True)


def count_score(instr, g, noun=None):
    L = R.choice([3, 4, 5])
    top = max(L, g + R.choice([0, 1, 2, 3]) if R.random() < 0.8 else g)
    cuts = sorted(R.sample(range(1, max(top, L) + 1), L - 1))
    edges = [0] + cuts
    levels = []
    for i, a in enumerate(edges):
        b = edges[i + 1] - 1 if i + 1 < len(edges) else None
        t = f"{a} or more" if b is None else (f"{a}" if a == b else f"{a} to {b}")
        levels.append(t + (f" {noun}" if noun else ""))
    gi = max(i for i, a in enumerate(edges) if g >= a)
    return {"type": "score", "instructions": instr, "criteria": levels}, str(gi)


def mc_parse(u):
    """'Question: q\\nChoices:\\nA. x\\n...Answer with the letter.' -> (q, [options]) or None."""
    m = re.search(r"Question: (.*?)\nChoices:\n(.*?)\nAnswer with the letter", u, re.S)
    if not m:
        return None
    opts = [clean(re.sub(r"^[A-Z]\. ", "", o)) for o in m.group(2).split("\n")]
    return m.group(1).strip(), opts


def subname(f):
    return "aokvqa" if "A-OKVQA" in f else f.split("/")[-2]


class Out:
    def __init__(self):
        self.items = defaultdict(list)

    def add(self, source, dataset, kind, src, state, q, gold, **extra):
        f, r = src[0][:2]
        self.items[source].append({"id": None, "source": source, "dataset": dataset, "kind": kind, "_src": src,
                                   "group": f"{subname(f)}/{f.split('/')[-1][:14]}:{r}",
                                   "state": state, "question": q, "gold": gold, **extra})


O = Out()


# ---------- image resolution ----------
def save(b, sub, edit=None):
    im = Image.open(io.BytesIO(b))
    fmt = im.format
    if edit or im.width * im.height > MAXPIX or fmt not in ("JPEG", "PNG"):
        im = im.convert("RGB")
        if edit:
            d = ImageDraw.Draw(im)
            for x0, y0, x1, y1 in edit:
                d.rectangle([x0 * im.width, y0 * im.height, x1 * im.width, y1 * im.height], fill=(128, 128, 128))
        s = min(1.0, math.sqrt(MAXPIX / (im.width * im.height)))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        ext = "png" if fmt == "PNG" else "jpg"
        im.save(buf, "PNG" if ext == "png" else "JPEG", quality=92)
        b = buf.getvalue()
    else:
        ext = "jpg" if fmt == "JPEG" else "png"
    p = f"{D}/images/{sub}/{hashlib.sha256(b).hexdigest()[:24]}.{ext}"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b)
    return p


def resolve(items):
    need, by_file = defaultdict(set), defaultdict(list)
    for it in items:
        for f, r, k, e in it["_src"]:
            need[f].add(r)
            by_file[f].append((f, r, k, e))
    done = {}
    for f, rows in need.items():
        names = pq.ParquetFile(f).schema_arrow.names
        col = pq.read_table(f, columns=["images" if "images" in names else "image"]).column(0)
        sub = subname(f)
        for ff, r, k, e in by_file[f]:
            kk = (f, r, k, json.dumps(e))
            if kk not in done:
                x = col[r].as_py()
                done[kk] = save((x[k] if isinstance(x, list) else x)["bytes"], sub, e)
        del col
        print("images", sub, len(done), flush=True)
    for it in items:
        it["images"] = [done[(f, r, k, json.dumps(e))] for f, r, k, e in it.pop("_src")]


# ---------- rule / record wrappers shared by several datasets ----------
CNT_RULES = [
    ("Approve the listing if the photo shows at least {k} {noun}.", "Is the listing approved under the rule?", lambda g, k: g >= k),
    ("Reject the delivery if fewer than {k} {noun} are visible in the photo.", "Is the delivery rejected under the rule?", lambda g, k: g < k),
    ("The setup passes inspection only if the photo shows at most {k} {noun}.", "Does the setup pass inspection?", lambda g, k: g <= k),
    ("The order lists {k} {noun}. Accept it only if the photo shows exactly that many.", "Is the order accepted?", lambda g, k: g == k),
]
CNT_RULES_Q = [
    ("Approve if the answer to the count question, counted in the image, is at least {k}.", "Is it approved under the rule?", lambda g, k: g >= k),
    ("Reject if the answer to the count question, counted in the image, is below {k}.", "Is it rejected under the rule?", lambda g, k: g < k),
    ("Pass only if the answer to the count question, counted in the image, is exactly {k}.", "Does it pass?", lambda g, k: g == k),
]
SIMPLE_HOWMANY = re.compile(r"^how many ([a-z ]{2,30}?) (?:are there|are in the (?:photo|picture|image)|are pictured|are visible|"
                            r"are shown|can you see|do you see|are in this (?:photo|picture|image))\??$")


def count_rule(dataset, src, q, g, unk_pair=True):
    m = SIMPLE_HOWMANY.match(q.lower())
    k = max(1, g + R.choice([-2, -1, 0, 0, 1, 1, 2]))
    if m:
        tpl, qq, f = R.choice(CNT_RULES)
        noun = m.group(1)
        state = {"rule": tpl.format(k=k, noun=noun)}
    else:
        tpl, qq, f = R.choice(CNT_RULES_Q)
        noun = None
        state = {"count_question": q, "rule": tpl.format(k=k)}
    O.add("rule-count", dataset, "rule", src, state, noul_q(qq), "true" if f(g, k) else "false")
    if unk_pair and R.random() < 0.18:  # same rule with the threshold moved to a state field that is given / not given
        base = {"count_question": q} if noun is None else {"item": noun}
        rule = (f"Approve if the photo shows at least the minimum number of {noun} given in the order." if noun else
                "Approve if the answer to the count question, counted in the image, is at least the order's minimum.")
        o_id = f"ORD-{R.randint(10000, 99999)}"
        O.add("ctrl-field", dataset, "rule", src, {**base, "order": {"id": o_id, "minimum": k}, "rule": rule},
              noul_q("Is the order approved under the rule?"), "true" if g >= k else "false")
        O.add("unk-field", dataset, "unknown", src, {**base, "order": {"id": o_id}, "rule": rule},
              noul_q("Is the order approved under the rule?"), UNK)


def fmt_like(gold, v):
    """Render number v with gold's decimals, thousands commas and unit."""
    m = NUM.fullmatch(gold.strip())
    dec = decimals(gold)
    body = f"{abs(v):,.{dec}f}" if "," in m.group(3) else f"{abs(v):.{dec}f}"
    return ("-" if v < 0 else "") + m.group(2) + body + m.group(4)


def threshold_rule(dataset, src, q, gold, doc):
    g = num(gold)
    step = max(abs(g) * R.uniform(0.04, 0.3), 10 ** -decimals(gold))
    ts = fmt_like(gold, g + R.choice([-1, 1]) * step)
    t = num(ts)
    if t is None or t == g:
        return
    kind = R.random()
    if kind < 0.4:
        state = {"metric": q, "rule": f"Flag the report if the metric, as shown in the {doc}, is above {ts}."}
        qq, gl = "Is the report flagged under the rule?", g > t
    elif kind < 0.75:
        state = {"metric": q, "rule": f"Approve if the metric, as shown in the {doc}, is at most {ts}."}
        qq, gl = "Is it approved under the rule?", g <= t
    else:
        v = gold if R.random() < 0.5 else ts
        state = {"metric": q, "submitted_value": v, "rule": f"Accept the submission only if the submitted value matches the {doc}."}
        qq, gl = "Is the submission accepted?", v == gold
    O.add("rule-threshold", dataset, "rule", src, state, noul_q(qq), "true" if gl else "false")
    if R.random() < 0.15:
        rule = f"Flag the report if the metric, as shown in the {doc}, is above the limit set in the policy."
        pol = {"name": R.choice(["Quarterly review", "Audit policy B", "Desk check"])}
        O.add("ctrl-field", dataset, "rule", src, {"metric": q, "policy": {**pol, "limit": ts}, "rule": rule},
              noul_q("Is the report flagged under the rule?"), "true" if g > t else "false")
        O.add("unk-field", dataset, "unknown", src, {"metric": q, "policy": pol, "rule": rule},
              noul_q("Is the report flagged under the rule?"), UNK)


def claim(dataset, src, q, gold, distractor, what="image"):
    v = gold if R.random() < 0.5 else distractor
    state = {"claim": {"question": q, "answer": v}}
    O.add("rule-claim", dataset, "rule", src, state, noul_q(f"Is the claimed answer correct according to the {what}?"),
          "true" if v == gold else "false")
    if R.random() < 0.12:
        O.add("unk-field", dataset, "unknown", src, {"claim": {"question": q}, "note": "The claimed answer was not submitted."},
              noul_q(f"Is the claimed answer correct according to the {what}?"), UNK)


def multi_check(dataset, src, qas):
    """qas: [(question, gold, distractor)] -> all/any rule with expected answers, half of them swapped."""
    mode = R.choice(["all", "all", "any"])
    n_wrong = R.choice([0, 1]) if mode == "all" else R.choice([len(qas), len(qas) - 1])
    wrong = set(R.sample(range(len(qas)), n_wrong))
    checks = [{"question": q, "expected": d if i in wrong else g} for i, (q, g, d) in enumerate(qas)]
    rule = ("Approve only if every check's expected answer is what the image shows." if mode == "all" else
            "Approve if at least one check's expected answer is what the image shows.")
    gl = (not wrong) if mode == "all" else len(wrong) < len(qas)
    O.add("rule-multi", dataset, "rule", src, {"checks": checks, "rule": rule}, noul_q("Is it approved under the rule?"),
          "true" if gl else "false")


# ---------- VQAv2 with full annotations + COCO instances ----------
CAT_SYN = {  # word in question -> COCO categories that must all be absent for the premise to be false
    "dog": ["dog"], "puppy": ["dog"], "cat": ["cat"], "kitten": ["cat"], "horse": ["horse"], "sheep": ["sheep"],
    "cow": ["cow"], "elephant": ["elephant"], "bear": ["bear", "teddy bear"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "bird": ["bird"], "bus": ["bus"], "train": ["train"], "truck": ["truck"], "boat": ["boat"], "airplane": ["airplane"],
    "plane": ["airplane"], "jet": ["airplane"], "bicycle": ["bicycle"], "bike": ["bicycle", "motorcycle"],
    "motorcycle": ["motorcycle"], "car": ["car"], "umbrella": ["umbrella"], "kite": ["kite"], "surfboard": ["surfboard"],
    "skateboard": ["skateboard"], "frisbee": ["frisbee"], "laptop": ["laptop"], "pizza": ["pizza"], "cake": ["cake"],
    "clock": ["clock"], "bench": ["bench"], "hydrant": ["fire hydrant"], "toilet": ["toilet"], "bed": ["bed"],
    "couch": ["couch"], "sofa": ["couch"], "refrigerator": ["refrigerator"], "fridge": ["refrigerator"], "vase": ["vase"],
    "banana": ["banana"], "sandwich": ["sandwich"], "donut": ["donut"], "broccoli": ["broccoli"], "carrot": ["carrot"],
    "suitcase": ["suitcase"], "teddy bear": ["teddy bear"], "racket": ["tennis racket"], "skis": ["skis"],
    "snowboard": ["snowboard"], "sink": ["sink"], "oven": ["oven"], "microwave": ["microwave"], "television": ["tv"],
    "tv": ["tv"], "hot dog": ["hot dog"], "giraffes": ["giraffe"], "elephants": ["elephant"], "zebras": ["zebra"],
}
PRESUP = re.compile(r"^(what colou?r (?:is|are) the|what is the \w+ (?:doing|holding|eating|made of|on|carrying)|"
                    r"what kind of|what type of|what brand|where is the|is the|are the|what is on the|what is in the)\b")
COVERQ = re.compile(r"^(what colou?r (?:is|are) the|what is the \w+ (?:doing|holding|eating|carrying)|what kind of|"
                    r"what type of|what is on the)\b")
TEXT_WORDS = re.compile(r"\b(say|says|written|sign|word|words|letter|letters|number|numbers|brand|name|read|text|logo|time)\b")
TEXT_CATS = {"book", "tv", "laptop", "cell phone", "stop sign", "clock", "keyboard", "remote", "parking meter"}
PHOTOS_NO_TEXT = []  # (src, image_id) for the domain-mismatch unknowns
PHOTO_PREMISE_QS = []  # (question, criteria, gold_key) about COCO objects, for the domain-mismatch unknowns


def syn_in(q):
    ql = " " + re.sub(r"[^\w\s]", " ", q.lower()) + " "
    return [s for s in CAT_SYN if f" the {s} " in ql or f" the {s}s " in ql]


def vqav2():
    Q = json.load(open(f"{D}/raw/v2_OpenEnded_mscoco_train2014_questions.json"))["questions"]
    A = {a["question_id"]: a for a in json.load(open(f"{D}/raw/v2_mscoco_train2014_annotations.json"))["annotations"]}
    by_q, per = defaultdict(set), defaultdict(dict)
    for q in Q:
        k = q["question"].strip().lower()
        by_q[k].add(q["image_id"])
        per[q["image_id"]][k] = (q["question"].strip(), A[q["question_id"]])
    del Q, A
    inst = json.load(open(f"{D}/raw/annotations/instances_train2014.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    size = {i["id"]: (i["width"], i["height"]) for i in inst["images"]}
    boxes = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        boxes[a["image_id"]][cats[a["category_id"]]].append((a["bbox"], a["iscrowd"]))
    del inst
    rows = load("vqav2")
    matched = []
    for f, r, texts in rows:
        ks = [first_line(t["user"]).lower() for t in texts]
        cand = set.intersection(*[by_q.get(k, set()) for k in ks]) if ks else set()
        if len(cand) == 1:
            matched.append((f, r, cand.pop()))
    print("vqav2 rows", len(rows), "matched to VQAv2 image ids", len(matched))

    # qa records with agreement
    recs, pool = [], defaultdict(list)
    for f, r, img in matched:
        for k, (q, a) in per[img].items():
            answers = [vnorm(x["answer"]) for x in a["answers"]]
            top, cnt = Counter(answers).most_common(1)[0]
            pool[a["question_type"]].append(top)
            for n in (5, 4, 3):
                pool[(qkey(q, n), a["answer_type"])].append(top)
            recs.append({"f": f, "r": r, "img": img, "q": q, "ans": top, "agree": cnt, "atype": a["answer_type"],
                         "qtype": a["question_type"], "ban": {key(x) for x in answers}})
    for x in recs:  # most specific question-prefix pool with enough distinct answers
        x["pool"] = next((pool[(qkey(x["q"], n), x["atype"])] for n in (5, 4, 3)
                          if len(set(pool[(qkey(x["q"], n), x["atype"])])) >= 15), pool[x["qtype"]])
    by_img = defaultdict(list)
    for x in recs:
        by_img[x["img"]].append(x)

    def options(x):
        if x["atype"] == "yes/no":
            return noul_q(x["q"]), "true" if x["ans"] == "yes" else "false"
        if x["atype"] == "number":
            return count_choice(x["q"], int(x["ans"]))
        ds = pick(x["pool"], x["ans"], R.choice([2, 3, 4]), ban=x["ban"])
        return choice_q(x["q"], x["ans"], ds) if ds else (None, None)

    src = lambda x, e=None: [(x["f"], x["r"], 0, e)]
    R.shuffle(recs)
    used = Counter()
    yn = {"yes": 0, "no": 0}
    n_choice = n_count = 0
    for x in recs:
        if used[x["img"]] >= 2:
            continue
        if x["atype"] == "yes/no" and x["agree"] >= 9 and x["ans"] in yn and yn[x["ans"]] < 450:
            yn[x["ans"]] += 1
        elif x["atype"] == "number" and x["agree"] >= 8 and x["ans"].isdigit() and int(x["ans"]) <= 15 and n_count < 400:
            n_count += 1
            g = int(x["ans"])
            q, gold = count_choice(x["q"], g) if R.random() < 0.6 else count_score(x["q"], g)
            O.add("vqav2", "vqav2", "plain", src(x), plain_state(), q, gold)
            used[x["img"]] += 1
            x["used"] = True
            continue
        elif x["atype"] == "other" and x["agree"] >= 8 and n_choice < 800:
            n_choice += 1
        else:
            continue
        q, gold = options(x)
        if q:
            O.add("vqav2", "vqav2", "plain", src(x), plain_state(), q, gold)
            used[x["img"]] += 1
            x["used"] = True

    # rules: counts, listing colour, claims, multi-checks
    nc = nl = ncl = nm = 0
    for x in recs:
        if x.get("used") or x["agree"] < 8:
            continue
        if x["atype"] == "number" and x["ans"].isdigit() and int(x["ans"]) <= 15 and nc < 300:
            nc += 1
            count_rule("vqav2", src(x), x["q"], int(x["ans"]))
            x["used"] = True
        elif (m := re.match(r"^what colou?r is the ([a-z ]{2,25})\?$", x["q"].lower())) and " and " not in x["ans"] and nl < 250:
            ds = pick(x["pool"], x["ans"], 1, ban=x["ban"])
            if not ds or " and " in ds[0]:
                continue
            nl += 1
            c = x["ans"] if R.random() < 0.5 else ds[0]
            O.add("rule-record", "vqav2", "rule", src(x), {"listing": {"item": m.group(1), "color": c},
                  "rule": "The listing is accurate only if the stated color matches the photo."},
                  noul_q(f"Does the {m.group(1)} in the photo match the listing's color?"), "true" if c == x["ans"] else "false")
            if R.random() < 0.15:
                O.add("unk-field", "vqav2", "unknown", src(x), {"listing": {"item": m.group(1)},
                      "rule": "The listing is accurate only if the stated color matches the photo."},
                      noul_q(f"Does the {m.group(1)} in the photo match the listing's color?"), UNK)
            x["used"] = True
        elif x["atype"] == "other" and ncl < 150:
            ds = pick(x["pool"], x["ans"], 1, ban=x["ban"])
            if ds:
                ncl += 1
                claim("vqav2", src(x), x["q"], x["ans"], ds[0], "photo")
                x["used"] = True
    for img, xs in by_img.items():
        xs = [x for x in xs if not x.get("used") and x["agree"] >= 9 and x["atype"] in ("yes/no", "other")]
        if len(xs) >= 2 and nm < 150:
            qas = []
            for x in R.sample(xs, 2):
                d = ("no" if x["ans"] == "yes" else "yes") if x["atype"] == "yes/no" else (pick(x["pool"], x["ans"], 1, ban=x["ban"]) or [None])[0]
                if d:
                    qas.append((x["q"], x["ans"], d))
            if len(qas) == 2:
                nm += 1
                multi_check("vqav2", [(xs[0]["f"], xs[0]["r"], 0, None)], qas)

    # premise-absent unknowns (question from image A about an object that image B does not contain)
    img_src = {img: (f, r) for f, r, img in matched}
    present = {img: set(boxes[img]) for img in img_src}
    text_of = {img: " " + " ".join(re.sub(r"[^\w\s]", " ", (y["q"] + " " + y["ans"]).lower()) for y in by_img[img]) + " "
               for img in img_src}
    imgs = list(img_src)
    for img in imgs:
        if not present[img] & TEXT_CATS and not TEXT_WORDS.search(text_of[img]):
            PHOTOS_NO_TEXT.append(img_src[img])
    cands = [x for x in recs if x["agree"] >= 8 and x["atype"] != "number" and PRESUP.match(x["q"].lower())
             and syn_in(x["q"]) and not re.match(r"^(is|are) (there|this|these|that|it)\b", x["q"].lower())]
    R.shuffle(cands)
    n1 = 0
    for x in cands:
        if n1 >= 600:
            break
        syns = syn_in(x["q"])
        need_absent = {c for s in syns for c in CAT_SYN[s]}
        if not any(c in present[x["img"]] for c in need_absent):
            continue  # premise not even confirmed on its own image
        q, gold = options(x)
        if not q:
            continue
        PHOTO_PREMISE_QS.append(q)
        for _ in range(30):
            b = R.choice(imgs)
            if b == x["img"] or present[b] & need_absent or any(f" {s} " in text_of[b] or f" {s}s " in text_of[b] for s in syns):
                continue
            f, r = img_src[b]
            O.add("unk-premise", "vqav2", "unknown", [(f, r, 0, None)], plain_state(), q, UNK, premise_from=x["img"], image_id=b)
            n1 += 1
            if not x.get("used"):  # the same question where the premise holds, so absence is the only cue
                O.add("vqav2", "vqav2", "plain", src(x), plain_state(), q, gold)
                x["used"] = True
            break

    # covered-evidence unknowns + controls where an unrelated object is covered
    n2 = 0
    for x in recs:
        if n2 >= 350 or x["agree"] < 8 or x["atype"] == "number" or not COVERQ.match(x["q"].lower()):
            continue
        syns = syn_in(x["q"])
        if not syns:
            continue
        cs = {c for s in syns for c in CAT_SYN[s]} & present[x["img"]]
        W, H = size[x["img"]]
        bx = [(b, cr) for c in cs for b, cr in boxes[x["img"]][c]]
        if not bx or len(bx) > 3 or any(cr for _, cr in bx):
            continue

        def grow(b, m=0.15):
            x0, y0, w, h = b
            return [max(0, (x0 - m * w) / W), max(0, (y0 - m * h) / H), min(1, (x0 + w * (1 + m)) / W), min(1, (y0 + h * (1 + m)) / H)]
        cover = [grow(b) for b, _ in bx]
        area = sum((c[2] - c[0]) * (c[3] - c[1]) for c in cover)
        if area > 0.4 or area < 0.01:
            continue
        q, gold = options(x)
        if not q:
            continue
        f, r = img_src[x["img"]]
        O.add("unk-cover", "vqav2", "unknown", [(f, r, 0, cover)], plain_state(), q, UNK, image_id=x["img"])
        n2 += 1
        others = [grow(b) for c, bl in boxes[x["img"]].items() if c not in cs for b, cr in bl if not cr]
        def overlap(a, c):
            return not (a[2] <= c[0] or c[2] <= a[0] or a[3] <= c[1] or c[3] <= a[1])
        others = [o for o in others if not any(overlap(o, c) for c in cover) and 0.01 < (o[2] - o[0]) * (o[3] - o[1]) < 0.3]
        if others:
            O.add("ctrl-cover", "vqav2", "plain", [(f, r, 0, R.sample(others, min(len(others), 2)))], plain_state(), q, gold, image_id=x["img"])
    print("vqav2 premise", n1, "cover", n2, "no-text photos", len(PHOTOS_NO_TEXT))


# ---------- counting ----------
def tallyqa():
    rows = load("tallyqa", 2)
    recs = [(f, r, first_line(t["user"]), int(clean(t["assistant"]))) for f, r, ts in rows for t in ts if clean(t["assistant"]).isdigit()]
    R.shuffle(recs)
    per_val, seen, n, nr = Counter(), set(), 0, 0
    for f, r, q, g in recs:
        if (f, r) in seen or g > 15:
            continue
        cap = {0: 70, 1: 120, 2: 120, 3: 100}.get(g, 90)
        if per_val[g] < cap and n < 550:
            per_val[g] += 1
            n += 1
            qq, gold = count_choice(q, g) if R.random() < 0.55 else count_score(q, g)
            O.add("tallyqa", "tallyqa", "plain", [(f, r, 0, None)], plain_state(), qq, gold)
        elif nr < 300:
            nr += 1
            count_rule("tallyqa", [(f, r, 0, None)], q, g)
        else:
            continue
        seen.add((f, r))


CLEVR_VOC = [["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"], ["cube", "sphere", "cylinder"],
             ["rubber", "metal"], ["small", "large"]]


def clevr():
    rows = load("clevr", 2)
    R.shuffle(rows)
    n = Counter()
    for f, r, ts in rows:
        qas = [(first_line(t["user"]), clean(t["assistant"]).lower()) for t in ts]
        R.shuffle(qas)
        q, a = qas[0]
        if a in ("yes", "no") and n["yn"] < 300:
            n["yn"] += 1
            O.add("clevr", "clevr", "plain", [(f, r, 0, None)], plain_state(), noul_q(q), "true" if a == "yes" else "false")
        elif a.isdigit() and n["cnt"] < 200:
            n["cnt"] += 1
            qq, gold = count_choice(q, int(a)) if R.random() < 0.5 else count_score(q, int(a))
            O.add("clevr", "clevr", "plain", [(f, r, 0, None)], plain_state(), qq, gold)
        elif a.isdigit() and n["rule"] < 200:
            n["rule"] += 1
            count_rule("clevr", [(f, r, 0, None)], q, int(a))
        elif (voc := next((v for v in CLEVR_VOC if a in v), None)) and n["attr"] < 300:
            n["attr"] += 1
            ds = R.sample([v for v in voc if v != a], min(len(voc) - 1, R.choice([2, 3, 4])))
            qq, gold = choice_q(q, a, ds)
            O.add("clevr", "clevr", "plain", [(f, r, 0, None)], plain_state(), qq, gold)
        elif len(qas) >= 2 and n["multi"] < 150:
            pair = []
            for q2, a2 in qas[:2]:
                voc = next((v for v in CLEVR_VOC if a2 in v), None)
                d = ("no" if a2 == "yes" else "yes") if a2 in ("yes", "no") else (str(int(a2) + R.choice([-1, 1]) if int(a2) else 1)
                    if a2.isdigit() else (R.choice([v for v in voc if v != a2]) if voc else None))
                if d:
                    pair.append((q2, a2, d))
            if len(pair) == 2:
                n["multi"] += 1
                multi_check("clevr", [(f, r, 0, None)], pair)


# ---------- statements ----------
def vsr():
    recs = []
    for f, r, ts in load("vsr"):
        for t in ts:
            m = re.search(r'"([^"]+)"', t["user"])
            if m and clean(t["assistant"]).lower() in ("yes", "no"):
                recs.append((f, r, m.group(1), clean(t["assistant"]).lower() == "yes"))
    R.shuffle(recs)
    n = Counter()
    for f, r, cap, y in recs:
        if n[y] >= 300:
            continue
        n[y] += 1
        tpl = R.choice(['Is this statement about the image true: "{c}"', 'Does the caption "{c}" correctly describe the image?',
                        'Is it true that {c0}'])
        O.add("vsr", "vsr", "plain", [(f, r, 0, None)], plain_state(),
              noul_q(tpl.format(c=cap, c0=cap[0].lower() + cap[1:].rstrip(".") + "?")), "true" if y else "false")


def nlvr2():
    recs = []
    for f, r, ts in load("nlvr2", 2):
        for t in ts:
            m = re.search(r'"([^"]+)"', t["user"])
            if m and clean(t["assistant"]).lower() in ("yes", "no"):
                recs.append((f, r, m.group(1), clean(t["assistant"]).lower() == "yes"))
    R.shuffle(recs)
    n, seen = Counter(), set()
    for f, r, s, y in recs:
        if n[y] >= 300 or (f, r) in seen:
            continue
        seen.add((f, r))
        n[y] += 1
        O.add("nlvr2", "nlvr2", "plain", [(f, r, 0, None), (f, r, 1, None)],
              {"images": "Image 1 is the left image and image 2 is the right image."},
              noul_q(f'Is this statement true of the two images: "{s}"'), "true" if y else "false")


# ---------- multiple choice sets ----------
def mc_set(sub, target, nocorrect, shards=None):
    recs = []
    for f, r, ts in load(sub, shards):
        for t in ts:
            p = mc_parse(t["user"])
            m = re.match(r"Answer: ([A-Z])", t["assistant"])
            if not p or not m:
                continue
            q, opts = p
            gi = ord(m.group(1)) - 65
            if gi >= len(opts) or len({key(o) for o in opts}) < len(opts) or not all(key(o) for o in opts) or max(map(len, opts)) > 120:
                continue
            recs.append((f, r, q, opts, gi))
    R.shuffle(recs)
    seen, n, nu = set(), 0, 0
    for f, r, q, opts, gi in recs:
        if (f, r) in seen:
            continue
        seen.add((f, r))
        gold = opts[gi]
        if n < target:
            n += 1
            qq, g = choice_q(q, gold, [o for o in opts if o != gold])
            O.add(sub, sub, "plain", [(f, r, 0, None)], plain_state(), qq, g)
        elif nu < nocorrect and len(opts) >= 3:
            nu += 1
            rest = [o for o in opts if o != gold]
            R.shuffle(rest)
            O.add("unk-nocorrect", sub, "unknown", [(f, r, 0, None)], plain_state(),
                  {"type": "choice", "instructions": q, "criteria": {o: None for o in rest}}, UNK)
        else:
            break


def aokvqa(target=500):
    """Original A-OKVQA train: not difficult_direct_answer, >= 4 of 10 direct answers equal the correct choice and none
    equals a distractor, so the choice is unambiguous."""
    import ast
    recs = []
    for f in sorted(glob.glob("/root/.cache/huggingface/hub/datasets--HuggingFaceM4--A-OKVQA/snapshots/*/data/train-*.parquet")):
        cols = ["question", "choices", "correct_choice_idx", "direct_answers", "difficult_direct_answer"]
        for r, x in enumerate(pq.read_table(f, columns=cols).to_pylist()):
            ch, gi = x["choices"], x["correct_choice_idx"]
            da = [vnorm(a) for a in ast.literal_eval(x["direct_answers"])]
            if (x["difficult_direct_answer"] or len({key(c) for c in ch}) < len(ch) or sum(a == vnorm(ch[gi]) for a in da) < 4
                    or any(vnorm(c) in da for i, c in enumerate(ch) if i != gi)):
                continue
            recs.append((f, r, x["question"], ch, ch[gi]))
    print("aokvqa unambiguous", len(recs))
    for f, r, q, ch, gold in R.sample(recs, target):
        qq, g = choice_q(q, gold, [o for o in ch if o != gold])
        O.add("aokvqa", "aokvqa", "plain", [(f, r, 0, None)], plain_state(), qq, g)


# ---------- text reading (open answers -> choice / noul / rules) ----------
META = {"unanswerable", "answering does not require reading text in the image", "unsuitable", "unsuitable image"}
MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b.*\d|\d.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}")


def atype(a):
    if num(a) is not None:
        return "year" if re.fullmatch(r"(19|20)\d\d", a.strip()) else ("pct" if "%" in a else "num")
    if MONTH.search(a.lower()):
        return "date"
    return "numtext" if re.match(r"[$€£]|\d", a.strip()) else "text"


def qkey(q, n):
    return " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split()[:n])


def open_set(sub, plain, doc="image", rules=None, shards=None, lower=False):
    rules = rules or {}
    recs = []
    for f, r, ts in load(sub, shards):
        for t in ts:
            a = clean(t["assistant"])
            if lower:
                a = a.lower()
            q = first_line(t["user"])
            if a and q and len(a) <= 80 and a.lower() not in META:
                recs.append({"f": f, "r": r, "q": q, "a": a, "t": atype(a)})
    pools = defaultdict(list)
    for x in recs:
        if x["a"].lower() not in ("yes", "no"):
            for n in (5, 4, 3, 2):
                pools[(qkey(x["q"], n), x["t"])].append(x["a"])
            pools[x["t"]].append(x["a"])

    def distr(x, k):
        for n in (5, 4, 3, 2):
            p = pools.get((qkey(x["q"], n), x["t"]), [])
            if len(set(p)) >= 15:
                break
        else:
            p = pools[x["t"]]
        return pick(p, x["a"], k, numeric=x["t"] in ("num", "pct", "year"))

    R.shuffle(recs)
    seen, n = set(), Counter()
    for x in recs:
        if (x["f"], x["r"]) in seen:
            continue
        src = [(x["f"], x["r"], 0, None)]
        yn = x["a"].lower() in ("yes", "no")
        if n["plain"] < plain:
            if yn:
                if n["yn"] > plain * 0.25:
                    continue
                n["yn"] += 1
                q, gold = noul_q(x["q"]), "true" if x["a"].lower() == "yes" else "false"
            else:
                ds = distr(x, R.choice([2, 3, 4]))
                if not ds:
                    continue
                q, gold = choice_q(x["q"], x["a"], ds, sort_numeric=x["t"] in ("num", "pct", "year") and R.random() < 0.5)
            n["plain"] += 1
            O.add(sub, sub, "plain", src, plain_state(), q, gold)
            TEXT_QS.append((q, sub))
        elif not yn and x["t"] in ("num", "pct") and n["thr"] < rules.get("threshold", 0):
            n["thr"] += 1
            threshold_rule(sub, src, x["q"], x["a"], doc)
        elif not yn and n["claim"] < rules.get("claim", 0):
            ds = distr(x, 1)
            if not ds:
                continue
            n["claim"] += 1
            claim(sub, src, x["q"], x["a"], ds[0], doc)
        else:
            continue
        seen.add((x["f"], x["r"]))
    print(sub, dict(n))


TEXT_QS = []  # (question, source) of text-reading choice items for the domain-mismatch unknowns


def ocrvqa():
    rows = load("ocrvqa", 2)
    books = []
    for f, r, ts in rows:
        d = {}
        for t in ts:
            q = first_line(t["user"]).lower()
            if q.startswith("who wrote this book") or q.startswith("who is the author"):
                d["author"] = clean(t["assistant"])
            elif q.startswith("what is the title"):
                d["title"] = clean(t["assistant"])
        if "author" in d and "title" in d and len(d["title"]) <= 80:
            books.append((f, r, d))
    R.shuffle(books)
    for f, r, d in books[:250]:
        o = R.choice(books)[2]
        swap = R.choice([None, None, "title", "author"])
        rec = dict(d)
        if swap and key(o[swap]) != key(d[swap]):
            rec[swap] = o[swap]
        else:
            swap = None
        O.add("rule-record", "ocrvqa", "rule", [(f, r, 0, None)],
              {"order": {"title": rec["title"], "author": rec["author"]},
               "rule": "Ship the order only if the cover in the photo is the ordered book: both the title and the author must match."},
              noul_q("Should the order be shipped?"), "false" if swap else "true")
    open_set("ocrvqa", 400, "book cover", shards=2)


# ---------- domain mismatch unknowns ----------
def domain_mismatch(n_text=150, n_photo=100):
    R.shuffle(PHOTOS_NO_TEXT)
    pool = [x for x in TEXT_QS if x[1] in ("docvqa", "chartqa", "infographic_vqa", "ocrvqa")]
    for (q, sub), (f, r) in zip(R.sample(pool, min(n_text, len(pool))), PHOTOS_NO_TEXT):
        O.add("unk-domain", sub, "unknown", [(f, r, 0, None)], plain_state(), q, UNK)
    charts = [it for s in ("chartqa", "docvqa") for it in O.items[s]]
    for q, it in zip(R.sample(PHOTO_PREMISE_QS, min(n_photo, len(PHOTO_PREMISE_QS))), R.sample(charts, n_photo)):
        O.add("unk-domain", "vqav2", "unknown", list(it["_src"]), plain_state(), q, UNK)


LICENSES = {
    "vqav2": "VQAv2 annotations CC BY 4.0; COCO train2014 images (Flickr, COCO terms of use); COCO instances CC BY 4.0",
    "tallyqa": "TallyQA (research; licence not stated); COCO / Visual Genome images (VG CC BY 4.0)",
    "clevr": "CLEVR CC BY 4.0", "vsr": "VSR Apache-2.0; COCO images", "nlvr2": "NLVR2 CC BY 4.0 annotations; web images, research use",
    "textvqa": "TextVQA CC BY 4.0; OpenImages images (CC BY 2.0)", "st_vqa": "ST-VQA research use (licence not stated)",
    "docvqa": "DocVQA research use (UCSF Industry Documents images)", "chartqa": "ChartQA GPL-3.0",
    "infographic_vqa": "InfographicVQA research use (licence not stated)", "ocrvqa": "OCR-VQA research use (Amazon book covers)",
    "ai2d": "AI2D CC BY-SA 4.0 (AllenAI)", "tqa": "TQA CC BY-NC 3.0", "scienceqa": "ScienceQA CC BY-NC-SA 4.0",
    "iconqa": "IconQA CC BY-SA 4.0", "aokvqa": "A-OKVQA Apache-2.0; COCO images", "visual7w": "Visual7W MIT; COCO images",
}


def main():
    vqav2()
    tallyqa()
    clevr()
    vsr()
    nlvr2()
    mc_set("ai2d", 500, 80)
    mc_set("tqa", 300, 60)
    mc_set("scienceqa", 350, 50)
    mc_set("iconqa", 400, 60)
    mc_set("visual7w", 500, 80, shards=2)
    aokvqa()
    open_set("textvqa", 700, "image", {"claim": 200}, shards=4, lower=True)
    open_set("st_vqa", 400, "image", {"claim": 100})
    open_set("docvqa", 600, "document", {"threshold": 100, "claim": 150}, shards=6)
    open_set("chartqa", 600, "chart", {"threshold": 300, "claim": 80})
    open_set("infographic_vqa", 400, "infographic", {"threshold": 150, "claim": 100})
    ocrvqa()
    domain_mismatch()
    os.makedirs(f"{D}/items", exist_ok=True)
    total = 0
    resolve([it for items in O.items.values() for it in items])
    for source, items in O.items.items():
        for i, it in enumerate(items):
            it["id"] = f"{source}-{i:05d}"
        with open(f"{D}/items/{source}.jsonl", "w") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        total += len(items)
        print(f"{source:18s} {len(items):6d}  gold {Counter(it['gold'] for it in items).most_common(4)}")
    json.dump(LICENSES, open(f"{D}/items/LICENSES.json", "w"), indent=1)
    print("total", total)


def check():
    """Every item is well formed: gold is one of its option labels, images exist."""
    import sys
    sys.path.insert(0, "/root/big")
    import jevsrv
    n = 0
    for f in glob.glob(f"{D}/items/*.jsonl"):
        for line in open(f):
            it = json.loads(line)
            labels, shown = jevsrv.options_for(it["question"])
            assert it["gold"] in labels, (it["id"], it["gold"], labels)
            assert len(set(labels)) == len(labels) and 3 <= len(labels) <= 7, it["id"]
            assert it["images"] and all(os.path.exists(p) for p in it["images"]), it["id"]
            n += 1
    print("check ok", n)


if __name__ == "__main__":
    main()
    check()
