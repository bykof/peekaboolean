# Typed visual questions on a laptop: what worked and what did not

This report describes how the v8b checkpoint was built, from v1 to v8b. Negative
results are included: several of them decided the design.

## 1. The task

Applications often need a handful of decisions about one image, not a caption: *what
kind of image is this, how sharp is it, is a person visible, which of these four
descriptions fits best*. The caller already knows the possible answers. A generative
VLM can answer this, but on a laptop it is slow, its output needs parsing, and its
confidence is not a probability.

The request format used here has three question types, all with caller-written
options:

| Type | Caller supplies | Model returns |
| --- | --- | --- |
| choice | named options, each with a free-text description | distribution over the options |
| score | an ordered rubric of any length | distribution over the levels and its expected index |
| noul | a yes/no question, optionally a wording for yes and for no | P(yes) |

A request carries one `state` (the context: a sentence, a persona, a JSON object, or
nothing) and any number of named questions. The target was a p95 below 500 ms for a
realistic request (six questions, 28 options) on an M1 Pro MacBook.

## 2. Design: score options, never generate

Every option becomes its own sequence: `image + state + question + proposed answer`.
The model outputs one number per sequence, and a softmax over a question's options
gives its distribution. From that follow the properties the format needs:

- **Any number of options, any rubric.** Nothing in the architecture fixes K. Training
  rebins native rating histograms to a random number of levels and rewrites option keys
  as letters, numbers, slugs or the answer text, so an unseen rubric is not out of
  distribution.
- **No parsing, no hallucinated answers.** The output is always one of the supplied
  options.
- **Independence.** Options cannot attend to each other, so an answer does not depend
  on option order or on the other questions. The cost: the model cannot compare two
  options that differ only by contrast.
- **Calibration.** Per question type, number of options and image size, one temperature
  is fitted on a held-out split (`calibrate.py`, `postprocess.py`).

### Serving cost

A naive implementation encodes the image once per option: 28 options mean 28 vision
passes. `serve.py --mode shared` encodes the image and state once, keeps the KV cache,
and scores each option as a short suffix against it. `--mode single` packs everything
into one forward pass, which wins on MPS for small requests because launch overhead
dominates there. `serve --check` asserts that all paths agree within 1e-3 in fp32.

## 3. Version history

| Version | Backbone | Change | Outcome |
| --- | --- | --- | --- |
| v1–v2 | Qwen3-VL-4B | scalar head on photo aesthetics (AVA, AADB) | the head worked, but aesthetics alone is not the task; aesthetic scores stayed near a text-only prior ([legacy doc](legacy-qwen-v1-v2.md)) |
| v3–v5 | SmolVLM-256M | typed questions from public VQA data (The Cauldron) | good on benchmarks, weak on requests that look like real usage (teacher choice 0.59) |
| v6 | SmolVLM-256M | + 247k questions written and labelled by a local Qwen3-VL-30B-A3B teacher | teacher choice 0.59 → 0.77, noul 0.66 → 0.87, score Spearman 0.37 → 0.71; public groups unchanged |
| v7 | SmolVLM-500M | teacher v7: blind filter, score target levels, negated twins; aesthetics dropped | 500M fits the latency budget |
| v7a/v7b | SmolVLM-500M | yes/no head instead of a fresh scalar head | best general model; teacher noul 0.94 |
| (side test) | Qwen3.5-0.8B | untrained, yes/no head | 5–10 s per request on the Mac; rejected |
| v8/v8b | SmolVLM-500M | fine-tune v7b on FairFace age, gender, child/adult | age Spearman 0.74 → 0.81, child 0.93 → 0.97, gender 0.88 → 0.96; general groups unchanged |

## 4. Findings

### 4.1 Public benchmarks teach a benchmark dialect

The Cauldron's VQA sets teach the three primitives in a narrow form: no state, bare
answer strings as options, score only for aesthetics, yes/no only as hard 0/1. v5 was
good on those sets (VQAv2 choice 0.90, TextVQA 0.95) but reached only 0.59 on requests
in the served shape: a state, descriptive options, rubrics with words instead of
numbers. Adding teacher-written requests in that shape (v6) raised choice accuracy by 18 points,
yes/no balanced accuracy by 21 points and score Spearman from 0.37 to 0.71, and left
every public group within its 95% confidence interval.

### 4.2 A local teacher: write, label, filter

`prepare_teacher.py` runs Qwen3-VL-30B-A3B via vLLM on the training machine, so no image
leaves it. Per image, in separate prompts:

1. **Author** one request in the served shape: a state and 3–6 named questions of a
   prescribed type mix. Some yes/no questions are steered toward "no" so the labels stay
   balanced; score questions are steered toward a random target level.
2. **Label** every question as a lettered multiple choice and read the teacher's
   next-token probabilities over the letters, in two option orders. The two
   distributions are mapped back and averaged, so position bias cancels. Questions where
   the orders disagree, or where the letters carry little probability mass, are dropped.
3. **Blind check** (v7): ask again with no image. A question answered confidently and
   identically without the image is answerable from its wording and is dropped. Before
   this filter, a text-only baseline reached a Spearman of 0.47 on teacher scores, and
   the teacher, shown a blank image, still matched its own choice labels 48% of the time
   (chance ≈ 28%).

The author never sees the labeller's prompt, so "this one should be false" cannot leak
into a label. 61,761 images produced about 175k kept questions.

The teacher's probabilities are soft labels, but they are not calibrated. On public rows
with known answers the teacher reached 96.5% (choice), 92.1% (noul) and 79.9% (score),
and its distributions were tempered by T = 1.6, 1.9 and 2.2 before training
(`prepare_v6.py`).

### 4.3 Reuse the backbone's own yes/no instead of a fresh head

v1–v6 put a new scalar MLP on the pooled hidden state. It starts at zero knowledge and
must learn everything from the training mix. v7a replaced it with the pretrained model's
own judgment: each option becomes the prompt *"Question: … Proposed answer: … Is the
proposed answer correct? Answer yes or no."*, and the score is `logit(Yes) − logit(No)`,
initialised from the LM head. A yes/no question is asked directly and scored as a sigmoid.

Untrained, this already reaches 0.83 balanced accuracy on VQAv2 yes/no and 0.76 on AI2D.
After training, it beats the scalar head in most groups on the same data (v7a vs v6: AI2D 0.79 → 0.89, CLEVR 0.74 → 0.78, teacher score 0.71 → 0.74). One caveat: VQAv2
yes/no still drops slightly with training (0.83 untrained → 0.80 trained), so the
training mix pulls against some pretrained knowledge.

### 4.4 The latency budget decides the backbone

| Backbone (M1 Pro, MPS, fp32, 512 px) | six questions / 28 options, p95 |
| --- | --- |
| SmolVLM-256M | ≈ 230 ms |
| SmolVLM-500M | ≈ 380–400 ms |
| Qwen3.5-0.8B, one pass | ≈ 10 s |
| Qwen3.5-0.8B, shared prefix | ≈ 3 s (estimate from the 3.3× CPU speedup: 15.0 s → 4.5 s) |

Qwen3.5-0.8B, a larger and newer backbone, is compute-bound on the Mac: 0.75B language parameters × about 55 suffix tokens × 28 options. Its hybrid
linear-attention layers carry a recurrent state instead of a plain KV cache, so the
shared-prefix path copies convolution and recurrent states per row. It works (and is
in `serve.py`), but it still misses a 1 s budget. SmolVLM-500M was the largest model
that fit.

### 4.5 Wording in synthetic data becomes the task

FairFace is a set of single-face crops. v8 asked them "Is there a child in the picture?"
In a single-face crop that question means "is *this* person a child", and that is what
the model learned. On scenes with several people this is the wrong question. v8b asks
person-level questions ("Is the person in the picture a child?", "the woman", "the
man") and keeps the teacher's scene-level questions for scenes. Generated labels
are only as good as the question text that goes with them.

### 4.6 A macro metric hides small groups

v8b was fine-tuned from v7b with FairFace rows added. The selection metric (mean NLL
over validation groups, teacher groups up-weighted) barely moved, because age and person
questions are a few groups among dozens. Early stopping therefore kept step 0, the
unchanged v7b. The checkpoint at step 4,000 improved age and person metrics a lot and
left the other groups unchanged, and it was chosen by hand. Next time: explicit group
weights, or report the groups that matter to the release separately.

### 4.7 Aesthetics: not learnable at this scale

AVA and AADB aesthetic scores stayed below a text-only prior in v5 and v6. From v7 on they were dropped from training. The
teacher's questions about those images (sharpness, lighting, content) were kept.

## 5. Results (v8b, held-out test split, 512 px)

| Group | n | metric | question prior | v8b |
| --- | --- | --- | --- | --- |
| all choice | 8,669 | accuracy | 0.33 | 0.86 |
| all noul | 4,287 | balanced accuracy | – | 0.87 |
| all score | 3,636 | Spearman of expected level | – | 0.75 |
| teacher choice | 2,936 | accuracy | 0.33 | 0.78 |
| teacher noul | 2,675 | balanced accuracy | – | 0.94 |
| teacher score | 2,952 | Spearman | – | 0.73 |
| FairFace age (10 bins) | 2,014 | Spearman | – | 0.81 |
| FairFace child / gender | 885 / 1,503 | balanced accuracy | – | 0.97 / 0.96 |

"Question prior" is a text-only baseline that sees the question and options but not
the image. Per-source numbers are in the README and in the JSON reports on the release.

## 6. Limitations

- **Teacher agreement is not ground truth.** The teacher groups measure how well the
  student reproduces a 30B model. The student inherits the teacher's mistakes.
- **No public human-labelled acceptance set** for request-shaped questions. This is the
  most useful thing to add.
- **Benchmark overlap.** The backbone's pretraining includes The Cauldron. The internal
  splits measure what the adapter adds, not generalisation to unseen sources.
- **Faces.** Age, gender and child/adult estimates carry FairFace's biases, and the model
  also answers them for drawings and cartoon characters. Do not use them to make
  decisions about individual people.
- **Independent options.** Contrast questions ("the larger one") are out of scope.

## 7. Reproducing v8b

Commands as run; each stage was a detached job (`python -m peekaboolean.background start
--run-dir runs/<name> -- <command>`).

```bash
# 1. public typed questions from The Cauldron (training partitions)
python -m peekaboolean.prepare_general --out data/general-v3
python -m peekaboolean.prepare_general --out data/general-v5 --reuse-sources data/general-v3

# 2. teacher requests (separate vLLM environment)
VLLM_USE_FLASHINFER_SAMPLER=0 python src/peekaboolean/prepare_teacher.py \
  --out data/teacher-v7 --chunk 2048 --gpu-memory 0.85
VLLM_USE_FLASHINFER_SAMPLER=0 python src/peekaboolean/prepare_teacher.py \
  --out data/teacher-v7 --splits-from data/general-v6 --calibrate 6000

# 3. mixture (v7), then FairFace, then the v8 mixture
python -m peekaboolean.prepare_v6 --public data/general-v5 --teacher data/teacher-v7 \
  --out data/general-v7 --drop-train-sources ava,aadb --balance-teacher-noul 0.6
python -m peekaboolean.prepare_age --out data/age
python -m peekaboolean.prepare_v6 --public data/general-v5 --teacher data/teacher-v7 \
  --out data/general-v8b --drop-train-sources ava,aadb --balance-teacher-noul 0.6 --extra data/age

# 4. v7b: 500M, yes/no head
python -m peekaboolean.pipeline --data data/general-v7 --run runs/v7b \
  --model HuggingFaceTB/SmolVLM-500M-Instruct --head yesno --epochs 2 --max-hours 20 \
  --eval-every 1000 --patience 6 --eval-limit 4000 --workers 12 --micro 8 --max-edge 512 \
  --image-sizes 256,384,512,512 --ablation 240 --eval-teacher-weight 3

# 5. v8b: fine-tune from v7b
python -m peekaboolean.pipeline --data data/general-v8b --run runs/v8b \
  --model HuggingFaceTB/SmolVLM-500M-Instruct --head yesno --init-adapter runs/v7b/step-034000 \
  --lr 3e-5 --epochs 0.5 --max-hours 6 --eval-every 1000 --patience 4 --eval-limit 4000 \
  --workers 12 --micro 8 --max-edge 512 --image-sizes 256,384,512,512 --ablation 240 \
  --eval-teacher-weight 3
```

The teacher needs `VLLM_USE_FLASHINFER_SAMPLER=0` on hosts without `nvcc`. The teacher
images include AVA and AADB, which must be downloaded separately (`prepare_ava.py`,
`prepare_aadb.py`).

## 8. Next steps

- A small human-labelled set of real requests as the acceptance test
- Contrast between options (a second pass that sees all options together)
- An MLX port for the Mac, which would make a larger backbone affordable
- A commercially licensable retrain without DocVQA, ChartQA, AVA and AADB
