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
| v9/v9b | SmolVLM-500M | v8b's recipe; teacher v9: Qwen3.6-35B-A3B writes and labels, Qwen3-VL-30B-A3B labels again, mean of both | on the v9b test split, teacher groups +3 to +5.5 points over v8b, also graded by the old teacher alone; public groups and FairFace unchanged (release 0.2.0) |
| v10/v10b | LFM2.5-VL-450M | v9a's and v9b's data and recipe on a new backbone ([newer-models-and-jev.md](newer-models-and-jev.md)) | v10b against v9b on the same test split: teacher choice / noul / score +6 / +3 / +9 points, CLEVR +16, DocVQA +8, FairFace age Spearman 0.81 → 0.86, selection NLL 0.507 → 0.424; AI2D and Screen2Words 2–3 points lower. request6 on the Mac takes 200 ms, against v0.2.0's 243 ms through the same path (180 ms through `tree`, which LFM2 cannot use) (release 0.3.0) |
| v11 | LFM2.5-VL-450M | fine-tune v10b with a trained can't-tell candidate (`--unknown`) on v9b's rows plus 72,735 teacher questions labelled with a can't-tell option, some with record or rule states | on the v12 test split, abstains on 79% / 30% / 84% of the teachers' can't-tell choice / noul / score questions and on at most 1.5% of the others; the other groups hold. ImajevBench dev + calibration 110 → 118 of 254, but 0 of 24 Unknown items; 247 ms on the Mac (release 0.4.0) |

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

### 4.7 Two teachers, averaged, beat either one

For v9 the teacher was compared on 18,345 public rows with known answers, each asked in
both option orders ([alternative-backbones.md](alternative-backbones.md) §9b).
- Qwen3.6-35B-A3B was more accurate than Qwen3-VL-30B-A3B on choice and score, but worse
  on noul: it says "yes" too often when the answer is no. A logit offset does not fix
  that.
- As an author it kept 24% more questions per image and wrote fewer that are answerable
  without the image.
- The mean of both teachers' distributions beat either teacher alone on every type.

v9 therefore lets Qwen3.6 write and label each request, and Qwen3-VL-30B-A3B label the
same questions again (`prepare_teacher.py --relabel-from`). A question stays only if both
teachers pass the mass and order-agreement filters: 198,814 of 212,733. Trained on this
data with v8b's recipe, v9b beats v8b on the teacher groups of the v9b test split. The
gain holds when those rows are graded by Qwen3-VL-30B-A3B's labels alone, the labeller
v8b learned from.

### 4.8 A newer backbone beat a better teacher

v0.2.0's teacher upgrade moved the teacher groups by 3 to 5.5 points (§4.7). Swapping SmolVLM-500M
for LFM2.5-VL-450M, with v9a's data, settings and seed unchanged, moved them by another 3 to 7. On
the v9b test split:
- teacher choice 0.829 → 0.879, noul 0.895 → 0.921, score 0.780 → 0.851;
- CLEVR choice 0.778 → 0.925, CLEVR count rubrics 0.756 → 0.902, DocVQA 0.869 → 0.940;
- AI2D (−3.1, n = 253) and Screen2Words (−1.6, n = 386) are about one standard error lower.

The new model also learned faster:
- At step 0 it was ahead on 13 of 16 validation groups, with the untrained yes/no head.
- At step 3,000 of 37,761 it was ahead of v9a's final checkpoint.
- Early stopping kept step 15,000.

The backbone is also cheaper to serve: its 16-layer LM, 10 of whose layers are short convolutions,
runs the 28 option suffixes of request6 in 120 ms, against 187 ms for SmolLM2's 32 layers. That
holds even with 2.5–4× more visual tokens. The screen of eight candidate backbones and their Mac
timings are in [newer-models-and-jev.md](newer-models-and-jev.md) §3.

### 4.9 Aesthetics: not learnable at this scale

AVA and AADB aesthetic scores stayed below a text-only prior in v5 and v6. From v7 on they were dropped from training. The
teacher's questions about those images (sharpness, lighting, content) were kept.

### 4.10 A can't-tell learns the kind of unknown it was shown

v11 adds one candidate to every question, "It cannot be determined from the image and the given
information.", scored by the same yes/no head. The teachers label it too: every labelling view
offers the can't-tell option, and its mass becomes the target.

The first author prompt asked for one question per request "whose honest answer cannot be
determined". On a 32-image smoke test, the teacher wrote questions about visibility ("is the base
fully in view?"), which have an answer, and the labeller gave those questions 0.05 can't-tell mass.
Asked instead for a concrete fact the image hides (a name, number or text that is cut off, too
small or out of frame), with concrete options only, the teacher's questions changed. Across the
full 24,063-image run, 47% of the 5,905 steered questions came out as can't-tell, against 3.6% of
the other 72,190.

v11 learned exactly that kind of unknown. On the test split it abstains on 79% of the teachers'
can't-tell choice questions and 84% of the score ones, but only on 30% of the 67 noul ones. It
abstains on at most 1.5% of the others, and the other groups hold. On ImajevBench it abstains on
none of the 24 Unknown items. Those are mostly rules that the photo and the record cannot decide,
and its can't-tell mass there (0.067) is the same as on answerable items (0.066). All 11
abstentions were text-only items answered against a blank grey image: to v11 a blank image is a
hidden fact. The next can't-tell data needs rules that cannot be decided, and requests without an
image.

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
Results for v9b (release 0.2.0), with v8b re-scored on the same rows, are in
[alternative-backbones.md](alternative-backbones.md) §9c. v10b (release 0.3.0) against v9b on
those rows is in the README and in [newer-models-and-jev.md](newer-models-and-jev.md) §3c.

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

## 7. Reproducing v8b, v9b, v10b and v11

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

v9b reuses steps 1 and 4-5 with the two-teacher data in step 2. The run went through
`prepare_general --out data/general-v5` once, and its calibration set was a mixture
without teacher rows. Each stage ran as a detached job.

```bash
# known-answer rows plus count rubrics, for the teacher calibration only
python -m peekaboolean.prepare_v6 --public data/general-v5 --teacher data/no-teacher --out data/general-v9pub

# first teacher writes and labels; second teacher labels the same questions (vLLM environment)
VLLM_USE_DEEP_GEMM=0 python src/peekaboolean/prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --splits-from data/general-v5 --out data/teacher-v9 --chunk 2048 --gpu-memory 0.85
VLLM_USE_DEEP_GEMM=0 python src/peekaboolean/prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --splits-from data/general-v9pub --calibrate 20000 --out data/teacher-v9 --gpu-memory 0.85
python src/peekaboolean/prepare_teacher.py --relabel-from data/teacher-v9 --out data/teacher-v9-2t \
  --chunk 2048 --gpu-memory 0.85
python src/peekaboolean/prepare_teacher.py --relabel-from data/teacher-v9 --splits-from data/general-v9pub \
  --calibrate 20000 --out data/teacher-v9-2t --gpu-memory 0.85

# mixtures, then steps 4 and 5 with data/general-v9 -> runs/v9a and data/general-v9b -> runs/v9b
python -m peekaboolean.prepare_v6 --public data/general-v5 --teacher data/teacher-v9-2t \
  --out data/general-v9 --drop-train-sources ava,aadb --balance-teacher-noul 0.6
python -m peekaboolean.prepare_v6 --public data/general-v5 --teacher data/teacher-v9-2t \
  --out data/general-v9b --drop-train-sources ava,aadb --balance-teacher-noul 0.6 --extra data/age
```

v9a's best checkpoint was step 36,000. For v9b, early stopping again kept step 0 (§4.6).
Step 3,000 was chosen by hand on validation. `VLLM_USE_DEEP_GEMM=0` is needed for FP8
checkpoints on hosts without a CUDA toolkit.

v10 and v10b (release 0.3.0) are v9a and v9b on LFM2.5-VL-450M: the same data, settings and
seed. LFM2 batches carry more tensors, and 1,024 file descriptors run out between DataLoader
workers, hence the `ulimit`.

```bash
ulimit -n 65536
python -m peekaboolean.pipeline --data data/general-v9 --run runs/v10-lfm25 --model LiquidAI/LFM2.5-VL-450M \
  --head yesno --epochs 2 --max-hours 20 --eval-every 1000 --patience 6 --eval-limit 4000 --workers 12 \
  --micro 8 --max-edge 512 --image-sizes 256,384,512,512 --ablation 240 --eval-teacher-weight 3
python -m peekaboolean.train_general --train data/general-v9b/train.jsonl --val data/general-v9b/val.jsonl \
  --out runs/v10b --model LiquidAI/LFM2.5-VL-450M --head yesno --init-adapter runs/v10-lfm25/step-015000 \
  --lr 3e-5 --epochs 0.5 --max-hours 6 --eval-every 1000 --log-every 25 --eval-limit 4000 --patience 4 \
  --workers 12 --micro 8 --max-edge 512 --image-sizes 256,384,512,512 --ablation 240 --eval-teacher-weight 3
# write the picked step to runs/v10b/best.txt (rule below), then calibrate and test it
python -m peekaboolean.postprocess --run runs/v10b --data data/general-v9b
python -m peekaboolean.full_test --adapter runs/v10b/step-008000 --data data/general-v9b --out full-test-v10b.json
```

v10's early stopping kept step 15,000. For v10b the pick rule was fixed before the run, because
for v9b selection NLL alone kept step 0 (§4.6): the best mean of FairFace age Spearman, child and
gender balanced accuracy among checkpoints whose selection NLL is at most step 0's + 0.005.
It picked step 8,000: selection NLL 0.4065 against step 0's 0.4102, and the best FairFace mean.
Early stopping on selection NLL alone would have kept step 5,000 (0.3991).

v11 (release 0.4.0): a new teacher run with the can't-tell option, a new seed and a limit of
15,000 training images (plus every validation, calibration and test image), relabelled by the
second teacher, mixed with v9b's rows as they are, then a fine-tune of v10b.

```bash
# teacher and calibration (vLLM environment; FP8 needs VLLM_USE_DEEP_GEMM=0)
python src/peekaboolean/prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 --seed 41 --unknown \
  --splits-from data/general-v5 --limit 15000 --out data/teacher-v12 --chunk 2048 --gpu-memory 0.85
python src/peekaboolean/prepare_teacher.py --model Qwen/Qwen3.6-35B-A3B-FP8 --seed 41 --unknown \
  --splits-from data/general-v9pub --calibrate 20000 --out data/teacher-v12 --gpu-memory 0.85
python src/peekaboolean/prepare_teacher.py --unknown --relabel-from data/teacher-v12 --out data/teacher-v12-2t \
  --chunk 2048 --gpu-memory 0.85
python src/peekaboolean/prepare_teacher.py --unknown --relabel-from data/teacher-v12 --splits-from data/general-v9pub \
  --calibrate 20000 --seed 41 --out data/teacher-v12-2t --gpu-memory 0.85
# v9b's rows plus the new teacher rows, twice in train
python -m peekaboolean.prepare_v6 --public data/general-v9b --teacher data/teacher-v12-2t --out data/general-v12 \
  --count-sources none --balance-teacher-noul 0.6 --teacher-repeat 2
python -m peekaboolean.train_general --train data/general-v12/train.jsonl --val data/general-v12/val.jsonl \
  --out runs/v11 --model LiquidAI/LFM2.5-VL-450M --head yesno --init-adapter runs/v10b/step-008000 --unknown \
  --lr 3e-5 --epochs 0.5 --max-hours 8 --eval-every 1000 --log-every 25 --eval-limit 4000 --patience 4 --workers 12 \
  --micro 8 --max-edge 512 --image-sizes 256,384,512,512 --ablation 240 --eval-teacher-weight 3
python -m peekaboolean.postprocess --run runs/v11 --data data/general-v12
python -m peekaboolean.full_test --adapter runs/v11/step-005000 --data data/general-v12 --out full-test-v11.json
```

Early stopping kept step 5,000.

The teacher needs `VLLM_USE_FLASHINFER_SAMPLER=0` on hosts without `nvcc`. The teacher
images include AVA and AADB, which must be downloaded separately (`prepare_ava.py`,
`prepare_aadb.py`).

## 8. Next steps

- A small human-labelled set of real requests as the acceptance test
- Can't-tell data for rules that the photo and the record cannot decide, and for requests
  without an image (§4.10). The can't-tell of v0.4.0 only knows facts the image hides
- Contrast between options (a second pass that sees all options together)
- The prompt rewrite that leaves each option only its own text: about −57% latency, needs a
  fine-tune ([newer-models-and-jev.md](newer-models-and-jev.md) §5). An MLX port no longer looks
  faster than torch 2.14 on MPS (§1 there)
- A commercially licensable retrain without DocVQA, ChartQA, AVA and AADB. LFM2.5's licence caps
  commercial use by revenue, so that retrain needs an Apache or MIT backbone (SmolVLM, InternVL3-1B)
