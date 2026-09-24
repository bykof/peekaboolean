# Legacy: Qwen3-VL aesthetics experiments (v1/v2)

Moved from the README on 2026-09-23. Kept for reproducibility of v1/v2; the current
pipeline is described in the README. Commands below use the legacy `train.py`.


The documentation below describes v1/v2 and is retained for reproducibility.
The old v2 run was stopped at the user's request; its checkpoint is preserved.
Its narrow photo-aesthetics results are not general image-question benchmarks.

Trains a scalar candidate-scoring head on a Qwen3-VL backbone, so a served model can
answer Choice / Score / Noul questions about an image with calibrated probabilities,
using whatever rubric arrives in the request.

Every command below is run from this folder (`peekaboolean/`), the project root — the one
holding `pyproject.toml`. Paths like `data/ava` and `runs/v1` are relative to it.

## Environment (RTX PRO 6000 Blackwell, sm_120)

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it once, then
let it build the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if you already have uv
cd peekaboolean
uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

The last line should print the card and `(12, 0)`.

`uv sync` creates `.venv/` from `pyproject.toml` + `uv.lock` and installs the `peekaboolean`
package itself in editable mode, so edits under `src/peekaboolean/` take effect immediately.
There is nothing to activate: prefix commands with `uv run`, or `source .venv/bin/activate`
if you prefer a shell with the environment on `PATH`.

Blackwell needs a recent CUDA, so the CUDA-linked packages — `torch` and `torchvision`
— come from PyTorch's cu128 index, pinned in `pyproject.toml` under `[[tool.uv.index]]`
/ `[tool.uv.sources]`. The default PyPI wheels have no sm_120 kernels and fail at the
first matmul. To move to a newer CUDA, change that index URL and re-run
`uv lock && uv sync`. `torchvision` is not optional: transformers builds the Qwen3-VL
processor with a video processor that imports it, so `AutoProcessor.from_pretrained`
raises without it.

Adding a dependency is `uv add <pkg>` (which updates `pyproject.toml`, the lockfile and
the environment together) rather than `pip install`. If the new package must match the
CUDA build, add it plainly and bind it in `[tool.uv.sources]` by hand — passing
`--index pytorch-cu128` on the command line also lets that mirror serve unrelated
packages, which quietly downgrades things like numpy and tqdm to its pinned copies. The
code uses
`attn_implementation="sdpa"` rather than flash-attention on purpose: flash-attn
frequently lags new architectures by months, and SDPA is fast enough here. Try flash-attn
later as an optimization, not now.

## Run order

**2. Fetch and convert real data.** Both sources come off the Hub ungated:

```bash
uv run peekaboolean-prepare-ava --out data/ava            # ~3.4GB, 25.5k images
uv run python -c "from huggingface_hub import hf_hub_download as d; \
    d('Iceclear/AADB','AADB.zip',repo_type='dataset',local_dir='data/raw')"
uv run peekaboolean-prepare-aadb --zip data/raw/AADB.zip --out data/aadb
uv run peekaboolean-prepare-mix --root data --part ava --part aadb:19437 --out data/mix
```

AVA ships `rating_counts`, the native 10-bin vote histogram, which goes into `hist`
untouched at its native length for the loader to rebin — nothing is reconstructed from
mean scores. AADB becomes one row per (image, attribute) with the attribute named in
`instructions`, so the model must condition on the question instead of emitting one
global quality score. Read the docstring in `prepare_aadb.py` before trusting its
numbers: AADB's 11 attributes are signed [-1, 1] where negative means the attribute
*detracts* (only `score` is [0, 1]), and it records a mean with no dispersion, so its
targets are spikes while AVA's are real vote spreads. `--smooth` widens them.

The mix step exists because AADB emits 12 questions per image and AVA one: concatenated
raw, it is ~84% attribute questions. `aadb:19437` caps it to parity with AVA. Equal row
counts is a starting point, not a tuned ratio.

**2b. Derive the other two primitives.** Everything above is a `score` row, so choice
and noul would train on nothing but the smoke set. Both labels already exist in these
sources; they are just not written down as questions.

```bash
uv run peekaboolean-prepare-derived --from ava  --part data/ava  --out data/ava-noul
uv run peekaboolean-prepare-derived --from aadb --part data/aadb --out data/aadb-pick --kinds choice
uv run peekaboolean-prepare-derived --from aadb --part data/aadb --out data/aadb-noul --kinds noul
uv run peekaboolean-prepare-mix --root data --part ava --part aadb:19437 \
    --part ava-noul:13000 --part aadb-pick --out data/mix2
```

AVA's histogram is a vote over 50+ raters, so the share of them putting a picture above
a threshold is a *measured* probability with its dispersion intact — the supervision a
calibrated noul wants, and rare: most vision datasets give one hard label and so teach
accuracy without teaching uncertainty. Those values spread properly (median 0.20, 10th
to 90th percentile 0.03 to 0.63).

AADB's 11 signed attributes say which one a picture is strongest in, which is a real
choice label. Where the top two are rated within noise of each other there is no true
answer, and `--min-margin` drops those rather than teaching a coin flip — about a third
of them, at the default. Its *nouls* are the weak part and are kept in their own part
for that reason: AADB records a mean with no dispersion, so the value is a monotone
re-expression of that mean rather than a measured rate, and 45% of them land within 0.05
of neutral. Add `--part aadb-noul:N` if you want them; the default mixture above leaves
them out and comes to 62% score, 21% noul, 17% choice.

**3. Train.**

```bash
uv run peekaboolean-train --train data/mix/train.jsonl --val data/mix/val.jsonl \
    --image-root data --model Qwen/Qwen3-VL-8B-Instruct \
    --out runs/v1 --accum 16 --lr 1e-4 --epochs 1 --log-every 25 --eval-every 200
```

`--eval-every` doubles as the checkpoint interval, so on a long run set it to something
you would be willing to lose. Prefix long runs with `PYTHONUNBUFFERED=1` when
redirecting to a file, or Python holds the progress lines in an 8KB buffer and the log
stays empty for hours.

**4. Calibrate on held-out, in-domain rows.**

```bash
uv run peekaboolean-calibrate --adapter runs/v1/final --val data/mix/calib.jsonl \
    --image-root data --model Qwen/Qwen3-VL-8B-Instruct
```

One temperature is not enough. The softmax runs over K candidates and K is whatever the
request asked for, so the scale that makes a 2-level rubric honest is not the one that
makes a 10-level rubric honest; temperatures are fitted per (type, K) and shrunk toward
the global fit so a thin bucket borrows instead of chasing noise. `--levels` decides
which K each score row is calibrated at, which matters more than it looks: without the
sweep the whole calibration set arrives at one K and the per-bucket map is a single
temperature wearing a hat. `serve.py` reads the map and falls back to the global fit for
a bucket that was never seen.

Read the reliability table, not the single number, and note what it is a table of: every
(row, outcome) pair, not top-1 accuracy. When the model says a tenth of the vote lands
on a level, does a tenth land there? That is what `probabilities` promises, and top-1 is
the wrong question for a score — putting 0.45 on the right level and 0.45 on its
neighbour is a good answer, not a failure. Stated 0.9 against actual 0.6 in a
well-populated bin is a model no temperature can rescue.

**5. Check rubric robustness**, which no metric in the training loop covers.

```bash
uv run peekaboolean-robustness --adapter runs/v1/final --val data/ava/val.jsonl \
    --image-root data/ava/images --model Qwen/Qwen3-VL-8B-Instruct --levels 3,5,8
```

Each score row is answered at 3, 5 and 8 levels and every prediction is reduced to its
implied mean on [0, 1], which is comparable across level counts. Read the per-row spread
against the MAE: a spread as large as the MAE means the rubric's shape moves the answer
as much as the model's own error does, and the model is reading the rubric rather than
the picture.

That is now a verdict rather than a hint. The run prints PASS or FAIL against
`--gate-ratio` (spread under half the MAE by default) and exits non-zero on failure, so
it can gate a release the way a test would. It is the one property the API sells and the
one no public benchmark covers, so it deserves the exit code more than `val_score` does.

`--consistency N` adds three invariants that no loss term enforces and that a released
model should be measured against:

- a two-level score and the noul asking the same thing should agree — here they are the
  same computation, so a gap is the primitive talking rather than the picture;
- swapping which of yes and no comes first must not move the answer. This one holds by
  construction, not by training — candidates are scored in separate sequences, so there
  is no position for a prior to attach to — and it measures 0.0000 exactly. It earns its
  place as a regression test: that isolation is what the API sells, and a single-pass
  forward over shared candidates is precisely what would break it without a word;
- the negated question should return one minus the answer. TypeSafe documents this as
  *not* guaranteed by the request format; in this architecture it is at least measurable, and a
  consistency term over negation pairs could enforce it — which would be a property to
  sell rather than a caveat to publish.

Note that `evaluate()` groups metrics by question *type*, and both AVA and AADB emit
`score` rows, so a mixed run reports one blended `val_score`. To see them apart, eval
against `data/ava/val.jsonl` and `data/aadb/val.jsonl` separately.

**6. Answer a request.**

```bash
uv run peekaboolean-serve --adapter runs/v1/final --model Qwen/Qwen3-VL-8B-Instruct \
    --image data/ava/images/10859.jpg --demo
```

A request is one state and a map of named questions, each Choice, Score or Noul with
its own criteria; the reply is the answers keyed the same way, with `probabilities` over
the option keys or level positions, `confidence` as `(K*pmax - 1) / (K - 1)`, and for a
score the probability-weighted level index. `--request req.json` sends your own.

Questions are built through `data.to_example`, the same function the trainer calls, so a
rubric cannot reach the model in one shape during training and another when served. That
sounds like tidiness and is not: skew of that kind shows up as a model that is quietly
worse rather than as an error, and no metric here would say so.

`--mode shared` (the default) encodes the state once and scores every candidate of every
question as a short suffix against its KV cache, so a request costs one vision pass
instead of one per candidate. `--mode naive` keeps the one-at-a-time path. The two are
not supposed to differ in what they answer, so check that on your own adapter before
trusting the fast one, and measure what it buys on your own images:

```bash
uv run peekaboolean-serve --adapter runs/v1/final --image <img> --check
uv run peekaboolean-serve --adapter runs/v1/final --image <img> --bench 8x5,1x64,1x218 --chunk 16
```

`--check` asserts the two agree to within batching arithmetic. Worth knowing before you
read too much into a small gap: the naive path disagrees with *itself* by a comparable
margin when you only change the batch split, and that is the floor any comparison of
this kind is measured against. Bench shapes are questions x candidates — `1x218` is the
size of request that scores a whole document's lines in one call, against an API limit
of 255 options per Choice. The win grows with the picture, because what is shared is the
image: on this card a 640x480 state went 6.4x to 12.8x across those shapes and a
1024x1024 one 12.3x to 22.6x, the largest request falling from 55s to 2.4s.

Every command above is a console script declared in `pyproject.toml`; the module form
(`uv run python -m peekaboolean.train …`) takes the same arguments and is equivalent.

## What to watch on the first real run

- **Peak memory** is printed every log interval. A 10-level rubric on a 1024px image is
  the worst case; if that fits, everything fits. Expect the 4B to sit low enough that you
  can raise `--max-edge` or move to the 8B.
- **Throughput in questions/sec.** Each training question costs K forward passes because
  the vision tower re-encodes the same image per candidate. Serving no longer pays that
  — `serve.py` encodes the state once and runs the candidates against its KV cache — but
  training still does, and the trick is harder here because the backward pass needs the
  graph. Measure before attempting it.
- **Rubric robustness.** Not measured by the training loop. After training, score the
  same images with 3, 5 and 8 levels and check the implied means agree. This is the
  property your API sells and the one no public benchmark covers.

## Files

| File | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Dependencies, torch's cu128 index, console scripts |
| `src/peekaboolean/model.py` | Backbone + LoRA + scalar head, pooling, save/load |
| `src/peekaboolean/data.py` | Row schema, rubric augmentation, histogram rebinning, prompt construction |
| `src/peekaboolean/train.py` | Training loop, EMD / CE / BCE losses, evaluation |
| `src/peekaboolean/calibrate.py` | Temperature fitting, ECE and reliability table |
| `src/peekaboolean/prepare_ava.py` | AVA subset from the Hub -> score rows, native histograms |
| `src/peekaboolean/prepare_aadb.py` | AADB -> one row per (image, attribute), scale mapping |
| `src/peekaboolean/prepare_mix.py` | Merge prepared sets, cap per source, shared image root |
| `src/peekaboolean/prepare_derived.py` | Choice and noul rows derived from AVA votes and AADB attributes |
| `src/peekaboolean/robustness.py` | Same images at 3/5/8 levels, implied means compared |
| `src/peekaboolean/serve.py` | Request -> answers, shared-prefix batching, `--check` and `--bench` |

## Known simplifications

Candidates are scored in independent forward passes, so they cannot attend to each other.
That is correct for the contract — options are judged on their own merits rather than by
debate, which is what lets the rubric be whatever arrives at request time — but it costs
the ability to separate two options that differ only by contrast. In training the image
is also encoded K times; serving is not (step 6). The head pools the final token of each
sequence; mean-pooling over the candidate span is worth an ablation. Only LoRA and the
head train; the vision tower stays frozen, which is right for a first run but is the
obvious next thing to unfreeze if aesthetic quality plateaus.
