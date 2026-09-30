# Newer backbones and Jev-style scoring (research and measurements, 2026-09-30)

Question: can a newer vision-language model on the Hugging Face Hub, or a newer way of
answering Jev-shaped requests, make peekaboolean better or faster? This follows
[alternative-backbones.md](alternative-backbones.md), written one day earlier, and does not
repeat it.

Sources are model cards, config and tokenizer files, the Hub API, papers and official
repositories; each claim links to one. Measurements are this session's:
- **Mac**: M1 Max (32-core GPU, 64 GB), PyTorch MPS, fp16, `benchmark.py` unless noted. The
  latency budget is set on an M1 Pro; expect 1.15–2× slower there (§4).
- **GPU**: the RTX PRO 6000 training server, bf16.

## 1. Summary

Ranked by value for the cost:

1. **torch 2.14 on macOS: −22%, no retraining. Done**, as `pyproject.toml`'s macOS pin.
   request6 p50 on SmolVLM v0.2.0 goes 309 → 243 ms. Against CPU fp32 on the 42 demo images,
   MPS fp16 differs by at most 0.0035 on 2.14 and 0.0038 on 2.11. The likely cause is the
   Metal prefill attention kernels added in torch 2.13; no A/B isolates it (§4).
2. **Tree-packed scoring: a further −27%, exact, no retraining. Done**, as `--mode tree`,
   which `auto` picks on SmolVLM and InternVL.
   - The whole request becomes one forward pass over image+state → question → option, with a
     4D mask that keeps options apart.
   - `serve --check` finds tree = naive to under 1e-5.
   - request6 on torch 2.14 goes 247 → 180 ms, so v0.2.0 as a whole goes 309 → 180 ms.
   - Plain-attention LMs with 1D positions only (§4).
3. **LFM2.5-VL-450M as the backbone: faster, better zero-shot, and it trains faster.**
   - Speed: request6 takes 221 ms against SmolVLM's 309 ms on torch 2.11, and 201 against
     242 ms on 2.14.
   - Quality: with the untrained yes/no head it beats untrained SmolVLM-500M in 13 of 16
     groups. Teacher choice goes 0.55 → 0.69 and ChartQA 0.55 → 0.75.
   - Training: on v9a's data and recipe, after 3,000 of 37,761 steps it is already ahead of
     v9a's final checkpoint, the base of v0.2.0. Selection NLL is 0.492 against 0.506, teacher
     choice / noul / score 0.838 / 0.900 / 0.782 against 0.818 / 0.874 / 0.772. The run
     continues (§3c).
   - Licence: the LFM Open License caps commercial use at US$10M annual revenue. That is
     fine for the CC BY-NC weights, but not for a commercial retrain.
4. **Rewrite the prompt so each option is only its own text: −57% with the tree, needs a
   fine-tune.** request6 costs 124 ms of model time on torch 2.14. The headroom would let
   InternVL3-1B fit (≈ 266 ms estimated, §5).
5. **Jev.** peekaboolean's request format is TypeSafe's text-only Jev API plus an image.
   - Open image-capable Jev models appeared in September 2026: imajev, Visual Jev, Jev-Omni.
   - Two independent image boards exist: Image JevBench and ImajevBench. A
     `POST /v1/systemone` endpoint would give the repo its first third-party numbers (§2).
6. **Bigger models are better but over budget.**
   - Zero-shot selection NLL: LFM2.5-VL-1.6B 0.609 and InternVL3.5-1B 0.716, against 0.821
     for LFM2.5-VL-450M.
   - Their request6 p50 on the M1 Max is 557 ms (torch 2.14) and 653 ms (2.11).
   - VisionPsy-Nano-460M (Apache-2.0, SmolVLM's shapes) is no better than SmolVLM
     zero-shot in this setting (§3a). Its reported benchmark lead does not carry over.

Not worth it now:
- **JEPA-style embedding scoring.** VL-JEPA weights are not downloadable, and text embeddings
  fail on negation (§6).
- **Qwen3.5-class hybrids, including Intern-Decision, in PyTorch MPS.** One request6-sized row
  takes 1,169 ms.
- **InternVL3.5-1B.** request6 p95 is 668 ms.
- **Visual-token pruning.** Visual tokens sit only in the once-encoded prefix, so it saves
  under about 10% here.
- **An MLX-first port.** The one same-chip MLX prefill number (FastVLM, M1 Max) is no faster
  than torch 2.14 measured here.
- **A new teacher.** Nothing that fits one 96 GB card clearly beats the current pair (§7).

## 2. Jev, and who else answers it with images

**What Jev is.** Jev is TypeSafe AI's hosted decision model at `POST /v1/systemone`. It takes
a `state` and named `noul` / `choice` / `score` questions, evaluates "every question against
[the state] in parallel", and is "Text only … No image, audio, or video input"
([models](https://docs.typesafe.ai/models.md), [API](https://docs.typesafe.ai/api)). The repo
already follows it: `processor.jev_prompt`, and `robustness.py` cites TypeSafe's docs.
peekaboolean is Jev plus an image. The weights and architecture are not published.

**Open Jev-shaped models that take images:**

| Model | Backbone | How it scores | Evidence |
|---|---|---|---|
| [Visual Jev](https://arxiv.org/abs/2609.25845) (2026-09-22, [code](https://github.com/guanxuyu-sv/Visual-Jev), Apache-2.0) | Qwen3-VL-4B + LoRA | Image and context encoded once; one isolated suffix per *question* that lists its options; LM-head letter logits | macro 0.706 → 0.761; a typed head adds +0.000 over the letter logits; 5.7 ms per question amortised at 32 questions (RTX 5090) |
| [imajev-2b/4b/9b](https://huggingface.co/mohit67890/imajev-2b) (2026-09-23, Apache-2.0) | Qwen3.5 + LoRA | One pass per question; a 255-code readout; four option orders averaged; an explicit `unknown` option | ~1M decisions, 300k human-labelled from 21 licence-audited image sources; Image JevBench 71.9% public / 71.9% sealed |
| [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | Gemma 4 12B | 256-option head | Image JevBench 67.1 / 80.7 |
| [Intern-Decision-0.8B/2B/4B](https://huggingface.co/internlm/Intern-Decision-0.8B) (2026-09-26, Apache-2.0) | Qwen3.5 (Gated DeltaNet hybrid) | Every question in **one** prompt; reads single-token option symbols before a `<decision>` marker | all 10,751 benchmark rows are text-only, so its image ability is unmeasured; hard labels only; training data undisclosed ([GitHub](https://github.com/InternLM/Intern-Decision)) |

**What transfers to peekaboolean:**
- **Listing options once per question** (Visual Jev) cuts request6's suffix text from 1,035
  to 395 tokens. It also lets options see each other: the contrast that REPORT §6 rules out,
  and that listwise rerankers gain from (LightOn-rerank LW beats its pointwise twin by
  +2.8 nDCG@10, [card](https://huggingface.co/lightonai/LightOn-rerank-LW-2B)). The price is
  option-order bias, which the teacher already handles by averaging two orders (§5).
- **Intern-Decision's all-questions-in-one-prompt** lets questions leak into each other,
  which peekaboolean avoids by design. Keep questions isolated.
- **Endpoint compatibility.** Serving the Jev envelope at `POST /v1/systemone` lets the
  public harnesses call the model.
- **Licence receipts.** imajev's audited image sources (ABO, VizWiz, VisA/DAGM/BTAD, PD12M,
  Wikimedia Commons, Open Images) bear on the commercial retrain in REPORT §8.
- **Checkpoint soup.** imajev ships a 0.5/0.5 average of two adapters. For the hand-picked
  checkpoints of REPORT §4.6 that is a cheap, untested idea.

**Evaluation sets.** None is human-labelled, image-based and request-shaped; REPORT §6's gap
stands. The nearest:
- [Image JevBench](https://benchmarkheaven.com/image-jev-bench): 684 items, one question each.
  The public items keep dataset labels; 333 fresh sealed items are model-authored with
  generated images.
- [ImajevBench](https://huggingface.co/datasets/mohit67890/imajev-bench): 533 AI-generated
  images with a state, audited by a model.
- [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): text only,
  but request-shaped (a state plus five named questions).

## 3. Backbones, measured

### 3a. Zero-shot quality

Each backbone is scored with an **untrained** yes/no head, i.e. fresh LoRA. That measures
what the pretrained model brings before any training (REPORT §4.3). The rows are every row
of the v9b validation split: 21,819 questions at 512 px. v9b is the trained v0.2.0 model on
the same rows, for scale.

| Group (n) | Metric | v9b (trained) | SmolVLM-500M | **LFM2.5-VL-450M** |
|---|---|---|---|---|
| teacher choice (3,459) | acc | 0.824 | 0.552 | **0.691** |
| teacher noul (3,261) | bal. acc | 0.886 | 0.707 | **0.800** |
| teacher score (2,917) | Spearman | 0.784 | 0.377 | **0.478** |
| VQAv2 choice / noul (1,910 / 1,126) | acc / bal. acc | 0.905 / 0.810 | 0.716 / 0.813 | **0.843** / 0.814 |
| DocVQA / ChartQA / TextVQA (1,472 / 570 / 482) | acc | 0.854 / 0.868 / 0.952 | 0.667 / 0.554 / 0.815 | **0.780 / 0.751 / 0.913** |
| AI2D / CLEVR (277 / 584) | acc | 0.903 / 0.807 | 0.736 / 0.462 | **0.758 / 0.606** |
| Screen2Words (403) | acc | 0.953 | **0.814** | 0.797 |
| count rubrics VQAv2 / CLEVR (313 / 113) | Spearman | 0.755 / 0.700 | 0.336 / 0.460 | **0.568 / 0.521** |
| FairFace age (1,920) | Spearman | 0.811 | **0.688** | 0.627 |
| FairFace child / gender (876 / 1,411) | bal. acc | 0.965 / 0.960 | 0.948 / 0.950 | **0.977 / 0.967** |
| selection NLL / macro NLL | | 0.503 / 0.575 | 0.938 / 0.919 | **0.840 / 0.826** |
| wrong-image ΔNLL | | 1.075 | 0.956 | 1.124 |

LFM2.5-VL-450M is better in 13 of 16 groups, equal on VQAv2 yes/no, and worse on
Screen2Words and FairFace age. It also leans on the image more: a wrong image costs it more
NLL.

The other backbones ran on a 4,000-row sample of the same split. The sampling is
training's validation sampling, `eval_indices` with teacher groups × 3. One row whose text
contains `<video>` is dropped (§8). Groups have n ≈ 163 each and the teacher groups
n ≈ 490, so differences under about 0.05 are noise.

| Group | v9b (trained) | SmolVLM-500M | VisionPsy-Nano-460M | **LFM2.5-VL-450M** | InternVL3-1B | InternVL3.5-1B | Qwen3-VL-2B | LFM2.5-VL-1.6B |
|---|---|---|---|---|---|---|---|---|
| teacher choice / noul / score | 0.826 / 0.889 / 0.813 | 0.540 / 0.725 / 0.454 | 0.590 / 0.704 / 0.318 | **0.723 / 0.809 / 0.574** | 0.704 / 0.791 / 0.495 | 0.741 / 0.790 / 0.501 | 0.830 / 0.796 / 0.668 | 0.797 / 0.894 / 0.504 |
| VQAv2 choice / noul | 0.920 / 0.853 | 0.736 / 0.854 | 0.853 / 0.808 | 0.883 / 0.812 | 0.840 / 0.796 | 0.896 / 0.775 | 0.914 / 0.796 | 0.896 / 0.860 |
| DocVQA / ChartQA / TextVQA | 0.865 / 0.865 / 0.939 | 0.669 / 0.552 / 0.804 | 0.693 / 0.460 / 0.871 | 0.877 / 0.742 / 0.883 | 0.896 / 0.675 / 0.914 | 0.883 / 0.742 / 0.926 | 0.982 / 0.847 / 0.957 | 0.939 / 0.804 / 0.957 |
| AI2D / CLEVR / Screen2Words | 0.890 / 0.804 / 0.951 | 0.718 / 0.442 / 0.834 | 0.816 / 0.521 / 0.681 | 0.748 / 0.607 / 0.834 | 0.767 / 0.583 / 0.877 | 0.798 / 0.693 / 0.883 | 0.883 / 0.681 / 0.896 | 0.847 / 0.589 / 0.834 |
| count rubrics VQAv2 / CLEVR | 0.775 / 0.701 | 0.396 / 0.463 | 0.346 / 0.549 | 0.545 / 0.521 | 0.639 / 0.635 | 0.686 / 0.481 | 0.806 / 0.297 | 0.693 / 0.444 |
| FairFace age / child / gender | 0.847 / 0.991 / 0.964 | 0.684 / 0.959 / 0.976 | 0.705 / 0.977 / 0.963 | 0.610 / 0.977 / 0.982 | 0.529 / 0.959 / 0.969 | 0.792 / 0.950 / 0.938 | 0.807 / 0.901 / 0.969 | 0.845 / 0.977 / 0.982 |
| **selection NLL** / macro NLL | 0.489 / 0.564 | 0.922 / 0.905 | 0.959 / 0.976 | **0.821** / 0.819 | 0.782 / 0.825 | 0.716 / 0.770 | 0.801 / 0.817 | **0.609** / 0.658 |
| M1 Max request6 p50, torch 2.11 (§3b) | 309 ms | 309 ms | ≈ 309 ms (same shapes; not measured) | **221 ms** | 454 ms | 653 ms | 981 ms (alternative-backbones.md §9a) | 644 ms |

- **Inside the budget, LFM2.5-VL-450M is the best starting point.** SmolVLM and VisionPsy
  cost the same, and both start far behind it.
- **VisionPsy-Nano-460M**, measured through a scratch shim over its own code:
  - The shim reproduces its processor's token ids and pixels, and its forward's yes/no score
    to four decimals.
  - It ran in fp32, because its attention does not run in bf16.
  - It is no better than SmolVLM here. Its published scores come from inputs tiled up to
    2,048 px; this is one 512 px tile and a yes/no verification prompt.
- **LFM2.5-VL-1.6B is the best untrained model by a wide margin** (selection 0.609), but it
  costs 557 ms on the M1 Max even with torch 2.14 (§3b).
- InternVL3.5-1B is better than InternVL3-1B. Both are slower than LFM2.5-VL-450M, and not
  clearly better on the teacher groups.
- SmolVLM2-500M, the zero-code control, was skipped: its processor needs `num2words`, which
  the environment does not have.

### 3b. Mac latency

request6 (6 questions, 28 options), 512 px, fp16, 30 repeats, stage medians from
`--breakdown`. Two images: the 960 × 618 street photo (512 × 330 after resizing) and its
square centre crop (512 × 512), because LFM2's NaFlex tokens follow the aspect ratio. The
new backbones run with an untrained adapter; latency does not depend on the weights. The
first two rows are interleaved reruns.

| Backbone | Visual tokens | torch 2.11: p50 / p95 (ms) | prefix / suffixes (ms) | torch 2.14: p50 / p95 |
|---|---|---|---|---|
| SmolVLM-500M (v0.2.0) | 64 | 308–309 / 313–319 (square 316 / 335) | 97–98 / 186–187 | 242 / 244 |
| **LFM2.5-VL-450M** | 160 (square 256) | **221 / 228–229** (square 249 / 260) | 80 / 120 | **201 / 203** |
| InternVL3-1B | 256 (one 448 px tile) | 454 / 473 (square 448 / 467) | 211 / 222 | – |
| InternVL3.5-1B | 256 | 653 / 668 | 233 / 401 | – |
| Qwen3-VL-2B (alternative-backbones.md §9a) | ~160 | 981 / 990 | – | – |
| LFM2.5-VL-1.6B | 160 | 644 / 680 | 298 / 326 | 557 / 571 |

- LFM2.5's 16-layer LM (10 of them short convolutions) runs the 28 suffixes in 120 ms,
  against 187 ms for SmolLM2's 32 layers. That holds even with 2.5–4× more visual tokens.
- fp16 moves request6 probabilities against fp32 by at most:
  - 0.0034 for SmolVLM and both LFM2.5 models, under both torch versions;
  - 0.005 for InternVL3-1B;
  - 0.011 for InternVL3.5-1B.
- The small requests (noul, choice4) take 96–120 ms on LFM2.5 against 102–154 ms on SmolVLM.
  LFM2.5's choice4 takes 120 ms through `shared` and 161 ms through `single`, because a long
  visual prefix repeated per row outweighs the second pass. `serve.py` now switches to
  `shared` above 2 candidates for every family except SmolVLM.
- **Hybrid linear attention does not come back in a one-row layout.** Qwen3.5-0.8B, the
  backbone of Intern-Decision-0.8B, takes 1,169 ms p50 for one row of 754 tokens (160 of them
  visual). That is the Jev-style one-prompt layout of request6 (fp16, torch 2.11). The
  transformers torch fallback of Gated DeltaNet is the cost (alternative-backbones.md §3.8). A
  third party runs a Qwen3.5-0.8B decision model in 36.6 ms through Core ML
  ([Kev-0.8B Core ML](https://huggingface.co/FluidInference/kev-0.8b-coreml)), so a custom
  runtime could change that.

### 3c. LFM2.5-VL-450M trained on the v9 data

The run uses v9a's exact recipe and data: `general-v9`, 2 epochs, LoRA r = 32,
lr = 5e-5, `--micro 8`, sizes 256/384/512/512, seed 31. The batches at a given step are
the same questions as v9a's, and both are validated on the same 4,000 rows. The run is
`runs/v10-lfm25` on the server:

```bash
ulimit -n 65536   # LFM2 batches carry more tensors; 1,024 descriptors ran out after the first eval
python -m peekaboolean.pipeline --data data/general-v9 --run runs/v10-lfm25 --model LiquidAI/LFM2.5-VL-450M \
  --head yesno --epochs 2 --max-hours 20 --eval-every 1000 --patience 6 --eval-limit 4000 --workers 12 \
  --micro 8 --max-edge 512 --image-sizes 256,384,512,512 --ablation 240 --eval-teacher-weight 3
```

| Step | Model | selection NLL | teacher choice / noul / score | VQAv2 choice | CLEVR | DocVQA | VQAv2 count |
|---|---|---|---|---|---|---|---|
| 0 | v9a (SmolVLM-500M) | 0.980 | 0.561 / 0.719 / 0.347 | 0.735 | 0.423 | 0.656 | 0.353 |
| 0 | LFM2.5-VL-450M | 0.869 | 0.687 / 0.812 / 0.479 | 0.783 | 0.550 | 0.794 | 0.521 |
| 1,000 | v9a | 0.675 | 0.717 / 0.834 / 0.548 | 0.852 | 0.698 | 0.751 | 0.594 |
| 1,000 | LFM2.5 | 0.532 | 0.819 / 0.879 / 0.755 | 0.868 | 0.852 | 0.894 | 0.703 |
| 2,000 | v9a | 0.643 | 0.719 / 0.831 / 0.602 | 0.852 | 0.693 | 0.772 | 0.594 |
| 2,000 | LFM2.5 | 0.488 | 0.825 / 0.884 / 0.759 | 0.873 | 0.884 | 0.921 | 0.743 |
| 3,000 | v9a | 0.611 | 0.747 / 0.856 / 0.652 | 0.852 | 0.693 | 0.810 | 0.619 |
| 3,000 | **LFM2.5** | **0.492** | **0.838 / 0.900 / 0.782** | **0.889** | **0.857** | **0.910** | 0.715 |
| 36,000 (best) | v9a, the base of v0.2.0 | 0.506 | 0.818 / 0.874 / 0.772 | 0.873 | 0.815 | 0.841 | 0.733 |

After 3,000 of 37,761 steps, LFM2.5-VL-450M is ahead of v9a's final checkpoint on every
column but the counting rubric (n = 164). Early stopping kept step 15,000 (patience 6, so the
run stopped at 21,000). This is v10. On every row of the v9b test split (21,996 questions,
512 px), against v9a, which was trained on the same data:

| Group (n) | metric | v9a (SmolVLM-500M) | v10 (LFM2.5-VL-450M) |
|---|---|---|---|
| teacher choice / noul / score (3,460 / 3,249 / 2,856) | acc. / bal. acc. / Spearman | 0.829 / 0.895 / 0.780 | **0.879 / 0.921 / 0.851** |
| VQAv2 choice / noul (2,006 / 1,160) | acc. / bal. acc. | 0.914 / 0.799 | 0.925 / 0.806 |
| DocVQA / ChartQA / TextVQA choice (1,538 / 530 / 417) | accuracy | 0.869 / 0.874 / 0.962 | **0.940 / 0.906** / 0.964 |
| AI2D / CLEVR / Screen2Words choice (253 / 603 / 386) | accuracy | 0.897 / 0.778 / 0.943 | 0.866 / **0.925** / 0.927 |
| counting rubrics VQAv2 / CLEVR (315 / 119) | Spearman | 0.788 / 0.756 | 0.787 / **0.902** |
| FairFace age / child / gender (2,014 / 885 / 1,503), not trained on | Spearman / bal. acc. | 0.755 / 0.964 / 0.937 | 0.829 / 0.965 / 0.944 |
| selection / macro NLL | | 0.508 / 0.573 | **0.434 / 0.489** |
| wrong-image ΔNLL | | 1.085 | 1.369 |

AI2D (−3.1 points) and Screen2Words (−1.6) are about one standard error lower; everything
else is level or better. The larger wrong-image gap says the new model leans on the image
more.

v10b repeats v9b's FairFace fine-tune from v10's step 15,000: `general-v9b`, lr 3e-5, half an
epoch, patience 4. A pick rule was fixed before the run: the best mean of FairFace age Spearman,
child and gender balanced accuracy among checkpoints whose selection NLL is at most step 0's +
0.005. It picked step 8,000; early stopping on selection NLL alone would have kept step 5,000.
v9b's validation got worse after step 0 (REPORT §4.6). v10b's instead fell from 0.410 to 0.399
at step 5,000, and every checkpoint from step 3,000 on was inside the window.

| Group (full test split) | v9b (release 0.2.0) | v10 | v10b (release 0.3.0) |
|---|---|---|---|
| teacher choice / noul / score | 0.830 / 0.892 / 0.779 | 0.879 / 0.921 / 0.851 | **0.888** / 0.920 / **0.866** |
| FairFace age / child / gender | 0.805 / 0.968 / 0.960 | 0.829 / 0.965 / 0.944 | **0.860 / 0.981 / 0.971** |
| CLEVR / DocVQA choice | 0.769 / 0.860 | 0.925 / 0.940 | **0.934 / 0.941** |
| AI2D / Screen2Words choice | **0.885 / 0.948** | 0.866 / 0.927 | 0.862 / 0.922 |
| selection / macro NLL | 0.507 / 0.574 | 0.434 / 0.489 | **0.424 / 0.485** |
| wrong-image ΔNLL | 1.178 | 1.369 | 1.642 |

The fine-tune added 3 points of age Spearman and 1.6 to 2.7 points of child and gender accuracy
to v10. No other group lost more than half a point; teacher score rose 1.5, CLEVR choice 0.9
and the CLEVR count rubrics 2.6 (n = 119).
Serving v10b:
- Mac (M1 Max, MPS fp16, torch 2.14): request6 200 / 202 ms p50 / p95. Against CPU fp32 on the
  42 demo images, the largest probability difference is 0.0078 and no answer changes.
- RTX PRO 6000 (bf16): 57 / 59 ms.
- Training throughput: 22 questions/s against v9a's 35. LFM2.5 has 2.5–4× more visual tokens.
- transformers falls back to the reference PyTorch `causal_conv1d`, because the CUDA
  package would need a CUDA toolkit on the host.

### 3d. Other new backbones

| Model | Released | Licence | Class | Notes |
|---|---|---|---|---|
| [qvac/VisionPsy-Nano-460M](https://huggingface.co/qvac/VisionPsy-Nano-460M) | 2026-07-13 | Apache-2.0 | 98M vision / 315M LM blocks, 32 × 960 SmolLM2, 64 tokens per 512 px tile: SmolVLM-500M's shapes | In its authors' VLMEvalKit harness: MMStar 47.6 / RealWorldQA 60.0 / POPE 87.9 / MME 1541, against 38.3 / 50.1 / 82.7 / 1455 for SmolVLM2-500M, and ahead of LFM2.5-VL-450M on 16 of 17 benchmarks. That harness reproduces OpenCompass's SmolVLM2-500M numbers within about 1 point. Custom nanoVLM code: ChatML prompt, `<\|image\|>` tokens. Its scores come from inputs tiled up to 2,048 px. **Measured here (§3a): at one 512 px tile its untrained yes/no head is no better than SmolVLM's**, so it is only worth a retrain for its licence |
| [CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) | 2026-08-10 | Apache-2.0 | 1.4B LM, 256 tokens; 4.9× compute | Trails Qwen3-VL-2B on CountBench and HallusionBench in Cohere's own table. Skip for the Mac |
| Intern-Decision, imajev | | | Qwen3.5 hybrids | §2 and §3b |

No new ≤ 2.5B VLM appeared on HuggingFaceTB, OpenGVLab, Qwen, Google, Microsoft, Apple, IBM,
NVIDIA or AllenAI since June 2026. The Hub API sweep lists what each org released. None of
the 2026 models has an independent OpenCompass number: its public data is still the
2025-09-17 snapshot.

## 4. Faster serving without retraining

**torch 2.14.** torch 2.13 added Metal prefill kernels for scaled-dot-product attention; 2.11
still sends q_len > 8 to MPSGraph
([2.13 release](https://github.com/pytorch/pytorch/releases/tag/v2.13.0),
[#181575](https://github.com/pytorch/pytorch/pull/181575)). 2.14 adds a Metal Performance
Primitives path ([#182256](https://github.com/pytorch/pytorch/pull/182256)). Its review
notes that the path is unvalidated on some GPU families.
- Measured with the repo's benchmark on the M1 Max: SmolVLM v0.2.0 request6 309 → 242 ms p50,
  noul 102 → 83 ms, choice4 154 → 126 ms. Probability differences against fp32 are
  unchanged (0.0032).
- Against CPU fp32, an independent implementation, over the 42 demo images × 11 questions
  (462 answers), MPS fp16 differs:
  - on torch 2.14: by at most 0.0035, with 1 top answer changed;
  - on torch 2.11: by at most 0.0038, with 2 changed.

  Every change is a near-tie below 0.003, for example 0.4511 against 0.4491. So the new
  kernel path is no worse on this M1 Max. The M1 Pro remains to be measured.
- The cu128 index the Linux pins use stops at torch 2.11 for cp313 (cu130 has 2.13 and 2.14,
  [index](https://download.pytorch.org/whl/cu128/torch/)). `pyproject.toml` therefore pins
  2.14 / torchvision 0.29 on macOS only (commit `deps: torch 2.14 on macOS`).

**Tree-packed scoring.**
- One forward pass holds the prefix, then each question's shared text once, then each
  option's own tokens.
- Positions: every option continues from the end of its question.
- Mask: a token attends to its ancestor segments and to earlier tokens of its own segment.
- Readout: the last token of each option.
- transformers 5.17 hands a custom 4D mask straight to SDPA
  (`masking_utils.py`, "If the mask is already 4D, simply return as-is").
- This is SpecInfer's tree attention applied to scoring
  ([arXiv 2305.09781](https://arxiv.org/abs/2305.09781)).
- It is implemented as `serve.score_tree` (`--mode tree`), which `auto` picks on
  `TREE_FAMILIES` (idefics3/smolvlm, internvl). Requests over 4,096 packed tokens fall back to
  `shared`.
- `serve --check` compares it with naive, and it agrees to under 1e-5 on both backbones and
  two requests.

End to end with the repo's benchmark (M1 Max, fp16, torch 2.14, p50 / p95):

| Backbone | request | `shared` | **`tree`** |
|---|---|---|---|
| SmolVLM v0.2.0 | request6 | 247 / 253 ms | **180 / 187 ms** |
| SmolVLM v0.2.0 | choice4 / noul | 124 / 105 ms | **99 / 87 ms** |
| InternVL3-1B (untrained) | request6 | 374 / 403 ms | **283 / 298 ms** |

Model-only time from the scratch script `treebench.py`, SmolVLM-500M, request6, fp16. It
includes vision and the LM, and excludes templating and preprocessing. Run on an idle
machine.

| Layout | LM token positions | torch 2.11 p50 / p95 | torch 2.14 p50 / p95 |
|---|---|---|---|
| today: prefix pass + 27 suffixes padded to 45 | 70 + 1,215 | 286 / 291 ms | 225 / 229 ms |
| **tree, same prompt** (exact: fp32 max \|Δ\| 4e-5 against naive) | 777 | **213 / 219** | **165 / 167** |
| tree, rewritten prompt (needs a fine-tune, §5) | 500 | 151 / 152 | 124 / 125 |
| one full sequence per option (naive reference) | 27 × 115 | 516 / 520 | 402 / 419 |

The vision tower alone takes 68 ms on 2.11 and 56 ms on 2.14.

Limits:
- **Plain attention only.** LFM2's causal conv (kernel 3) and Qwen3.5's DeltaNet mix
  neighbouring packed tokens, so siblings would leak into each other. For those, a three-pass
  cache (prefix → questions → options) is the variant, at about 30 ms per extra pass on MPS.
- On torch 2.11, cost per token rises above about 800 packed tokens. On 2.14 it stays flat at
  about 140 µs per token (LM only).
- Training can use the same packing. LM tokens per question fall about 2.2× at K = 4 and 2.9×
  at K = 10 (estimate).

## 5. With retraining: prompt layout and contrast

**Rewrite: instruction before the option.** Today every option row repeats the question and
an 18-token trailer, "Is the proposed answer correct? Answer yes or no." plus the chat tail.
Moving the trailer before `Proposed answer:` makes each option leaf just
`name: description<end_of_utterance>\nAssistant:`. This is the order Qwen3-Reranker uses
([card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)).
- Measured: 124 ms model time on torch 2.14 with the tree (§4).
- Accuracy: no primary source ablates this move for yes/no verification. The nearest is a
  Qwen3-VL-2B reranker where prompt order spread 10.9 points zero-shot but only 3.4 after LoRA
  ([arXiv 2606.10759](https://arxiv.org/abs/2606.10759)).
- So fine-tune from v9b under a new prompt-style name, so that old adapters keep loading, then
  compare on the full test split.
- A milder variant keeps a short trailer: `Proposed answer: X\nCorrect?`.

**InternVL3-1B with tree + rewrite.** The stage sums come to ≈ 266 ms on the M1 Max at torch
2.11 (estimate, random inputs). That is below SmolVLM's shared path today. Its 151 ms vision
tower becomes the largest item, and on an M1 Pro it is borderline (≈ 306–532 ms). The zero-shot
screen (§3a) says whether its quality is worth that. Qwen3-VL-2B stays at ≈ 750 ms even with
both.

**Listing options (contrast).** Visual Jev's per-question listing (§2) answers contrast
questions ("the larger one"), which are currently out of scope.
- It costs order bias. Language models pick option symbols unevenly
  ([PriDe, arXiv 2309.03882](https://arxiv.org/abs/2309.03882)).
- Symbol binding "varies greatly by model"
  ([Robinson et al., arXiv 2210.12353](https://arxiv.org/abs/2210.12353)), which is a risk at
  500M. Average two listing orders, as the teacher already does.
- A version that keeps per-option independence for everything else: an opt-in
  `"contrast": true` question. It gets a second question node listing all options, each option
  still reads its own yes/no, and two orders are averaged. Untested anywhere.

## 6. JEPA

[VL-JEPA](https://arxiv.org/abs/2512.10942) is a 1.6B non-generative model: a frozen V-JEPA 2
ViT-L, a predictor built from Llama-3.2-1B layers, and an EmbeddingGemma-300M text encoder.
It answers discriminative VQA by predicting an answer embedding and picking the nearest
candidate. It reports TallyQA 69.9 and POPE 85.7, against 44.8 and 85.8 for SmolVLM-500M in
its own evaluation.
- The Hub repos are gated under a FAIR non-commercial licence and hold only a README
  ([delong-chen/VL-JEPA](https://huggingface.co/delong-chen/VL-JEPA)).
- Embedding scoring would replace per-option passes with dot products. But bi-encoders reach
  8–39% pairwise accuracy on the NevIR negation benchmark (random 25%), against 51–65% for
  cross-encoders ([arXiv 2502.13506](https://arxiv.org/abs/2502.13506)). Negated and
  descriptive options are core to noul and choice.
- Two hybrids remain speculative:
  - an embedding pre-filter for very large K (the API allows 255 options);
  - an LLM-JEPA-style auxiliary loss during LoRA training
    ([arXiv 2509.14252](https://arxiv.org/abs/2509.14252)), for which evidence exists only on
    text LLMs.

## 7. Teacher and training

- **Teacher: keep Qwen3.6-35B-A3B plus the Qwen3-VL-30B-A3B relabel.** The only sourced gain
  that fits one 96 GB card is Qwen3.8-27B's +0.6 on RealWorldQA
  ([card](https://huggingface.co/Qwen/Qwen3.8-27B)), at about 9× the active parameters. The
  cheap test is a third labeller from another family, averaged with `--relabel-from` and
  scored with `--calibrate`: Gemma 4 31B or Muse Glimmer-30B, both Apache-2.0 and in vLLM 0.30.
- **Training levers with published evidence**, thin at ≤ 1B:
  - **A 3-point LR sweep.** LoRA matches full fine-tuning when its learning rate is about 10×
    higher ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)).
  - **DoRA**, about +1 point on VL-BART and LLaVA
    ([arXiv 2402.09353](https://arxiv.org/abs/2402.09353)).
  - **Type-dependent teacher weights in `soften`.** Qwen3.6 is better on choice and score,
    Qwen3-VL-30B on noul (alternative-backbones.md §9b).
  - **Ground-truth counts.** Replace the teachers' count labels, which are only 81–84%
    accurate, with counts from instance annotations.
  - **Balanced statement-form noul rows** behind one fixed wrapper, for the "statements get
    P ≈ 0.5–0.8" failure. VQAScore uses "Does this figure show '{text}'?"
    ([arXiv 2404.01291](https://arxiv.org/abs/2404.01291)).

## 8. Code on the `next-backbones` branch

- `model.py`:
  - LFM2-VL and InternVL encode each image once and scatter it into the rows that share it
    (`_llava_hidden`).
  - LoRA also targets LFM2's `out_proj`/`in_proj`/`w1`–`w3`.
  - The tokenizer's right padding now beats a processor's own default (InternVL's is left,
    which would break last-token pooling).
  - InternVL uses one 448 px tile. LFM2's token cap follows the image size, and its floor
    follows the cap. Without that, 256 px images were upscaled past the fixed patch padding,
    and batches failed to collate.
- `data.py`, `serve.py`:
  - Images are passed as one list per prompt, which LFM2 requires. SmolVLM and Qwen get
    identical inputs either way.
  - `spatial_shapes` is carried along.
  - `_expand_cache` guards LFM2's empty recurrent slot.
  - The single-pass cutoff is per family (§3b).
  - `score_tree` / `--mode tree` is new (§4).
- `pyproject.toml` / `uv.lock`: torch 2.14 and torchvision 0.29 on macOS, 2.11 elsewhere (§4).
- `serve --check` passes, with worst difference 0.00000, for SmolVLM v0.2.0,
  LFM2.5-VL-450M and InternVL3-1B, on torch 2.11 and 2.14. The tree path is included where it
  applies. `tests/test_image_size.py` covers the processor settings.
- Known issue: user text containing a processor's placeholder strings makes the request
  fail.
  - InternVL raises `UnboundLocalError` on `<video>`, and one v9b validation row has it.
  - SmolVLM raises `ValueError` on `<image>` ("The total number of <image> tokens …"), and
    LFM2 does the same.
  - It is a clean error, not a wrong answer. Still, escape these strings in `render` before
    serving untrusted text.

## 8b. imajev, and peekaboolean on ImajevBench

[imajev](https://github.com/mohit67890/imajev) is #1 of 49 on Image JevBench v0.1.3 (imajev-4b,
composite 76.4, sealed accuracy 83.3%). It is the nearest relative of this project; everything below is
from its README, code and the [ImajevBench dataset](https://huggingface.co/datasets/mohit67890/imajev-bench)
unless marked as measured.

**What it is.**
- **Base and adapter.** Qwen3.5-2B/4B/9B (Apache-2.0) with LoRA (r16/α32; the 4B's latest is r64) on
  every language projection, including DeltaNet. The vision tower is frozen.
- **Readout.** A 255-code decision head at one position per question: options are listed in the prompt,
  and each question is one pass. Four option orders are averaged ("rotations").
- **`unknown`.** A first-class trained option. Answers carry `unknown_probability` and `abstained`.
- **Data.** About 1M decisions:
  - 504k human-labelled, from 36 licence-checked sources, 21 of them image sources;
  - 475k pseudo-labelled by its 9B, kept only when two option orders agree;
  - 72k photo-vs-record and two-photo decisions;
  - 18k hard typed questions, kept only when open teachers agree;
  - 40k soft-target rows.

  The shipped adapter is a 50/50 weight average of the last two stages. The whole project used
  about $676 of GPU time.
- **Calibration.** One temperature per size (1.3–1.7), plus a photo-only bucket.
- **Serving.** `POST /v1/systemone` with `images` (data URLs), up to two images. About 0.1 s raw per
  decision on an H100, 0.35 s with four rotations; MLX weights for the Mac.

**Measured: peekaboolean on ImajevBench v2.0-lite, through its own harness.**
- The harness is `imajev_bench run --adapter imajev-http` against peekaboolean's new
  `POST /v1/systemone`, on the M1 Max at 512 px, with the benchmark's own `_correct`.
- Test gold is withheld (279 items), so the test predictions exist but have to be scored by the
  maintainer.
- The dev and calibration splits have gold: 254 items, 24 of them "can't tell".

| Track (dev + calibration) | n | v0.2.0 (SmolVLM) | LFM2.5-VL-450M, step 9,000 (uncalibrated) | v0.3.0 (v10b) |
|---|---|---|---|---|
| all | 254 | 104 (40.9%) | 110 (43.3%) | 110 (43.3%) |
| visual | 103 | 50 (48.5%) | 59 (57.3%) | 58 (56.3%) |
| joint (photo + record or rule) | 113 | 41 (36.3%) | 40 (35.4%) | 43 (38.1%) |
| text only, answered against a blank image | 38 | 13 (34.2%) | 11 (28.9%) | 9 (23.7%) |
| of these, "can't tell" references | 24 | 0 | 0 | 0 |

For scale, on the test split: SmolVLM2-2.2B scores 28.7%, untuned Qwen3.5-2B 60.2%, imajev-2b
70.3–71.7% and imajev-4b 83.9%. Different split, so compare loosely.

**Where the gap is**, by family (dev + calibration, v0.2.0 / LFM2.5):
- **Rules from the state applied to the photo** (`threshold_rule`, `rule_exception`,
  `multi_step_rule`): about 35–40%. Choice-type threshold rules are the worst: 6/16 and 1/16.
  The teachers never wrote requests like these. The states in the training data are personas,
  contexts or records, but questions rarely require applying a rule stated there.
- **"Can't tell"**: 0/24. There is no abstain output. On the test split this caps peekaboolean
  at 92.5%.
- **Comparison** (31 items): 13/31, the independent-options limit (REPORT §6).
- **Counting** (20): 5 → 8. Text reading (31) is already 23/31 for both, and attribute state 4 → 9/9.
- **Text-only reasoning** (date arithmetic, numerical reconciliation): peekaboolean was never
  trained without an image.

**What peekaboolean can adopt, ranked:**
1. **A trained `unknown`.** Add an "it cannot be determined" candidate row per question, scored
   like any option, and return `unknown_probability` / `abstained` for real. The labels come
   from the teachers: offer a "cannot be determined" letter in `prepare_teacher.py`'s labelling
   prompt and keep its mass.
2. **Rule- and record-grounded requests.**
   - A teacher authoring mode whose state is a record or a rule, with questions that need it
     applied to what is visible: thresholds, exceptions, photo against listing fields.
   - imajev's generators for these are Apache-2.0: `scripts/v2/state_grounded/` builds the
     photo-vs-record pairs and edits.
3. **Contrast.** The opt-in listing node of §5 for comparison questions.
4. **Ground-truth counts** instead of teacher counts (§7).
5. **imajev's licence-checked, human-labelled image sources**, with receipts from `scripts/v2/write_license_receipts*.py` and fetchers for Commons, Open Images and PD12M, for label
   quality and for the commercial retrain.
6. **Checkpoint soup** (50/50 LoRA average) as a cheap alternative to picking checkpoints by hand.

**What doesn't fit:**
- **Qwen3.5 backbones.** Too slow in PyTorch MPS (§3b).
- **Four rotations.** peekaboolean's pointwise scoring is order-invariant by construction.
- **imajev's reasoning-heavy text stages.** peekaboolean serves image requests.

## 9. Next steps

1. **Finish `runs/v10-lfm25` and compare it with v9a: done** (§3c).
2. **Fine-tune it on `general-v9b` with v9b's recipe: done.** v10b is release 0.3.0 (§3c),
   with a model card that states the LFM Open License. Still to do: `benchmark.py` on the
   M1 Pro.
3. **torch 2.14 on macOS: done.** Still to do: measure it on the M1 Pro.
4. **Tree-packed scoring: done** for SmolVLM and InternVL. LFM2 would need a per-question
   cache pass instead.
5. **The prompt rewrite.** It is a fine-tune experiment on the chosen backbone, compared on
   the full test split. For LFM2 the latency gain also needs the per-question pass.
6. **Jev.**
   - `POST /v1/systemone` with the Jev envelope in the local server: done.
   - The ImajevBench test run of v0.3.0 went to the maintainer for scoring, as an issue on
     [mohit67890/imajev](https://github.com/mohit67890/imajev/issues).
   - The weights are public on the Hub as
     [bykof/peekaboolean-450m](https://huggingface.co/bykof/peekaboolean-450m), which Image
     JevBench re-scores on its own schedule; its paid priority evaluation was not requested.

   On that board the smallest image entries are 0.8B (0.35–0.65 public accuracy), and
   imajev-2B reaches 0.72.
7. **Commercial path.** Neither LFM2.5 (revenue cap) nor VisionPsy (weak in this setting)
   solves it. InternVL3-1B with tree + rewrite, or SmolVLM itself, stays the Apache/MIT
   route, together with imajev's licence-audited image sources.
8. **LFM2.5-VL-1.6B, only if the target moves** to newer Macs or GPU serving. A 3,000-step run
   on the same data sizes its gain after training.
9. **Escape placeholder strings in request text: done.** `data.candidate_prompts` puts a space
   after the `<` of any tag-shaped text, so `<image>` no longer makes the processor count a
   second image and `<|im_end|>` no longer closes the user turn.
10. **A trained can't-tell (§8b, items 1 and 2).** The code is in (`--unknown`). A 96-image
    smoke test fixed the author prompt: asked for a question "whose answer cannot be
    determined", the teacher asked whether things were visible, which has an answer. Asked
    for a concrete fact the image hides, 9 of 22 such questions got most of the can't-tell
    mass, against 1 of 291 other questions. The full teacher run and v11 follow.
