# Alternative backbones for peekaboolean (survey, 2026-09-29)

Question: is there an open vision-language model that would be a better or faster backbone
than SmolVLM-500M-Instruct for scoring options with `logit(Yes) − logit(No)` under the
M1 Pro budget (p95 < 500 ms, six questions / 28 options, MPS, fp32, 512 px)?

Sources are model cards, config files, papers, the transformers 5.17.0 source installed in
this repo's `.venv`, and the OpenCompass leaderboard data. Parameter counts come from
instantiating each config on the `meta` device with transformers 5.17.0 (no weights
downloaded) and splitting the parameters into vision + projector, LM blocks and embeddings.
"Self" marks numbers reported by a model's authors, "OC" marks independent
OpenCompass/VLMEvalKit numbers ([data](http://opencompass.openxlab.space/assets/OpenVLM.json),
used by the [leaderboard space](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard);
its `time` field is 20250917, so nothing released after mid-September 2025 has an
independent number).

## 1. Summary

Nothing at ≤ 2B is both clearly better and clearly cheaper than SmolVLM-500M under an
Apache-style licence. The one like-for-like swap is **LFM2.5-VL-450M**: its LM costs about
0.9× SmolVLM-500M's per token with half the layers, and its authors report better scores on
every perception benchmark they list. Its licence caps commercial use at US$10M annual
revenue. The clear quality step is **InternVL3-1B / InternVL3.5-1B** (MME ≈ 1,910 against
≈ 1,400 for SmolVLM-500M, MMBench about 70 against 50), at an estimated 1.6–1.8× the
compute, about 600–700 ms in fp32. That only fits the budget if fp16 or an MLX port buys
about 1.5×. **Qwen3-VL-2B** has the best perception and counting numbers at 2B and its
family is already wired into `model.py`, but at about 4.9× the compute it is a probe for
the MLX-port question, not a drop-in. Qwen3.5-based models (Qwen3.5-2B, MiniCPM-V-4.6,
Intern-Decision) inherit the hybrid linear-attention problem that sank Qwen3.5-0.8B. The
cause is the pure-PyTorch gated-delta-rule loop on MPS, not the parameter count (§3.8).

Shortlist, in the order to probe:

1. **LFM2.5-VL-450M.** Same compute class, 16 instead of 32 LM layers, better self-reported
   POPE, MMStar, RealWorldQA and CountBench, and first-party MLX weights. Main risk: the LFM
   Open License v1.0 revenue cap, plus hybrid conv layers that need two small code changes
   (§8). No independent numbers exist yet.
2. **InternVL3-1B-hf, then InternVL3.5-1B-HF.** Apache/MIT, a plain Qwen KV cache, native
   in transformers, and a large perception gain in both OC and self-reported numbers. Main
   risk: roughly 1.6–1.8× the compute (a 300M vision tower and a longer LM) → likely over
   500 ms in fp32.
3. **Qwen3-VL-2B-Instruct.** Strongest ≤ 2B on counting (CountBench 88.4) and
   hallucination, Apache-2.0, zero new code. Main risk: about 4.9× the compute, around
   1.9 s estimated in fp32. It only matters if the MLX port happens.

## 2. Comparison

### 2a. Architecture, licence, cost

"Compute" is the estimated FLOP ratio for the 28-option request against SmolVLM-500M (§2c).
"Est. p95" scales the measured 390 ms linearly by that ratio. It is a first-order guess, not
a measurement.

| Model | Released | Licence | Vision+proj / LM blocks / emb | LM layers × hidden | Visual tok @512 px | LM attention | transformers 5.17 / MLX | Yes/No single token | Compute | Est. p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SmolVLM-500M** (current) | 2025-01-20 | Apache-2.0 | 98M / 315M / 47M (+47M untied head) | 32 × 960 | 64 | plain, KV cache | `idefics3` ✓ / mlx-vlm ✓ | ✓ | 1.00 | 390 ms (measured) |
| SmolVLM2-500M-Video | 2025-02-11 | Apache-2.0 | same as above | 32 × 960 | 64 | plain | `smolvlm` ✓ / ✓ | ✓ | 1.00 | ≈ 390 ms |
| **LFM2.5-VL-450M** | 2026-04-08 | LFM Open License v1.0 (revenue cap) | 94M / 287M / 67M | 16 × 1024 (10 conv, 6 attn) | 256 (tunable 64–256) | hybrid: short conv (L=3) + GQA | `lfm2_vl` ✓ (card: ≥ 5.1) / ✓, first-party MLX | ✓ | 1.01 (0.80 at 64 tok) | ≈ 395 ms |
| LFM2-VL-450M | 2025-08-12 | LFM Open License v1.0 | 94M / 287M / 67M | 16 × 1024 | 256 | hybrid conv + attn | ✓ (card: ≥ 4.57) / ✓ | ✓ | 1.01 | ≈ 395 ms |
| **InternVL3-1B-hf** | 2025-04-18 | MIT + Apache-2.0 (card text; metadata links the Qwen licence) | 309M / 358M / 136M | 24 × 896 | 256 (one 448 px tile) | plain (Qwen2.5) | `internvl` ✓ / ✓ | ✓ | 1.57 | ≈ 610 ms |
| **InternVL3.5-1B-HF** | 2025-08-29 | Apache-2.0 | 309M / 441M / 156M | 28 × 1024 | 256 (one 448 px tile) | plain (Qwen3) | `internvl` ✓ / ✓ | ✓ | 1.82 | ≈ 710 ms |
| **Qwen3-VL-2B-Instruct** | 2025-10-19 | Apache-2.0 | 407M / 1,409M / 311M | 28 × 2048 | 256 | plain (Qwen3) | `qwen3_vl` ✓ (already in `model.py`) / ✓ | ✓ | 4.9 | ≈ 1.9 s |
| LFM2.5-VL-1.6B | 2026-01-05 | LFM Open License v1.0 | 426M / 1,036M / 134M | 16 × 2048 | 256 | hybrid conv + attn | ✓ / ✓ | ✓ | 3.8 | ≈ 1.5 s |
| Gemma 4 E2B-it | 2026-03-02 | Apache-2.0 | 169M (+ audio) / 1,877M / 2,752M (PLE) | 35 × 1536 (28 sliding, 7 global) | ≤ 280 soft tokens (count at 512 px unverified) | plain, sliding + global | `gemma4` ✓ / ✓ | ✓ | ≈ 5.4 | ≈ 2.1 s |
| FastVLM-0.5B | 2025-08-25 | apple-amlr (research only) | FastViTHD 125M (paper) / 358M / 136M | 24 × 896 | 64 at 512 px (256 at 1024) | plain (Qwen2) | `fast_vlm` ✓ via a third-party HF conversion / ✓ | unchecked | ≈ 1.1 + vision | unverified |
| Qwen3.5-0.8B (tried) | 2026-02-28 | Apache-2.0 | 101M / 498M / 254M | 24 × 1024 (18 linear, 6 full) | 256 | hybrid Gated DeltaNet | `qwen3_5` ✓ / ✓ | ✓ | 1.65 | predicted 645 ms; measured 3–10 s |

Sources: `config.json`, `preprocessor_config.json`/`processor_config.json` and `tokenizer.json`
of each repo (e.g. [LFM2.5-VL-450M config](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/resolve/main/config.json),
[processor_config](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/resolve/main/processor_config.json),
[InternVL3.5-1B-HF config](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF/resolve/main/config.json),
[Qwen3-VL-2B config](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json),
[Gemma 4 E2B config](https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json));
release dates from the Hub API `createdAt`; licences from each card's metadata. The yes/no check
encodes `"Yes"`, `"No"`, `" Yes"` and `" No"` with each `tokenizer.json`; all four are single
tokens wherever it says ✓. The transformers column is the
`MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` of the installed 5.17.0. The MLX column is the
[mlx-vlm model list](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models) at v0.7.4
(2026-09-28), which has `smolvlm`, `idefics3`, `lfm2_vl`, `internvl_chat`, `qwen3_vl`,
`qwen3_5`, `gemma4`, `fastvlm`. None of the shortlisted repos ships a CoreML export.
FastVLM's transformers docs load `KamilaMila/FastVLM-0.5B`, not Apple's repo
([doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fast_vlm.md)).
The LFM2.5-VL-450M card recommends `min_image_tokens=32`, while its `processor_config.json`
ships 64 ([card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)).

### 2b. Quality numbers

"–" means not reported by that source. The MMBench split differs by source: OC uses
TEST_EN_V11, LiquidAI "dev en", Qwen and InternVL "DEV_EN_V1.1 / v1.1 EN". Numbers from
different sources are not strictly comparable.

| Model | Src | MME sum (perc.) | POPE | MMBench | MMStar | AI2D | TextVQA | DocVQA | ChartQA | RealWorldQA | CountBench | HallusionBench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolVLM-500M | OC | 1394.8 (1157.6) | 84.3 | 50.0 | 38.3 | 59.2 | 60.2 | – | 62.8 | 41.0 | – | 31.1 |
| SmolVLM-500M | self [paper T1](https://arxiv.org/abs/2504.05299) | – | – | – | 38.3 | 59.2 | 60.2 | 70.5 | 62.8 | – | – | – |
| SmolVLM2-500M | OC | 1448.3 (1211.2) | 82.7 | 49.8 | 38.2 | 57.3 | 60.3 | – | 59.6 | 49.9 | – | 27.7 |
| SmolVLM2-500M | LiquidAI's VLMEvalKit run | – | 82.67 | 52.32 | 38.20 | – | – | – | – | 49.90 | 61.81 | – |
| LFM2-VL-450M | self [card](https://huggingface.co/LiquidAI/LFM2-VL-450M) | 1229.91 | 83.79 | 56.27 | 40.87 | – | – | – | – | 52.03 | 47.64 | – |
| **LFM2.5-VL-450M** | self [card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) | – | 86.93 | 60.91 | 43.00 | – | – | – | – | 58.43 | 73.31 | – |
| **InternVL3-1B** | OC | 1912.4 (1497.7) | 89.5 | 69.9 | 52.3 | 69.7 | 75.0 | – | 68.1 | 57.0 | – | 37.2 |
| InternVL3-1B | self [InternVL3.5 paper T4–T6](https://arxiv.org/abs/2508.18265) | 1934.4 | 90.7 | 69.9 | 51.5 | 69.4 | 74.1 | 81.9 | 75.3 | 58.2 | – | 41.4 |
| **InternVL3.5-1B** | self [paper T4–T6](https://arxiv.org/abs/2508.18265) | 1910.2 | 86.8 | 69.9 | 51.9 | 71.1 | 71.5 | 85.6 | 77.7 | 57.6 | – | 41.0 |
| **Qwen3-VL-2B** | self [card figure](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_2b_32b_vl_instruct.jpg) | – | – | 77.8 | 58.3 | 76.9 | – | 93.3 | – | 63.9 | 88.4 | 51.4 |
| LFM2.5-VL-1.6B | self [card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | – | – | – | 50.67 | – | – | – | – | 64.84 | – | – |
| FastVLM-0.5B (1024 px) | self [card](https://huggingface.co/apple/FastVLM-0.5B) | – | – | – | – | 68.0 | 64.5 | 82.5 | 76.0 | 56.1 | – | – |

MME is scored with yes/no questions and POPE is yes/no object hallucination, so these two
are the closest public proxies for the noul head. Gemma 4 E2B's card reports no perception
benchmarks, only MMMU-Pro 44.2, MATH-Vision 52.4 and OmniDocBench
([card](https://huggingface.co/google/gemma-4-E2B-it)), so its quality for this task is
unverified. No source reports aesthetics or dating ("era"), two of peekaboolean's known
weak spots.

### 2c. Compute arithmetic

Forward FLOPs ≈ 2 × parameters × tokens. With `--mode shared` the vision tower runs once
over the patches, the LM runs once over the prefix P (visual tokens + about 40 state and
template tokens), and then over the suffixes S = 28 × 55 = 1,540 tokens. Attention of the
suffixes over a ≤ 300-token prefix adds under 5% and is ignored. Embeddings are lookups;
the vocabulary head is skipped (`_inner`).

- SmolVLM-500M: LM 2 × 0.315B × (104 + 1,540) = 1.03 TFLOP; vision 2 × 0.098B × 1,024 patches = 0.20 → **1.24 TFLOP**.
  At a measured 380–400 ms this is about 3 TFLOP/s effective, which is consistent with the
  REPORT's observation that the 28-option request is arithmetic-bound.
- LFM2.5-VL-450M: 2 × 0.287B × (296 + 1,540) = 1.06; vision 2 × 0.094B × 1,024 = 0.19 → 1.25 (**1.01×**).
  With `max_image_tokens=64` (the image drops to 256 px equivalent): 0.94 + 0.05 → 0.99 (0.80×).
- InternVL3-1B: 2 × 0.358B × 1,836 = 1.31; vision 2 × 0.309B × 1,024 = 0.63 → 1.95 (**1.57×**).
- InternVL3.5-1B: 2 × 0.441B × 1,836 = 1.62; vision 0.63 → 2.25 (**1.82×**).
- Qwen3-VL-2B: 2 × 1.409B × 1,836 = 5.17; vision 2 × 0.407B × 1,024 = 0.83 → 6.0 (**4.9×**).
- LFM2.5-VL-1.6B: 2 × 1.036B × 1,836 = 3.80; vision 2 × 0.426B × 1,024 = 0.87 → 4.7 (3.8×).
- Gemma 4 E2B: 2 × 1.877B × (≈ 153 + 1,540) = 6.36; vision ≈ 0.35 → 6.7 (≈ 5.4×; visual token count assumed).
- Qwen3.5-0.8B: 2 × 0.498B × 1,836 = 1.83; vision 0.21 → 2.0 (1.65×), yet it measured 8–25× slower. FLOPs do not predict hybrid linear attention on MPS (§3.8).

Suffix length depends on the tokenizer. 55 is SmolLM2's count, and the Qwen and LFM
tokenizers are not longer for English text (unverified per model). fp32 weight memory is
parameters × 4 bytes: LFM2.5-VL-450M 1.8 GB, InternVL3-1B 3.8 GB, InternVL3.5-1B 4.2 GB,
Qwen3-VL-2B 8.5 GB, Gemma 4 E2B 20 GB (its PLE tables are loaded too), against 2.0 GB for
SmolVLM-500M (4.1 GiB process RAM measured).

## 3. Candidate notes

**3.1 LFM2.5-VL-450M** ([card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)). LFM2.5-350M
LM plus a SigLIP2 NaFlex 86M encoder, which is the same size class as SmolVLM's SigLIP-B/16.
It handles images up to 512 × 512 natively and tiles larger ones. The card's VLMEvalKit
table puts it above SmolVLM2-500M on every perception benchmark listed: MMStar 43.0 vs 38.2,
RealWorldQA 58.4 vs 49.9, POPE 86.9 vs 82.7, CountBench 73.3 vs 61.8. It trails only on
MMMU (32.7 vs 34.1). Its predecessor
LFM2-VL-450M scored *lower* on MME (1229.9 vs 1448.3, [card](https://huggingface.co/LiquidAI/LFM2-VL-450M)),
and LFM2.5's card does not report MME. So the yes/no benchmark is the open question. The
cards claim "2× faster inference speed on GPUs", which does not transfer to MPS fp32. The
[licence](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/resolve/main/LICENSE) is
Apache-like, but §5 conditions commercial use on the licensee staying below a US$10,000,000
annual-revenue "Threshold". Non-profits are exempt for research. That is a revenue cap, not
a non-commercial ban, but it conflicts with REPORT §8's "commercially licensable retrain"
for anyone above the cap. The architecture is hybrid: 10 of 16 layers are gated short
convolutions with a 3-token cache. In transformers 5.17 the off-CUDA path is one depthwise
`F.conv1d` per layer (`models/lfm2/modeling_lfm2.py`, `causal_conv1d_fn`), not a Python
loop, so the Qwen3.5 slowdown should not repeat. This is an inference from the code; §8
measures it.

**3.2 InternVL3-1B / InternVL3.5-1B** ([3.5 paper](https://arxiv.org/abs/2508.18265),
[3.5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF), [3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)).
InternViT-300M plus Qwen2.5-0.5B (v3) or Qwen3-0.6B (v3.5), 448 px tiles, 256 tokens each,
plain decoder with a standard KV cache. InternVL3.5's gains over InternVL3 are in reasoning
(MMMU 44.2 vs 43.4, MathVista 59.3 vs 45.8, paper Table 3). On perception, 3.5-1B is equal
or slightly worse (MME 1910 vs 1934, POPE 86.8 vs 90.7, paper Table 6). For this task
InternVL3-1B is therefore the cheaper and at least as good first probe. The InternVL3-1B
card's text says MIT, with Qwen2.5 as an Apache-2.0 component, while its metadata links the
Qwen licence ([card](https://huggingface.co/OpenGVLab/InternVL3-1B)); read it before a
commercial retrain. InternVL3.5 is plainly Apache-2.0. Independent OC numbers exist for
InternVL3-1B and agree with the self-reported ones within about 1–2 points.

**3.3 Qwen3-VL-2B-Instruct** ([card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)).
The best ≤ 2B perception numbers found: MMBench 77.8, MMStar 58.3, HallusionBench 51.4, and
CountBench 88.4, which bears on the weak counting (Spearman 0.79). It is the teacher's family,
already handled by `_qwen_hidden` and the shared-prefix code, so a probe needs no new code.
At 4.9× the compute it cannot meet 500 ms in PyTorch fp32 on an M1 Pro. It sizes the MLX
question: if fp16 or MLX brings it near 1 s, a 2B backbone is still out of budget.

**3.4 SmolVLM2-500M-Video-Instruct** ([card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)).
Same weights layout as the current backbone, with video added to training. OC shows a wash
on images (MME +54, RealWorldQA +8.9, POPE −1.6, AI2D −1.9, ChartQA −3.2). It is a
zero-code control, not a candidate.

**3.5 FastVLM-0.5B** ([card](https://huggingface.co/apple/FastVLM-0.5B), [paper](https://arxiv.org/abs/2412.13303)).
Designed for Apple silicon. The paper times it on an M1 Max with the vision encoder in
CoreML on the Neural Engine and the LM in MLX fp16: 166 ms TTFT at 1024 px for the 0.5B
model. That is one prefill, not 28 scored suffixes, and not PyTorch MPS. Rejected on
licence: the Apple ML Research licence limits use and derivatives to "Research Purposes",
excluding "use in any commercial product or service"
([LICENSE](https://huggingface.co/apple/FastVLM-0.5B/resolve/main/LICENSE)).

**3.6 Gemma 4 E2B** ([card](https://huggingface.co/google/gemma-4-E2B-it)). Apache-2.0, a
welcome change from the Gemma Terms of Gemma 3/3n. "2.3B effective (5.1B with embeddings)"
through per-layer embeddings, and a configurable visual token budget of 70–1,120. The LM
alone is about 6× SmolVLM's per token, the model carries a ~300M audio tower, and fp32
weights are about 20 GB. Out of budget for the Mac, and the card gives no perception scores.

**3.7 LFM2.5-VL-1.6B** ([card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)). The natural
second size if an MLX port lands (3.8×; MMStar 50.7 and RealWorldQA 64.8, above
InternVL3.5-1B's 50.3 and 57.1 in the same LiquidAI table). Same licence cap as 3.1.

**3.8 Why Qwen3.5-based models stay out.** With no `fla` package (CUDA/Triton only, and not
installed here), transformers 5.17 runs Gated DeltaNet through `torch_chunk_gated_delta_rule`
(`models/qwen3_5/modeling_qwen3_5.py`, lines 300ff). That function contains Python loops
over the chunk size (`for i in range(1, chunk_size)`, line 399) and over chunks (line 418),
in each of the 18 linear-attention layers. On MPS every iteration is a batch of small
kernel launches. This matches Qwen3.5-0.8B measuring 8–25× slower than its 1.65× FLOP
ratio predicts. The following all share that LM and that cost:
[MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) (1.3B = SigLIP-400M + the
Qwen3.5-0.8B text stack, per its config),
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and
[Intern-Decision-0.8B/2B](https://huggingface.co/internlm/Intern-Decision-0.8B). Intern-Decision,
released 2026-09-26, is worth reading as design prior art. It takes the same
state + named-questions request and scores *all* questions in one forward pass through
letter-symbol logits, so options can see each other (the contrast peekaboolean lacks) at
the price of order dependence. It reports 34 ms per query on an RTX 4090, which does not
transfer to MPS. Its benchmarks are its own, not comparable.

## 4. Rejected

- SmolVLM2-2.2B: LM 1.7B, 24 × 2048, about 5× compute ([card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)). No newer SmolVLM exists on the HuggingFaceTB org as of 2026-09-29.
- LFM2-VL-3B / LFM2.5-VL-3B: 30 × 2048 LM, about 6× compute, revenue-capped licence ([card](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)). `LFM2.5-VL-3B-DSpark` (279M) is a speculative-decoding draft model, not a standalone VLM ([card](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark)).
- Qwen3.5-0.8B / 2B, MiniCPM-V-4.6, Intern-Decision: hybrid Gated DeltaNet LM, slow torch fallback on MPS (§3.8).
- Gemma 3 4B, Gemma 3n E2B/E4B, PaliGemma 2 3B: Gemma Terms of Use (gated), ≥ 2B LMs, superseded by Gemma 4 ([gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it), [paligemma2-3b-pt-448](https://huggingface.co/google/paligemma2-3b-pt-448)). PaliGemma 2 is pretrained-only.
- Gemma 4 E2B / E4B: Apache-2.0 but ≈ 5.4× (E2B) and ≈ 12× LM compute (E4B: 3.97B LM-block parameters), with 20–32 GB fp32 weights (§3.6).
- FastVLM-0.5B / 1.5B: research-only licence (§3.5).
- moondream2 (Phi-1.5 LM, 1.9B, custom code): about 4× compute; Moondream 3 / 3.1 are 9B MoE under a custom "moondream-model-license-1.0" ([3.1](https://huggingface.co/moondream/moondream3.1-9B-A2B)).
- Florence-2: encoder-decoder task-prompt model with no instruction-following VQA; not a yes/no judge ([card](https://huggingface.co/microsoft/Florence-2-base)).
- Ovis2-1B (Apache-2.0, OC MME 1719.5, POPE 87.7): custom `trust_remote_code` visual-tokenizer architecture; the native transformers `ovis2` port exists but offers no advantage over InternVL3-1B at similar cost. Ovis2.5-2B: Qwen3-1.7B LM, about 4.5×.
- Granite Vision 3.3-2B / 4.1-4B: 40 × 2048 and 40 × 2560 LMs, at least 5× compute ([3.3-2b](https://huggingface.co/ibm-granite/granite-vision-3.3-2b), [4.1-4b](https://huggingface.co/ibm-granite/granite-vision-4.1-4b)).
- H2OVL-Mississippi-800M: 2024 model, custom code, OC MME 1468.9 and MMStar 39.5, no better than SmolVLM-500M at about 1.7× compute ([card](https://huggingface.co/h2oai/h2ovl-mississippi-800m)).
- MiniCPM-V 4 / 4.5: 4.1B / 8.7B ([4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5)).
- MolmoE-1B: 7.2B-total MoE, custom code, OC MME 1471.1. Molmo2-4B: 36 × 2560 LM and not in transformers 5.17's auto mapping ([card](https://huggingface.co/allenai/Molmo2-4B)).
- Perception LM 1B: FAIR non-commercial research licence, gated, Llama-3.2-1B LM at about 3× ([card](https://huggingface.co/facebook/Perception-LM-1B)).
- Ministral 3 3B: 26 × 3072 LM, 3.8B total ([card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)).
- Eagle2-1B / 2B: CC BY-NC 4.0 ([card](https://huggingface.co/nvidia/Eagle2-1B)). Nemotron Nano VL is 8B / 12B ([12B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)).
- DeepSeek-VL2-Tiny: 3.4B-total MoE, DeepSeek licence, not in transformers' auto mapping ([card](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)).
- jina-vlm (2.4B): CC BY-NC 4.0 ([card](https://huggingface.co/jinaai/jina-vlm)).
- OCR specialists (PaddleOCR-VL-1.6, DeepSeek-OCR 2, Nemotron Parse, jina-ocr-v1): not general VQA.

## 5. Non-generative scorers

**Multimodal rerankers** score a (query, document) pair with one forward pass and no text
generation. This is peekaboolean's shape, with "query" = state + question and
"document" = image + proposed answer.

- [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) (Apache-2.0,
  2026-01-07; [report](https://arxiv.org/abs/2601.04720)) is a Qwen3-VL-2B whose score is
  the yes/no token logit. The card's vLLM example sets `classifier_from_token: ["no", "yes"]`,
  and the model is instruction-aware. It is the same head as ours, trained for retrieval
  relevance rather than answer correctness. A possible initialisation, but at Qwen3-VL-2B's
  4.9× cost.
- [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0): Qwen2-VL-2B with ranking
  losses, CC BY-NC 4.0, so it cannot be used for a commercial retrain.
- [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2):
  1.7B (SigLIP2-400M + a bidirectional Llama-3.2-1B), a mean-pooled binary classification
  head, trained on document-page retrieval. NVIDIA Open Model License plus the Llama 3.2
  licence. Its domain is documents, not photos.

**Contrastive encoders**: [SigLIP 2](https://arxiv.org/abs/2502.14786) (Apache-2.0,
e.g. [base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex)),
[Perception Encoder](https://arxiv.org/abs/2504.13181) (PE-Core, Apache-2.0,
[B16](https://huggingface.co/facebook/PE-Core-B16-224)) and MobileCLIP2
([S0](https://huggingface.co/apple/MobileCLIP2-S0), apple-amlr, research only). The cost is one
image embedding per request plus one short text embedding per option. The text embeddings
cache across images and no LM runs, so it is far below the 28-suffix LM cost; no M1 timing
was found in a primary source. The limits are fundamental. The score is a similarity of
two independent embeddings, so a `state` or instruction can only be folded into the option
text. The model never sees the question and the candidate together with the image. These
encoders also handle negation poorly ([NegBench](https://arxiv.org/abs/2501.09425),
"Vision-Language Models Do Not Understand Negation") and counting poorly
([Teaching CLIP to Count to Ten](https://arxiv.org/abs/2302.12066)). Both are core to noul
and score questions. They are useful as a cheap pre-filter or as an extra frozen feature,
not as the scorer.

## 6. Teacher

Yes, there is a clearly better open teacher of the same class: **Qwen3.6-35B-A3B**
([card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Apache-2.0, 2026-04-15), or its
predecessor Qwen3.5-35B-A3B. On Qwen's own model cards, against the current
Qwen3-VL-30B-A3B-Instruct ([figure](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/table_nothinking_vl-30a3.jpg)):
RealWorldQA 85.3 vs 73.7, MMBench-v1.1 92.8 vs 87.0, HallusionBench 69.8 vs 61.5. The
Qwen3.5-35B-A3B card adds MMStar 81.9 vs 72.1 and CountBench 97.8 vs 89.8
([card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)). All of these are self-reported by
the same organisation. It has about 3B active parameters, like the current teacher, so
labelling throughput should be similar (unverified). Its 35.95B parameters make about 72 GB
in bf16, which fits one 96 GB card with a thin KV budget at `--gpu-memory 0.85`; the official
[FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) halves that. vLLM
registers the architecture `Qwen3_5MoeForConditionalGeneration`
([registry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)),
and the card recommends `vllm>=0.19.0`. Next-token logprobs are standard vLLM sampling
output, as the current pipeline already uses. Two changes are needed. The model thinks by
default, so labelling needs `chat_template_kwargs={"enable_thinking": False}`; otherwise the
first token is `<think>`, not a letter. And the tempering temperatures (T = 1.6 / 1.9 / 2.2)
must be refitted. Alternatives: Gemma 4 26B-A4B (Apache-2.0) scores lower on the same Qwen
table (RealWorldQA 72.2). Qwen3.8-27B ([card](https://huggingface.co/Qwen/Qwen3.8-27B),
Apache-2.0, 2026-08-05) reaches RealWorldQA 85.9 but is dense, about 9× the active
parameters per token. Qwen3.8-Flash-Next (≈ 180B) does not fit.

## 7. Runtime

The repo's `benchmark.py` already has `--dtype float16` ("the one to try on MPS"). It keeps
an fp32 reference to compare answers, and that is the cheapest test of whether InternVL3-1B
(1.57×) can fit. No primary source gives a PyTorch-MPS fp16-vs-fp32 speedup for these
models on an M1, so the gain is unverified until measured. An MLX port is the bigger lever.
mlx-vlm v0.7.4 implements all three shortlisted families, and LiquidAI ships first-party MLX
weights (bf16, 8-bit down to 4-bit) for LFM2.5-VL. Apple's FastVLM paper benchmarks exactly
this split on an M1 Max: the vision encoder in CoreML on the Neural Engine, the LM prefill
through MLX in fp16 (`mlx_lm.cache_prompt`) ([paper, "Benchmarking" paragraph](https://arxiv.org/abs/2412.13303)).
mlx-vlm is generate-oriented, though. A port would need our own prefix-cache + batched-suffix
forward, the LoRA merge and the yes/no head. If MLX fp16 gives about 2× over PyTorch fp32,
the 1.6–1.8× InternVL models fit and Qwen3-VL-2B (4.9×) still does not. That 2× is a guess
to measure, not a sourced number.

## 8. How to verify on the Mac

Each probe uses the *untrained* yes/no head. `CandidateScorer(model_id, head="yesno")` with
fresh LoRA leaves the model's behaviour unchanged, as in REPORT §4.3. That separates backbone
speed and prior knowledge from training. Run on the M1 Pro, MPS, 512 px, with the
`request6` shape of `python -m peekaboolean.benchmark` (six questions, 28 options),
`--repeats 30 --breakdown`. The benchmark takes an adapter directory, so save an untrained
adapter with `save_adapter` first. Pass means p95 < 500 ms (`--target-ms 500`), with answers
matching fp32 wherever a lower precision is used.

1. **LFM2.5-VL-450M.** Time `--mode shared` and `--mode single` at 256 and at 64 image tokens.
   Pass: p95 ≤ 400 ms at 256 tokens, meaning no slower than today. Then run the untrained
   zero-shot check on the 4,000-row validation sample. Pass: VQAv2 yes/no balanced accuracy
   ≥ 0.83 and AI2D ≥ 0.76, SmolVLM-500M's untrained numbers (REPORT §4.3). Also run
   `serve --check`, which must pass below 1e-3. If `--breakdown` shows the prefix or conv
   layers dominating, the conv path is slower on MPS than the code suggests.
2. **InternVL3-1B-hf, then InternVL3.5-1B-HF.** Same timing in fp32 and `--dtype float16`.
   Pass: p95 < 500 ms in any dtype whose answers match fp32. Fallback: fp32 p95 ≤ 750 ms and
   a zero-shot score far enough above SmolVLM-500M to justify waiting for MLX. Same
   zero-shot check.
3. **Qwen3-VL-2B.** This needs no new code. Time it in fp32 and fp16, and read the vision,
   prefix and suffix split to calibrate the §2c model. The zero-shot check gives the quality
   ceiling at 2B. There is no pass/fail on latency: fp16 p95 ≤ 1 s would mean MLX needs
   only about 2× more to reach the budget.

Code paths a new family touches (point only; nothing is implemented here):

- `model.py` `LORA_SUFFIXES` / `language_lora_targets`: LFM2 names its layers `out_proj`,
  `w1`, `w2`, `w3` and `in_proj`. With today's suffixes only `q/k/v_proj` in its 6 attention
  layers would get LoRA (checked on the meta model). InternVL and Qwen use the existing
  names. `VISION_PAT` matches `vision_tower` for both new families; their
  `multi_modal_projector` is neither frozen nor LoRA'd, as today.
- `model.py` `CandidateScorer.hidden`: only idefics3/smolvlm and the Qwen families encode
  the image once and share it across rows. Other families fall through to
  `self.inner(**inputs)`, which drops `image_counts`, so one image against K rows fails in
  `score_single` and in training batches. LFM2-VL and InternVL need an encode-once branch
  like `_qwen_hidden` (scatter features into image-token positions with 1D positions).
- `model.py` `configure_image_size`: add processor branches, for LFM2-VL
  `do_image_splitting=False` and `max_image_tokens`/`min_image_tokens`, for InternVL a
  single 448 px tile (`crop_to_patches=False`). `answer_token_ids` already derives the
  leading space from the chat template. Both new templates end the assistant header with a
  newline, so it picks `"Yes"`.
- `serve.py` `_expand_cache`: LFM2 conv layers are `LinearAttentionLayer`s (transformers
  `cache_utils.py`, `"conv": LinearAttentionLayer`) whose `recurrent_states` is `{0: None}`.
  The existing branch would call `.repeat` on `None` and needs a None guard; the conv state
  itself is tiny (hidden × 3). `score_shared` positions: both new families use 1D positions,
  so the non-Qwen `start = prefix_len` branch applies. Re-measure `SINGLE_PASS_MAX` for
  each backbone.
- `data.py` `build_inputs` / `candidate_prompts`: the prompt is built with
  `apply_chat_template` and an `{"type": "image"}` part, which both processors accept.
  `serve --check` confirms the split point tokenizes identically (`_split_point`).

Claims above marked "unverified" (Gemma 4 visual-token count at 512 px, FastVLM's vision
cost at 512 px, MPS fp16 and MLX speedups, LFM2 conv speed on MPS, teacher throughput) are
the first things these measurements settle.

## 9. Measured, 2026-09-29

### 9a. Mac: fp16 and Qwen3-VL-2B

Setup: M1 Max, MPS, torch 2.11, transformers 5.17, 512 px, 30 repeats, `benchmark.py`
with `--breakdown`, one 960×618 street photo. Qwen3-VL-2B has an untrained yes/no head.

| Model | dtype | request6 p50 / p95 (ms) | Max prob. diff vs fp32 | Peak footprint |
|---|---|---|---|---|
| SmolVLM-500M (v8b) | fp32 | 332 / 342 | – | 4.4 GB |
| SmolVLM-500M | fp16 | 310 / 317 | 0.0009 | 3.0 GB |
| SmolVLM-500M | bf16 | 416 / 448 | 0.0138 | – |
| Qwen3-VL-2B | fp32 | 1172 / 1204 | – | 13.0 GB |
| Qwen3-VL-2B | fp16 | 981 / 990 | 0.0027 | 7.2 GB |
| Qwen3-VL-2B, square image | fp32 | 1510 / 1726 | – | – |

- fp16 buys 1.08× on request6 for SmolVLM and 1.22× for Qwen, well short of the ~1.5× the
  InternVL-1B models need (§7). For SmolVLM the shared prefix forward stays at ~100 ms in
  fp16; the whole gain comes from the suffixes. fp16 does halve the memory.
- bf16 on M1 is slower than fp32 and changes answers. Don't use it.
- Qwen3-VL-2B costs 3.5× SmolVLM on this landscape photo (~160 visual tokens) and
  4.5–5.1× on a square one (256 tokens). That confirms §2c's 4.9×. Projected to the M1 Pro:
  about 1.1 s p95 in fp16, 2.0 s for square images in fp32. It does not fit the budget.
- For Qwen, `SINGLE_PASS_MAX = 8` is the wrong cutoff. choice4 through `shared` takes
  443 ms, through `single` 663 ms (p95, fp16).

Two bugs in the `qwen3_vl` path, found by `serve --check` (shared and naive differ by up to
0.245). SmolVLM is not affected. Both are now fixed, and `serve --check` passes for
Qwen3-VL-2B (≤ 1e-5):

1. `model.py:175` keeps only `get_image_features(...).pooler_output`, and `model.py:183`
   calls the LM without deepstack features. Training, the naive path and `score_single`
   therefore run Qwen3-VL without its deepstack injection into LM layers 0–2. The shared
   prefix (`serve.py:226`) passes `pixel_values` and gets it.
2. `serve.py:135` `_split_point` backs the cut off the `\n\n` run. Qwen tokenizes `.\n\n` as one
   token, so prefix + suffix ids differ from the one-piece prompt whenever the state ends in
   punctuation.

The Qwen timings above were measured with a scratch patch equivalent to the fixes.

fp16 is now the serving default on MPS (`serve.load`; training keeps fp32). Over the 42
demo images × 7 questions, fp16 moved probabilities by at most 0.0022 and changed no top
answer. request6 p95 on the M1 Max is 324 ms, against 342 ms in fp32.

### 9b. Teacher: Qwen3.6-35B-A3B vs Qwen3-VL-30B-A3B

Setup: RTX PRO 6000, vLLM 0.30.0, `prepare_teacher.py --calibrate`. The known-answer set
has 18,345 rows: a small `prepare_general` run over all splits plus count rubrics. Every
question is asked in two option orders. Qwen3.6 runs with `enable_thinking: False`.

| Group (n) | Qwen3-VL-30B-A3B acc / NLL | Qwen3.6 FP8 acc / NLL | Qwen3.6 bf16 acc / NLL | Mean of old + new |
|---|---|---|---|---|
| choice (13,016) | 0.955 / 0.145 | **0.962** / 0.120 | 0.962 / 0.117 | 0.964 |
| noul (4,231) | **0.919** / 0.268 | 0.898 / 0.251 | 0.900 / 0.253 | 0.928 |
| score, count rubrics (1,098) | 0.811 / 0.581 | **0.830** / 0.423 | 0.821 / 0.428 | 0.840 |
| run time (incl. load) | 744 s | 768 s | 783 s | – |

- The differences are well outside noise. On rows where exactly one teacher is right,
  Qwen3.6 wins 298:198 on choice and 92:71 on score, and loses 167:257 on noul. The largest
  single gains are CLEVR (choice 0.905 → 0.949, count rubrics 0.802 → 0.874). The largest
  loss is ChartQA noul (0.850 → 0.770).
- Noul: Qwen3.6 says "yes" too often when the answer is no. Accuracy on no-answers drops
  0.952 → 0.891, on yes-answers it rises 0.890 → 0.907. This is not FP8 (bf16 is the same),
  and a fitted logit offset does not fix it (held-out NLL-optimal offset ≈ +0.1). The old
  teacher leans "no"; an offset of +1.8 brings it to 0.928. So the gap is real error, not
  threshold placement.
- Authoring and labelling on 400 images (`--limit 100` per split): Qwen3.6 keeps 1,440
  questions against 1,160 (+24%). It writes 3 invalid requests against 12, drops 55 for
  order disagreement against 123 and 411 as text-answerable against 589. The yes-share of
  its kept nouls is 0.59 against 0.68, so it is more balanced.
- The mean of both teachers beats each one on every type. `prepare_teacher.py
  --relabel-from` implements it: the second teacher answers the first one's kept questions,
  and `prepare_v6.soften` averages all four views. The commands are in its docstring.
  Run for real with Qwen3-VL-30B-A3B as the second teacher:
  - On the 400-image Qwen3.6 output it kept 1,333 of 1,440 questions. The rest were
    dropped by the second teacher's order-disagreement and mass filters.
  - The teachers agree on the top answer for 91% of choice, 96% of noul and 74% of score
    questions. The noul yes-share is 0.57.
  - The merged calibration (18,345 rows) fits these temperatures, with NLL after fitting:

    | Type | Qwen3.6 alone: T / NLL | Both teachers: T / NLL |
    |---|---|---|
    | choice | 0.8 / 0.115 | 0.8 / 0.102 |
    | noul | 1.5 / 0.235 | 1.3 / 0.185 |
    | score | 1.2 / 0.417 | 1.2 / 0.386 |

    Accuracy with both teachers is 0.964 / 0.930 / 0.837.

Setup notes: on this host FP8 needs `VLLM_USE_DEEP_GEMM=0`, because DeepGEMM JIT-compiles
and there is no CUDA toolkit. Qwen3.6 needs `chat_template_kwargs={"enable_thinking":
False}` in the three `llm.chat` calls of `prepare_teacher.py`, or its first token is
`<think>`. The old teacher ignores the kwarg.
