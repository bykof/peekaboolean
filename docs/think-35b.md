# peekaboolean-think-35b against imajev

peekaboolean-think-35b is a typed-decision model that reasons before it answers: Qwen3.6-35B-A3B (Apache-2.0) with a
LoRA trained on its own correct reasoning traces, read out through the option-letter probabilities after the
reasoning. It speaks Jev's `POST /v1/systemone` with imajev's extensions (`images`, `unknown_probability`,
`abstained`), so every number below comes from the same requests sent to both systems. Code: [`think/`](../think/).

This report covers what was measured, how, and what did not work. Quality was the only target; latency is reported
but was not optimised.

## Summary

| | peekaboolean-think-35b | imajev-4b | ours − imajev, paired 95% CI |
|---|---|---|---|
| ImajevBench dev + calibration (254) | **245 (96.5%)**, ECE 0.014 | 203–208 (79.9–81.9%), ECE 0.057 | +37 to +42 items |
| RealJev, 2,100 real-image items | **78.4%**, ECE 0.038*, Brier 0.338* | 73.1%, ECE 0.080, Brier 0.364 | +5.3 [+3.5, +7.1] |
| imajev's held-out real-photo exam, 1,779 items | **65.6%**, ECE 0.055*, Brier 0.504* | 58.0%, ECE 0.072, Brier 0.563 | +7.6 [+5.1, +10.3] |
| JevBench hard / original / easy (text only) | **103** / 72 / 48 | 79 / 71 / 48 | +24 items on hard |

\* with the shipped real-photo calibration map (`calibration.json`); raw numbers are in the tables below.

- It wins on every benchmark, on accuracy and on Brier score, with paired intervals that exclude zero.
- It loses on three of the 13 sub-panels: LIVE-in-the-Wild image-quality scoring (−24.8 points), POPE adversarial
  object presence (−4.0) and the 60 rebuilt Image JevBench preview items (−5.0, not significant).
- It runs on a 64 GB M1 Max: the 8-bit MLX model answers the ImajevBench calibration split 76/81, as on the server.
- Training added +2.1 ± 1.8 points over the untrained base on the 382 RealJev items both were run on, but cost
  text-only reasoning on JevBench hard (103 against the base's 108 of 111). Most of the margin over imajev comes from
  the base model and the reasoning readout, not from the fine-tune.

## Setup

**Systems.**
- **imajev-4b** (`mohit67890/imajev-4b@c9e5f132`, its phase-3 adapter, the one that is #1 on Image JevBench v0.1.4),
  run by us on the same server through its own `scripts/playground/server.py` as shipped: 4 option rotations and its
  `calibration-rot4.json`. On ImajevBench we also ran its leaderboard protocol (`run_local_v2.py`, full rotations).
  Its server rejects requests over 4,096 tokens; the 92 RealJev items it rejected were retried with images downscaled
  to 1,280 px, and its best answer is the one scored.
- **Qwen3.6-35B-A3B zero-shot**: the same front end (`think/jevsrv.py`) on Qwen's FP8 checkpoint, no training.
- **peekaboolean-think-35b**: the merged LoRA model, served with online FP8 through the same front end.
- **Qwen3.8-27B zero-shot**: tried as a stronger base mid-way (see "What did not work").

**Readout.** The prompt lists the options as letters with "Unknown" last. The model reasons, ends with
`Answer: <letter>`, and the letter-token probabilities at that position are the answer distribution. Four reasoning
paths are sampled (T = 0.6) and their distributions averaged. Abstention is "Unknown has the highest probability".

**Calibration.** `think/calib.py` maps the averaged distribution through p' ∝ (p + ε)^(1/T), which never changes an
answer or an abstention. It is fitted on held-out items only, never on an evaluation set: 490 held-out training-style
items (`calibration-q36-v1.json`) and 558 items from the unused remainder of RealJev's seven public sources, disjoint
from RealJev by item, image and cluster (`build_realjev_calib.py`). Both raw and calibrated numbers are shown.

**Evaluation sets.**
- **ImajevBench v2.0-lite** dev + calibration (254 items with gold), through imajev's own harness and scorer
  (`imajev_bench score --allow-draft`). The test split's gold is withheld by its maintainer; our predictions for it
  are prepared for submission (`runs/q36-v1-test`), not sent.
- **RealJev**: 2,100 human-labelled real-image items from MMStar, RealWorldQA, HallusionBench, POPE (adversarial),
  CV-Bench, BLINK, MME-RealWorld-lite and the rebuildable Image JevBench public-preview items, converted mechanically
  ([`think/evals/REALJEV.md`](../think/evals/REALJEV.md)).
- **imajev's held-out real-photo exam**, rebuilt with imajev's own converters (MM-UPD + TUBench abstention, CountQA,
  Fashionpedia, LIVE-in-the-Wild, MuirBench two-image pairs); a seeded 1,779-item subsample (400 per panel, all pairs).
- **JevBench** public splits (text only), through its own harness and the TypeSafe adapter.

Correct means the gold option, or abstaining when the gold is Unknown. Errors count as wrong. ECE is 10-bin top-label
over the full distribution including Unknown; Brier is the multi-class Brier score.

## Results

Calibration map: eps 1e-10, T 8, fitted on 548 held-out items (runs/realjevcal-q36-v1/predictions.jsonl).

### ImajevBench v2.0-lite, dev + calibration (254 items with gold; the benchmark's own harness and scorer)

| System | dev (173) | calibration (81) | all (254) | ECE | ECE, calibrated | contrast sets all right (45) | Unknown items abstained (24) | false abstentions (230) |
|---|---|---|---|---|---|---|---|---|
| imajev-4b (its HTTP server, as shipped) | 137 | 66 | 203 (79.9%) | 0.057 |  | 29 | 21 | 6 |
| imajev-4b (its leaderboard protocol) | 140 | 68 | 208 (81.9%) | 0.057 |  | 29 | 22 | 3 |
| Qwen3.6-35B-A3B zero-shot | 170 | 77 | 247 (97.2%) | 0.021 |  | 41 | 20 | 0 |
| Qwen3.8-27B zero-shot | 168 | 77 | 245 (96.5%) | 0.038 |  | 44 | 24 | 9 |
| **peekaboolean-think-35b** | 169 | 76 | 245 (96.5%) | 0.014 | 0.145 | 40 | 19 | 0 |

### RealJev (2,100 human-labelled real-image items from 8 public sets)

| System | all | blink | cvbench | hallusionbench | jevbench_preview | mme_realworld_lite | mmstar | pope_adversarial | realworldqa | ECE | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| imajev-4b | 73.1 | 71.7 | 85.7 | 73.7 | 88.3 | 47.3 | 65.3 | 88.3 | 76.3 | 0.080 | 0.364 |
| Qwen3.6-35B-A3B zero-shot (first 404 items only) | 76.7 | – | – | – | – | – | 75.4 | – | 80.4 | 0.180 | 0.393 |
| Qwen3.8-27B zero-shot (first 222 items only) | 79.7 | – | – | – | – | – | 79.7 | – | – | 0.173 | 0.380 |
| **peekaboolean-think-35b** | 78.4 | 78.8 | 89.3 | 78.3 | 83.3 | 57.7 | 77.0 | 84.3 | 82.3 | 0.166 | 0.378 |
| peekaboolean-think-35b, calibrated | 78.4 | 78.8 | 89.3 | 78.3 | 83.3 | 57.7 | 77.0 | 84.3 | 82.3 | 0.038 | 0.338 |

Ours: false_abstain 3.5%; imajev-4b: false_abstain 2.6%.

Paired, ours − imajev-4b (group bootstrap 95% CI):

```
A=runs/realjev-q36-v1
B=runs/realjev-imajev4b-best
                             n      A      B    A-B  95% CI (group bootstrap)
ALL                       2100   78.4   73.1   +5.3  [+3.5, +7.1]
blink                      240   78.8   71.7   +7.1  [+1.2, +13.3]
cvbench                    300   89.3   85.7   +3.7  [+0.0, +7.3]
hallusionbench             300   78.3   73.7   +4.7  [+0.3, +8.9]
jevbench_preview            60   83.3   88.3   -5.0  [-13.3, +3.2]
mme_realworld_lite         300   57.7   47.3  +10.3  [+5.0, +15.6]
mmstar                     300   77.0   65.3  +11.7  [+6.3, +17.3]
pope_adversarial           300   84.3   88.3   -4.0  [-7.6, -0.3]
realworldqa                300   82.3   76.3   +6.0  [+1.3, +10.7]
```

### imajev's held-out real-photo exam (its own converters; 1,779-item subsample)

| System | all | heldout_abstention | heldout_countqa | heldout_fashionpedia | heldout_livewild | heldout_pairs | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| imajev-4b | 58.0 | 78.5 | 38.2 | 58.8 | 57.2 | 55.9 | 0.072 | 0.563 |
| Qwen3.6-35B-A3B zero-shot (first 104 items only: all abstention panel) | 95.2 | 95.2 | – | – | – | – | 0.037 | 0.092 |
| **peekaboolean-think-35b** | 65.6 | 89.8 | 65.8 | 68.0 | 32.5 | 79.9 | 0.256 | 0.583 |
| peekaboolean-think-35b, calibrated | 65.6 | 89.8 | 65.8 | 68.0 | 32.5 | 79.9 | 0.055 | 0.504 |

Ours: unk_recall 65.7%, false_abstain 4.0%; imajev-4b: unk_recall 60.1%, false_abstain 8.6%.

Paired, ours − imajev-4b (group bootstrap 95% CI):

```
A=runs/heldout-q36-v1
B=runs/heldout-imajev4b
                             n      A      B    A-B  95% CI (group bootstrap)
ALL                       1779   65.6   58.0   +7.6  [+5.1, +10.3]
heldout_abstention         400   89.8   78.5  +11.2  [+7.3, +15.3]
heldout_countqa            400   65.8   38.2  +27.5  [+21.4, +33.3]
heldout_fashionpedia       400   68.0   58.8   +9.3  [+3.8, +14.5]
heldout_livewild           400   32.5   57.2  -24.8  [-30.8, -19.3]
heldout_pairs              179   79.9   55.9  +24.0  [+16.5, +31.5]
```

### JevBench public splits (text only; the benchmark's own harness and scorer)

| System | hard (111) | original (72) | easy (48) |
|---|---|---|---|
| imajev-4b | 79/111 (ECE 0.074) | 71/72 (ECE 0.067) | 48/48 (ECE 0.006) |
| Qwen3.6-35B-A3B zero-shot | 108/111 (ECE 0.041) | 72/72 (ECE 0.009) | 48/48 (ECE 0.009) |
| **peekaboolean-think-35b** | 103/111 (ECE 0.036) | 72/72 (ECE 0.007) | 48/48 (ECE 0.009) |

## How it was trained

1. **Items with known answers** (`think/datagen/build_items.py`), from human-labelled *train* splits only:
   the_cauldron subsets (VQAv2, A-OKVQA, NLVR2 with two images, ScienceQA, TextVQA, DocVQA, ChartQA, AI2D, CLEVR,
   VSR, ...), rule and record wrappers whose answer is computed from the gold (thresholds, claims, listing fields),
   and Unknown items made by removing the evidence (false premise, covered region, a rule needing a missing field),
   each with a matched answerable control. Images matching any evaluation image by sha256 or perceptual hash were
   dropped (35 items).
2. **Rejection-sampled self-distillation** (`rft.py`): 3,604 items sampled 2–4 times through the exact serving prompt;
   a trace is kept only when its final letter is the gold one. 2,982 traces (13.5% Unknown) for training, 134 for
   validation. The base already answers 92.5% of these correctly, so the traces mostly sharpen behaviour it has:
   answering "best option" questions instead of abstaining, and abstaining when the evidence is really missing.
3. **LoRA** (`think/train/train_lora.py`): r = 32 on the language model's attention, Gated DeltaNet projections and
   shared-expert MLP (42M parameters; routed experts, router and vision frozen), bf16 base, 2 epochs, loss on the
   assistant turn only. The training sequence is token-identical to what vLLM generates (`check_template.py`).
   Held-out loss was lowest at step 100 of 208 (0.142, from 0.144 at step 25), and that checkpoint ships.
   One RTX PRO 6000 (96 GB): 75 GB peak, ~2,300 tokens/s, 1.5 h.

## Running it

GPU: vLLM with the merged checkpoint and `think/jevsrv.py`. Mac (64 GB): the 8-bit MLX conversion (36 GB) and
`think/jevmlx.py`. Commands: [`think/README.md`](../think/README.md).

**Measured on a MacBook Pro M1 Max, 64 GB** (`think/jevmlx.py`, mlx-vlm 0.7.4, the 8-bit conversion of the merged
model, 36 GB on disk): ImajevBench calibration split 76/81 with one reasoning sample per question, the same as the
server's 4-sample run (76/81); ECE 0.062 raw; latency p50 14 s, p90 60 s per question. Four samples, as on the
server, take four times as long.

## What did not work, or is not shown

- **Image-quality scoring is a real weakness.** On LIVE-in-the-Wild (5-level perceived quality) the model is right
  32.5% of the time against imajev's 57.2%. Reasoning does not help a subjective ordinal judgement, and nothing in our
  training data covers it.
- **One calibration map does not fit every domain.** The model is right ~97% of the time on ImajevBench's synthetic
  scenes and ~78% on real photos, while its vote shares look alike. The shipped `calibration.json` is fitted on 548
  real-photo items from the unused remainder of RealJev's sources (disjoint by item, image and cluster); it brings
  RealJev ECE to 0.038 and the held-out exam (other sources, fully out of sample) to 0.055, but raises
  ImajevBench ECE from 0.014 to 0.145. `calibration-renders.json`, fitted on held-out training-style items, gives
  ImajevBench 0.046 and RealJev 0.098. Calibration never changes an answer.
- **Qwen3.8-27B was tried as the base and dropped.** Zero-shot it tied the 35B-A3B: ImajevBench 245 against 247,
  1,204 training items 93.5% against 93.5%, the first 215 RealJev items 79.5% against 77.2% (+2.3 ± 4.2). It is dense,
  so it costs about 9× the compute per token here and is 5–10× slower on a Mac, and its full pipeline did not fit in
  the time. Its partial runs are kept (`think/drive38.sh`).
- **Base and trained model were served at slightly different precision** (Qwen's block-FP8 checkpoint against online
  FP8 of the merged bf16 weights). The trained-vs-base numbers are matched by item but not by quantisation scheme.
- **Not shown:** the ImajevBench test split (gold withheld; predictions prepared, not submitted) and Image JevBench
  (run by its maintainer on request). Both need a public submission.
- **Licences.** The base is Apache-2.0. The adapter was trained on traces over datasets with mixed licences
  (some non-commercial or research-only), so the adapter and the merged weights are released under CC BY-NC 4.0,
  like the earlier peekaboolean releases.
- **Speed was not a goal.** A decision takes seconds on one GPU (four reasoning samples of a few hundred tokens each)
  against imajev's ~0.35 s.
