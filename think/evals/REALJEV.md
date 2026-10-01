# RealJev

2,100 human-labelled items from real-image evaluation sets, converted mechanically into typed Jev requests
(`POST /v1/systemone`, imajev's wire format). No item was written, relabelled or filtered by a model. Built by
`../build_realjev.py` (seed `20261001`); `stats.json` has the per-source skip counts.

## Files

- `records.jsonl`: one record per line: `{id, source, group, wire, gold, task?}`.
  - `wire` is the Jev request with `images` as paths relative to this directory (`jrun.py` inlines them as data URLs):
    `{"state": ..., "questions": {"decision": {"type": "choice"|"noul", "instructions": ..., "criteria": ...}}, "images": [...]}`.
  - `gold` is the option key: the option text for `choice`, `"true"`/`"false"` for `noul`.
  - `group` is the cluster for the bootstrap: the source image (sha1 of its bytes) unless noted below.
  - `task` is the source's own task/category where it has one (MMStar category, CV-Bench type/task, BLINK task, ...).
- `images/<source>/`: the source image bytes unchanged (JPEG/PNG/WEBP). None needed re-encoding.

## Sources

| source | upstream (split) | licence | kept | type | group |
|---|---|---|---:|---|---|
| `mmstar` | `Lin-Chen/MMStar` (val, 1,500) | none stated on the HF card or the GitHub repo; images from the benchmarks MMStar draws on | 300 | choice | image |
| `realworldqa` | `xai-org/RealworldQA` (test, 765) | CC BY-ND 4.0 | 300 (216 choice, 84 yes/no) | choice / noul | image |
| `hallusionbench` | `lmms-lab/HallusionBench` (image split, 951) | BSD-3-Clause (tianyi-lab/HallusionBench) | 300 | noul | category/subcategory/set_id (132 clusters) |
| `pope_adversarial` | `lmms-lab/POPE` Full/adversarial (3,000) | MIT (AoiDragon/POPE); COCO val2014 images under the COCO/Flickr terms | 300 | noul | COCO image (238 clusters) |
| `cvbench` | `nyu-visionx/CV-Bench` (test, 2D 1,438 + 3D 1,200) | Apache-2.0; images from ADE20K, COCO, Omni3D | 300: 75 each Count, Relation, Depth, Distance | choice | image |
| `blink` | `BLINK-Benchmark/BLINK` (val) | Apache-2.0 | 240: 40 each Counting, Relative_Depth, Spatial_Relation, Object_Localization, Multi-view_Reasoning (2 images), Visual_Correspondence (2 images) | choice | image |
| `mme_realworld_lite` | `yifanzhang114/MME-RealWorld-lite-lmms-eval` (1,919; the authors' lmms-eval copy of MME-RealWorld-Lite) | Apache-2.0 (MME-RealWorld card) | 300 | choice | image |
| `jevbench_preview` | Image JevBench public preview, `items-extended-real.json` (128 items), image fetched from each item's source row | per item: CLEVR-HOPE CC BY 4.0, Geometry3K MIT, ArxivQA CC BY-SA 4.0 | 60 of 128 | choice | image |

Total 2,100 (1,416 choice, 684 noul; 80 two-image items).

## Conversion rules

- Seeded sampling: a shuffle seeded with `20261001:<source>`, then the first N items that pass the filters (per-task quotas
  for CV-Bench and BLINK).
- `state` is `"Answer from the image."` for every item except `jevbench_preview`, whose own `state` is kept minus its
  `image` asset-path key (e.g. `{"source": "ArxivQA"}`).
- The question is the source question verbatim. Only answer-format instructions are removed: RealWorldQA's last line
  `Please answer directly with ...`, MMStar's MathVista-style `Hint: Please answer the question and provide the correct option
  letter ...` line (the stem after `Question:` is kept), and BLINK's `Select from the following choices/options.` line. For BLINK the
  stem is the `prompt` field without its option block, because the short `question` field drops the task context (what the drawn
  marks A/B/REF mean).
- Listed options become a `choice` with `criteria = {option text: null}` in source order; gold is the gold letter's option text.
  MMStar `Options: A: .., B: ..` and `Choices:\n(A) ..`, RealWorldQA `A. ..` lines and MME-RealWorld `(A) ..` strings are parsed;
  items whose letters are not A, B, C, ... in order, or with duplicate option texts, are skipped. MME-RealWorld's own
  "The image does not feature the object" option is kept as an ordinary option.
- Yes/no questions without listed options become `noul` (HallusionBench `gt_answer` 1/0, POPE yes/no, RealWorldQA open
  questions answered Yes/No). RealWorldQA open questions with any other answer (numbers, words) are skipped (174).
- Skipped: items needing more than 2 images (none of the chosen tasks), and items whose request with inlined images exceeds
  2 MB (counted while sampling: RealWorldQA 19, HallusionBench 32, MME-RealWorld 17).
- `jevbench_preview`: rubric and gold verbatim. The source row is located by index and checked against the item text
  (CLEVR-HOPE: the config is not named; `HOP00/test_complex_ood` is the only config whose rows match all 20 items; Geometry3K test
  `problem`; ArxivQA train shard 0 question and answer). 68 of 128 cannot be rebuilt exactly and are skipped: FinQA 20
  (`bevaya/FinQA` has no image; the preview image is the builder's own table rendering), ScreenSpot 36 and Multimodal-Mind2Web 12
  (the question asks for a "labelled marker A-E" drawn by the preview builder; the source screenshot has no markers and the
  distractor positions are not published).

## Running

```
venv/bin/python jrun.py --records realjev/records.jsonl --endpoint http://127.0.0.1:8765/v1/systemone --out runs/NAME --workers 32
venv/bin/python jscore.py --records realjev/records.jsonl runs/NAME
venv/bin/python jscore.py --records realjev/records.jsonl --compare runs/A runs/B
```
