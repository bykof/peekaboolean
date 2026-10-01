# peekaboolean-think-35b

A typed-decision model that reasons before it answers. It takes 0–2 images, a `state` and named `noul` / `choice` /
`score` questions and returns, for every question, a probability per option you wrote, plus `unknown_probability` and
`abstained`. The request and response are Jev's `POST /v1/systemone` with imajev's extensions, so imajev's and
JevBench's harnesses call it unchanged.

- **Base:** [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0, 35B total / 3B active, vision).
- **Adapter:** LoRA r=32 on the language model (attention, Gated DeltaNet projections, shared-expert MLP), trained by
  rejection-sampled self-distillation on 2,982 of the model's own correct reasoning traces over human-labelled
  training items (no eval data, no teacher labels). Routed experts, router and vision tower are frozen.
- **Readout:** the model thinks, ends with `Answer: <letter>`; the option probabilities are the letter-token
  probabilities at that position, averaged over 4 sampled reasoning paths (T = 0.6). "Unknown" is always the last
  option. An optional calibration map (`calib.py`) only reshapes probabilities; it never changes an answer.
- **Runs on:** one 96 GB GPU (vLLM, FP8) or a 64 GB Apple-silicon Mac (MLX 8-bit, 36 GB).

Results against imajev-4b, and how they were measured: [../docs/think-35b.md](../docs/think-35b.md).

## Serve on a GPU (vLLM)

```bash
# merged checkpoint (base + adapter), online FP8; MTP speculative decoding is exact and ~2x faster
VLLM_USE_DEEP_GEMM=0 vllm serve <merged-checkpoint> --served-model-name q36-v1 --quantization fp8 \
  --max-model-len 32768 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 2}' --enable-prefix-caching \
  --max-logprobs 50 --api-server-count 8 --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
python think/jevsrv.py --model q36-v1 --name peekaboolean-think-35b --samples 4 --calibration calibration.json
curl -s localhost:8765/v1/systemone -d @request.json
```

`train/merge.py --base Qwen/Qwen3.6-35B-A3B --adapter <adapter dir> --out <merged-checkpoint>` builds the merged
checkpoint on the CPU (about a minute). vLLM can also serve the base with `--enable-lora --lora-modules`.
`--api-server-count 8` matters: with one API process, building per-token logprobs saturates its CPU.

## Run on a Mac (MLX, 64 GB)

```bash
uv venv --python 3.12 .mlx && VIRTUAL_ENV=.mlx uv pip install mlx-vlm==0.7.4
hf download bykof/peekaboolean-think-35b-mlx-8bit --local-dir peekaboolean-think-35b-mlx-8bit
.mlx/bin/python think/jevmlx.py --model-path peekaboolean-think-35b-mlx-8bit --samples 4 --calibration calibration.json
```

`jevmlx.py` is `jevsrv.py` with mlx-vlm generating; prompt, readout and response are the same code. On an M1 Max one
reasoning sample takes about 25–50 s, so `--samples 1` is the practical setting for interactive use. The 8-bit model
was converted with `mlx_vlm convert -q --q-bits 8` from the merged bf16 checkpoint (it also runs on Linux, CPU only).

## How it was built (scripts in this directory; paths assume the training server's `/root/big` layout)

1. `evals/`: RealJev (2,100 human-labelled real-image items from 8 public sets, `build_realjev.py`, `REALJEV.md`),
   imajev's own held-out exam rebuilt with imajev's converters (`convert_heldout.py`), JevBench through its harness
   (`jevbench_run.sh`), a parallel runner and scorer (`jrun.py`, `jscore.py`), and `prun.py` for ImajevBench through
   its own harness and scorer.
2. `datagen/`: training items with known answers from human-labelled train splits (`build_items.py`: the_cauldron
   subsets, rule and record wrappers computed from gold, Unknown items made by removing the evidence), image
   decontamination against every eval image (`decontam.py`), RFT sampling through the exact jevsrv prompt (`rft.py`),
   and the SFT export (`export_sft.py`).
3. `train/`: LoRA SFT (`train_lora.py`, loss on the assistant turn only, token-identical to what vLLM generates,
   verified by `check_template.py`), merge (`merge.py`).
4. `drive.sh` / `drive36b.sh`: train → pick by held-out loss → merge → serve → every eval. `drive38.sh` is the
   Qwen3.8-27B attempt (see the report).
5. `calib.py`: fit the calibration map on 500 held-out training-style items, apply it to any run.

Self-checks: `python think/test_calib.py`, `python think/evals/test_jeval.py`.
