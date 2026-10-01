# LoRA SFT for Qwen3.6-35B-A3B (jevsrv decisions)

Everything here runs from `/root/big/train` on the training server, with the venv at `/root/train-env`.

| file | what it does |
|---|---|
| `train_lora.py` | the trainer: SFT with LoRA, loss on the assistant turn only, token-budget micro-batches, grad accumulation, eval, save, resume |
| `merge.py` | merges an adapter into the bf16 base and writes a full HF checkpoint (CPU only) |
| `check_template.py` | checks that the training ids match, token for token, what the running vLLM receives and samples |
| `check_fla.py` | checks that the Gated DeltaNet layers run the flash-linear-attention Triton kernels |
| `generate.py` | transformers: teacher-forced loss and greedy generation, for a merged model or base + `--adapter` |
| `vllm_check.py` | sends rows to a vLLM server in jevsrv's format and scores the final letter |
| `smoke.sh`, `make_smoke.py` | end-to-end smoke test on Qwen3.5-0.8B: train, resume, merge, generate |
| `make_tiny_moe.py`, `vllm_tiny.sh`, `prof_step.py` | random-weight Qwen3_5Moe models with the 35B layout for merge, vLLM LoRA and speed tests |

## Environment

```bash
uv venv /root/train-env --python /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
uv pip install --python /root/train-env/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/train-env/bin/python "transformers==5.17.*" peft accelerate flash-linear-attention pillow
```

This installs torch 2.11.0+cu128 (sm_120 works), transformers 5.17.0, peft 0.21.1, flash-linear-attention 0.5.2 and
triton 3.6.0. The venv has to use uv's managed Python because the system Python 3.13 has no `Python.h`, and without
it Triton cannot build its driver, so fla falls back to CPU. `causal_conv1d` is not installed (it needs nvcc), so the
depthwise conv uses the torch fallback, which costs about 2.5% of step time.

## Data format

Each JSONL row looks like `{"images": ["/abs/a.jpg", ...], "prompt": "<jevsrv PROMPT, filled in>", "response": "..."}`.

- The user turn is built the way jevsrv builds it: the images first, then the text, rendered with the model's chat
  template and `add_generation_prompt`. For Qwen3.6 the rendered prompt ends with `<|im_start|>assistant\n<think>\n`.
- `response` is exactly what vLLM returns as `choices[i].message.content` when no reasoning parser is set (the
  current server has none). That is everything generated after `<think>\n`:
  `<reasoning>\n</think>\n\n<final text>\n\nAnswer: X`. Put it in the row verbatim and do not strip `</think>`.
- Training ids are the prompt ids, followed by `tokenizer(response)`, followed by `<|im_end|>`. The loss covers only
  the response and `<|im_end|>`.
- `check_template.py` verified this. It sent 12 samples to the running vLLM (300 to 5,600 tokens each) and every one
  was identical: the prompt ids (1,726 and 1,242, image pads included) and the sampled ids, with vLLM's
  `token_ids` including `<|im_end|>`.
- Images go through the processor defaults, which is what vLLM uses (`longest_edge` = 16.7 MP). If you pass
  `--image-max-pixels`, merge.py writes the same value into the merged `preprocessor_config.json`. When serving base +
  LoRA instead, pass the same value to vLLM yourself through `--mm-processor-kwargs`.
- Rows longer than `--max-len` are dropped and counted in the log. Real thinking traces run up to about 5.6k tokens
  plus 1.2k to 1.7k of prompt, so check `dropped_over_max_len` and raise `--max-len` if many rows are dropped.

## The 35B run (only after you say the GPU is free: the vLLM on :8000 and jevsrv must be stopped first)

```bash
cd /root/big/train && mkdir -p logs && ulimit -n 65536
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# train
(nohup /root/train-env/bin/python train_lora.py --model Qwen/Qwen3.6-35B-A3B \
   --train data/train.jsonl --eval data/eval.jsonl --out runs/q36-v1 \
   --epochs 2 --lr 1e-4 --warmup 0.03 --micro-tokens 16384 --grad-accum 8 --max-len 8192 \
   --eval-every 100 --save-every 200 > logs/q36-v1.log 2>&1 < /dev/null &)
# resume: the same command with "--resume auto" (latest runs/q36-v1/step-*) or "--resume runs/q36-v1/step-000400".
# Keep --train/--epochs/--micro-tokens/--grad-accum/--seed unchanged, because the data plan is replayed from them.
grep -E '"step"' logs/q36-v1.log | tail      # loss, eval_loss, tok_s, max_mem_gib
```

Defaults: LoRA r=32, alpha=64, dropout 0.05 on 310 language-model Linears. That is q/k/v/o in the 10 full-attention
layers, in_proj_qkv/z/b/a and out_proj in the 30 DeltaNet layers, and the shared-expert gate/up/down in all 40, for
42.3M parameters. The router, routed experts and vision tower are frozen. `--train-experts` adds PEFT
`target_parameters` on `mlp.experts.{gate_up_proj,down_proj}`; PEFT then forces dropout to 0. The optimizer is AdamW
(fused) with cosine decay and warmup, and grad clipping at 1.0. A micro-batch is length-bucketed rows with
batch × longest ≤ `--micro-tokens`. One step is `--grad-accum` micro-batches, and its loss is the per-token mean over
all labelled tokens in the step. lm_head plus CE runs in checkpointed chunks of 2,048 tokens, so the 248k-vocab
logits are never materialized at once.

### Merge (CPU, while the GPU does anything else)

```bash
/root/train-env/bin/python merge.py --base Qwen/Qwen3.6-35B-A3B --adapter runs/q36-v1/step-00NNNN --out merged/q36-v1
```

PEFT computes the merge, and the output keeps the base's 26 shards with only the LoRA-touched tensors replaced
(asserted: changed tensors == LoRA layers, 310). `mtp.*`, the ViT, dtypes and file names stay as in the base, and the
tokenizer, chat template, processor and config files are copied over. With a random 310-layer adapter on the real 35B, the merge took
49 s with the shards in page cache, and the merged model is held in RAM (about 70 GB). Spot check: q_proj comes out
exactly W + 2·B@A, and `mtp.fc`, ViT qkv, routed experts and router are byte-identical to the base.

### Serve

```bash
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1   # same as the current server; there is no nvcc
# merged bf16 checkpoint (67 GB of weights). Add "--quantization fp8" for online FP8 at about the current memory.
(nohup /root/vllm-env/bin/vllm serve /root/big/train/merged/q36-v1 --served-model-name q36-v1 --port 8000 \
   --gpu-memory-utilization 0.92 --max-model-len 32768 --limit-mm-per-prompt '{"image": 2}' \
   --enable-prefix-caching --max-logprobs 50 > /root/big/vllm-q36-v1.log 2>&1 < /dev/null &)
# or base + adapter, without merging (good for A/B tests: the base is the request's "model", the adapter is "q36-v1")
(nohup /root/vllm-env/bin/vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name q36 --port 8000 \
   --enable-lora --max-lora-rank 32 --lora-modules q36-v1=/root/big/train/runs/q36-v1/step-00NNNN \
   --gpu-memory-utilization 0.92 --max-model-len 32768 --limit-mm-per-prompt '{"image": 2}' \
   --enable-prefix-caching --max-logprobs 50 > /root/big/vllm-q36-v1.log 2>&1 < /dev/null &)
# an expert adapter (--train-experts) is passed as JSON: '{"name": "q36-v1", "path": "...", "is_3d_lora_weight": true}'
cd /root/big && python jevsrv.py --model q36-v1 --name q36-v1-think --port 8765 --samples 4
```

For a fair zero-shot vs trained comparison, serve both at the same precision. The zero-shot numbers so far come from
Qwen's block-FP8 checkpoint, while the merged model is bf16 or online FP8. On the Mac:
`python -m mlx_vlm.convert --hf-path merged/q36-v1 --mlx-path q36-v1-8bit -q --q-bits 8`.

## Memory and speed for the 35B, measured on real-dimension slices

`make_tiny_moe.py --layers 4` and `--layers 8` build random Qwen3_5Moe models with the real dimensions (256 experts,
real ViT). Both were trained with `train_lora.py` on rows of about 3.7k and 6.3k tokens (1.5k image tokens each),
at `--micro-tokens 8192`.

| | weights | peak | tok/s |
|---|---|---|---|
| 4 layers | 9.0 GiB | 15.1 GiB | ~4,500 |
| 8 layers | 15.4 GiB | 21.5 GiB | ~3,300 |
| per extra layer | 1.6 GiB | +1.6 GiB (activations flat with checkpointing) | 20 µs per token |
| **40 layers (estimate)** | 65.4 GiB | **~73 GiB at ≤8k tokens per micro-batch, ~79 GiB at 16k** | **~1,050** |

A 95 GiB card therefore takes `--micro-tokens 16384` with about 15 GiB to spare. Use 8192 if anything else is on the
GPU, and 24k is the edge. At about 1k tokens/s, 1M tokens takes about 16 minutes, so 70M tokens (20k rows × 3.5k)
take about 18 h per epoch. In the profile (`prof_step.py`), about 30% of wall time is host overhead and the CE softmax
is about 11% of GPU time. If speed matters, the options are Liger fused CE, the causal-conv1d hub kernel, or larger
micro-batches.

## Verification done (logs/ on the server)

- **fla**: `check_fla.py` shows `chunk_gated_delta_rule -> fla.ops.gated_delta_rule.chunk`. At T=4096, fwd+bwd takes
  5.5 ms with fla against 77 ms with torch, with relative max diff ≤1e-2 (bf16). Profiling a Qwen3.5-0.8B fwd+bwd
  shows 7 fla kernels launched (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_bwd_kernel_dqkwg`, ...).
- **template identity**: see Data format above.
- **smoke** (`smoke.sh`, Qwen3.5-0.8B, 96 rows, 50 steps):
  - train loss 1.29 → 0.29 (step 5) → 0.007 (10) → 1e-4 (20) → 0.0 (50); eval loss 0.002 at step 10, 1e-4 at 20.
  - Resuming from step-40 replays steps 41 to 50 with identical lr and loss_tokens.
  - The merge changed 186 of 186 LoRA layers. Merged and base + adapter give the same loss (3e-5) and 8/8 letters,
    against 1.37 and 0/8 for the base.
  - vLLM 0.30 served the merged checkpoint (8/8) and base + `--enable-lora` (16/16, trained format).
- **35B merge**: done on the real checkpoint with a random adapter, 310/310 tensors changed, 26 shards (see Merge).
- **35B architecture**: on a tiny Qwen3_5Moe, vLLM `--enable-lora` with the default adapter and with a 3D expert
  adapter gives greedy text identical to the merged checkpoints. Merge counts are 31/31 and 39/39, and losses match
  between merged and adapter.
