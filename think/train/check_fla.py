"""Show that the Gated DeltaNet layers run flash-linear-attention's Triton kernels, not the torch fallback.

1. what transformers bound `chunk_gated_delta_rule` to at import time (closure of the hub/fallback wrapper);
2. fla vs the torch reference on random inputs: max abs diff of out and grads, and time;
3. a profiled fwd+bwd of the real model: the CUDA kernels it launched with 'delta'/'chunk' in their names.
python check_fla.py --model Qwen/Qwen3.5-0.8B
"""
import argparse, time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--tokens", type=int, default=4096)
a = ap.parse_args()

fn = m.torch_chunk_gated_delta_rule
impl = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))["implementation"]
print("chunk_gated_delta_rule ->", impl.__module__, impl.__name__)
assert impl.__module__.startswith("fla."), "torch fallback in use"


def run(f, q, k, v, g, b):
    q, k, v, g, b = (x.detach().requires_grad_() for x in (q, k, v, g, b))
    torch.cuda.synchronize()
    t = time.time()
    o, _ = f(q, k, v, g=g, beta=b, use_qk_l2norm_in_kernel=True)
    o.float().pow(2).sum().backward()
    torch.cuda.synchronize()
    return o, [x.grad for x in (q, k, v, g, b)], time.time() - t


torch.manual_seed(0)
T, H, D = a.tokens, 16, 128
q, k, v = (torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
g = -torch.rand(1, T, H, device="cuda").mul(0.5)
b = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16)
for _ in range(2):  # 2nd round timed after Triton autotune / warmup
    o1, g1, t1 = run(impl, q, k, v, g, b)
    o2, g2, t2 = run(fn.__wrapped__, q, k, v, g, b)
err = lambda x, y: ((x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-6)).item()
print(f"T={T}: fla {t1 * 1e3:.1f} ms vs torch {t2 * 1e3:.1f} ms fwd+bwd; rel max diff out {err(o1, o2):.2e}, "
      f"grads {[f'{err(x, y):.1e}' for x, y in zip(g1, g2)]}")

del o1, g1, o2, g2, q, k, v, g, b
torch.cuda.empty_cache()
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
ids = torch.randint(0, 10000, (1, a.tokens), device="cuda")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(input_ids=ids, use_cache=False, logits_to_keep=64).logits.float().sum().backward()
names = sorted({e.key for e in prof.key_averages() if any(s in e.key.lower() for s in ("delta", "chunk", "recurrent"))})
print(f"{len(names)} DeltaNet-ish CUDA kernels in model fwd+bwd:", names[:12])
assert names, "no fla kernels launched"
print("FLA PATH OK")
