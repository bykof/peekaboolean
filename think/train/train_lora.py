"""LoRA SFT for Qwen3.5 / Qwen3.6 VL models on jevsrv decisions.

Rows (JSONL): {"images": [abs paths], "prompt": user text, "response": assistant text}.
The user turn is jevsrv's: the images first, then the text, rendered with the model's chat template and
add_generation_prompt (Qwen3.6 ends it with "<|im_start|>assistant\n<think>\n"). `response` is what vLLM returns as
`message.content` without a reasoning parser, i.e. everything generated after that "<think>\n":
"<reasoning>\n</think>\n\n<final text ending in Answer: X>". Training ids = prompt ids + tokenizer(response) +
<|im_end|>; the loss covers the response and <|im_end|> only. check_template.py verifies this against vLLM.

python train_lora.py --model Qwen/Qwen3.6-35B-A3B --train T.jsonl --eval E.jsonl --out runs/x
"""
import argparse, glob, json, math, os, random, time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

# Language model only: full attention, Gated DeltaNet, shared expert (MoE) or the dense MLP (Qwen3.5 dense / 27B).
# Router (mlp.gate, not a Linear), routed experts (3D params) and the vision tower never match.
TARGETS = (r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj"
           r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
           r"|mlp\.(shared_expert\.)?(gate|up|down)_proj)")
EXPERTS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


class Encoder:
    def __init__(self, processor, template_kwargs):
        self.p, self.tk = processor, template_kwargs
        self.end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prompt_text(self, row):
        content = [{"type": "image"} for _ in row["images"]] + [{"type": "text", "text": row["prompt"]}]
        return self.p.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                          add_generation_prompt=True, **self.tk)

    def response_ids(self, row):
        return self.p.tokenizer(row["response"], add_special_tokens=False)["input_ids"] + [self.end]

    def lengths(self, row):
        """(total, response) token counts without decoding the images."""
        ip = self.p.image_processor
        n = len(self.p.tokenizer(self.prompt_text(row), add_special_tokens=False)["input_ids"])
        for path in row["images"]:
            w, h = Image.open(path).size
            n += ip.get_number_of_image_patches(h, w, {}) // ip.merge_size ** 2 - 1  # one <|image_pad|> -> N
        r = len(self.response_ids(row))
        return n + r, r

    def __call__(self, row):
        images = [Image.open(p).convert("RGB") for p in row["images"]] or None
        enc = self.p(text=[self.prompt_text(row)], images=images, return_tensors="pt")
        resp = torch.tensor([self.response_ids(row)])
        enc["labels"] = torch.cat([torch.full_like(enc["input_ids"], -100), resp], 1)
        for k in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if k in enc:
                fill = resp if k == "input_ids" else torch.full_like(resp, 1 if k == "attention_mask" else 0)
                enc[k] = torch.cat([enc[k], fill], 1)
        return enc


def collate(encs, pad_id):
    """Right padding: causal attention and the left-to-right DeltaNet never see the pads of a real token."""
    L = max(e["input_ids"].shape[1] for e in encs)
    out = {}
    for k, pad in (("input_ids", pad_id), ("attention_mask", 0), ("mm_token_type_ids", 0), ("labels", -100)):
        if k in encs[0]:
            out[k] = torch.stack([F.pad(e[k][0], (0, L - e[k].shape[1]), value=pad) for e in encs])
    for k in ("pixel_values", "image_grid_thw"):
        xs = [e[k] for e in encs if k in e]
        if xs:
            out[k] = torch.cat(xs)
    return out


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows, enc, pad_id):
        self.rows, self.enc, self.pad = rows, enc, pad_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):  # idx = one micro-batch (list of row indices), see batch_sampler below
        return collate([self.enc(self.rows[i]) for i in idx], self.pad)


def micro_batches(lens, budget, rng):
    """Shuffle, sort within windows by length, pack until batch * longest > budget padded tokens, shuffle batches."""
    idx = list(range(len(lens)))
    rng.shuffle(idx)
    out = []
    for w in range(0, len(idx), 256):
        cur, longest = [], 0
        for i in sorted(idx[w:w + 256], key=lambda i: lens[i]):
            if cur and (len(cur) + 1) * max(longest, lens[i]) > budget:
                out.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, lens[i])
        out.append(cur)
    rng.shuffle(out)
    return out


def load_rows(path, enc, max_len):
    rows, lens, resp, dropped = [], [], [], 0
    for line in open(path):
        row = json.loads(line)
        n, r = enc.lengths(row)
        if n > max_len:
            dropped += 1
            continue
        rows.append(row)
        lens.append(n)
        resp.append(r)
    print(json.dumps({"file": path, "rows": len(rows), "dropped_over_max_len": dropped, "tokens": sum(lens),
                      "loss_tokens": sum(resp)}), flush=True)
    return rows, lens, resp


def loss_sum(model, batch, chunk=2048):
    """Summed CE over the labelled tokens. lm_head + CE run in checkpointed chunks, so the full-vocab logits of an
    8k sequence (248k vocab) are never materialized at once."""
    labels = batch.pop("labels")
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = hf.model(**batch, use_cache=False).last_hidden_state
    y = labels[:, 1:]
    m = y != -100
    h, y = h[:, :-1][m], y[m]
    ce = lambda hh, yy: F.cross_entropy(hf.lm_head(hh).float(), yy, reduction="sum")
    return sum(checkpoint(ce, h[i:i + chunk], y[i:i + chunk], use_reentrant=False) for i in range(0, len(y), chunk))


def loader(ds, plan, workers):
    return torch.utils.data.DataLoader(ds, sampler=plan, batch_size=None, num_workers=workers,
                                       pin_memory=True, persistent_workers=False, prefetch_factor=4 if workers else None)


@torch.no_grad()
def evaluate(model, ds, plan, resp, workers):
    model.eval()
    tot = n = 0
    for mb, batch in zip(plan, loader(ds, plan, workers)):
        with torch.autocast("cuda", torch.bfloat16):
            tot += loss_sum(model, {k: v.cuda(non_blocking=True) for k, v in batch.items()}).item()
        n += sum(resp[i] for i in mb)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", help="a step-NNNNNN dir, or 'auto' for the latest one in --out")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per forward (batch * longest)")
    ap.add_argument("--grad-accum", type=int, default=8, help="micro-batches per optimizer step")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--image-max-pixels", type=int, help="default: the processor's (= what vLLM uses)")
    ap.add_argument("--image-min-pixels", type=int)
    ap.add_argument("--template-kwargs", default="{}", help='e.g. \'{"enable_thinking": true}\'; must match serving')
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--train-experts", action="store_true", help="also LoRA the routed experts (target_parameters); PEFT then needs dropout 0 everywhere")
    ap.add_argument("--experts-implementation", default="grouped_mm")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)

    processor = AutoProcessor.from_pretrained(a.model)
    if a.image_max_pixels:
        processor.image_processor.size["longest_edge"] = a.image_max_pixels
    if a.image_min_pixels:
        processor.image_processor.size["shortest_edge"] = a.image_min_pixels
    enc = Encoder(processor, json.loads(a.template_kwargs))
    pad_id = processor.tokenizer.pad_token_id
    rows, lens, resp = load_rows(a.train, enc, a.max_len)
    train_ds = Rows(rows, enc, pad_id)
    if a.eval:
        erows, elens, eresp = load_rows(a.eval, enc, a.max_len)
        eval_ds, eval_plan = Rows(erows, enc, pad_id), micro_batches(elens, a.micro_tokens, random.Random(0))

    plan = [mb for e in range(a.epochs) for mb in micro_batches(lens, a.micro_tokens, random.Random(a.seed + e))]
    total = len(plan) // a.grad_accum
    if a.max_steps:
        total = min(total, a.max_steps)

    moe = "moe" in AutoConfig.from_pretrained(a.model).model_type  # dense models reject experts_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        **({"experts_implementation": a.experts_implementation} if moe else {}))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0 if a.train_experts else a.dropout, target_modules=TARGETS,
                     target_parameters=EXPERTS if a.train_experts else None, bias="none")
    model = get_peft_model(model, cfg)  # adapters in fp32 (PEFT autocast_adapter_dtype), matmuls under bf16 autocast
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999), fused=True)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(a.warmup * total), total)
    step = 0
    if a.resume:
        d = a.resume if a.resume != "auto" else max(glob.glob(os.path.join(a.out, "step-*")), default=None)
        if d:
            set_peft_model_state_dict(model, load_file(os.path.join(d, "adapter_model.safetensors")))
            st = torch.load(os.path.join(d, "state.pt"), weights_only=True)
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            step = st["step"]
            print(json.dumps({"resumed": d, "step": step}), flush=True)

    def save(step):
        d = os.path.join(a.out, f"step-{step:06d}")
        model.save_pretrained(d)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, os.path.join(d, "state.pt"))

    rest = plan[step * a.grad_accum: total * a.grad_accum]  # resume replays the same plan from `step`
    it = iter(loader(train_ds, rest, a.workers))
    model.train()
    t0, tok = time.time(), 0
    for step, mbs in enumerate((rest[i:i + a.grad_accum] for i in range(0, len(rest), a.grad_accum)), step + 1):
        n_loss = sum(resp[i] for mb in mbs for i in mb)
        tot = 0.0
        for _ in mbs:
            batch = {k: v.cuda(non_blocking=True) for k, v in next(it).items()}
            tok += int(batch["attention_mask"].sum())
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_sum(model, batch) / n_loss
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0:
            dt = time.time() - t0
            print(json.dumps({"step": step, "of": total, "loss": round(tot, 4), "grad_norm": round(gn, 3),
                              "lr": sched.get_last_lr()[0], "tok_s": round(tok / dt), "loss_tokens": n_loss,
                              "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
            t0, tok = time.time(), 0
        if a.eval and (step % a.eval_every == 0 or step == total):
            print(json.dumps({"step": step, "eval_loss": round(evaluate(model, eval_ds, eval_plan, eresp, a.workers), 4)}),
                  flush=True)
        if step % a.save_every == 0 or step == total:
            save(step)


if __name__ == "__main__":
    main()
