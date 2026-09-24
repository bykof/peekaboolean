#!/usr/bin/env python3
"""Train the candidate scorer.

  python -m peekaboolean.train --train data/train.jsonl --val data/val.jsonl \
      --image-root data/images --model Qwen/Qwen3-VL-4B-Instruct --out runs/v1

One question per micro-step (its K candidates are the batch), with gradient
accumulation for the effective batch. This keeps memory predictable regardless of
how many levels a rubric has.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DecisionDataset, build_inputs
from .model import CandidateScorer


def emd_loss(logits: torch.Tensor, target: torch.Tensor, r: int = 2) -> torch.Tensor:
    """Earth Mover's Distance between predicted and target distributions.

    Score levels are ORDERED. Cross-entropy would punish "off by one level" exactly
    as hard as "off by four", which is wrong and produces a model whose probability
    mass is scattered instead of concentrated near the truth. This is the single
    most important loss choice in the project.
    """
    p = torch.softmax(logits, dim=-1)
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_t = torch.cumsum(target, dim=-1)
    return ((cdf_p - cdf_t).abs() ** r).mean() ** (1.0 / r)


def loss_for(qtype: str, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if qtype == "score":
        return emd_loss(logits, target)
    if qtype == "choice":
        # unordered, so ordinary cross-entropy against the one-hot target
        return -(target * torch.log_softmax(logits, dim=-1)).sum()
    if qtype == "noul":
        p_yes = torch.softmax(logits, dim=-1)[1]
        return F.binary_cross_entropy(p_yes.clamp(1e-6, 1 - 1e-6), target[1])
    raise ValueError(qtype)


@torch.no_grad()
def evaluate(model, processor, dataset, device, limit: int = 200) -> dict:
    model.eval()
    totals: dict[str, list[float]] = {}
    correct = seen = 0
    for i in range(min(limit, len(dataset))):
        ex = dataset[i]
        batch = build_inputs(ex, processor).to(device)
        logits = model(batch)
        l = loss_for(ex.qtype, logits, ex.target.to(device))
        totals.setdefault(ex.qtype, []).append(l.item())
        if ex.qtype in ("choice", "noul"):
            correct += int(logits.argmax().item() == ex.target.argmax().item())
            seen += 1
    model.train()
    out = {f"val_{k}": sum(v) / len(v) for k, v in totals.items()}
    if seen:
        out["val_acc"] = correct / seen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=16, help="questions per optimizer step")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-edge", type=int, default=1024)
    ap.add_argument("--min-levels", type=int, default=2, help="narrowest generated rubric")
    ap.add_argument("--max-levels", type=int, default=10, help="widest; the API accepts 10")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_ds = DecisionDataset(args.train, args.image_root, augment=True, seed=args.seed,
                               min_levels=args.min_levels, max_levels=args.max_levels)
    val_ds = DecisionDataset(args.val, args.image_root, augment=False, seed=args.seed) if args.val else None

    model = CandidateScorer(args.model, lora_r=args.lora_r).to(device)
    processor = CandidateScorer.load_processor(args.model)

    # The head is new and random, so it wants a much higher LR than the adapters.
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.0, betas=(0.9, 0.95))

    n_questions = int(len(train_ds) * args.epochs)
    total_steps = max(1, n_questions // args.accum)

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    loader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=0)

    print(f"[train] {len(train_ds)} questions, {total_steps} optimizer steps, device={device}")
    model.train()
    step = seen = 0
    running: list[float] = []
    t0 = time.time()

    for epoch in range(math.ceil(args.epochs)):
        for ex in loader:
            if seen >= n_questions:
                break
            try:
                batch = build_inputs(ex, processor, args.max_edge).to(device)
                logits = model(batch)
                loss = loss_for(ex.qtype, logits, ex.target.to(device)) / args.accum
                loss.backward()
                running.append(loss.item() * args.accum)
            except torch.cuda.OutOfMemoryError:
                # A pathological rubric (10 levels on a large image) can spike.
                # Skip it rather than lose the run.
                print(f"[train] OOM at question {seen}, skipping")
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            seen += 1

            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                    rate = seen / (time.time() - t0)
                    print(f"step {step}/{total_steps}  loss {sum(running)/len(running):.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} q/s  peak {mem:.1f} GB")
                    running.clear()

                if val_ds and step % args.eval_every == 0:
                    print(f"[eval] {evaluate(model, processor, val_ds, device)}")
                    model.save_adapter(str(out_dir / "checkpoint"))

    model.save_adapter(str(out_dir / "final"))
    if val_ds:
        print(f"[eval final] {evaluate(model, processor, val_ds, device, limit=1000)}")
    print(f"[train] done in {(time.time()-t0)/60:.1f} min -> {out_dir/'final'}")


if __name__ == "__main__":
    main()
