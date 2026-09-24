"""Resumable training with source metrics and image ablations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import torch
import torch.nn.functional as F
from .data import DecisionDataset, build_inputs, collate
from .model import CandidateScorer, PROMPT_FOR_HEAD, configure_image_size, select_device, device_dtype
from .baselines import QuestionPriors


def loss_for(qtype, logits, target):
    ce = -(target * F.log_softmax(logits.float(), -1)).sum()
    if qtype == "score":
        delta = torch.softmax(logits.float(), -1).cumsum(-1) - target.cumsum(-1)
        return delta.square().mean() + 0.2 * ce
    return ce


class TrainStream(torch.utils.data.IterableDataset):
    """Training questions from `start`, built into model inputs by DataLoader workers.

    Worker w takes positions start+w, start+w+W, ...; the loader interleaves workers
    round-robin, so the stream is the same position order the single-process loop
    walked and a resumed run continues exactly where it stopped. Image decode,
    resizing and tokenization -- most of a step on this backbone -- leave the GPU loop.

    With micro > 1, each item is `micro` consecutive questions collated into one
    forward at one image size: a 256M backbone scoring 2-8 short sequences at a time
    leaves the GPU idle on kernel launches, so questions are scored together.
    """
    def __init__(self, dataset, processor, sizes, seed, start, stop, micro=1):
        super().__init__()
        self.dataset, self.processor, self.sizes = dataset, processor, sizes
        self.seed, self.start, self.stop, self.micro = seed, start, stop, micro

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        order, order_epoch = None, -1
        pad_id = self.processor.tokenizer.pad_token_id
        for first in range(self.start + worker * self.micro, self.stop, workers * self.micro):
            size = random.Random(f"{self.seed}:size:{first}").choice(self.sizes)
            configure_image_size(self.processor, size)
            items = []
            for position in range(first, min(first + self.micro, self.stop)):
                epoch, index = divmod(position, len(self.dataset))
                if epoch != order_epoch:
                    order = list(range(len(self.dataset)))
                    random.Random(f"{self.seed}:order:{epoch}").shuffle(order); order_epoch = epoch
                self.dataset.epoch = epoch
                ex = self.dataset[order[index]]
                items.append((position, ex, build_inputs(ex, self.processor, size)))
            if self.micro == 1:
                yield items[0]
            else:
                yield ([i[0] for i in items], [i[1] for i in items], collate([i[2] for i in items], pad_id))


class EvalBatches(torch.utils.data.Dataset):
    """Evaluation rows in order, `micro` questions per collated forward."""
    def __init__(self, dataset, processor, ids, max_edge, micro=8):
        self.dataset, self.processor, self.ids, self.max_edge, self.micro = dataset, processor, ids, max_edge, micro

    def __len__(self): return math.ceil(len(self.ids) / self.micro)

    def __getitem__(self, b):
        configure_image_size(self.processor, self.max_edge)
        examples = [self.dataset[i] for i in self.ids[b * self.micro:(b + 1) * self.micro]]
        inputs = [build_inputs(ex, self.processor, self.max_edge) for ex in examples]
        return examples, collate(inputs, self.processor.tokenizer.pad_token_id)


def _one_thread(_):
    torch.set_num_threads(1)


def loader(source, workers, **kwargs):
    if not workers:
        return iter(source) if isinstance(source, torch.utils.data.IterableDataset) else (source[i] for i in range(len(source)))
    return iter(torch.utils.data.DataLoader(source, batch_size=None, num_workers=workers, prefetch_factor=8,
                                            worker_init_fn=_one_thread, **kwargs))


def spearman(x, y):
    if len(x) < 3: return float("nan")
    rx = torch.tensor(x).argsort().argsort().float(); ry = torch.tensor(y).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(rx.norm() * ry.norm())
    return float((rx * ry).sum()) / denom if denom else float("nan")


def eval_indices(dataset, limit=600, seed=0, weights=None):
    """Round-robin over (source, type) groups; a source with weight w contributes w rows
    per round, so the groups that decide selection get tighter estimates."""
    groups = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[(row.get("source", "unknown"), row["type"])].append(i)
    rng = random.Random(seed)
    for ids in groups.values(): rng.shuffle(ids)
    selected = []
    while groups and (not limit or len(selected) < limit):
        for key in list(groups):
            for _ in range((weights or {}).get(key[0], 1)):
                if limit and len(selected) >= limit: break
                selected.append(groups[key].pop())
                if not groups[key]: del groups[key]; break
    return selected


# Photo aesthetics stays in the report but not in checkpoint selection: it is not the
# product, and on v5/v6 it never beat the text-only prior, so it only adds noise.
SELECTION_EXCLUDE = ("ava", "aadb")


@torch.no_grad()
def evaluate(model, processor, dataset, device, limit=600, max_edge=384, ablation=60, priors=None, workers=0,
             weights=None):
    was_training = model.training
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    ids = eval_indices(dataset, limit, weights=weights)
    wrong_images = []
    means = defaultdict(lambda: ([], [], []))   # score group -> predicted, target, prior implied means
    def scored():
        for examples, inputs in loader(EvalBatches(dataset, processor, ids, max_edge), workers):
            yield from zip(examples, model(inputs.to(device)).float().cpu().split(inputs["image_counts"].tolist()))
    configure_image_size(processor, max_edge)
    for n, (i, (ex, logits)) in enumerate(zip(ids, scored())):
        p, t = logits.softmax(-1), ex.target
        metrics = {"nll": float(-(t * logits.log_softmax(-1)).sum()),
                   "brier": float((p - t).square().sum()), "uniform_nll": math.log(len(t))}
        # Soft labels (teacher rows) count toward accuracy when they clearly favour one answer.
        decided = float(t.max()) >= 0.6
        prior = priors.predict(dataset.rows[i], ex) if priors else None
        if prior is not None:
            metrics["question_prior_nll"] = float(-(t * prior.clamp_min(1e-9).log()).sum())
            if ex.qtype in ("choice", "noul") and decided:
                # A tie (e.g. a uniform prior) is a fractional win, not a win for whichever
                # option happens to be listed first.
                top = (prior == prior.max()).float()
                metrics["question_prior_accuracy"] = float(top[t.argmax()] / top.sum())
        if ex.qtype == "choice" and decided: metrics["accuracy"] = float(p.argmax() == t.argmax())
        if ex.qtype == "noul":
            metrics["probability_mae"] = float(abs(p[1] - t[1]))
            if decided:
                metrics["accuracy"] = float(p.argmax() == t.argmax())
                metrics["positive_recall" if t[1] > 0.5 else "negative_recall"] = metrics["accuracy"]
        if ex.qtype == "score":
            centres = (torch.arange(len(t)) + 0.5) / len(t)
            metrics["emd"] = float((p.cumsum(-1) - t.cumsum(-1)).square().mean().sqrt())
            metrics["mean_mae"] = float(abs(((p - t) * centres).sum()))
            if prior is not None:
                metrics["question_prior_emd"] = float((prior.cumsum(-1) - t.cumsum(-1)).square().mean().sqrt())
                metrics["question_prior_mean_mae"] = float(abs(((prior - t) * centres).sum()))
            for group in (ex.qtype, f"{dataset.rows[i].get('source', 'unknown')}/{ex.qtype}"):
                means[group][0].append(float((p * centres).sum())); means[group][1].append(float((t * centres).sum()))
        source = dataset.rows[i].get("source", "unknown")
        for group in (ex.qtype, f"{source}/{ex.qtype}"):
            for k, value in metrics.items(): groups[group][k].append(value)
        if n < ablation:
            other = next((dataset.rows[j]["image"] for j in ids[n + 1:] + ids[:n]
                          if dataset.rows[j]["image"] != dataset.rows[i]["image"]), None)
            if other:
                ex.image = dataset.image_root / other
                wrong = model(build_inputs(ex, processor, max_edge).to(device)).float().cpu()
                wrong_images.append((f"{source}/{ex.qtype}", float(-(t * wrong.log_softmax(-1)).sum()) - metrics["nll"]))
    model.train(was_training)
    result = {g: {**{k: sum(v) / len(v) for k, v in m.items()}, "n": len(m["nll"])} for g, m in groups.items()}
    for metrics in result.values():
        if "positive_recall" in metrics and "negative_recall" in metrics:
            metrics["balanced_accuracy"] = (metrics["positive_recall"] + metrics["negative_recall"]) / 2
    for group, (pred, target, _) in means.items():
        result[group]["spearman"] = spearman(pred, target)
    source_losses = [m["nll"] for g, m in result.items() if "/" in g]
    macro = sum(source_losses) / max(1, len(source_losses))
    # Request-shaped (teacher) questions are what serving looks like; public benchmarks keep
    # the model honest on exact labels. Selection weighs the two halves equally.
    teacher = [m["nll"] for g, m in result.items() if g.startswith("teacher/")]
    public = [m["nll"] for g, m in result.items() if "/" in g and not g.startswith("teacher/")
              and g.split("/")[0] not in SELECTION_EXCLUDE]
    selection = (sum(teacher) / len(teacher) + sum(public) / len(public)) / 2 if teacher and public else macro
    by_group = defaultdict(list)
    for group, delta in wrong_images: by_group[group].append(delta)
    deltas = [d for _, d in wrong_images]
    return {"groups": result, "macro_nll": macro, "selection_nll": selection,
            "image_ablation": {"n": len(deltas),
              "wrong_image_minus_correct_nll": sum(deltas) / max(1, len(deltas)),
              "by_group": {g: {"n": len(v), "delta": sum(v) / len(v)} for g, v in sorted(by_group.items())}}}


def save_checkpoint(model, opt, sched, out, state):
    path = out / f"step-{state['step']:06d}"
    model.save_adapter(str(path))
    state = {**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temp = path / "training_state.tmp"
    torch.save(state, temp); temp.replace(path / "training_state.pt")
    temp = out / "latest.tmp"; temp.write_text(path.name); temp.replace(out / "latest.txt")
    print(f"[checkpoint] {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default="runs/v4")
    ap.add_argument("--resume", help="checkpoint path or 'latest'")
    ap.add_argument("--init-adapter", help="start from this trained adapter (weights only; fresh optimizer and schedule)")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-hours", type=float, default=72)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=None,
                    help="default 3e-4 for a fresh mlp head; --lr for the yesno head, which starts pretrained")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-edge", type=int, default=384)
    ap.add_argument("--image-sizes", default="256,384,512")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=2000)
    ap.add_argument("--ablation", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers building inputs (0 = inline)")
    ap.add_argument("--micro", type=int, default=1, help="questions per forward; must divide --accum")
    ap.add_argument("--head", choices=["mlp", "yesno"], default="mlp",
                    help="mlp: fresh scalar head; yesno: the backbone's own Yes-vs-No logit, trainable")
    ap.add_argument("--eval-teacher-weight", type=int, default=1,
                    help="validation rows per round for each teacher group relative to public groups")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()
    if args.micro < 1 or args.accum % args.micro: ap.error("--micro must divide --accum")
    if args.head_lr is None: args.head_lr = 3e-4 if args.head == "mlp" else args.lr
    if args.accum < 1 or args.epochs <= 0 or args.eval_every < 1: ap.error("positive accum, epochs and eval-every required")
    device = select_device(args.device)
    if device == "cpu": raise SystemExit("No accelerator found; refusing a silent CPU training run")
    torch.set_num_threads(4)
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "args.json").exists() and not args.resume: raise SystemExit("Run exists; use --resume latest or a fresh --out")
    resume = out / (out / "latest.txt").read_text().strip() if args.resume == "latest" else args.resume
    state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False) if resume else {}
    hashes = {key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest() for key in ("train", "val")}
    if state and state["data_hashes"] != hashes: raise SystemExit("Dataset changed since checkpoint")
    if state:
        previous = json.loads((out / "args.json").read_text())
        for key in ("model", "accum", "epochs", "lr", "head_lr", "lora_r", "seed", "image_sizes", "max_edge", "warmup", "head"):
            if previous.get(key, "mlp" if key == "head" else None) != getattr(args, key): raise SystemExit(f"Resume requires original --{key.replace('_', '-')}")
    else: (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    train_ds = DecisionDataset(args.train, args.image_root, augment=True, min_levels=2, max_levels=10, seed=args.seed)
    val_ds = DecisionDataset(args.val, args.image_root, augment=False, seed=args.seed)
    priors = QuestionPriors(train_ds.rows)
    model = CandidateScorer(args.model, lora_r=args.lora_r,
        gradient_checkpointing=args.gradient_checkpointing, dtype=device_dtype(device),
        adapter=str(resume or args.init_adapter) if (resume or args.init_adapter) else None,
        is_trainable=bool(resume or args.init_adapter), head=args.head).to(device)
    processor = CandidateScorer.load_processor(args.model, args.max_edge, prompt=PROMPT_FOR_HEAD[args.head])
    head = list(model.head.parameters()); head_ids = {id(p) for p in head}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], "lr": args.lr},
        {"params": head, "lr": args.head_lr}], betas=(0.9, 0.95), weight_decay=0.01)
    n_questions = math.ceil(len(train_ds) * args.epochs)
    total_steps = math.ceil(n_questions / args.accum)
    def schedule(step):
        if step < args.warmup: return (step + 1) / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * min(1, progress))) / 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    seen, step = state.get("seen", 0), state.get("step", 0)
    best, stale = state.get("best", float("inf")), state.get("stale", 0)
    if state:
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"]); random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and device == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
        print("[train] stop requested; checkpoint at next optimizer boundary", flush=True)
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    sizes = [int(s) for s in args.image_sizes.split(",")]
    start, initial_seen = time.monotonic(), seen
    elapsed_before = state.get("elapsed_seconds", 0)
    def checkpoint():
        return save_checkpoint(model, opt, sched, out, {"step": step, "seen": seen, "best": best,
            "stale": stale, "data_hashes": hashes, "elapsed_seconds": elapsed_before + time.monotonic() - start})
    def validate():
        configure_image_size(processor, args.max_edge)
        report = evaluate(model, processor, val_ds, device, args.eval_limit, args.max_edge, args.ablation, priors,
                          workers=args.workers, weights={"teacher": args.eval_teacher_weight})
        with (out / "metrics.jsonl").open("a") as fh: fh.write(json.dumps({"step": step, **report}) + "\n")
        print(f"[eval] step {step} {json.dumps(report)}", flush=True)
        return report.get("selection_nll", report["macro_nll"])
    if not resume:
        best = validate(); checkpoint(); (out / "best.txt").write_text(f"step-{step:06d}")
    print(f"[train] model={args.model} device={device} questions={n_questions} steps={total_steps} start={step}", flush=True)
    model.train(); running = []
    stream = loader(TrainStream(train_ds, processor, sizes, args.seed, seen, n_questions, args.micro), args.workers)
    while seen < n_questions:
        opt.zero_grad(set_to_none=True)
        group_size = min(args.accum, n_questions - seen)
        # OOM/data errors terminate visibly; never silently lose partial gradients.
        offset = 0
        while offset < group_size:
            positions, examples, inputs = next(stream)
            if args.micro == 1: positions, examples = [positions], [examples]
            if positions[0] != seen + offset: raise RuntimeError(f"stream out of order: {positions[0]} != {seen + offset}")
            logits = model(inputs.to(device))
            parts = logits.split(inputs["image_counts"].tolist()) if args.micro > 1 else [logits]
            losses = [loss_for(ex.qtype, part, ex.target.to(device)) for ex, part in zip(examples, parts)]
            loss = torch.stack(losses).sum()
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at rows {positions}")
            (loss / group_size).backward(); running.extend(float(l.detach()) for l in losses)
            offset += len(positions)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        opt.step(); sched.step(); seen += group_size; step += 1
        if step % args.log_every == 0:
            rate = (seen - initial_seen) / (time.monotonic() - start)
            print(f"step {step}/{total_steps} loss {sum(running)/len(running):.5f} {rate:.2f} q/s lr {sched.get_last_lr()[0]:.2e}", flush=True)
            running.clear()
        if step % args.eval_every == 0:
            value = validate(); improved = value < best - args.min_delta
            best, stale = (value, 0) if improved else (best, stale + 1)
            path = checkpoint()
            if improved: (out / "best.txt").write_text(path.name)
            if args.patience and stale >= args.patience:
                print("[train] early stopping: validation stopped improving", flush=True); break
        if stop["requested"] or (args.max_hours and elapsed_before + time.monotonic() - start >= args.max_hours * 3600):
            checkpoint(); print("[train] paused with resumable checkpoint", flush=True); return
    value = validate(); improved = value < best
    if improved: best = value
    path = checkpoint()
    if improved: (out / "best.txt").write_text(path.name)
    (out / "complete.json").write_text(json.dumps({"step": step, "best": (out / "best.txt").read_text().strip()}))
    print(f"[train] complete; best checkpoint: {(out / 'best.txt').read_text().strip()}", flush=True)


if __name__ == "__main__":
    main()
