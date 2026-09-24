"""Candidate-scoring model: a VLM backbone (SmolVLM, Qwen3-VL, Qwen3.5) + LoRA + a scalar or yes/no head.

The head takes one (image, state, instructions, candidate) sequence and emits a
single number. A question with K candidates produces K scores, and a softmax over
them gives the distribution. Nothing about the architecture fixes K, which is why
the API can accept whatever rubric the caller sends at request time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForImageTextToText

VISION_PAT = re.compile(r"visual|vision|image_encoder|patch_embed|merger")
QWEN_FAMILIES = ("qwen3_vl", "qwen3_5")
HEADS = ("mlp", "yesno")


def answer_token_ids(model_id: str) -> tuple[int, int]:
    """First token of "Yes" and "No" as the model would start its reply: after SmolVLM's
    "Assistant:" that is " Yes"; after Qwen's newline-terminated turn header, "Yes"."""
    proc = AutoProcessor.from_pretrained(model_id)
    prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "Q?"}]}],
                                      tokenize=False, add_generation_prompt=True)
    lead = "" if prompt[-1].isspace() else " "
    tok = proc.tokenizer
    return tok.encode(lead + "Yes", add_special_tokens=False)[0], tok.encode(lead + "No", add_special_tokens=False)[0]


LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def language_lora_targets(model: nn.Module) -> list[str]:
    """Collect LoRA targets from the language model only.

    PEFT matches target_modules by suffix, and the vision tower uses the same
    names (q_proj etc.). Passing the bare suffixes would silently LoRA the vision
    encoder we just froze, so we enumerate full module paths and filter.
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if VISION_PAT.search(name):
            continue
        if name.split(".")[-1] in LORA_SUFFIXES:
            names.append(name)
    if not names:
        raise RuntimeError("no LoRA targets found; check module naming for this backbone")
    return names


class CandidateScorer(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_vision: bool = True,
        gradient_checkpointing: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        adapter: str | None = None,
        is_trainable: bool = False,
        head: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        saved = Path(adapter) / "scorer_config.json" if adapter else None
        saved = json.loads(saved.read_text()) if saved and saved.exists() else {}
        # "mlp": a fresh scalar head (v1-v6). "yesno": the model's own logit(Yes) - logit(No)
        # at the answer position, as a trainable linear head initialized from the LM head,
        # so an untrained scorer already answers with what the pretrained model knows.
        self.head_kind = head or saved.get("head", "mlp")
        if self.head_kind not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")

        backbone = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa")
        self.family = backbone.config.model_type

        if freeze_vision:
            frozen = 0
            for name, param in backbone.named_parameters():
                if VISION_PAT.search(name):
                    param.requires_grad_(False)
                    frozen += 1
            # Frozen vision + pixel inputs that need no grad means autograd never
            # builds a graph through the tower, so the repeated encoding costs
            # compute but almost no memory.
            print(f"[model] froze {frozen} vision parameters")

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=language_lora_targets(backbone),
        )
        if adapter:
            from peft import PeftModel
            self.backbone = PeftModel.from_pretrained(backbone, adapter, is_trainable=is_trainable)
        else:
            self.backbone = get_peft_model(backbone, lora)
        self.backbone.print_trainable_parameters()

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.enable_input_require_grads()

        hidden = backbone.config.text_config.hidden_size if hasattr(backbone.config, "text_config") \
            else backbone.config.hidden_size
        # fp32 head: it is tiny, and scalar regression in bf16 is needlessly noisy.
        if self.head_kind == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.GELU(),
                nn.Linear(hidden // 4, 1),
            ).float()
        else:
            yes, no = answer_token_ids(model_id)
            lm = backbone.get_output_embeddings().weight.detach().float()
            self.head = nn.Linear(hidden, 1).float()
            with torch.no_grad():
                self.head.weight.copy_((lm[yes] - lm[no]).unsqueeze(0))
                self.head.bias.zero_()
        if adapter:
            self.head.load_state_dict(torch.load(Path(adapter) / "head.pt", map_location="cpu", weights_only=True))

    @property
    def inner(self):
        base = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        return base.model

    def hidden(self, batch: dict):
        inputs = {k: v for k, v in batch.items()
                  if k not in ("candidate_index", "mm_token_type_ids", "image_counts", "row_sign")}
        if self.family in QWEN_FAMILIES and "pixel_values" in inputs:
            return self._qwen_hidden(inputs, batch)
        if self.family == "qwen3_vl" and "mm_token_type_ids" in batch:
            inputs["mm_token_type_ids"] = batch["mm_token_type_ids"]
        if self.family in ("idefics3", "smolvlm") and "pixel_values" in inputs:
            # build_inputs supplies one image shared by every candidate. Encode only
            # the first copy. This path is also valid with a trainable vision tower.
            # A collated micro-batch carries one image per question and `image_counts`
            # candidates for each, in order.
            pixels = inputs.pop("pixel_values")
            mask = inputs.pop("pixel_attention_mask", None)
            counts = batch.get("image_counts")
            if counts is None:
                pixels, mask = pixels[:1], mask[:1] if mask is not None else None
            features = self.inner.get_image_features(pixels, pixel_attention_mask=mask).pooler_output
            inputs["image_hidden_states"] = (features.repeat(inputs["input_ids"].shape[0], 1, 1) if counts is None
                                             else features.repeat_interleave(counts.to(features.device), dim=0))
        return self.inner(**inputs, output_hidden_states=False, use_cache=False).last_hidden_state

    def _qwen_hidden(self, inputs: dict, batch: dict):
        """Qwen: encode each distinct image once, scatter its embeddings into every row
        that shares it, and compute the 3D rope positions ourselves (the model can only
        infer them when it is handed pixel values, i.e. would re-encode per row)."""
        inner = self.inner
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        pixels, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        counts = batch.get("image_counts")
        if counts is None:          # one image shared by every row (build_inputs)
            pixels, grid = pixels[:int(grid[0].prod())], grid[:1]
            counts = torch.tensor([ids.shape[0]])
        counts = counts.tolist()
        features = inner.get_image_features(pixels, grid).pooler_output
        rows = torch.cat([f.repeat(c, 1) for f, c in zip(features, counts)])
        embeds = inner.get_input_embeddings()(ids)
        image_mask = (ids == inner.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(image_mask, rows.to(embeds.dtype))
        row_grid = grid.repeat_interleave(torch.tensor(counts, device=grid.device), dim=0)
        positions, _ = inner.get_rope_index(ids, mm_token_type_ids=batch["mm_token_type_ids"],
                                            image_grid_thw=row_grid, attention_mask=mask)
        return inner(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                     use_cache=False).last_hidden_state

    def forward(self, batch: dict) -> torch.Tensor:
        """batch holds K candidate sequences. Returns a (K,) tensor of scores."""
        model_inputs = {k: v for k, v in batch.items() if k != "candidate_index"}
        hidden = self.hidden(model_inputs)                  # no unused vocabulary projection

        # Right padding, so the last real token is at sum(mask) - 1. That token has
        # attended to the whole sequence, which makes it the natural summary.
        lengths = model_inputs["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        scores = self.head(pooled.float()).squeeze(-1)      # (K,)
        if "row_sign" in batch:
            scores = scores * batch["row_sign"].to(scores.device, scores.dtype)
        return scores

    def save_adapter(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.head.state_dict(), f"{path}/head.pt")
        (Path(path) / "scorer_config.json").write_text(json.dumps(
            {"model_id": self.model_id, "family": self.family, "head": self.head_kind,
             "prompt": PROMPT_FOR_HEAD[self.head_kind]}))

    @staticmethod
    def load_processor(model_id: str, image_size: int = 512, prompt: str = "judge"):
        proc = AutoProcessor.from_pretrained(model_id)
        # Right padding matches the pooling above. Left padding would break it.
        if hasattr(proc, "tokenizer"):
            proc.tokenizer.padding_side = "right"
        # The prompt style is part of the model contract; data.build_inputs and the
        # server read it from here, so training and serving cannot disagree.
        proc.jev_prompt = prompt
        configure_image_size(proc, image_size)
        return proc


PROMPT_FOR_HEAD = {"mlp": "judge", "yesno": "yesno"}


def configure_image_size(processor, size: int):
    """One square image, no hidden tiling; match the projected visual-token count."""
    if processor.__class__.__name__.startswith(("Idefics3", "SmolVLM")):
        if size not in (256, 384, 512):
            raise ValueError("compact image size must be 256, 384 or 512")
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": size}
        processor.image_processor.max_image_size = {"longest_edge": size}
        # SmolVLM-256M/500M use patch size 16 and spatial compression factor 4.
        processor.image_seq_len = (size // 64) ** 2
    elif hasattr(processor, "image_processor") and getattr(processor.image_processor, "merge_size", None):
        # Qwen: dynamic resolution bounded by pixel count. load_image already fits the
        # longest edge to `size`, so allow up to size*size pixels and never upscale much.
        processor.image_processor.size = {"longest_edge": size * size, "shortest_edge": 32 * 32}


def select_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_dtype(device):
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
