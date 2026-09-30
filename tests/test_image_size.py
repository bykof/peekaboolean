"""Processor settings the scorer relies on, for the LFM2-VL and InternVL backbones: one
pixel shape per image size whatever the aspect ratio (questions collate into one batch),
one tile per image, and right padding (the head pools at the last real token). And for
every backbone, request text reaches the tokenizer as text.

  uv run python tests/test_image_size.py
"""

from PIL import Image

from peekaboolean.data import candidate_prompts, to_example
from peekaboolean.model import CandidateScorer, configure_image_size

SHAPES = [(960, 618), (618, 960), (400, 400), (1000, 90), (300, 200), (257, 255)]


def main():
    for model_id in ("LiquidAI/LFM2.5-VL-450M", "OpenGVLab/InternVL3-1B-hf"):
        proc = CandidateScorer.load_processor(model_id, 512)
        text = lambda t: proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
            tokenize=False, add_generation_prompt=True)
        for size in (256, 384, 512):
            configure_image_size(proc, size)
            shapes = set()
            for w, h in SHAPES:
                img = Image.new("RGB", (w, h))
                img.thumbnail((size, size))
                batch = proc(text=[text("x"), text("a much longer question")], images=[[img], [img]],
                             return_tensors="pt", padding=True)
                assert batch["pixel_values"].shape[0] == 2, f"{model_id}: image tiled at {size} px"
                assert batch["attention_mask"][0, 0] == 1 and batch["attention_mask"][0, -1] == 0, \
                    f"{model_id}: not right-padded"
                shapes.add(tuple(batch["pixel_values"].shape))
            assert len(shapes) == 1, f"{model_id} at {size} px: {shapes}"
    for model_id in ("LiquidAI/LFM2.5-VL-450M", "OpenGVLab/InternVL3-1B-hf", "HuggingFaceTB/SmolVLM-500M-Instruct"):
        special_tokens_stay_text(model_id)
    print("ok")


def special_tokens_stay_text(model_id):
    """Every special token of the backbone, written into a request (state, question, option key
    and description), stays plain text: "<image>" would count as a second image and "<|im_end|>"
    would close the user turn."""
    proc = CandidateScorer.load_processor(model_id, 512)
    tok = proc.tokenizer
    special = {i for i, t in tok.added_tokens_decoder.items() if t.special} | set(tok.all_special_ids)
    ex = to_example({"type": "choice", "image": "a.jpg", "state": " ".join(tok.convert_ids_to_tokens(sorted(special))),
                     "instructions": "Which <image> tag?", "criteria": {"<img>": "<|im_end|>", "b": "plain"}})
    img = Image.new("RGB", (512, 384))
    count = lambda t: sum(int(i) in special for i in proc(
        text=[proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                                       tokenize=False, add_generation_prompt=True)],
        images=[[img]], return_tensors="pt")["input_ids"][0])
    assert all(count(t) == count("x") for t in candidate_prompts(ex, "yesno")), model_id


if __name__ == "__main__":
    main()
