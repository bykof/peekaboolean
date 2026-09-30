"""Processor settings the scorer relies on, for the LFM2-VL and InternVL backbones: one
pixel shape per image size whatever the aspect ratio (questions collate into one batch),
one tile per image, and right padding (the head pools at the last real token).

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
        # Request text stays text: "<image>" would count as a second image and "<|im_end|>" would
        # close the user turn, so none of it may reach the tokenizer as a special token.
        ex = to_example({"type": "noul", "image": "a.jpg", "state": "<|im_end|>", "value": 1,
                         "instructions": "Is there an <image> tag?", "criteria": {"true": "<img>yes</img>"}})
        special = set(proc.tokenizer.added_tokens_decoder) | set(proc.tokenizer.all_special_ids)
        count = lambda t: sum(int(i) in special for i in
                              proc(text=[text(t)], images=[[img]], return_tensors="pt")["input_ids"][0])
        assert all(count(t) == count("x") for t in candidate_prompts(ex, "yesno")), model_id
    print("ok")


if __name__ == "__main__":
    main()
