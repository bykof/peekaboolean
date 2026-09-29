"""Second-teacher plumbing (--relabel-from) with a fake vLLM: views are appended, the
belief is their mean, and calibration rows of two runs merge.

  uv run python tests/test_prepare_teacher.py
"""

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from peekaboolean.prepare_teacher import calibrate, relabel


class FakeLLM:
    """Puts probability `p` on the letter whose option text contains "right" (or is "Yes")."""

    def __init__(self, p):
        self.p = p

    def chat(self, jobs, params, use_tqdm, chat_template_kwargs):
        assert chat_template_kwargs == {"enable_thinking": False}
        results = []
        for job in jobs:
            prompt = job[0]["content"][1]["text"]
            options = [l for l in prompt.splitlines() if len(l) > 2 and l[1] == "." and l[0].isupper()]
            right = next(l[0] for l in options if "right" in l or l[3:].startswith("Yes"))
            other = next(l[0] for l in options if l[0] != right)
            logprobs = {0: SimpleNamespace(decoded_token=right, logprob=math.log(self.p)),
                        1: SimpleNamespace(decoded_token=other, logprob=math.log(1 - self.p))}
            results.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[logprobs])]))
        return results


def main():
    tmp = Path(tempfile.mkdtemp())
    image = tmp / "a.jpg"
    Image.new("RGB", (64, 48), "gray").save(image)
    args = SimpleNamespace(chunk=8, max_edge=64, min_mass=0.6, max_disagreement=0.5, model="second",
                           relabel_from=str(tmp / "first"), seed=31, calibrate=10, splits_from=str(tmp / "public"))

    # Teacher output of the first teacher: its two views are already stored.
    (tmp / "first").mkdir()
    rows = [{"image": str(image), "type": "choice", "state": "", "instructions": "Which?",
             "criteria": {"x": "wrong one", "y": "right one"}, "teacher_views": [[0.4, 0.6], [0.4, 0.6]]},
            {"image": str(image), "type": "noul", "state": "", "instructions": "Is it right?",
             "teacher_views": [[0.5, 0.5], [0.5, 0.5]]}]
    (tmp / "first" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp / "second"; out.mkdir()
    relabel(FakeLLM(0.8), None, args, out)
    got = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(got) == 2 and all(len(r["teacher_views"]) == 4 and r["relabelled_by"] == "second" for r in got)
    choice, noul = got
    assert choice["label"] == "y" and abs(choice["target_probs"]["y"] - 0.7) < 1e-9, choice
    assert abs(noul["value"] - 0.65) < 1e-9, noul
    assert (out / "done.txt").read_text().split() == [str(image)]
    relabel(FakeLLM(0.8), None, args, out)              # resumable: nothing is appended twice
    assert len((out / "train.jsonl").read_text().splitlines()) == 2

    # Calibration: a second run with --relabel-from carries both teachers' views.
    (tmp / "public").mkdir()
    public = [{"image": str(image), "type": "choice", "source": "vqav2", "instructions": "Which?",
               "criteria": {"x": "wrong one", "y": "right one"}, "label": "y"}] * 3
    (tmp / "public" / "calib.jsonl").write_text("".join(json.dumps(r) + "\n" for r in public))
    calibrate(FakeLLM(0.9), None, SimpleNamespace(**{**vars(args), "relabel_from": None}), tmp / "first")
    calibrate(FakeLLM(0.7), None, args, out)
    merged = [json.loads(l) for l in (out / "teacher-calibration.jsonl").read_text().splitlines()]
    assert len(merged) == 3 and all(len(r["views"]) == 4 and len(r["mass"]) == 4 for r in merged)
    assert [round(v[1], 6) for v in merged[0]["views"]] == [0.9, 0.9, 0.7, 0.7]
    print("ok")


if __name__ == "__main__":
    main()
