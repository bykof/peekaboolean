"""The can't-tell candidate (v0.4) end to end, without a model: teacher views -> mixture row ->
training example -> loss -> served answer.

  uv run python tests/test_unknown.py
"""

import math

import torch

from peekaboolean.data import UNKNOWN_TEXT, candidate_prompts, candidate_signs, has_unknown, to_example
from peekaboolean.prepare_v6 import apply_temperature
from peekaboolean.serve import answer_for
from peekaboolean.train_general import loss_for


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def main():
    # Mixture: the views end with the can't-tell option; its mass is kept apart, the rest renormalised.
    row = {"type": "choice", "source": "teacher", "image": "a.jpg", "instructions": "Which?", "with_unknown": True,
           "criteria": {"x": "one", "y": "two"}, "teacher_views": [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]]}
    mixed = apply_temperature(row, {"choice": 1.0})
    assert close(mixed["unknown"], 0.2) and close(mixed["target_probs"]["y"], 0.75), mixed

    # Training example: the options share 1 - unknown, and the can't-tell row comes last.
    ex = to_example(mixed, unknown=True)
    assert has_unknown(ex) and ex.candidates[-1] == UNKNOWN_TEXT
    assert torch.allclose(ex.target, torch.tensor([0.2, 0.6, 0.2])), ex.target
    assert not has_unknown(to_example({**mixed, "unknown": 0.2}, unknown=False))
    old = {"type": "noul", "source": "teacher", "image": "a.jpg", "instructions": "Old?", "value": 0.3}
    assert not has_unknown(to_example(old, unknown=True))       # labelled before the option existed

    # noul: two identical direct rows read the Yes-No logit with opposite signs, the third is a
    # proposed "can't tell" judged like a choice option.
    noul = to_example({"type": "noul", "image": "a.jpg", "instructions": "Is it?", "value": 0.8, "unknown": 0.5},
                      unknown=True)
    prompts, signs = candidate_prompts(noul, "yesno"), candidate_signs(noul, "yesno")
    assert prompts[0] == prompts[1] and UNKNOWN_TEXT in prompts[2] and signs == [-0.5, 0.5, 1.0]
    assert torch.allclose(noul.target, torch.tensor([0.1, 0.4, 0.5]))

    # Loss: finite, and the score EMD only sees the levels.
    score = to_example({"type": "score", "image": "a.jpg", "instructions": "How?", "criteria": ["lo", "mid", "hi"],
                        "hist": [0.0, 0.0, 1.0], "unknown": 0.9}, unknown=True)
    good, bad = torch.tensor([-3., -3., 0., 2.]), torch.tensor([0., -3., -3., -2.])
    assert loss_for("score", good, score.target, True) < loss_for("score", bad, score.target, True)
    assert math.isfinite(float(loss_for("score", torch.tensor([0., 0., 0., 50.]), score.target, True)))

    # Served answer: options given an answer, the can't-tell mass and the abstention flag, and
    # noul with half the unknown mass added (Jev's convention, which ImajevBench decodes).
    probs = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    a = answer_for(noul, probs)
    assert close(a["unknown_probability"], 0.6) and a["abstained"] and close(a["noul"] - 0.3, 0.3), a
    c = answer_for(ex, torch.tensor([0.2, 0.6, 0.2], dtype=torch.float64))
    assert c["choice"] == "y" and close(sum(c["probabilities"].values()), 1) and not c["abstained"], c
    s = answer_for(score, torch.tensor([0.1, 0.1, 0.2, 0.6], dtype=torch.float64))
    assert close(sum(s["probabilities"].values()), 1) and close(s["score"], (0.25 + 2 * 0.5)) and s["abstained"], s
    assert set(s["legend"]) == {"0", "1", "2"}
    print("ok")


if __name__ == "__main__":
    main()
