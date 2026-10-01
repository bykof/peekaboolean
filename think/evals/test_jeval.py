"""venv/bin/python test_jeval.py  -- self-check of jrun.decode and jscore metrics."""
from jrun import decode
from jscore import correct, metrics, boot_ci

noul = {"type": "noul", "instructions": "q", "criteria": {"true": None, "false": None}}
p = decode({"answers": {"decision": {"type": "noul", "noul": 0.7, "unknown_probability": 0.2, "abstained": False}}}, noul)
assert p["status"] == "answered" and p["value"] == "true"
assert abs(p["probabilities"]["true"] - 0.6) < 1e-9 and abs(p["probabilities"]["false"] - 0.2) < 1e-9

ch = {"type": "choice", "instructions": "q", "criteria": {"red": None, "blue": None}}
p = decode({"answers": {"decision": {"probabilities": {"red": 0.25, "blue": 0.75}, "unknown_probability": 0.6, "abstained": True}}}, ch)
assert p["status"] == "abstained" and p["value"] is None
assert abs(p["probabilities"]["blue"] - 0.3) < 1e-9 and p["probabilities"]["__unknown__"] == 0.6
p2 = decode({"answers": {"decision": {"probabilities": {"red": 0.9, "blue": 0.1}}}}, ch)  # plain Jev: no unknown
assert p2["value"] == "red" and p2["probabilities"]["__unknown__"] == 0.0
try:
    decode({"answers": {"decision": {"probabilities": {"red": 1.0}}}}, ch)
    raise AssertionError("missing key must fail")
except ValueError:
    pass

sc = {"type": "score", "instructions": "q", "criteria": ["bad", "ok", "good"]}
p3 = decode({"answers": {"decision": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "unknown_probability": 0.0, "abstained": False}}}, sc)
assert p3["value"] == "2"

recs = [{"id": "a", "gold": "red", "group": "g1"}, {"id": "b", "gold": "__unknown__", "group": "g1"},
        {"id": "c", "gold": "blue", "group": "g2"}, {"id": "d", "gold": "red", "group": "g3"}]
preds = {"a": {"status": "answered", "value": "red", "probabilities": {"red": 0.9, "blue": 0.1, "__unknown__": 0.0}},
         "b": p, "c": {"status": "error", "value": None, "probabilities": None}}
assert [correct(r, preds.get(r["id"])) for r in recs] == [True, True, False, False]
m = metrics(recs, preds)
assert m["acc"] == 0.5 and m["errors"] == 2 and m["n_valid"] == 2 and m["unk_recall"] == 1.0
# a: conf .9 correct (bin 9: |1-.9|); b: top __unknown__ .6 correct (bin 6: |1-.6|) -> ECE = (.1 + .4) / 2
assert abs(m["ece"] - 0.25) < 1e-9, m["ece"]
lo, hi = boot_ci(recs, {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}, samples=500)
assert 0.0 <= lo <= 0.5 <= hi <= 1.0
print("ok")
