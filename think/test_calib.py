"""python think/test_calib.py: the calibration map keeps every answer and fits an 80%-right one-hot model to ~0.8."""
import calib

p = {"A": 1.0, "B": 0.0, "__unknown__": 0.0}
q = calib.calibrate(p, 1e-3, 2.0)
assert max(q, key=q.get) == "A" and abs(sum(q.values()) - 1) < 1e-9
rows = [(p, "A")] * 8 + [(p, "B")] * 2
_, e, t = calib.fit(rows)
assert abs(calib.calibrate(p, e, t)["A"] - 0.8) < 0.02, calib.calibrate(p, e, t)
print("ok")
