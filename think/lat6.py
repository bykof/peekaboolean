"""p50/p95 wall time of benchmark.py's request6 (6 questions, 28 options) against a Jev /v1/systemone endpoint.

python lat6.py --endpoint http://127.0.0.1:8765/v1/systemone --image square.jpg --repeats 10
"""
import argparse, base64, json, statistics, time, urllib.request

Q = {
    "kind": {"type": "choice", "instructions": "What kind of image is this?",
             "criteria": {"photo": "a photograph of a real scene", "screenshot": "an app or website screenshot",
                          "document": "a scanned or photographed page", "chart": "a chart or diagram"}},
    "setting": {"type": "choice", "instructions": "Where does the scene take place?",
                "criteria": {k: f"the setting is {k.replace('_', ' ')}" for k in
                             ("indoors_home", "office", "street", "nature", "vehicle", "unclear")}},
    "clutter": {"type": "score", "instructions": "How cluttered is the image?",
                "criteria": ["empty", "sparse", "moderate", "busy", "chaotic"]},
    "legibility": {"type": "score", "instructions": "How legible is any text in the image?",
                   "criteria": [f"{i} out of 10" for i in range(1, 11)]},
    "person": {"type": "noul", "instructions": "Is a person visible?"},
    "damage": {"type": "noul", "instructions": "Does anything shown look damaged?",
               "criteria": {"true": "visible damage", "false": "no visible damage"}},
}
ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--warmup", type=int, default=1)
a = ap.parse_args()
body = json.dumps({"state": "", "questions": Q,
                   "images": ["data:image/jpeg;base64," + base64.b64encode(open(a.image, "rb").read()).decode()]}).encode()
times = []
for i in range(a.warmup + a.repeats):
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(a.endpoint, data=body, headers={"Content-Type": "application/json"}), timeout=3600) as r:
        ans = json.loads(r.read())["answers"]
    if i >= a.warmup:
        times.append(time.perf_counter() - t)
times.sort()
p95 = times[min(len(times) - 1, round(0.95 * (len(times) - 1)))]
print(json.dumps({"repeats": len(times), "p50_s": round(statistics.median(times), 2), "p95_s": round(p95, 2),
                  "answers": {k: v.get("choice", v.get("noul", v.get("score"))) for k, v in ans.items()}}))
