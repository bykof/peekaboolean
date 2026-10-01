"""Downscale the images of records imajev rejected (4096-token limit) and write a retry records file."""
import json, sys
from pathlib import Path
from PIL import Image
run, out, edge = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
bad = {json.loads(l)["id"] for l in open(run / "predictions.jsonl") if json.loads(l)["status"] == "error"}
root = Path("realjev"); small = root / f"images_{edge}"; small.mkdir(exist_ok=True)
with open(out, "w") as f:
    for l in open(root / "records.jsonl"):
        r = json.loads(l)
        if r["id"] not in bad: continue
        paths = []
        for p in r["wire"].get("images", []):
            im = Image.open(root / p); im.thumbnail((edge, edge))
            q = Path(f"images_{edge}") / Path(p).name
            im.convert("RGB").save(root / q.with_suffix(".jpg"), quality=92); paths.append(str(q.with_suffix(".jpg")))
        r["wire"]["images"] = paths
        f.write(json.dumps(r) + "\n")
print(len(bad), "retry records ->", out)
