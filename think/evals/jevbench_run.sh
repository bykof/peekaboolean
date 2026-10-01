#!/bin/sh
# JevBench public splits (hard/original/easy) through the harness's own typesafe adapter and summarize scorer.
# usage: jevbench_run.sh ENDPOINT_BASE OUT_DIR   (ENDPOINT_BASE without /v1/systemone, e.g. http://127.0.0.1:8765)
# The harness runs serially; each split is sharded (hard 8 + original 5 + easy 3 = 16 concurrent requests) and merged.
# Resumable: rerunning with the same OUT_DIR runs only tasks without an ok result (failed ones are retried).
# Only deviation from the stock harness: the typesafe adapter's fixed 120 s HTTP timeout is raised to 1800 s.
set -e
EP=$1; OUT=$2
PY=/root/big/evals/venv/bin/python; JB=/root/big/evals/jevbench
RUN='import sys
from jevbench.adapters.typesafe import TypeSafeAdapter as T
i = T.__init__
def init(s, *a, **k):
    i(s, *a, **k)
    s.timeout_s = 1800.0
T.__init__ = init
from jevbench.cli import main
sys.exit(main())'
mkdir -p "$OUT/shards" "$OUT/raw"
if [ ! -e "$OUT/shards/easy.00" ]; then
  for spec in hard:8 original:5 easy:3; do
    tier=${spec%:*}; k=${spec#*:}
    split -n r/$k -d "$JB/datasets/public/$tier.jsonl" "$OUT/shards/$tier."
  done
fi
for s in "$OUT"/shards/*.[0-9][0-9]; do
  # todo = tasks without an ok result; move a failed attempt's raw file aside (the harness refuses to overwrite it)
  n=$($PY - "$s" "$OUT/raw" <<'PYEOF'
import glob, hashlib, json, os, sys
s, raw = sys.argv[1:]
files = sorted(glob.glob(s + ".results*.jsonl"))
ok = {r["task_id"] for f in files for r in map(json.loads, open(f)) if r["ok"]}
todo = [l for l in open(s) if json.loads(l)["id"] not in ok]
for l in todo:
    p = os.path.join(raw, hashlib.sha256(json.loads(l)["id"].encode()).hexdigest() + ".json")
    if os.path.exists(p):
        os.replace(p, p + f".try{len(files)}")
open(s + ".todo", "w").writelines(todo)
print(len(files) if todo else -1)
PYEOF
)
  [ "$n" = -1 ] && continue
  sfx=$([ "$n" = 0 ] && echo "" || echo ".$n")
  PYTHONPATH=$JB $PY -c "$RUN" run --tasks "$s.todo" --adapter typesafe --endpoint "$EP" --key-env "" \
    --results "$s.results$sfx.jsonl" --ledger "$s.ledger$sfx.jsonl" --raw-dir "$OUT/raw" --manifest "$s.manifest$sfx.json" \
    > "$s.log$sfx" 2>&1 &
done
wait
: > "$OUT/scores.txt"
for tier in hard original easy; do
  # merge: last ok result per task, else its last failed one
  $PY - "$OUT"/shards/$tier.[0-9][0-9].results*.jsonl > "$OUT/$tier.results.jsonl" <<'PYEOF'
import json, sys
best = {}
for f in sys.argv[1:]:
    for r in map(json.loads, open(f)):
        if r["ok"] or not best.get(r["task_id"], {}).get("ok"):
            best[r["task_id"]] = r
print("".join(json.dumps(r) + "\n" for r in best.values()), end="")
PYEOF
  PYTHONPATH=$JB $PY -m jevbench.cli summarize --tasks "$JB/datasets/public/$tier.jsonl" \
    --results "$OUT/$tier.results.jsonl" > "$OUT/$tier.summary.json"
  $PY - "$OUT/$tier.summary.json" $tier <<'PYEOF' | tee -a "$OUT/scores.txt"
import json, sys
s = json.load(open(sys.argv[1]))
r = lambda x: None if x is None else round(x, 4)
print(sys.argv[2], f"{s['n_correct']}/{s['n_planned']}", "acc", r(s["accuracy"]), "ece", r(s["ece"]["ece"]),
      "brier", r(s["brier_mean"]), "valid", s["n_valid"], "attempted", s["n_attempted"])
PYEOF
done
