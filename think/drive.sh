#!/bin/bash
# Train → pick → merge → serve trained → eval trained → serve base (same precision) → eval base.
# Resumable by stage: each stage touches $D/<stage>.done. Run detached:
#   (nohup bash /root/big/drive.sh > /root/big/drive/drive.log 2>&1 < /dev/null &)
set -u
B=/root/big; D=$B/drive; mkdir -p $D; cd $B
ulimit -n 65536
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
VPY=/root/vllm-env/bin/python; TPY=/root/train-env/bin/python; EPY=$B/evals/venv/bin/python
RUN=${RUN:-q36-v1}
TRAIN=${TRAIN:-$B/data/sft/train_all.jsonl}; EVAL=${EVAL:-$B/data/sft/val_all.jsonl}
log() { echo "$(date +%H:%M:%S) $*"; }
stage() { [ -e $D/$1.done ] && { log "skip $1"; return 1; }; log "start $1"; return 0; }
done_() { touch $D/$1.done; log "done $1"; }

stop_gpu() {  # vLLM, jevsrv and imajev off the GPU
  pkill -f "[v]llm serve" ; pkill -f "[j]evsrv.py" ; pkill -f "[p]layground/server.py"
  for i in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break; sleep 5; done
  pkill -9 -f "[v]llm serve"; pkill -9 -f "VLLM::"; sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

serve() {  # serve MODEL NAME [extra vllm args]: vLLM on :8000 + jevsrv on :8765
  local model=$1 name=$2; shift 2
  stop_gpu
  (nohup /root/vllm-env/bin/vllm serve $model --served-model-name $name --port 8000 --api-server-count 8 \
     --gpu-memory-utilization 0.88 --max-model-len 32768 --max-num-seqs 384 --limit-mm-per-prompt '{"image": 2}' \
     --enable-prefix-caching --max-logprobs 50 --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
     "$@" > $D/vllm-$name.log 2>&1 < /dev/null &)
  for i in $(seq 120); do curl -s -m 3 localhost:8000/v1/models >/dev/null && break; sleep 10; done
  curl -s -m 3 localhost:8000/v1/models >/dev/null || { log "vLLM for $name did not come up"; exit 1; }
  (nohup $VPY jevsrv.py --model $name --name $name-think --samples 4 > $D/jevsrv-$name.log 2>&1 < /dev/null &)
  sleep 5
}

evals() {  # evals TAG: every suite through jevsrv on :8765, in parallel
  local t=$1 E=http://127.0.0.1:8765/v1/systemone
  [ -e $B/data/calib/jrecords.jsonl ] || $VPY - <<'EOF'
import json
with open("/root/big/data/calib/jrecords.jsonl", "w") as f:
    for l in open("/root/big/data/calib/items.jsonl"):
        r = json.loads(l)
        f.write(json.dumps({"id": r["id"], "source": r["source"], "group": r.get("group", r["id"]), "gold": r["gold"],
                            "wire": {"state": r["state"], "questions": {"decision": r["question"]}, "images": r["images"]}}) + "\n")
EOF
  cd $B/evals
  $EPY jrun.py --records $B/data/calib/jrecords.jsonl --endpoint $E --out runs/calib-$t --workers 24 > runs/calib-$t.log 2>&1 &
  $EPY jrun.py --records realjev/records.jsonl --endpoint $E --out runs/realjev-$t --workers 48 > runs/realjev-$t.log 2>&1 &
  $EPY jrun.py --records heldout/records-sub.jsonl --endpoint $E --out runs/heldout-$t --workers 32 > runs/heldout-$t.log 2>&1 &
  ./jevbench_run.sh http://127.0.0.1:8765 $B/evals/jevbench-runs/$t > jevbench-runs/$t.log 2>&1 &
  cd $B
  for s in dev calibration; do
    [ -e runs/$t-$s/completion.json ] || { rm -rf runs/$t-$s; PYTHONPATH=imajev/src $VPY prun.py --records imajev-bench/records/records-public.jsonl \
      --root imajev-bench --split $s --output runs/$t-$s --endpoint $E --workers 32; }
    [ -e runs/$t-$s/score.json ] || PYTHONPATH=imajev/src $VPY -m imajev_bench score --allow-draft --records imajev-bench/records/records-public.jsonl \
      --root imajev-bench --split $s --predictions runs/$t-$s/predictions.jsonl --output runs/$t-$s/score.json --bootstrap-samples 1000
  done
  wait
  cd $B/evals  # one retry pass for request errors (jrun reruns only missing/errored ids)
  $EPY jrun.py --records $B/data/calib/jrecords.jsonl --endpoint $E --out runs/calib-$t --workers 24 >> runs/calib-$t.log 2>&1 &
  $EPY jrun.py --records realjev/records.jsonl --endpoint $E --out runs/realjev-$t --workers 48 >> runs/realjev-$t.log 2>&1 &
  $EPY jrun.py --records heldout/records-sub.jsonl --endpoint $E --out runs/heldout-$t --workers 32 >> runs/heldout-$t.log 2>&1 &
  ./jevbench_run.sh http://127.0.0.1:8765 $B/evals/jevbench-runs/$t >> jevbench-runs/$t.log 2>&1 &
  wait; cd $B
}

if stage train; then
  stop_gpu
  cd $B/train && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TPY train_lora.py --model Qwen/Qwen3.6-35B-A3B \
    --train $TRAIN --eval $EVAL --out runs/$RUN --epochs 2 --lr 1e-4 --warmup 0.05 --micro-tokens 16384 --grad-accum 4 \
    --max-len 12288 --eval-every 25 --save-every 25 $( [ -d runs/$RUN ] && echo --resume auto ) > $D/train.log 2>&1
  rc=$?; cd $B; [ $rc = 0 ] || { log "training failed rc=$rc"; exit 1; }
  done_ train
fi

if stage pick; then  # lowest held-out loss; ties → later step
  $VPY - <<EOF > $D/best.txt
import json, re
ev = [json.loads(l) for l in open("$D/train.log") if l.startswith('{"step"') and "eval_loss" in l]
best = min(ev, key=lambda r: (r["eval_loss"], -r["step"]))
print(f"$B/train/runs/$RUN/step-{best['step']:06d}")
EOF
  cat $D/best.txt; [ -d "$(cat $D/best.txt)" ] || { log "no checkpoint dir for best step"; exit 1; }
  done_ pick
fi

if stage merge; then
  $TPY train/merge.py --base Qwen/Qwen3.6-35B-A3B --adapter $(cat $D/best.txt) --out $B/train/merged/$RUN > $D/merge.log 2>&1 \
    || { log "merge failed"; exit 1; }
  done_ merge
fi

if stage eval-trained; then
  serve $B/train/merged/$RUN $RUN --quantization fp8
  evals $RUN
  done_ eval-trained
fi

if stage eval-base; then
  serve Qwen/Qwen3.6-35B-A3B q36 --quantization fp8
  evals base8
  done_ eval-base
fi
log "all done"; touch $D/done
