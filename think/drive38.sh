#!/bin/bash
# Qwen3.8-27B: zero-shot evals + RFT sampling (one server) → export → LoRA → pick → merge → trained evals.
# Waits for drive.sh (the 35B run) to finish first. Resumable by stage ($D/<stage>.done).
#   (nohup bash /root/big/drive38.sh > /root/big/drive38/drive.log 2>&1 < /dev/null &)
set -u
NOSPEC=${NOSPEC-1}  # MTP needs extra DeltaNet state slots per sequence: ~75 running seqs with it on this card
B=/root/big; D=$B/drive38; mkdir -p $D; cd $B
ulimit -n 65536
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
VPY=/root/vllm-env/bin/python; TPY=/root/train-env/bin/python; EPY=$B/evals/venv/bin/python
RUN=q38-v1; BASE=Qwen/Qwen3.8-27B
log() { echo "$(date +%H:%M:%S) $*"; }
stage() { [ -e $D/$1.done ] && { log "skip $1"; return 1; }; log "start $1"; return 0; }
done_() { touch $D/$1.done; log "done $1"; }
# the 35B driver's helpers; fewer running sequences, because at 384 the KV cache filled and vLLM kept preempting
source <(sed -n '/^stop_gpu() {/,/^}/p; /^serve() {/,/^}/p; /^evals() {/,/^}/p' $B/drive.sh | sed "s/--max-num-seqs 384/--max-num-seqs ${MAXSEQS:-256}/; s/--enable-prefix-caching/${PREFIX:---enable-prefix-caching}/; ${NOSPEC:+s/--speculative-config [^ ]* [^ ]* [^ ]* [^ ]*//}")

evals_light() {  # evals_light TAG: calibration set, ImajevBench dev+cal, JevBench (small prompts)
  local t=$1 E=http://127.0.0.1:8765/v1/systemone
  cd $B/evals
  $EPY jrun.py --records $B/data/calib/jrecords.jsonl --endpoint $E --out runs/calib-$t --workers 24 > runs/calib-$t.log 2>&1 &
  ./jevbench_run.sh http://127.0.0.1:8765 $B/evals/jevbench-runs/$t > jevbench-runs/$t.log 2>&1 &
  cd $B
  for s in dev calibration; do
    [ -e runs/$t-$s/completion.json ] || { rm -rf runs/$t-$s; PYTHONPATH=imajev/src $VPY prun.py --records imajev-bench/records/records-public.jsonl \
      --root imajev-bench --split $s --output runs/$t-$s --endpoint $E --workers 32; }
    [ -e runs/$t-$s/score.json ] || PYTHONPATH=imajev/src $VPY -m imajev_bench score --allow-draft --records imajev-bench/records/records-public.jsonl \
      --root imajev-bench --split $s --predictions runs/$t-$s/predictions.jsonl --output runs/$t-$s/score.json --bootstrap-samples 1000
  done
  wait
}

until [ -e $B/drive/done ]; do sleep 60; done; log "35B driver done"

if stage zs-and-rft; then
  serve $BASE-FP8 q38
  (cd $B/train && PYTHONPATH=$B/train:$B $TPY check_template.py --model $BASE --served q38 \
    --image $B/imajev-bench/assets/f5c651efe765736d.jpg --save $D/check_template.json > $D/check_template.log 2>&1)
  tail -3 $D/check_template.log
  mkdir -p $B/data/rft38
  (cd $B/data && $VPY -u rft.py --model q38 --ids ids38.txt --n 2 --concurrency 64 --out rft38/full.jsonl \
     --n-source '{"aokvqa": 4, "rule-claim": 4, "rule-count": 4, "nlvr2": 4, "ctrl-field": 4, "visual7w": 4, "vsr": 4, "st_vqa": 4, "rule-multi": 4, "rule-threshold": 4, "rule-record": 4}' \
     > $D/rft.log 2>&1) &
  evals_light q38zs
  wait
  (cd $B/data && $VPY -u rft.py --model q38 --ids ids38.txt --n 2 --concurrency 64 --out rft38/full.jsonl \
     --n-source '{"aokvqa": 4, "rule-claim": 4, "rule-count": 4, "nlvr2": 4, "ctrl-field": 4, "visual7w": 4, "vsr": 4, "st_vqa": 4, "rule-multi": 4, "rule-threshold": 4, "rule-record": 4}' \
     >> $D/rft.log 2>&1)  # resume pass for request errors
  done_ zs-and-rft
fi

if stage export; then
  mkdir -p $B/data/sft38
  (cd $B/data && RFT_DIR=$B/data/rft38 SFT_DIR=$B/data/sft38 $VPY export_sft.py > $D/export.log 2>&1) || { log "export failed"; exit 1; }
  wc -l $B/data/sft38/*.jsonl
  done_ export
fi

if stage train; then
  stop_gpu
  cd $B/train && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TPY train_lora.py --model $BASE \
    --train $B/data/sft38/train_all.jsonl --eval $B/data/sft38/val_all.jsonl --out runs/$RUN --epochs 2 --lr 1e-4 \
    --warmup 0.05 --micro-tokens 12288 --grad-accum 6 --max-len 12288 --eval-every 25 --save-every 25 \
    $( ls -d runs/$RUN/step-* >/dev/null 2>&1 && echo --resume auto ) > $D/train.log 2>&1
  rc=$?; cd $B; [ $rc = 0 ] || { log "training failed rc=$rc"; exit 1; }
  done_ train
fi

if stage pick; then
  $VPY - <<EOF > $D/best.txt
import json
ev = [json.loads(l) for l in open("$D/train.log") if l.startswith('{"step"') and "eval_loss" in l]
best = min(ev, key=lambda r: (r["eval_loss"], -r["step"]))
print(f"$B/train/runs/$RUN/step-{best['step']:06d}")
EOF
  cat $D/best.txt; [ -d "$(cat $D/best.txt)" ] || { log "no checkpoint dir for best step"; exit 1; }
  done_ pick
fi

if stage merge; then
  $TPY train/merge.py --base $BASE --adapter $(cat $D/best.txt) --out $B/train/merged/$RUN > $D/merge.log 2>&1 \
    || { log "merge failed"; exit 1; }
  done_ merge
fi

if stage eval-trained; then
  serve $B/train/merged/$RUN $RUN --quantization fp8
  evals $RUN
  done_ eval-trained
fi

if stage eval-zs-rest; then  # the zero-shot base on everything evals_light skipped (completed runs are skipped)
  serve $BASE-FP8 q38
  evals q38zs
  done_ eval-zs-rest
fi
log "all done"; touch $D/done
