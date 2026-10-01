#!/bin/bash
# The 35B path, if Qwen3.8 does not win: full evals of the trained q36-v1, then the base at the same precision.
#   (nohup bash /root/big/drive36b.sh > /root/big/drive36b/drive.log 2>&1 < /dev/null &)
set -u
B=/root/big; D=$B/drive36b; mkdir -p $D; cd $B
ulimit -n 65536
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
VPY=/root/vllm-env/bin/python; TPY=/root/train-env/bin/python; EPY=$B/evals/venv/bin/python
log() { echo "$(date +%H:%M:%S) $*"; }
stage() { [ -e $D/$1.done ] && { log "skip $1"; return 1; }; log "start $1"; return 0; }
done_() { touch $D/$1.done; log "done $1"; }
# the 35B driver's helpers; 128 running sequences (at 384 the KV cache filled and vLLM kept preempting)
source <(sed -n '/^stop_gpu() {/,/^}/p; /^serve() {/,/^}/p; /^evals() {/,/^}/p' $B/drive.sh | sed "s/--max-num-seqs 384/--max-num-seqs 128/")

if stage eval-trained; then
  rm -rf runs/q36-v1-dev runs/q36-v1-calibration  # cut short by the restart on 2026-10-01
  serve $B/train/merged/q36-v1 q36-v1 --quantization fp8
  evals q36-v1
  done_ eval-trained
fi
if stage eval-base8; then
  serve Qwen/Qwen3.6-35B-A3B q36 --quantization fp8
  evals base8
  done_ eval-base8
fi
log "all done"; touch $D/done
