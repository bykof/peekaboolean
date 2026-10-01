#!/bin/bash
# Raw upstream files for imajev's v1 held-out exam, laid out where scripts/v1/convert_heldout_*.py expect them.
set -euo pipefail
C=/root/big/evals/imajev-src/.cache/datasets/v1
PY=/root/big/evals/venv/bin/python
mkdir -p $C/heldout_abstention $C/heldout_countqa $C/heldout_fashionpedia $C/heldout_livewild/x $C/heldout_pairs
$PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("MM-UPD/MM-UPD", repo_type="dataset", local_dir="$C/heldout_abstention/mmupd",
    allow_patterns=[f"data/{n}.tsv" for n in ["mmaad_aad_20240303_base", "mmaad_standard_20240303_base", "mmiasd_iasd_20240303_base", "mmivqd_ivqd_20240303_base"]])
snapshot_download("He-Xingwei/TUBench", repo_type="dataset", local_dir="$C/heldout_abstention/tubench")
snapshot_download("Jayant-Sravan/CountQA", repo_type="dataset", local_dir="$C/heldout_countqa", allow_patterns=["data/*.parquet"])
snapshot_download("MUIRBENCH/MUIRBENCH", repo_type="dataset", local_dir="$C/heldout_pairs", allow_patterns=["data/*.parquet"])
snapshot_download("chaofengc/IQA-PyTorch-Datasets", repo_type="dataset", local_dir="$C/heldout_livewild/dl", allow_patterns=["live_challenge.tgz"])
PYEOF
tar -xzf $C/heldout_livewild/dl/live_challenge.tgz -C $C/heldout_livewild/x
cd $C/heldout_fashionpedia
curl -sSfLO https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
curl -sSfLO https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
echo FETCH_DONE
