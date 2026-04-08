#!/usr/bin/env bash
# Download + merge + shuffle the chankhavu/c2_eagle3_train training data
# for Nemotron-Cascade-2-30B-A3B Eagle3 training.
#
# Produces:
#   $WORK_DIR/data/all_data_shuffled.jsonl   (mixed-stage Experiment 2)
#
# The dataset has three splits:
#   * cascade2_sft_train.jsonl   (off-policy SFT, ~20k rows)
#   * c2_traces_train.jsonl      (on-policy reasoning traces T=1.0, top_p=0.95)
#   * c2_traces_cot_train.jsonl  (on-policy CoT reasoning traces)
#
# We concatenate all three and pre-shuffle with a deterministic seed (42).
# SpecForge's dataloader will reshuffle each epoch on top.

set -euo pipefail

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export HF_HOME

mkdir -p "$WORK_DIR/data"

echo "[prepare-data] HF_HOME=$HF_HOME"
echo "[prepare-data] WORK_DIR=$WORK_DIR"
echo "[prepare-data] downloading chankhavu/c2_eagle3_train ..."

python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "chankhavu/c2_eagle3_train",
    repo_type="dataset",
    allow_patterns=["*.jsonl", "*.json"],
    max_workers=8,
)
print(f"[prepare-data] dataset snapshot at {p}")
with open(os.path.join(os.environ.get("WORK_DIR", "."), "data", ".dataset_path"), "w") as f:
    f.write(p)
PY

DATA=$(cat "$WORK_DIR/data/.dataset_path")
echo "[prepare-data] merging+shuffling splits from $DATA"

cat \
    "$DATA/cascade2_sft_train.jsonl" \
    "$DATA/c2_traces_train.jsonl" \
    "$DATA/c2_traces_cot_train.jsonl" \
    | shuf --random-source=<(yes 42) \
    > "$WORK_DIR/data/all_data_shuffled.jsonl"

n=$(wc -l < "$WORK_DIR/data/all_data_shuffled.jsonl")
echo "[prepare-data] wrote $WORK_DIR/data/all_data_shuffled.jsonl ($n rows)"
echo "[prepare-data] DONE"
