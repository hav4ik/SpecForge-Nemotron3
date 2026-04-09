#!/usr/bin/env bash
# V2 data prep for the Nemotron-Cascade-2 Eagle3 follow-up iteration.
#
# V2 differs from V1's stage 2 in two ways:
#   1. Includes c2_traces_cot_train.jsonl (V1 stage 2 excluded the CoT
#      set for fast iteration; V2 folds it back in to get the longer
#      reasoning traces).
#   2. Single-stage training -- no separate SFT bootstrap stage. V2
#      warm-starts from V1's stage 2 final checkpoint (which already
#      has the SFT foundation baked in via V1's stage 1).
#
# Output:
#   $WORK_DIR/data/all_data_v2.jsonl
#       (c2_traces_train.jsonl + c2_traces_cot_train.jsonl concatenated
#        and pre-shuffled with deterministic seed 42)

set -euo pipefail

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export HF_HOME

mkdir -p "$WORK_DIR/data" "$WORK_DIR/data/extra_traces"

echo "[prepare-data-v2] HF_HOME=$HF_HOME"
echo "[prepare-data-v2] WORK_DIR=$WORK_DIR"
echo "[prepare-data-v2] downloading chankhavu/c2_eagle3_train ..."

python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "chankhavu/c2_eagle3_train",
    repo_type="dataset",
    allow_patterns=["*.jsonl", "*.json"],
    max_workers=8,
)
print(f"[prepare-data-v2] dataset snapshot at {p}")
with open(os.path.join(os.environ.get("WORK_DIR", "."), "data", ".dataset_path"), "w") as f:
    f.write(p)
PY

DATA=$(cat "$WORK_DIR/data/.dataset_path")
echo "[prepare-data-v2] traces files:"
echo "[prepare-data-v2]   $DATA/c2_traces_train.jsonl"
echo "[prepare-data-v2]   $DATA/c2_traces_cot_train.jsonl"

# Optional extra traces files (same convention as V1).
EXTRA_FILES=()
if compgen -G "$WORK_DIR/data/extra_traces/*.jsonl" > /dev/null; then
    for f in "$WORK_DIR/data/extra_traces/"*.jsonl; do
        EXTRA_FILES+=("$f")
        n=$(wc -l < "$f")
        echo "[prepare-data-v2] extra traces file: $f ($n rows)"
    done
fi

cat \
    "$DATA/c2_traces_train.jsonl" \
    "$DATA/c2_traces_cot_train.jsonl" \
    "${EXTRA_FILES[@]}" \
    | shuf --random-source=<(yes 42) \
    > "$WORK_DIR/data/all_data_v2.jsonl"

n_total=$(wc -l < "$WORK_DIR/data/all_data_v2.jsonl")
n_traces=$(wc -l < "$DATA/c2_traces_train.jsonl")
n_cot=$(wc -l < "$DATA/c2_traces_cot_train.jsonl")
echo "[prepare-data-v2] wrote $WORK_DIR/data/all_data_v2.jsonl"
echo "[prepare-data-v2]   total rows: $n_total ($n_traces traces + $n_cot CoT + extras)"

# Stage validation file the same way the V1 prep scripts do, in case
# the user re-ran with a fresh WORK_DIR.
if [[ -f "$DATA/c2_traces_validation.jsonl" ]]; then
    cp -f "$DATA/c2_traces_validation.jsonl" "$WORK_DIR/data/c2_traces_validation.jsonl"
    n_eval=$(wc -l < "$WORK_DIR/data/c2_traces_validation.jsonl")
    echo "[prepare-data-v2] staged eval file $WORK_DIR/data/c2_traces_validation.jsonl ($n_eval rows)"
fi

echo "[prepare-data-v2] DONE"
