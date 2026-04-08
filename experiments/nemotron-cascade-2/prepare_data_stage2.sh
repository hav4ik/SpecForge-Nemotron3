#!/usr/bin/env bash
# Stage 2 (on-policy fine-tune) data prep for the 2-stage Eagle3
# training of Nemotron-Cascade-2-30B-A3B.
#
# Stage 2 specializes the draft (already trained on a large off-policy
# SFT pool in stage 1) to the on-policy reasoning trace distribution
# that the model will actually emit at inference time. Smaller corpus,
# higher distribution match to the deployment setting.
#
# Sources of stage-2 data, all concatenated:
#   1. c2_traces_train.jsonl (~9.6k on-policy reasoning traces, T=1.0
#      top_p=0.95)
#   2. Any *.jsonl files placed under $WORK_DIR/data/extra_traces/
#      (optional; for adding additional rollouts)
#
# NOT included (intentionally, as of 2026-04-08):
#   * c2_traces_cot_train.jsonl -- the CoT traces have substantially
#     higher mean tokens-per-sample than the plain traces, which makes
#     stage 2 wall-clock much slower. We're prioritizing fast iteration
#     for the first baseline; if the deployed draft underperforms on
#     CoT-heavy reasoning later, fold this back in.
#
# Output:
#   $WORK_DIR/data/all_data_stage2.jsonl  (concatenated and pre-shuffled
#                                          with deterministic seed 42)
#
# All extra files MUST follow the same conversation schema.

set -euo pipefail

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export HF_HOME

mkdir -p "$WORK_DIR/data" "$WORK_DIR/data/extra_traces"

echo "[prepare-data-stage2] HF_HOME=$HF_HOME"
echo "[prepare-data-stage2] WORK_DIR=$WORK_DIR"
echo "[prepare-data-stage2] downloading chankhavu/c2_eagle3_train ..."

python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "chankhavu/c2_eagle3_train",
    repo_type="dataset",
    allow_patterns=["*.jsonl", "*.json"],
    max_workers=8,
)
print(f"[prepare-data-stage2] dataset snapshot at {p}")
with open(os.path.join(os.environ.get("WORK_DIR", "."), "data", ".dataset_path"), "w") as f:
    f.write(p)
PY

DATA=$(cat "$WORK_DIR/data/.dataset_path")
echo "[prepare-data-stage2] base traces file: $DATA/c2_traces_train.jsonl (CoT set excluded for fast iteration)"

EXTRA_FILES=()
if compgen -G "$WORK_DIR/data/extra_traces/*.jsonl" > /dev/null; then
    for f in "$WORK_DIR/data/extra_traces/"*.jsonl; do
        EXTRA_FILES+=("$f")
        n=$(wc -l < "$f")
        echo "[prepare-data-stage2] extra traces file: $f ($n rows)"
    done
else
    echo "[prepare-data-stage2] (no extra traces files in $WORK_DIR/data/extra_traces/ -- using base only)"
fi

cat \
    "$DATA/c2_traces_train.jsonl" \
    "${EXTRA_FILES[@]}" \
    | shuf --random-source=<(yes 42) \
    > "$WORK_DIR/data/all_data_stage2.jsonl"

n_total=$(wc -l < "$WORK_DIR/data/all_data_stage2.jsonl")
echo "[prepare-data-stage2] wrote $WORK_DIR/data/all_data_stage2.jsonl ($n_total rows)"

# Stage the validation file alongside the train file (same rationale as
# stage 1). Stage 2 launcher evals against c2_traces_validation.jsonl
# at three length caps (16384/32768/65536).
if [[ -f "$DATA/c2_traces_validation.jsonl" ]]; then
    cp -f "$DATA/c2_traces_validation.jsonl" "$WORK_DIR/data/c2_traces_validation.jsonl"
    n_eval=$(wc -l < "$WORK_DIR/data/c2_traces_validation.jsonl")
    echo "[prepare-data-stage2] staged eval file $WORK_DIR/data/c2_traces_validation.jsonl ($n_eval rows)"
else
    echo "[prepare-data-stage2] WARN: c2_traces_validation.jsonl not found in dataset snapshot at $DATA" >&2
    echo "[prepare-data-stage2] WARN: training will start without --eval-data-path" >&2
fi

echo "[prepare-data-stage2] DONE"
