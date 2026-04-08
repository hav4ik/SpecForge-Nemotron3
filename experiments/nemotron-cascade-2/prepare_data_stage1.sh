#!/usr/bin/env bash
# Stage 1 (off-policy SFT) data prep for the 2-stage Eagle3 training of
# Nemotron-Cascade-2-30B-A3B.
#
# Stage 1 trains the draft on a LARGER pool of off-policy SFT data drawn
# from cascade2's pre-training distribution. The goal is to give the draft
# a strong general-language foundation before stage 2 specializes it on
# the on-policy reasoning trace distribution.
#
# Sources of stage-1 data, all concatenated:
#   1. cascade2_sft_train.jsonl from chankhavu/c2_eagle3_train (the
#      original 20000 SFT conversations -- always included)
#   2. Any *.jsonl files placed under $WORK_DIR/data/extra_sft/ (optional;
#      this is where you drop additional cascade2-pretraining-style
#      conversations to scale up stage 1)
#
# Output:
#   $WORK_DIR/data/all_data_stage1.jsonl  (concatenated and pre-shuffled
#                                          with deterministic seed 42)
#
# All extra files MUST follow the same conversation schema as
# cascade2_sft_train.jsonl: each line is a JSON object with a
# `"conversations"` field containing a list of {role, content, ...}
# messages. SpecForge's safe_conversations_generator will normalize the
# message keys for any extra fields (tool_calls, name, tool_call_id) so
# tool-use SFT data is fine.

set -euo pipefail

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export HF_HOME

mkdir -p "$WORK_DIR/data" "$WORK_DIR/data/extra_sft"

echo "[prepare-data-stage1] HF_HOME=$HF_HOME"
echo "[prepare-data-stage1] WORK_DIR=$WORK_DIR"
echo "[prepare-data-stage1] downloading chankhavu/c2_eagle3_train ..."

python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "chankhavu/c2_eagle3_train",
    repo_type="dataset",
    allow_patterns=["*.jsonl", "*.json"],
    max_workers=8,
)
print(f"[prepare-data-stage1] dataset snapshot at {p}")
with open(os.path.join(os.environ.get("WORK_DIR", "."), "data", ".dataset_path"), "w") as f:
    f.write(p)
PY

DATA=$(cat "$WORK_DIR/data/.dataset_path")
echo "[prepare-data-stage1] base SFT file: $DATA/cascade2_sft_train.jsonl"

# Collect: the canonical cascade2_sft_train.jsonl + any extra *.jsonl in
# $WORK_DIR/data/extra_sft/
EXTRA_FILES=()
if compgen -G "$WORK_DIR/data/extra_sft/*.jsonl" > /dev/null; then
    for f in "$WORK_DIR/data/extra_sft/"*.jsonl; do
        EXTRA_FILES+=("$f")
        n=$(wc -l < "$f")
        echo "[prepare-data-stage1] extra SFT file: $f ($n rows)"
    done
else
    echo "[prepare-data-stage1] (no extra SFT files in $WORK_DIR/data/extra_sft/ -- using base only)"
fi

cat \
    "$DATA/cascade2_sft_train.jsonl" \
    "${EXTRA_FILES[@]}" \
    | shuf --random-source=<(yes 42) \
    > "$WORK_DIR/data/all_data_stage1.jsonl"

n_total=$(wc -l < "$WORK_DIR/data/all_data_stage1.jsonl")
echo "[prepare-data-stage1] wrote $WORK_DIR/data/all_data_stage1.jsonl ($n_total rows)"

# Stage the validation file alongside the train file. Both stage 1 and
# stage 2 launchers eval against c2_traces_validation.jsonl at three
# length caps (16384/32768/65536), so we materialize a stable copy in
# $WORK_DIR/data/ rather than relying on the HF snapshot path which
# changes per cache dir.
if [[ -f "$DATA/c2_traces_validation.jsonl" ]]; then
    cp -f "$DATA/c2_traces_validation.jsonl" "$WORK_DIR/data/c2_traces_validation.jsonl"
    n_eval=$(wc -l < "$WORK_DIR/data/c2_traces_validation.jsonl")
    echo "[prepare-data-stage1] staged eval file $WORK_DIR/data/c2_traces_validation.jsonl ($n_eval rows)"
else
    echo "[prepare-data-stage1] WARN: c2_traces_validation.jsonl not found in dataset snapshot at $DATA" >&2
    echo "[prepare-data-stage1] WARN: training will start without --eval-data-path; you can re-run this script after the dataset is updated" >&2
fi

echo "[prepare-data-stage1] DONE"
