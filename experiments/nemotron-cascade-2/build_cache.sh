#!/usr/bin/env bash
# Wrapper to build the offline tokenizer/loss-mask cache + vocab mapping
# for one of the Nemotron-Cascade-2 Eagle3 experiments. Run this BEFORE
# launching training so the train script hits the cache and skips
# preprocessing entirely.
#
# Usage examples:
#
#   # Default: long-context sw4k experiment (L=65536, sw4k draft config)
#   bash experiments/nemotron-cascade-2/build_cache.sh
#
#   # Baseline experiment (L=32768, no sliding window)
#   EXPERIMENT=baseline bash experiments/nemotron-cascade-2/build_cache.sh
#
# Override the work dir / data path / chunk count via env vars:
#
#   WORK_DIR=/scratch/eagle3 NUM_PROC=64 \
#       bash experiments/nemotron-cascade-2/build_cache.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
TRAIN_DATA=${TRAIN_DATA:-$WORK_DIR/data/all_data_shuffled.jsonl}
TARGET_MODEL=${TARGET_MODEL:-nvidia/Nemotron-Cascade-2-30B-A3B}
NUM_PROC=${NUM_PROC:-32}
EXPERIMENT=${EXPERIMENT:-sw4k}

case "$EXPERIMENT" in
    baseline)
        DRAFT_CONFIG="$SPECFORGE_ROOT/configs/nemotron-cascade-2-eagle3.json"
        MAX_LENGTH=${MAX_LENGTH:-32768}
        ;;
    sw4k)
        DRAFT_CONFIG="$SPECFORGE_ROOT/configs/nemotron-cascade-2-eagle3-sw4k.json"
        MAX_LENGTH=${MAX_LENGTH:-65536}
        ;;
    *)
        echo "Unknown EXPERIMENT=$EXPERIMENT (expected: baseline | sw4k)" >&2
        exit 1
        ;;
esac

mkdir -p "$WORK_DIR/cache" "$WORK_DIR/logs"

export HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export TOKENIZERS_PARALLELISM=false

echo "[build-cache] EXPERIMENT=$EXPERIMENT"
echo "[build-cache] TRAIN_DATA=$TRAIN_DATA"
echo "[build-cache] DRAFT_CONFIG=$DRAFT_CONFIG"
echo "[build-cache] MAX_LENGTH=$MAX_LENGTH"
echo "[build-cache] NUM_PROC=$NUM_PROC"
echo "[build-cache] CACHE_DIR=$WORK_DIR/cache"

python "$SCRIPT_DIR/build_cache_offline.py" \
    --target-model-path "$TARGET_MODEL" \
    --draft-model-config "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --chat-template nemotron-h \
    --max-length "$MAX_LENGTH" \
    --cache-dir "$WORK_DIR/cache" \
    --num-proc "$NUM_PROC" \
    --trust-remote-code
