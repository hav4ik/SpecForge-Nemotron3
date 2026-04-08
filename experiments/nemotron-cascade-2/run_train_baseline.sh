#!/usr/bin/env bash
# Baseline Eagle3 training for Nemotron-Cascade-2-30B-A3B.
#
# Configuration:
#   * max_length=32768  (truncates ~23% of conversations at the long-tail)
#   * ttt_length=2      (memory-conservative; ~1.7x speedup ceiling)
#   * NO sliding window (full causal attention on the draft)
#   * NO grad checkpoint, NO chunked MLP, NO chunked fused loss
#
# Why a separate baseline: this is the simplest config that fits comfortably
# on a 96 GB GPU per rank without any of the long-context engineering tricks.
# Useful as a sanity check / loss-curve reference for validating that the
# more aggressive sw4k experiment converges to similar trajectory shape.
#
# Use Experiment 2 (single-stage mixed) data layout: cascade2_sft_train +
# c2_traces_train + c2_traces_cot_train concatenated and pre-shuffled, see
# experiments/nemotron-cascade-2/prepare_data.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
TRAIN_DATA=${TRAIN_DATA:-$WORK_DIR/data/all_data_shuffled.jsonl}
TARGET_MODEL=${TARGET_MODEL:-nvidia/Nemotron-Cascade-2-30B-A3B}
NUM_GPUS=${NUM_GPUS:-2}

mkdir -p "$WORK_DIR/cache" "$WORK_DIR/checkpoints" "$WORK_DIR/logs"

export HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/cache/compiled_kernels
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    "$SPECFORGE_ROOT/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend hf \
    --trust-remote-code \
    --draft-model-config "$SPECFORGE_ROOT/configs/nemotron-cascade-2-eagle3.json" \
    --embedding-key backbone.embeddings.weight \
    --train-data-path "$TRAIN_DATA" \
    --chat-template nemotron-h \
    --cache-dir "$WORK_DIR/cache" \
    --output-dir "$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-baseline" \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 32768 \
    --ttt-length 2 \
    --tp-size 1 \
    --build-dataset-num-proc 32 \
    --dataloader-num-workers 4 \
    --dist-timeout 180 \
    --save-interval 2000 \
    --log-interval 20 \
    --warmup-ratio 0.02 \
    --model-card-template "$SCRIPT_DIR/MODEL_CARD_TEMPLATE_baseline.md" \
    --report-to wandb \
    --wandb-project nemotron-cascade-2-eagle3 \
    --wandb-name baseline-L32k-ttt2-mixed \
    "$@"
