#!/usr/bin/env bash
# Stage 2 of the 2-stage Eagle3 training for Nemotron-Cascade-2-30B-A3B.
#
# Stage 2: load the draft weights from a stage-1 checkpoint and
# fine-tune on the on-policy reasoning trace distribution
# (c2_traces_train + c2_traces_cot_train + any extras dropped under
# $WORK_DIR/data/extra_traces/) at a LOWER learning rate. The goal is
# to specialize the draft to the actual deployment distribution
# without forgetting the general SFT foundation from stage 1.
#
# Recommended schedule per @hav4ik:
#   * Stage 1: 1 epoch over the larger SFT pool (lr=1e-4)
#   * Stage 2: 2-3 epochs over the smaller traces pool (lr=5e-5)
#
# Why --ckpt-dir not --resume:
#   `--ckpt-dir <dir>` loads the draft model WEIGHTS from a previous
#   checkpoint but creates a FRESH optimizer + scheduler. This is what
#   we want for stage 2 because we're switching to a different LR and
#   want the optimizer to start from zero moments (otherwise the Adam
#   moments from stage 1 would override the new LR's intended dynamics).
#
#   `--resume` would also load the optimizer state -- that's the
#   "continue training as if uninterrupted" semantic and is the wrong
#   thing for a stage transition.
#
# Prerequisites:
#   1. Stage 1 must have completed and produced at least one checkpoint:
#      $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_<N>/
#   2. Stage 2 data file built via prepare_data_stage2.sh:
#      $WORK_DIR/data/all_data_stage2.jsonl
#   3. Tokenizer + loss-mask + vocab mapping cache built for stage 2:
#      TRAIN_DATA=$WORK_DIR/data/all_data_stage2.jsonl \
#          bash experiments/nemotron-cascade-2/build_cache.sh
#
# Usage:
#   CKPT_DIR=$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_<N> \
#       bash experiments/nemotron-cascade-2/run_train_stage2.sh
#
# Output:
#   $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2/

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
TRAIN_DATA=${TRAIN_DATA:-$WORK_DIR/data/all_data_stage2.jsonl}
EVAL_DATA=${EVAL_DATA:-$WORK_DIR/data/c2_traces_validation.jsonl}
TARGET_MODEL=${TARGET_MODEL:-nvidia/Nemotron-Cascade-2-30B-A3B}
NUM_GPUS=${NUM_GPUS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MAX_LENGTH=${MAX_LENGTH:-32768}
CACHE_DIR=${CACHE_DIR:-$WORK_DIR/cache_l${MAX_LENGTH}}

if [[ -z "${CKPT_DIR:-}" ]]; then
    echo "ERROR: CKPT_DIR must be set to the stage-1 checkpoint directory." >&2
    echo "Example:" >&2
    echo "  CKPT_DIR=\$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_15000 \\" >&2
    echo "      bash experiments/nemotron-cascade-2/run_train_stage2.sh" >&2
    exit 2
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: CKPT_DIR=$CKPT_DIR is not a directory." >&2
    exit 2
fi
if [[ ! -f "$CKPT_DIR/config.json" || ! -f "$CKPT_DIR/model.safetensors" ]]; then
    echo "ERROR: $CKPT_DIR does not look like a SpecForge checkpoint" >&2
    echo "       (expected config.json + model.safetensors)" >&2
    exit 2
fi

mkdir -p "$CACHE_DIR" "$WORK_DIR/checkpoints" "$WORK_DIR/logs"

export HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/cache/compiled_kernels
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[stage2] Loading stage-1 weights from CKPT_DIR=$CKPT_DIR"
echo "[stage2] Fine-tuning at lr=$LEARNING_RATE for $NUM_EPOCHS epochs on $TRAIN_DATA"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    "$SPECFORGE_ROOT/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend hf \
    --trust-remote-code \
    --draft-model-config "$SPECFORGE_ROOT/configs/nemotron-cascade-2-eagle3-sw4k.json" \
    --embedding-key backbone.embeddings.weight \
    --train-data-path "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --eval-lengths 16384,32768,65536 \
    --eval-interval 1000 \
    --chat-template nemotron-h \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2" \
    --ckpt-dir "$CKPT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size 1 \
    --learning-rate "$LEARNING_RATE" \
    --max-length "$MAX_LENGTH" \
    --ttt-length 6 \
    --draft-mlp-chunk-size 4096 \
    --fused-linear-loss \
    --fused-linear-loss-chunk-size 4096 \
    --tp-size 1 \
    --build-dataset-num-proc 32 \
    --dataloader-num-workers 4 \
    --dist-timeout 180 \
    --save-interval 2000 \
    --log-interval 20 \
    --warmup-ratio 0.02 \
    --model-card-template "$SCRIPT_DIR/MODEL_CARD_TEMPLATE_sw4k.md" \
    --report-to wandb \
    --wandb-project nemotron-cascade-2-eagle3 \
    --wandb-name "stage2-L${MAX_LENGTH}-ttt6-sw4k" \
    "$@"
