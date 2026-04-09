#!/usr/bin/env bash
# V2 of the Nemotron-Cascade-2 Eagle3 draft head training.
#
# V2 differs from V1 in five ways:
#
#   1. Single-stage training. V1 was 2-stage (SFT bootstrap + on-policy
#      traces). V2 collapses this into a single stage that warm-starts
#      from V1's stage 2 final checkpoint (which already has the SFT
#      foundation baked in via V1's stage 1).
#
#   2. ttt_length = 7 (V1 used 6). The Eagle3 paper / SGLang reference
#      default. We held at ttt=6 in V1 only as a memory workaround at
#      L=65536 pre-chunked-MLP. At V1's L=32768 with the chunked stack
#      we have plenty of headroom for ttt=7 without grad-ckpt.
#
#   3. CoT data included. V1 stage 2 trained on c2_traces_train.jsonl
#      only (9658 rows) and excluded c2_traces_cot_train.jsonl (908
#      rows of long CoT traces, ~50k mean tokens). V2 includes both.
#
#   4. Single-length eval. V1 ran eval at 16k/32k/64k separately
#      (~3x the eval cost). V1's data showed the per-length gap is
#      <0.005 across all positions -- the multi-length eval was
#      answering a question that turned out to be uninteresting.
#      V2 just evaluates at L=65536.
#
#   5. Length-bucketed sampling enabled. V1 stage 2 disabled bucketing
#      to keep gradient diversity per step on the narrow trace
#      distribution. V2 re-enables it because:
#         - The CoT data has wide length variance (50k mean vs 21k for
#           regular traces), so the straggler-rank bottleneck is now
#           significant on DP=2.
#         - We're warm-starting, so the optimizer doesn't need as much
#           gradient diversity per step as a from-scratch run does.
#
# Vocab mapping: REUSED unchanged from V1
# (`union_vocab_mapping_l32k.pt`). The lm_head index alignment is
# preserved end-to-end. The CoT data only adds 0.04% out-of-vocab
# tokens (analyzed in HANDOFF.md "Version 2" section); rebuilding the
# union mapping would force re-init of the lm_head and is not worth
# the engineering risk.
#
# Optimizer: lower LR + shorter warmup vs V1 stage 2 because we're
# warm-starting from already-converged weights, and the CoT addition
# might want gentler updates to avoid destabilizing the converged
# regions of weight space.
#
# Prerequisites:
#   1. V1 stage 2 final checkpoint, e.g.
#      $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2/epoch_2_step_12000/
#      (or wherever V1 was stopped)
#   2. V2 data file built via prepare_data_v2.sh:
#      $WORK_DIR/data/all_data_v2.jsonl
#   3. Cache built for the V2 data file:
#      TRAIN_DATA=$WORK_DIR/data/all_data_v2.jsonl \
#          CACHE_DIR=$WORK_DIR/cache_l32768 MAX_LENGTH=32768 \
#          bash experiments/nemotron-cascade-2/build_cache.sh
#   4. Union vocab mapping (built once for V1, reused as-is):
#      $WORK_DIR/data/union_vocab_mapping_l32k.pt
#
# Usage:
#   CKPT_DIR=$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2/epoch_2_step_12000 \
#       bash experiments/nemotron-cascade-2/run_train_v2.sh
#
# Output:
#   $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-v2/

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
TRAIN_DATA=${TRAIN_DATA:-$WORK_DIR/data/all_data_v2.jsonl}
EVAL_DATA=${EVAL_DATA:-$WORK_DIR/data/c2_traces_validation.jsonl}
TARGET_MODEL=${TARGET_MODEL:-nvidia/Nemotron-Cascade-2-30B-A3B}
NUM_GPUS=${NUM_GPUS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.01}
TTT_LENGTH=${TTT_LENGTH:-7}
MAX_LENGTH=${MAX_LENGTH:-32768}
CACHE_DIR=${CACHE_DIR:-$WORK_DIR/cache_l${MAX_LENGTH}}
VOCAB_MAPPING_PATH=${VOCAB_MAPPING_PATH:-$WORK_DIR/data/union_vocab_mapping_l32k.pt}

if [[ -z "${CKPT_DIR:-}" ]]; then
    echo "ERROR: CKPT_DIR must be set to the V1 final checkpoint directory." >&2
    echo "Example:" >&2
    echo "  CKPT_DIR=\$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2/epoch_2_step_12000 \\" >&2
    echo "      bash experiments/nemotron-cascade-2/run_train_v2.sh" >&2
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
if [[ ! -f "$VOCAB_MAPPING_PATH" ]]; then
    echo "ERROR: VOCAB_MAPPING_PATH=$VOCAB_MAPPING_PATH does not exist" >&2
    exit 2
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: TRAIN_DATA=$TRAIN_DATA does not exist (run prepare_data_v2.sh first)" >&2
    exit 2
fi

mkdir -p "$CACHE_DIR" "$WORK_DIR/checkpoints" "$WORK_DIR/logs"

export HF_HOME=${HF_HOME:-$WORK_DIR/hf_home}
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=$WORK_DIR/cache/compiled_kernels
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "[v2] Loading V1 final weights from CKPT_DIR=$CKPT_DIR"
echo "[v2] Training at L=$MAX_LENGTH ttt=$TTT_LENGTH lr=$LEARNING_RATE warmup=$WARMUP_RATIO for $NUM_EPOCHS epochs"
echo "[v2] Train data: $TRAIN_DATA"
echo "[v2] Eval data: $EVAL_DATA (single L=65536 pass)"
echo "[v2] Vocab mapping: $VOCAB_MAPPING_PATH (reused from V1, unchanged)"

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
    --eval-lengths 65536 \
    --eval-interval 1000 \
    --vocab-mapping-path "$VOCAB_MAPPING_PATH" \
    --with-data-bucketing \
    --chat-template nemotron-h \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-v2" \
    --ckpt-dir "$CKPT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size 1 \
    --learning-rate "$LEARNING_RATE" \
    --max-length "$MAX_LENGTH" \
    --ttt-length "$TTT_LENGTH" \
    --draft-mlp-chunk-size 4096 \
    --fused-linear-loss \
    --fused-linear-loss-chunk-size 4096 \
    --tp-size 1 \
    --build-dataset-num-proc 32 \
    --dataloader-num-workers 4 \
    --dist-timeout 180 \
    --save-interval 2000 \
    --log-interval 20 \
    --warmup-ratio "$WARMUP_RATIO" \
    --model-card-template "$SCRIPT_DIR/MODEL_CARD_TEMPLATE_sw4k.md" \
    --report-to wandb \
    --wandb-project nemotron-cascade-2-eagle3 \
    --wandb-name "v2-L${MAX_LENGTH}-ttt${TTT_LENGTH}-sw4k-bucketed-cot" \
    "$@"
