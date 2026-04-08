#!/usr/bin/env bash
# Stage 1 of the 2-stage Eagle3 training for Nemotron-Cascade-2-30B-A3B.
#
# Stage 1: train the draft on a LARGER pool of off-policy SFT data
# (cascade2_sft_train + any extra cascade2-pretraining-style files
# dropped under $WORK_DIR/data/extra_sft/). The goal is a strong
# general-language foundation that stage 2 then specializes to the
# on-policy reasoning trace distribution.
#
# Recommended schedule per @hav4ik:
#   * Stage 1: 1 epoch over the larger SFT pool (lr=1e-4),
#     length-bucketed sampling (--with-data-bucketing) at L=32768
#     for fast iteration
#   * Stage 2: 2-3 epochs over the smaller traces pool (lr=5e-5),
#     loaded from the stage-1 checkpoint via --ckpt-dir, NO bucketing
#     (we want maximum gradient diversity per step on the on-policy
#     distribution we're specializing to)
#
# Why bucketing in stage 1 only:
#   Stage 1 has a wide length distribution (~1k to ~32k tokens after
#   the user's >=1024 floor). With DP=2 and batch_size=1, every step
#   is bottlenecked by the slowest rank, so a (2k, 32k) pairing wastes
#   half the GPU. Length-bucketing pairs similar lengths within each
#   global step, killing the straggler. Stage 2 has a narrower length
#   distribution AND is the "specialize precisely" stage, so we leave
#   it un-bucketed to keep gradient diversity high.
#
# Why L=32768 (down from 65536):
#   Get a baseline trained head out the door first. Each step is
#   ~2-3x faster at L=32k vs L=65k, and most of the SFT pool fits
#   under 32k anyway. Once we have a working draft we'll iterate up
#   to L=65k for the long-context evaluation.
#
# Prerequisites:
#   1. Stage 1 data file built via prepare_data_stage1.sh:
#      $WORK_DIR/data/all_data_stage1.jsonl
#   2. Tokenizer + loss-mask + vocab mapping cache built for stage 1:
#      TRAIN_DATA=$WORK_DIR/data/all_data_stage1.jsonl \
#          bash experiments/nemotron-cascade-2/build_cache.sh
#
# Output:
#   $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/
#       epoch_0_step_<N>/
#           model.safetensors
#           config.json
#           training_state.pt
#           MODEL_CARD.md
#
# To run stage 2 from this checkpoint, set CKPT_DIR to the deepest
# epoch_0_step_<N> directory and invoke run_train_stage2.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
TRAIN_DATA=${TRAIN_DATA:-$WORK_DIR/data/all_data_stage1.jsonl}
EVAL_DATA=${EVAL_DATA:-$WORK_DIR/data/c2_traces_validation.jsonl}
TARGET_MODEL=${TARGET_MODEL:-nvidia/Nemotron-Cascade-2-30B-A3B}
NUM_GPUS=${NUM_GPUS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-32768}
CACHE_DIR=${CACHE_DIR:-$WORK_DIR/cache_l${MAX_LENGTH}}
# CRITICAL: must point at the UNION (stage1+stage2) vocab mapping so the
# draft lm_head trained in stage 1 stays index-aligned with stage 2's
# vocab when stage 2 loads from --ckpt-dir. Build it once via:
#   python experiments/nemotron-cascade-2/build_union_vocab_mapping.py ...
# See HANDOFF.md "vocab mapping bug" section for the full rationale.
VOCAB_MAPPING_PATH=${VOCAB_MAPPING_PATH:-$WORK_DIR/data/union_vocab_mapping_l32k.pt}

mkdir -p "$CACHE_DIR" "$WORK_DIR/checkpoints" "$WORK_DIR/logs"

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
    --draft-model-config "$SPECFORGE_ROOT/configs/nemotron-cascade-2-eagle3-sw4k.json" \
    --embedding-key backbone.embeddings.weight \
    --train-data-path "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --eval-lengths 16384,32768,65536 \
    --eval-interval 1000 \
    --vocab-mapping-path "$VOCAB_MAPPING_PATH" \
    --with-data-bucketing \
    --chat-template nemotron-h \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size 1 \
    --learning-rate 1e-4 \
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
    --wandb-name "stage1-L${MAX_LENGTH}-ttt6-sw4k-bucketed" \
    "$@"
