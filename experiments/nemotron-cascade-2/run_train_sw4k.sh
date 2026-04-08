#!/usr/bin/env bash
# Long-context Eagle3 training for Nemotron-Cascade-2-30B-A3B with
# sliding-window attention on the draft.
#
# Configuration:
#   * max_length=65536  (preserves ~96.5% of conversations fully)
#   * ttt_length=6      (~96% of ttt=7's theoretical speedup ceiling)
#   * sliding_window=4096 on the draft (collapses O(L^2) -> O(L*window))
#   * draft-mlp-grad-checkpoint  -> drops MLP saved-for-backward
#   * draft-mlp-chunk-size=4096   -> Liger-style chunked MLP forward,
#                                    drops per-step transient peak
#   * fused-linear-loss           -> chunked fused linear+soft-target CE
#                                    on the lm_head, never materializes
#                                    [B,T,V] logits tensor
#
# Hypothesis: most reasoning-trace tokens are local-pattern continuations,
# so the draft only needs to predict locally-coherent tokens. The verifier
# rejects globally-divergent draft proposals anyway, so a small attention
# window should give comparable acceptance rates at much lower memory.
#
# Inference compatibility: the trained checkpoint can be served by vLLM
# via:
#
#     vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \
#         --speculative-config '{
#             "model": "<path-to-checkpoint>",
#             "method": "eagle3",
#             "num_speculative_tokens": 5
#         }' \
#         --trust-remote-code \
#         --mamba-ssm-cache-dtype float32
#
# vLLM's LlamaAttention reads `layer_types` + `sliding_window` from the
# draft config.json automatically (Eagle3-aware sliding-window plumbing
# already exists in vllm/model_executor/models/llama.py). Critical: keep
# the draft's `max_position_embeddings` = the verifier's value
# (262144) -- vLLM clamps the entire serving max_model_len to
# min(draft, verifier) so a smaller draft would cripple long-context
# serving.
#
# This is the experiment described in the Nemotron-Cascade-2 Eagle3
# README; see experiments/nemotron-cascade-2/README.md for the OOM
# debugging history and memory math that led to this exact config.

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
# Allow PyTorch to grow CUDA segments instead of pre-reserving fixed
# blocks. Reclaims a few GiB of fragmentation headroom which matters at
# L=65536 where we sit near the 96 GB GPU ceiling.
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
    --chat-template nemotron-h \
    --cache-dir "$WORK_DIR/cache" \
    --output-dir "$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-sw4k" \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 65536 \
    --ttt-length 6 \
    --draft-mlp-grad-checkpoint \
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
    --report-to wandb \
    --wandb-project nemotron-cascade-2-eagle3 \
    --wandb-name sw4k-L65k-ttt6-fused-mixed \
    "$@"
