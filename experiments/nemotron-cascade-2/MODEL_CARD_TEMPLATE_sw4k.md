---
license: apache-2.0
library_name: specforge
base_model: nvidia/Nemotron-Cascade-2-30B-A3B
tags:
  - eagle3
  - speculative-decoding
  - draft-model
  - sliding-window-attention
  - long-context
  - nemotron
  - mamba
  - hybrid-state-space
language:
  - en
pipeline_tag: text-generation
---

# Eagle3 Long-Context Draft Head for Nemotron-Cascade-2-30B-A3B (Sliding-Window 4k)

This is an [Eagle3](https://arxiv.org/abs/2503.01840) speculative-decoding
**draft head** trained against
[`nvidia/Nemotron-Cascade-2-30B-A3B`](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
as the verifier. To our knowledge it is the first Eagle3 head trained
against a hybrid Mamba-Transformer MoE verifier, and the first to
explore **sliding-window attention on the draft** as a long-context
optimization.

## TL;DR

* **Verifier**: `nvidia/Nemotron-Cascade-2-30B-A3B` (30B-param hybrid
  Mamba-Transformer MoE, 52 layers = 31 Mamba + 15 MoE/MLP + 6 GQA
  attention).
* **Draft architecture**: 1-layer Llama transformer block with
  `hidden_size=2688` (matches verifier residual stream),
  `intermediate_size=8064` (~3x hidden), `head_dim=128`,
  `num_attention_heads=32`, `num_key_value_heads=2` (GQA),
  `sliding_window=4096` (band causal), `max_position_embeddings=262144`
  (matches verifier so vLLM does not clamp serving max_model_len),
  `vocab_size=131072`, `draft_vocab_size=32000`.
* **Aux hidden state layers** captured from verifier: layers
  **2 / 26 / 48** (~4% / 50% / 92% depth -- Mamba / Attention / Mamba).
  Layout follows NVIDIA's gpt-oss-120b long-context Eagle3 reference.
* **Trainable parameters**: ~196 M (excludes the frozen embedding
  layer loaded from the verifier).
* **Checkpoint**: `{ckpt_basename}` (epoch {epoch}, step {step}).

## Key training details

| | |
|---|---|
| **Training framework** | SpecForge fork at https://github.com/hav4ik/SpecForge-Nemotron3, branch `nemotron-cascade-2-experiments` |
| **Git commit** | `{git_commit}` |
| **Training date** | {train_date} |
| **Training data** | `chankhavu/c2_eagle3_train` (cascade2 SFT 20k + on-policy traces 9.6k + on-policy CoT 0.9k = 30566 conversations, pre-shuffled) |
| **Max sequence length** | {max_length} tokens (preserves ~96.5% of conversations fully at 65536) |
| **TTT length** | {ttt_length} (Eagle3 test-time-training unroll depth) |
| **Sliding window** | 4096 tokens on the draft attention |
| **Optimizer** | BF16Optimizer, lr={lr}, warmup_ratio={warmup_ratio} |
| **Epochs** | {num_epochs} |
| **Per-rank batch size** | {batch_size} (no sequence packing) |
| **Tensor parallel** | TP=1, DP={dp_size} |
| **Global effective batch per step** | {global_batch} conversations |

### Note on batch size

This draft was trained with **per-rank `batch_size=1`** and **no
sequence packing**. With `dp_size={dp_size}` data-parallel ranks the
global effective batch is **{global_batch} conversations per
optimizer step**. This is the standard configuration for long-context
Eagle3 training (NVIDIA's gpt-oss-120b long-context Eagle3 uses the
same per-rank batch=1 layout) -- packing into fixed-token-budget
batches gives diminishing returns at L=65536 because each
conversation already saturates the GPU activation budget.

Per-step **token counts vary by ~20x** because conversation lengths
in the training distribution range from ~30 to ~65000 tokens
(p25=4k, p50=12k, p75=28k, p95=62k tokens). Random per-rank shuffling
means per-step wall-clock time has the same variance, and the slowest
rank determines the global step time. Per-token gradient signal is
still ample (~`E[seq_len] * ttt_length = 70k+` position gradients per
rank per step, more than the original Eagle3 paper's batch=16 setup
on ~1k-token ShareGPT sequences). So the small global batch does NOT
hurt convergence per gradient step; it only hurts wall-clock
throughput. A length-bucketed `DistributedSampler` would close most
of the throughput gap without touching the math (tracked as future
work in the SpecForge fork's experiments folder).
| **Wandb run** | {wandb_run_url} |

## ⚠️ CRITICAL: Mamba SSM state precision

This draft was trained with the **Mamba SSM scan boundary state in
float32**, matching vLLM 0.19's default behavior for NemotronH
(`--mamba_ssm_cache_dtype float32`).

The Nemotron team has explicitly confirmed
([HF discussion](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions/8))
that downcasting the Mamba SSM state to bf16 causes a **~10% absolute
regression on AIME-class math benchmarks** (88.3% vs 99.17% on AIME 2025
was the SGLang-vs-vLLM gap they measured from this single issue alone).

**Both training and inference must use fp32 SSM state** to avoid this
precision regression. Concretely:

* **Training**: SpecForge fork includes a monkey-patch
  (`specforge/_mamba_fp32_patch.py`) that forces upstream `mamba_ssm`'s
  `_state_passing_fwd` to use `out_dtype=torch.float32`. The patch is
  applied at the very top of `scripts/train_eagle3.py` before any other
  module-level imports. It is bit-equivalent to vLLM 0.19's NemotronH
  default behavior.
* **Inference (vLLM 0.19+)**: works automatically for NemotronH-Cascade-2
  because the verifier's `config.json` ships with
  `mamba_ssm_cache_dtype: "float32"`, which triggers vLLM's
  `NemotronHForCausalLMConfig.verify_and_update_config` to set the SSM
  cache to fp32. **No explicit flag needed** when serving NemotronH-Cascade-2
  with vLLM 0.19+.
* **Inference (SGLang)**: pass `--mamba-ssm-dtype float32` (the SGLang
  flag is not on by default, per the Nemotron team's recommendation).
* **Inference (other frameworks / older vLLM)**: explicitly pass
  `--mamba_ssm_cache_dtype float32` (vLLM) or equivalent.

For the full audit story (call chain, the exact line numbers in
upstream `mamba_ssm` 2.3.1 where the bf16 downcast happens, vLLM's
4-layer override chain, etc.), see the
[HANDOFF.md "Mamba SSM precision -- the full intricacies" section](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/HANDOFF.md#mamba-ssm-precision----the-full-intricacies).

## Inference: serving with vLLM

```bash
vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \
    --speculative-config '{
        "model": "<path-to-this-checkpoint-or-hf-repo>",
        "method": "eagle3",
        "num_speculative_tokens": 5
    }' \
    --trust-remote-code \
    --max-model-len 262144 \
    --tensor-parallel-size 2
```

vLLM 0.19+ will automatically:
* Load the verifier with fp32 Mamba SSM state (auto-read from
  the verifier's config.json -- no explicit flag needed)
* Apply sliding-window attention to the draft using the
  `sliding_window: 4096` and `layer_types: ["sliding_attention"]`
  fields in this checkpoint's config.json. The Eagle3-aware
  `target_layer_count` offset is auto-computed from the verifier
  (no need to set it manually).
* Bound the draft KV cache to the 4096-token window via
  vLLM's `SlidingWindowSpec` in the hybrid KV cache manager.

## Inference: serving with SGLang

SGLang does not yet support sliding-window attention on Eagle3 drafts
(as of late 2025 -- check the SGLang changelog for status). When
SGLang adds support, the equivalent invocation will be:

```bash
python -m sglang.launch_server \
    --model nvidia/Nemotron-Cascade-2-30B-A3B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <path-to-this-checkpoint-or-hf-repo> \
    --speculative-num-steps {ttt_length} \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mamba-ssm-dtype float32 \
    --trust-remote-code \
    --tp 2
```

Until SGLang ships sliding-window support for Eagle3 drafts, **prefer
serving this draft via vLLM 0.19+**.

## Sliding-window attention -- the experimental hypothesis

The draft is a 1-layer Llama transformer; without windowing its
self-attention is O(L^2) per query position. The verifier
(Nemotron-Cascade-2) only has 6 attention layers and 31 Mamba layers,
so the verifier's compute scales gracefully with context length, but
the Eagle3 draft becomes the long-context bottleneck.

Hypothesis: **most reasoning-trace tokens are local-pattern continuations**
-- the draft only needs to predict locally-coherent next tokens; the
verifier rejects any draft proposal that diverges globally. So the
draft should be able to do well with a 4096-token sliding window,
collapsing its attention cost from O(L^2) to O(L*window) and freeing
us to push training context length up.

This checkpoint is the first published Eagle3 head built around this
hypothesis. Acceptance rates / downstream speedup numbers (when
available) will be added to a follow-up section here.

## Memory engineering required to train at L=65536

Training a 1-layer Llama-style draft against a 30B-param frozen
Mamba-Transformer verifier at L=65536 with `ttt_length=6` does not fit
on a 96 GB GPU per rank without significant engineering. The
SpecForge fork that produced this checkpoint adds the following
training-time optimizations (all required for this configuration):

1. **Grad-checkpoint on the draft MLP** (`--draft-mlp-grad-checkpoint`):
   recomputes `gate_proj/silu/up_proj/(gate*up)` intermediates during
   backward, drops them from the saved-for-backward set across TTT
   unrolls.
2. **Chunked MLP forward** (`--draft-mlp-chunk-size 4096`): Liger-style
   per-chunk MLP forward over the seq dim, drops the per-step
   transient peak from ~4 GiB to ~800 MiB per MLP forward.
3. **Chunked fused linear + soft-target CE**
   (`--fused-linear-loss --fused-linear-loss-chunk-size 4096`): chunks
   the lm_head + KL-distillation loss over the seq dim with per-chunk
   grad checkpointing, never materializes the full `[B, T, V]` logits
   tensor. Mathematically equivalent to the unchunked path (numerical
   equivalence verified in the SpecForge fork's
   `specforge/core/loss.py` `__main__` block).
4. **Sliding-window attention** on the draft (this checkpoint's config).
5. **Patch to free verifier full-vocab logits** before draft TTT
   unrolling so the ~16 GiB `[L, V]` tensor doesn't stay alive across
   the backward pass.
6. **In-place padding** for the per-step shift operations on big
   logit tensors.

See the
[experiments/nemotron-cascade-2/README.md](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/README.md)
for the complete memory math reference and the per-commit engineering
tour.

## License and attribution

* **Verifier model** (`nvidia/Nemotron-Cascade-2-30B-A3B`): refer to
  the verifier's HF page for its license. This draft head is
  derivative in the sense that it was trained against the verifier's
  outputs.
* **This draft head**: Apache 2.0 (matching the SpecForge license).
* **Training framework**: SpecForge fork at
  https://github.com/hav4ik/SpecForge-Nemotron3, derived from
  https://github.com/sgl-project/SpecForge with additional engineering
  patches for hybrid Mamba-Transformer verifiers (see branch
  `nemotron-cascade-2-experiments`).
