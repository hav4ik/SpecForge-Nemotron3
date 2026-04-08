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

## Training schedule: 2-stage (SFT bootstrap + on-policy specialization)

This draft head is trained in **two stages** to first build a general-language
foundation and then specialize to the on-policy reasoning trace distribution
that the verifier will actually emit at inference time.

| Stage | Source | Rows | LR | Epochs | Bucketing | Notes |
|---|---|---|---|---|---|---|
| **Stage 1** (SFT bootstrap) | `cascade2_sft_train.jsonl` | 20,000 | 1e-4 | 1 | **on** (length-bucketed sampler, stage 1 only) | Broad SFT distribution; long-tail length variance, so length-bucketed sampling kills the straggler-rank bottleneck on DP=2 |
| **Stage 2** (on-policy fine-tune) | `c2_traces_train.jsonl` | 9,658 | 5e-5 | 2-3 | **off** | Loaded from stage 1 checkpoint via `--ckpt-dir` (weight-only, fresh optimizer). Bucketing intentionally disabled to maximize gradient diversity per step on the narrower on-policy distribution we're specializing to. |

The CoT trace set (`c2_traces_cot_train.jsonl`, 0.9k rows) is **excluded
from this iteration** because it has significantly higher mean tokens-per-row
(~50k) than the plain traces (~21k), which would slow stage 2 wall-clock for
marginal benefit on the first baseline. It will be folded back in for a
follow-up stage 2 fine-tune if the deployed draft underperforms on
CoT-heavy reasoning.

## Key training details

| | |
|---|---|
| **Training framework** | SpecForge fork at https://github.com/hav4ik/SpecForge-Nemotron3, branch `nemotron-cascade-2-experiments` |
| **Git commit** | `{git_commit}` |
| **Training date** | {train_date} |
| **Max sequence length** | **{max_length} tokens** (this iteration is the L=32768 fast-iteration baseline; an L=65536 follow-up is planned once this baseline ships -- see "Iteration plan" section below) |
| **Truncation loss at this max_length** | Stage 1: ~26% of tokens lost to L=32768 truncation (avg tok/row 18904 → 13898). Stage 2: ~15% of tokens lost (avg 21468 → 18179). See "Training data accounting" section below for the per-stage table. |
| **TTT length** | {ttt_length} (Eagle3 test-time-training unroll depth) |
| **Sliding window** | 4096 tokens on the draft attention |
| **Optimizer** | BF16Optimizer, lr={lr}, warmup_ratio={warmup_ratio} |
| **Epochs (this stage)** | {num_epochs} |
| **Per-rank batch size** | {batch_size} (no sequence packing) |
| **Tensor parallel** | TP=1, DP={dp_size} |
| **Global effective batch per step** | {global_batch} conversations |

## Training data accounting (the full token-count picture)

Three different but equally valid token counts exist for this dataset, and
they don't agree because they measure different things. Documented here to
prevent future confusion:

| Metric | Stage 1 (cascade2_sft_train) | Stage 2 (c2_traces_train) | Source |
|---|---|---|---|
| Rows / conversations | 20,000 | 9,658 | dataset card + jsonl line count |
| **Untruncated total tokens** (1) | **378,073,468** | **207,336,404** | dataset card (`apply_chat_template(messages, tokenize=True)`) |
| **Truncated total tokens** at L=32768 (2) | 277,963,707 | 175,570,455 | this fork's tokenizer pipeline w/ `max_length=32768` |
| Truncation loss vs (1) | 100 M / 26.5% | 32 M / 15.3% | (1) - (2) |
| **Loss-masked tokens** (assistant turns only, what loss is computed on) (3) | **132,941,808** | **158,793,164** | this fork, `loss_mask == 1` positions in the tokenized sequence |
| Loss-mask fraction of (2) | 47.8% | 90.4% | (3) / (2) |
| Avg untruncated tok/row | 18,904 | 21,468 | dataset card |
| Avg truncated tok/row | 13,898 | 18,179 | (2) / rows |
| Avg loss-masked tok/row | 6,647 | 16,442 | (3) / rows |
| jsonl size on disk | 1521 MiB | 660 MiB | filesystem |

The three counts measure:

1. **Untruncated total** -- every token in the conversation including system
   prompt + user turns + assistant turns + special tokens, with no max-length
   cap. This is what the dataset card publishes.
2. **Truncated total** -- same thing but capped at the training `max_length`.
   Any sample longer than the cap loses its tail. **Stage 1 loses ~26% of
   total tokens at L=32768** because most SFT conversations are >18k tokens
   and many exceed 32k. Stage 2 loses ~15%. This is the main reason an L=65536
   follow-up is planned.
3. **Loss-masked** -- the subset of (2) where `loss_mask == 1`, i.e. only
   the assistant-turn positions. These are the only positions Eagle3's
   distillation loss is computed on. Stage 2's 90% loss-mask fraction
   reflects that reasoning traces are mostly the model's *output* (short
   user prompt, very long assistant trace). Stage 1's 48% reflects more
   balanced SFT dialogue.

## Eagle3 vocabulary pruning (the union d2t / t2d mapping)

Eagle3's draft head predicts logits over a **smaller draft vocabulary**
than the verifier's full vocab to keep the lm_head shape manageable.
Concretely: `target_vocab_size=131072` → `draft_vocab_size=32000`. The
mapping (`d2t` and `t2d` buffers on the draft model) is the **top-32000
most frequent target tokens** in the training set, computed from
loss-masked positions.

**Multi-stage gotcha** (fixed in this fork): the standard SpecForge
pipeline auto-generates the d2t / t2d from whatever training set the
current run is processing, and unconditionally calls
`draft_model.load_vocab_mapping(...)` AFTER loading the checkpoint passed
via `--ckpt-dir`. For 2-stage training this would mean stage 2 OVERWRITES
the d2t / t2d buffers loaded from the stage 1 checkpoint with a NEW
mapping derived from stage 2's train set. The lm_head weights loaded from
stage 1 would still be aligned to stage 1's mapping → silent index
permutation mismatch in every gradient step of stage 2.

**Fix**: a [union vocab mapping](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/build_union_vocab_mapping.py)
built from the *combined* token frequencies of stage 1 + stage 2,
saved once before training, and passed to BOTH stage launchers via
`--vocab-mapping-path`. This way the lm_head is index-aligned end to end.

**Per-stage coverage with the union top-32000:**

| Stage | Total loss-masked tokens | Unique token ids | Coverage by union top-32k | Tokens lost (out of vocab) | Unique-in-vocab |
|---|---|---|---|---|---|
| Stage 1 | 132,941,808 | 81,439 | **99.5328%** | 621,131 (0.47%) | 31,978 / 81,439 (39.27%) |
| Stage 2 | 158,793,164 | 36,510 | **99.9621%** | 60,180 (0.04%) | 26,017 / 36,510 (71.26%) |
| Union | 291,734,972 | 83,209 | **99.77%** | 671,311 (0.23%) | 32,000 / 83,209 (38.46%) |

Stage 2 has *better* coverage than stage 1 even though the union top-K
was built jointly -- because stage 2's reasoning-trace distribution is
much narrower (~half the unique tokens of stage 1 despite ~20% more
total tokens). Stage 2 contributes 1,770 unique tokens *not* present in
stage 1's loss-masked positions (mostly latex / reasoning markers); these
would have been silently dropped from the draft vocab if the mapping had
been built from stage 1 only.

Both stages comfortably exceed 99% frequency coverage with `draft_vocab_size=32000`,
so the chosen draft vocab is big enough -- no need to grow it.

## Iteration plan: this is the L=32768 fast-iteration baseline

This checkpoint is the **first** Eagle3 head trained against
Nemotron-Cascade-2 in this fork's iteration sequence:

| Iteration | max_length | Status | Goal |
|---|---|---|---|
| **L=32768** (this) | 32768 | in flight / shipped | Fast-iteration baseline. Get a working draft with all the engineering knobs validated end-to-end. Pays a ~26% truncation loss on stage 1 and ~15% on stage 2. |
| **L=65536** (next) | 65536 | planned | Long-context follow-up. Re-train both stages at the higher cap to recover the truncated tail. Will need `--draft-mlp-grad-checkpoint` re-enabled to fit on a 96 GB GPU per rank. |

The per-length eval data (16k / 32k / 64k validation sets, see below)
lets us measure how well the L=32k draft generalizes to longer contexts
*without* training at long context. If the L=32k → L=64k generalization
gap on the eval set is small, the L=65k follow-up may not be worth
re-training. If the gap is large, the follow-up is justified.

### Note on batch size + length-bucketed sampling

This draft is trained with **per-rank `batch_size=1`** and **no sequence
packing**. With `dp_size={dp_size}` data-parallel ranks the global
effective batch is **{global_batch} conversations per optimizer step**.
This is the standard configuration for long-context Eagle3 training
(NVIDIA's gpt-oss-120b long-context Eagle3 uses the same per-rank
batch=1 layout) -- packing into fixed-token-budget batches gives
diminishing returns at long context because each conversation already
saturates the GPU activation budget.

Per-step token counts vary by **~20x** because conversation lengths
in the training distribution range from ~30 to ~30000+ tokens. Without
length-aware sampling, every optimizer step is bottlenecked by the
slowest rank: a (2k, 32k) DP-pair wastes half the GPU while rank 1
sits idle waiting for rank 0.

**Stage 1 fixes this with length-bucketed sampling**
(`--with-data-bucketing`, see
[`specforge/data/utils.py:LengthBucketDistributedSampler`](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/specforge/data/utils.py)):
the sampler sorts the dataset by per-sample length, partitions the
sorted index into contiguous "global batches" of size `num_replicas *
batch_size`, shuffles only the *order* of those global batches per
epoch with a deterministic seed, and assigns each rank its slice of
every global batch. Effect: all ranks within a single optimizer step
see samples of similar length, eliminating the straggler-rank waste.

**Bias mitigation**: pure length-sorted iteration would bias gradient
updates by difficulty across an epoch (long samples first or last). The
per-epoch global-batch-order shuffle mitigates this; the intra-batch
order is intentionally NOT shuffled so each step still sees bucketed
lengths. Same tradeoff as HF Trainer's `group_by_length=True`.

**Stage 2 leaves bucketing OFF** intentionally. The on-policy reasoning
trace distribution is narrow enough that gradient diversity per step
matters more than throughput, and the per-step difficulty drift would
work against the "specialize precisely" goal of stage 2.

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

## Memory engineering required to train at long context

Training a 1-layer Llama-style draft against a 30B-param frozen
Mamba-Transformer verifier at long context (L=32k or L=65k) with
`ttt_length=6` does not fit on a 96 GB GPU per rank without
significant engineering. The SpecForge fork that produced this
checkpoint adds the following training-time optimizations:

1. **Chunked MLP forward** (`--draft-mlp-chunk-size 4096`,
   **active for this checkpoint**): Liger-style per-chunk MLP forward
   over the seq dim, drops the per-step transient peak from ~4 GiB to
   ~800 MiB per MLP forward. Cheap defensive memory saving with
   negligible compute overhead.
2. **Chunked fused linear + soft-target CE**
   (`--fused-linear-loss --fused-linear-loss-chunk-size 4096`,
   **active for this checkpoint**): chunks the lm_head + KL-distillation
   loss over the seq dim with per-chunk grad checkpointing, never
   materializes the full `[B, T, V]` logits tensor. Saves ~14 GiB at
   L=32k and ~28 GiB at L=65k. Mathematically equivalent to the
   unchunked path (numerical equivalence verified in the SpecForge
   fork's `specforge/core/loss.py` `__main__` block).
3. **Sliding-window attention** on the draft (`sliding_window=4096`,
   **active for this checkpoint**): collapses draft self-attention
   from O(L²) to O(L * window). See "Sliding-window attention" section.
4. **Mamba SSM fp32 monkey-patch** (always on): see Mamba SSM section.
5. **Patch to free verifier full-vocab logits** before draft TTT
   unrolling so the ~16 GiB `[L, V]` tensor doesn't stay alive across
   the backward pass.
6. **In-place padding** for the per-step shift operations on big
   logit tensors.

**Disabled at L=32k, re-enabled at L=65k**:

7. **Grad-checkpoint on the draft MLP** (`--draft-mlp-grad-checkpoint`,
   **OFF for this L=32k checkpoint**, will be ON for the L=65k follow-up):
   recomputes `gate_proj/silu/up_proj/(gate*up)` intermediates during
   backward, drops them from the saved-for-backward set across TTT
   unrolls. Required at L=65k to fit on 96 GB but costs ~25-30% on the
   draft forward, so we leave it off at L=32k where we have memory
   headroom and want maximum step throughput.

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
