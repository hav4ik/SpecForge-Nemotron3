---
license: apache-2.0
library_name: specforge
base_model: nvidia/Nemotron-Cascade-2-30B-A3B
tags:
  - eagle3
  - speculative-decoding
  - draft-model
  - nemotron
  - mamba
  - hybrid-state-space
language:
  - en
pipeline_tag: text-generation
---

# Eagle3 Baseline Draft Head for Nemotron-Cascade-2-30B-A3B

This is an [Eagle3](https://arxiv.org/abs/2503.01840) speculative-decoding
**draft head** trained against
[`nvidia/Nemotron-Cascade-2-30B-A3B`](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
as the verifier. To our knowledge it is the first Eagle3 head trained
against a hybrid Mamba-Transformer MoE verifier.

This is the **baseline** configuration -- a minimal-config sanity
training run with no sliding-window attention and no long-context
engineering tricks. For the long-context variant with sliding-window
attention, see the
[`-sw4k` checkpoints](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/MODEL_CARD_TEMPLATE_sw4k.md).

## TL;DR

* **Verifier**: `nvidia/Nemotron-Cascade-2-30B-A3B` (30B-param hybrid
  Mamba-Transformer MoE, 52 layers = 31 Mamba + 15 MoE/MLP + 6 GQA
  attention).
* **Draft architecture**: 1-layer Llama transformer block with
  `hidden_size=2688` (matches verifier residual stream),
  `intermediate_size=8064` (~3x hidden), `head_dim=128`,
  `num_attention_heads=32`, `num_key_value_heads=2` (GQA),
  **full causal attention (no sliding window)**,
  `max_position_embeddings=262144`, `vocab_size=131072`,
  `draft_vocab_size=32000`.
* **Aux hidden state layers** captured from verifier: layers
  **2 / 26 / 48** (~4% / 50% / 92% depth -- Mamba / Attention / Mamba).
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
| **Max sequence length** | {max_length} tokens |
| **TTT length** | {ttt_length} (Eagle3 test-time-training unroll depth) |
| **Sliding window** | **none** (full causal attention) |
| **Optimizer** | BF16Optimizer, lr={lr}, warmup_ratio={warmup_ratio} |
| **Epochs** | {num_epochs} |
| **Per-rank batch size** | {batch_size} (no sequence packing) |
| **Tensor parallel** | TP=1, DP={dp_size} |
| **Global effective batch per step** | {global_batch} conversations |

### Note on batch size

This draft was trained with **per-rank `batch_size=1`** and **no
sequence packing**. With `dp_size={dp_size}` data-parallel ranks the
global effective batch is **{global_batch} conversations per
optimizer step**. Per-step token counts vary substantially with
conversation length (the training distribution ranges from ~30 to
~32k tokens at this experiment's max_length). See the
[`-sw4k` model card](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/MODEL_CARD_TEMPLATE_sw4k.md#note-on-batch-size)
for the full discussion of batch sizing in long-context Eagle3
training.
| **Wandb run** | {wandb_run_url} |

## ⚠️ CRITICAL: Mamba SSM state precision

This draft was trained with the **Mamba SSM scan boundary state in
float32**, matching vLLM 0.19's default behavior for NemotronH
(`--mamba_ssm_cache_dtype float32`).

The Nemotron team has explicitly confirmed
([HF discussion](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions/8))
that downcasting the Mamba SSM state to bf16 causes a **~10% absolute
regression on AIME-class math benchmarks** (88.3% vs 99.17% on AIME 2025
was the SGLang-vs-vLLM gap they measured).

**Both training and inference must use fp32 SSM state.** Concretely:

* **Training**: SpecForge fork includes a monkey-patch
  (`specforge/_mamba_fp32_patch.py`) that forces upstream `mamba_ssm`'s
  `_state_passing_fwd` to use `out_dtype=torch.float32`. Bit-equivalent
  to vLLM 0.19's NemotronH default behavior.
* **Inference (vLLM 0.19+)**: works automatically for NemotronH-Cascade-2
  because the verifier's `config.json` ships with
  `mamba_ssm_cache_dtype: "float32"`.
* **Inference (SGLang)**: pass `--mamba-ssm-dtype float32` (not on by
  default).

For the full audit story, see the
[HANDOFF.md "Mamba SSM precision -- the full intricacies" section](https://github.com/hav4ik/SpecForge-Nemotron3/blob/nemotron-cascade-2-experiments/experiments/nemotron-cascade-2/HANDOFF.md#mamba-ssm-precision----the-full-intricacies).

## Inference: serving with vLLM

```bash
vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \
    --speculative-config '{
        "model": "<path-to-this-checkpoint-or-hf-repo>",
        "method": "eagle3",
        "num_speculative_tokens": 3
    }' \
    --trust-remote-code \
    --tensor-parallel-size 2
```

## License and attribution

* **Verifier model** (`nvidia/Nemotron-Cascade-2-30B-A3B`): refer to
  the verifier's HF page for its license.
* **This draft head**: Apache 2.0 (matching the SpecForge license).
* **Training framework**: SpecForge fork at
  https://github.com/hav4ik/SpecForge-Nemotron3, branch
  `nemotron-cascade-2-experiments`.
