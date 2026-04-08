# Eagle3 Draft Head Training for Nemotron-Cascade-2-30B-A3B

This folder contains the experiment scripts and reproduction instructions for
training a long-context Eagle3 speculative-decoding draft head against
[`nvidia/Nemotron-Cascade-2-30B-A3B`](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B),
a 30B-param hybrid Mamba-Transformer MoE model. To our knowledge this is the
first Eagle3 head trained against a hybrid Mamba-Transformer verifier.

The branch this folder lives on (`nemotron-cascade-2-experiments`) carries
all of the SpecForge engineering changes that this experiment required.
See the **What's in the branch** section below for the change list.

## Why this is non-trivial

Eagle3 papers and the canonical SpecForge configs assume a pure transformer
verifier. Nemotron-Cascade-2 is hybrid:

* **52 layers total**: 31 Mamba-2, 15 MoE/MLP, 6 GQA attention layers
* **Hidden size**: 2688 (Eagle3 draft hidden_size locked to this value)
* **Vocab size**: 131072
* **Layer pattern** (`hybrid_override_pattern`):
  `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (M=Mamba, E=MoE, *=Attention)
* **Aux hidden state layers selected for the Eagle3 draft**: layers
  **2 / 26 / 48** (~4% / 50% / 92% depth, Mamba/Attention/Mamba), following
  NVIDIA's gpt-oss-120b long-context Eagle3 layer-spacing convention.

The verifier is loaded via HF custom remote code (`trust_remote_code=True`,
`auto_map.AutoModelForCausalLM = modeling_nemotron_h.NemotronHForCausalLM`).
SpecForge's HF target backend already supports custom-remote-code models in
principle, but it needed several patches to actually work for this case --
see **What's in the branch**.

## Files in this folder

```
experiments/nemotron-cascade-2/
├── README.md                  # this file
├── HANDOFF.md                 # current run state, wandb links, gotchas
├── prepare_data.sh            # download + merge + shuffle the training data
├── build_cache.sh             # offline cache builder wrapper (calls below)
├── build_cache_offline.py     # actual cache builder (no GPU needed)
├── run_train_baseline.sh      # baseline launcher (L=32k, ttt=2, no window)
└── run_train_sw4k.sh          # long-context launcher (L=65k, ttt=6, sw4k, all good stuff)
```

## How to reproduce from scratch

```bash
# 1. Install SpecForge from this branch (editable)
git clone -b nemotron-cascade-2-experiments \
    https://github.com/hav4ik/SpecForge-Nemotron3.git
cd SpecForge-Nemotron3
pip install -e . --no-deps  # the deps are flexible -- see Environment below

# 2. Pick a workspace dir
export WORK_DIR=/path/to/scratch/eagle3-work

# 3. Download + merge + shuffle the training data
bash experiments/nemotron-cascade-2/prepare_data.sh
# → produces $WORK_DIR/data/all_data_shuffled.jsonl

# 4. Build the offline tokenizer/loss-mask cache (no GPU needed)
bash experiments/nemotron-cascade-2/build_cache.sh
# Default EXPERIMENT=sw4k -> max_length=65536 cache.
# For the baseline experiment: EXPERIMENT=baseline ... -> max_length=32768.
# Takes ~5-7 min on a 32-core CPU. The cache is keyed by
# (data_path, max_length, chat_template, target_model) so the train script
# will find it automatically.

# 5. Launch training (uses 2 GPUs by default)
bash experiments/nemotron-cascade-2/run_train_sw4k.sh
# OR for the baseline:
bash experiments/nemotron-cascade-2/run_train_baseline.sh
```

All scripts honor the following env vars (defaults in parentheses):

| var | default | what |
|-----|---------|------|
| `WORK_DIR` | `$(pwd)/eagle3-work` | working dir for caches, checkpoints, logs |
| `HF_HOME` | `$WORK_DIR/hf_home` | HuggingFace cache for models + datasets |
| `TRAIN_DATA` | `$WORK_DIR/data/all_data_shuffled.jsonl` | training data file |
| `TARGET_MODEL` | `nvidia/Nemotron-Cascade-2-30B-A3B` | the verifier (BF16 build) |
| `NUM_GPUS` | `2` | nproc_per_node passed to torchrun |
| `NUM_PROC` | `32` | dataset preprocessing worker count |

## The two experiments

### `baseline`: minimal-config sanity training

Config: `L=32768, ttt=2, no sliding window, no grad checkpointing, no chunking`.

Fits comfortably on 96 GB GPUs without any of the long-context engineering
tricks. Useful as a loss-curve reference.

* Script: `run_train_baseline.sh`
* Draft config: `configs/nemotron-cascade-2-eagle3.json`
* Wandb run name: `baseline-L32k-ttt2-mixed`

### `sw4k`: long-context with sliding-window attention + everything

Config: `L=65536, ttt=6, sliding_window=4096, draft-mlp-grad-checkpoint,
draft-mlp-chunk-size=4096, fused-linear-loss, fused-linear-loss-chunk-size=4096`.

Trains the draft on near-full-length reasoning traces (96.5% preserved) at
ttt=6 (~96% of ttt=7's theoretical speedup ceiling). All of the engineering
optimizations on this branch are required to make this fit on a 96 GB GPU.

* Script: `run_train_sw4k.sh`
* Draft config: `configs/nemotron-cascade-2-eagle3-sw4k.json`
* Wandb run name: `sw4k-L65k-ttt6-fused-mixed`

## Environment

What we know works:

* Python 3.12
* PyTorch 2.10 + CUDA 13.0 (we used the system venv at `/venv/main` on the
  training box; nightly torch from PyTorch's index)
* `transformers==4.57.6`
* `mamba_ssm` + `causal-conv1d` (built from source against torch 2.10) --
  needed for the verifier's Mamba layers' fast path
* `triton==3.6.0`
* `flashinfer` (for vLLM's NemotronH support; not strictly required for
  training)
* `datasets`, `accelerate`, `wandb`, `tensorboard`, `yunchang` (USP)
* No `flash_attn` package (we use the flex_attention backend on the draft)
* No `sglang` (we use the HF target backend, not the sglang backend)

The branch deliberately makes `sglang` and `yunchang` optional imports so
the codebase loads cleanly without them.

## What's in the branch

The `nemotron-cascade-2-experiments` branch carries 11 commits on top of
SpecForge `main`. Brief tour:

1. **`Make sglang and yunchang optional dependencies`** -- lazy imports so
   `specforge.args` and `specforge.distributed` load without sglang/yunchang.
2. **`HF backend: lazy sglang imports + backbone.layers discovery`** --
   makes `eagle3_target_model.py` import without sglang and teaches
   `HFEagle3TargetModel._get_transformer_layers()` to find layers under
   `model.backbone.layers` (the NemotronH layout).
3. **`100x faster loss-mask parser + tool-use schema fix`** -- replaces
   `GeneralParser.parse()`'s O(num_turns × seq_len) re-tokenization with a
   single-pass `return_offsets_mapping` + bisect approach. Without this
   change, dataset preprocessing for 30k long reasoning traces took 25+
   minutes per worker. With the patch, it's ~30 ms per conversation. Also
   fixes a pyarrow schema-cast crash on tool-use messages.
4. **`Free verifier full-vocab logits before draft TTT unrolling`** -- the
   single biggest training-memory win. The HF target backend returns the
   verifier's `[seq, full_vocab=131072]` logits as `target` (~16 GiB at
   L=65k). SpecForge held three independent references to it across the
   draft forward+backward pass, OOMing every long-context run. The fix
   precomputes the small `target_p_padded` (~4 GiB on draft vocab) inside
   `run_forward()` and frees the dataclass + local references BEFORE
   entering `eagle3_model.forward()`, plus drops the `.float()` upcast on
   the softmax that doubled the per-step `target_p` footprint.
5. **`Sliding-window attention + optional MLP grad checkpointing`** --
   adds `sliding_window` plumbing to `LlamaForCausalLMEagle3`
   (`_make_causal_mask`, `prepare_decoder_attention_mask`, `LlamaModel.forward`,
   `LlamaFlexAttention.forward`, and the `generate_eagle3_mask` flex_attention
   mask generator). The TTT suffix-mask block (single per-position K/V
   additions per unroll step) is intentionally **not** windowed. Also adds
   the optional `mlp._grad_checkpoint` toggle.
6. **`Add nemotron-h chat template and Nemotron-Cascade-2 draft configs`** --
   registers the `nemotron-h` chat template in
   `specforge/data/template.py` (same ChatML markers as Qwen) and adds two
   draft configs under `configs/`.
7. **`Add --draft-mlp-grad-checkpoint CLI flag`** -- wires the toggle.
8. **`Extend grad checkpointing to compute_logits`** -- companion to the
   MLP checkpoint, drops the per-step `[seq, draft_vocab]` logits from
   the saved-for-backward set across TTT unrolls.
9. **`Chunked fused linear + soft-target CE for Eagle3 lm_head`** -- the
   biggest single optimization for long context. Adds
   `fused_linear_log_softmax_loss()` in `specforge/core/loss.py` that
   re-uses the existing `LogSoftmaxLoss` triton kernel inside a chunked +
   grad-checkpointed wrapper, so the draft's `[B, T, V]` logits tensor is
   never materialized. Mathematically equivalent to the unchunked path
   (verified numerically in the file's `__main__` block at realistic
   shapes; both loss values and gradients on `hidden_states` AND
   `lm_head_weight` match at `rtol=1e-4 atol=1e-5`).
10. **`Chunked MLP forward to lower per-step transient peak`** -- the
    Liger-style sequence-chunked MLP. The MLP is pointwise along the seq
    dim, so we split into chunks and never materialize the full
    `[B, T, intermediate]` tensors. Per-step transient peak drops from
    ~4 GiB to ~800 MiB at L=65k. Composes cleanly with the outer
    `--draft-mlp-grad-checkpoint` (no nested checkpointing).
11. **`In-place chunked padding for huge logit tensors`** + a few smaller
    fixes (see `git log --oneline main..HEAD` on the branch).

## Why ttt=6 instead of ttt=7

ttt=7 is the canonical Eagle3 paper default and gives a theoretical ~3x
speedup ceiling. We use ttt=6 in the long-context experiment because the
worst-case 65k-token rank still OOMs at ttt=7 by ~64 MB even with all of
the engineering tricks above. ttt=6 yields ~96% of ttt=7's speedup (~2.9x
vs ~3.0x) for a 1-step reduction in saved attention activations. To push
back to ttt=7 we would need *also* attention checkpointing (handling
`past_key_values` cache mutation correctly) or FSDP2 verifier sharding
(saves ~31 GB per GPU by holding only half the verifier weights on each
rank).

## Known issues / future work

* **`acc1 > acc0` in early training** -- the SpecForge TTT loop's
  `padding(loss_mask, left=False)` shifts loss masks left between unrolls,
  zeroing the rightmost position each step. Reasoning traces tend to have
  harder-to-predict tail tokens, so dropping them makes the deeper steps
  look "easier" than step 0. Worth investigating if this persists past
  step 5-10k.
* **No sequence packing** -- with `batch_size=1` and per-rank shuffled
  sampling, step time variance is huge (0.6 s to 13 s) because some
  conversations are 30 tokens and others are 65k. A packed-batch loader
  would smooth this out and recover ~2-3x throughput.
* **No FSDP2** -- SpecForge uses FSDP1 for the draft (which is irrelevant
  for memory because the draft is ~196M trainable params). The big win
  would be FSDP2 sharding the *verifier* across both ranks (~31 GB per
  GPU saved). Wraps to `fully_shard()` of the loaded NemotronH module --
  estimated ~30-60 LOC in `HFEagle3TargetModel.from_pretrained`.
* **No FlashAttention on Blackwell** -- we use `flex_attention` on the
  draft because `flash_attn` doesn't have a Blackwell wheel for our torch
  version yet. flex_attention is fine but warns "called without
  torch.compile() - this will use an unfused implementation". Compiling
  would save another ~3-4 GB of attention scores memory.
* **Chunked attention not implemented** -- the draft attention path is
  not chunked or grad-checkpointed. ~2 GB per TTT step worth of attention
  activations still accumulate. Would unblock ttt=7 at L=65k.

## Scaling to more GPUs

**Yes -- this recipe scales linearly via pure data parallelism**, with no
code changes needed, up to roughly 8-16 GPUs. After that you start getting
diminished returns from convergence (effective batch grows with rank count).

### What scales

The current setup is `tp_size=1` + `dp_size=N` (one full verifier copy per
rank, draft model wrapped in FSDP1 across the DP group). Bumping `NUM_GPUS`
from 2 to 4/8/16 just changes `dp_size`:

* **Each rank still loads a full 63 GB verifier copy.** No memory savings
  per rank from going wide -- but no extra cost either.
* **Per-step compute is fixed per rank** (verifier forward + draft
  forward+backward on its own 1-conversation shard).
* **Draft gradient all-reduce** is on ~196M params (~400 MiB in bf16) each
  step → ~10-20 ms over NVLink. Negligible vs the 1-5 sec per-step
  compute.
* **Effective batch size = N**. Each rank processes `batch_size=1`
  conversations, so the global step sees N conversations.
* **Wall-clock per epoch divides by N**:

  | NUM_GPUS | dp_size | hr/epoch (L=65k, ttt=6) | steps/epoch |
  |---------:|--------:|------------------------:|------------:|
  | 2  | 2  | ~19    | 15283 |
  | 4  | 4  | ~9.5   | 7641  |
  | 8  | 8  | ~4.75  | 3820  |
  | 16 | 16 | ~2.4   | 1910  |

### What doesn't change

* **Per-GPU memory ceiling**: each rank still needs 96 GB to hold the full
  verifier + activations for a worst-case 65k-token conversation. The
  recipe **does not work on smaller GPUs** without further engineering.
* **No tensor parallelism**: SpecForge supports `--tp-size>1` which routes
  through HF's `tp_plan="auto"` mechanism, but NemotronH's custom remote
  code doesn't ship a `tp_plan`, so the verifier can't be sharded with
  this path. Adding a `tp_plan` to the modeling file would be the cleanest
  enabler for `tp_size>1`.

### Things to watch when scaling up

1. **Effective batch size growth**. At N=8 the effective batch is 8
   conversations per step, which is fine -- the Eagle3 paper trains with
   bs=16-32. At N=16+ you should consider raising `--learning-rate`
   (linear scaling rule applies fairly well to this loss).

2. **Steps per epoch shrinks**. At 30566 conversations / N=16 = 1910
   steps/epoch. Five epochs = ~9550 steps total. That's fine for a draft
   head, but the LR warmup ratio (`--warmup-ratio 0.02` ≈ 191 warmup
   steps) becomes more significant relative to total steps -- consider
   bumping warmup to 0.05 at very high N.

3. **Length-bucketed sampling becomes important**. With `batch_size=1` and
   shuffled DP sampling, individual ranks can pull pathologically-long
   conversations; per-step time has 20× variance. At higher N this
   becomes the bottleneck because the slowest rank determines the global
   step time. A length-bucketed `DistributedSampler` would smooth this
   out -- not implemented yet on this branch.

4. **Worst-case conversation OOM probability scales with N**. With more
   ranks drawing from the long-tail distribution each epoch, the chance
   that *some* rank pulls the deepest 65k-token conversation in the same
   global step approaches 1. The current recipe fits the worst case on
   96 GB by exactly the right margin -- but if you observe sporadic OOMs
   at N≥8, drop `ttt_length` from 6 to 5 (~96 → ~91% of speedup ceiling)
   to recover ~2.5 GB headroom per step.

### What FSDP2 verifier sharding would unlock

If you want to run on **smaller GPUs** (e.g., 8× 48 GB or 16× 24 GB), or
push **L=131072+** on 96 GB GPUs, the next engineering work is wrapping
the loaded verifier with FSDP2's `fully_shard()` so each rank only holds
1/N of the verifier weights:

```
Per-GPU verifier memory:    63 GiB → 63/N GiB
Per-GPU peak (L=131k, N=4):  ~95 GiB → ~50 GiB → fits on 48 GB
Per-GPU peak (L=131k, N=8):  ~95 GiB → ~30 GiB → fits on 48 GB
```

Cost: ~5-10% per-step overhead from gather-on-forward communication, plus
~30-60 LOC of patching in `HFEagle3TargetModel.from_pretrained`. The
verifier is frozen so there's no gradient/optimizer-state sharding to
worry about. Tracked in **Known issues / future work** above.

## Memory math reference

At `L=65536, ttt=6, intermediate=8064, draft_vocab=32k, bf16` per rank:

```
Verifier weights (frozen):                ~63.0 GiB
target_p_padded (chunked through fused):   ~4.2 GiB
Aux hidden states [3, L, 2688]:            ~1.0 GiB
Per-step saved-for-backward attention:    ~2.0 GiB × 6 = 12.0 GiB
Per-step saved MLP I/O (grad-ckpt):       ~0.7 GiB × 6 =  4.2 GiB
Current step transient (chunked MLP):     ~0.8 GiB
norm_hidden per step (alive transient):   ~0.4 GiB
Misc / fragmentation overhead:            ~5-8 GiB
                                           ─────────
Total:                                    ~88-91 GiB ✓ fits on 96 GB
```

For `ttt=7` the per-step counts go to ×7 instead of ×6, adding +2.7 GiB,
which historically OOMed by 64 MiB on the worst-case 65k conversation.
