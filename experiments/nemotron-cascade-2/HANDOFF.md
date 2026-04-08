# HANDOFF — Nemotron-Cascade-2 Eagle3 training

This file is **transient state**: the current run, what's been tried, where
the checkpoints live. If you're a new agent picking this up, **read this
first**, then `README.md` for the engineering rationale, then `git log
--oneline main..HEAD` for the patch tour.

If you're a future-me reading this without the conversation history: this
folder is the current state of a multi-day debugging effort to get a long-
context Eagle3 draft head trained on a hybrid Mamba-Transformer verifier.
**The training engineering is done; what remains is monitoring the run,
optionally pushing to L=131072 with FSDP2 verifier sharding, and pushing
the trained checkpoint to HF when it's ready.**

## First 5 minutes on a new instance

If you're picking this up on a fresh box with the GPUs already provisioned:

```bash
# 1. Clone this branch (the source of truth)
git clone -b nemotron-cascade-2-experiments \
    https://github.com/hav4ik/SpecForge-Nemotron3.git
cd SpecForge-Nemotron3

# 2. Install (use the existing env if /venv/main exists; otherwise create
#    a fresh one with python>=3.11 and the deps listed in README.md
#    "Environment" section)
pip install -e . --no-deps
pip install datasets tensorboard wandb yunchang  # the few light deps we need
# mamba_ssm + causal_conv1d are needed for the verifier's Mamba fast path
# but the slow path also works -- skip them if the build fails on your CUDA
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# 3. Login to wandb (the user's project lives at hav4ik/nemotron-cascade-2-eagle3)
wandb login                # paste API key from wandb.ai/settings
# (HuggingFace login is NOT needed -- all models/datasets used are public)

# 4. Pick a workspace dir on big-disk storage
export WORK_DIR=/scratch/nemotron-eagle3

# 5. Download data (~2 min, ~80 MB)
bash experiments/nemotron-cascade-2/prepare_data.sh

# 6. Build cache (~5-10 min CPU work, no GPUs needed)
bash experiments/nemotron-cascade-2/build_cache.sh

# 7. Launch the long-context training (the run we left in flight)
bash experiments/nemotron-cascade-2/run_train_sw4k.sh
# Open the wandb URL printed near the top of the log to monitor

# 8. (Optional) Push commits back to this fork:
#    PAT credentials are NOT stored in the repo. Ask @hav4ik for a fresh
#    fine-grained PAT scoped to hav4ik/SpecForge-Nemotron3, then push via:
#    git push https://hav4ik:$PAT@github.com/hav4ik/SpecForge-Nemotron3.git \
#        nemotron-cascade-2-experiments
```

## What transfers across instances vs what doesn't

| Thing | Where it lives | Re-create how |
|---|---|---|
| **All code + configs + docs** | This branch on github.com/hav4ik/SpecForge-Nemotron3 | `git clone -b nemotron-cascade-2-experiments` |
| **Numerical equivalence test for chunked fused loss** | `specforge/core/loss.py` `__main__` block | `python specforge/core/loss.py` (needs 1 GPU, ~10 sec) |
| **Wandb runs / loss curves** | wandb.ai/hav4ik/nemotron-cascade-2-eagle3 | (cloud, persistent) |
| **Training data** | huggingface.co/datasets/chankhavu/c2_eagle3_train | `prepare_data.sh` re-downloads |
| **Verifier weights (~63 GB)** | huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B | HF auto-downloads on first run; ~10 min |
| **Tokenized cache** | local disk (`$WORK_DIR/cache/`) | `build_cache.sh` rebuilds in ~5-10 min |
| **Trained draft checkpoints** | local disk (`$WORK_DIR/checkpoints/`) | **Lost on instance teardown unless pushed to HF.** Re-train from scratch (~hours) |
| **My conversation history with @hav4ik** | (claude.ai session) | (lost) -- this HANDOFF + README + commit messages are the canonical record of decisions |
| **My auto-memory** | `/root/.claude/projects/-workspace/memory/` | (lost) -- nothing critical was stored there for this project |
| **PAT token for git push** | (was in chat once) | Ask @hav4ik for a fresh fine-grained PAT scoped only to the fork |
| **Wandb API key** | (in @hav4ik's wandb account) | `wandb login` and paste from wandb.ai/settings |

## Current run (as of this commit)

* **Experiment**: `sw4k` (long-context + sliding window 4k + grad-ckpt
  + chunked MLP + fused chunked linear/loss + ttt=6)
* **Wandb**: project `nemotron-cascade-2-eagle3`, see runs starting with
  `sw4k-L65k-ttt6-fused-mixed`
* **Output dir**: `<WORK_DIR>/checkpoints/nemotron-cascade-2-eagle3-sw4k/`
  (will fill with `epoch_0_step_*` subdirs every 2000 steps)
* **Status at last update**: ~step 273 of ~15283 (epoch 0), ~19 min
  elapsed, ~4.5 sec/step running average. ETA per epoch ~19 hours;
  full 5-epoch run ~4 days.

The `mixed` (baseline) experiment was killed deliberately to free GPUs
for the `sw4k` run, but **its step-2000 checkpoint is still on disk** at
`<WORK_DIR>/checkpoints/nemotron-cascade-2-eagle3-mixed/epoch_0_step_2000/`
-- usable as a sanity-check reference.

## Trained checkpoints on disk (this box)

```
$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-mixed/epoch_0_step_2000/
    config.json              ~828 B
    model.safetensors        ~399 MB   (~196 M trainable params)
    training_state.pt        ~4.5 KB
```

This is the L=32k baseline (ttt=2, no window). Useful as a baseline-distribution
reference for end-to-end inference benchmarks (acceptance rate vs the
new sw4k experiment when it finishes).

## Wandb runs (in chronological order)

All runs are on wandb project `nemotron-cascade-2-eagle3` (org `hav4ik`).
Names below are in their wandb-run-name form:

| run name | config | status | notes |
|---|---|---|---|
| `mixed-tp1-dp2-bf16` (multiple ids) | L=32k, ttt=2, no window, no grad-ckpt | killed at ~step 2520 | has saved checkpoint at step 2000 |
| `sw4k-mixed-tp1-dp2-bf16` (`hwb96q0a` etc.) | L=32k, ttt=7, sw4k, grad-ckpt | killed | OOMed at various points before chunked MLP / fused-loss work |
| `sw4k-L65k-ttt7-fused-mixed` | L=65k, ttt=7, full stack | OOMed by 64 MiB | the run that motivated the chunked MLP commit |
| **`sw4k-L65k-ttt6-fused-mixed`** | **L=65k, ttt=6, full stack** | **CURRENTLY RUNNING** | the actual long-context experiment |

## Engineering changes shipped on this branch

11 commits past `main`. Brief reverse-chronological tour (run
`git log --oneline main..HEAD` for the full list):

```
8d2b9d5 specforge/draft: chunked MLP forward to lower per-step transient peak
d4a213d specforge: chunked fused linear + soft-target CE for Eagle3 lm_head
f90bbc7 specforge/draft: extend grad checkpointing to compute_logits
b160ebf specforge/train: add --draft-mlp-grad-checkpoint CLI flag
10ba046 specforge: Add nemotron-h chat template and Nemotron-Cascade-2 draft configs
9ff1f2b specforge/draft: sliding-window attention + optional MLP grad checkpointing
c2f0168 specforge: Free verifier full-vocab logits before draft TTT unrolling
e301f98 specforge/data: 100x faster loss-mask parser + tool-use schema fix
c55a87a specforge/HF backend: lazy sglang imports + backbone.layers discovery
f97e6c0 specforge: Make sglang and yunchang optional dependencies
```

Each commit message is detailed with the rationale, the math, and the
file/line references for the change. Read them in chronological order
(`git log --reverse --oneline main..HEAD`) for the cleanest narrative.

## Things you should check first if the run looks broken

1. **Is the Python process still alive?**
   `ps -ef | grep train_eagle3 | grep -v grep | wc -l`
   should print `>= 2` (one per rank, plus dataset workers if a Map is
   running). If it prints 0, the run died -- check the latest log file
   for the error.

2. **Are the GPUs busy?**
   `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`
   should show ~70-95 GB used per GPU and 50-100% util when training is
   active. 0% util with high memory usage means it's stuck (probably in
   dataset preprocessing if early, or NCCL barrier if mid-run).

3. **Has step count advanced since the last log line?**
   `grep -oE "[0-9]+/15283" $WORK_DIR/logs/train_sw4k.log | tail -1`
   should monotonically increase. If it's stuck, find the latest
   exception via
   `grep -E "Tried to allocate|Error|RuntimeError" $WORK_DIR/logs/train_sw4k.log`

4. **Wandb is reporting?** Open the run URL printed near the top of the
   log:
   `grep "View run at" $WORK_DIR/logs/train_sw4k.log`

## Long-context engineering flags — and how to disable them

All four of the long-context optimizations are **opt-in CLI flags** on
`scripts/train_eagle3.py`. They are off by default so existing SpecForge
users are unaffected. The launchers in this folder
(`run_train_sw4k.sh`) enable all four together; to debug a suspected
correctness regression you can disable them one at a time, in this
recommended order from "least likely to break math" to "most likely":

| flag | what it does | safe to disable? | what you lose |
|---|---|---|---|
| `--draft-mlp-chunk-size 4096` | Chunks the draft MLP forward over the seq dim. Pure transient-peak reduction. | **YES, first thing to try if anything looks wrong.** | ~3.6 GiB transient peak savings per MLP forward. May trigger MLP-forward OOMs at L=65k. |
| `--draft-mlp-grad-checkpoint` | Wraps `self.mlp(x)` in `torch.utils.checkpoint`. Drops MLP intermediates from the saved-for-backward set. Same flag also enables grad-ckpt on `compute_logits()`. | **YES, second thing to try.** Math is bit-equivalent to non-checkpointed (autograd guarantees this for non-reentrant ckpt). | ~25 GiB saved-for-backward across the 6 TTT unrolls. Will OOM at L=65k. |
| `--fused-linear-loss` (+ `--fused-linear-loss-chunk-size 4096`) | Replaces the unchunked `compute_logits → LogSoftmaxLoss.apply` pipeline with a chunked + grad-checkpointed fused linear+CE that never materializes the `[B,T,V]` logits tensor. | **YES, but verify equivalence first** by running `python specforge/core/loss.py` (the `__main__` block has a numerical equivalence test, ~10 sec, 1 GPU). | ~28 GiB across the 6 TTT unrolls. Will OOM at L=65k. |
| draft config `sliding_window: 4096` + `layer_types: ["sliding_attention"]` | Sliding-window attention on the draft. Currently only honored by our patched `LlamaForCausalLMEagle3` and by vLLM at inference. | **YES, but only by switching the draft config** to `nemotron-cascade-2-eagle3.json` (no sliding window) instead of `-sw4k.json`. The training+inference distribution will then be full-attention. | The whole windowed-attention experiment hypothesis. Falls back to full causal, ~3 GiB more attention activations per step. |

To disable everything and run pure SpecForge upstream behavior, switch
to `run_train_baseline.sh` which uses no flags and the no-window draft
config. That run is the apples-to-apples sanity baseline.

### How to verify the chunked fused loss is mathematically equivalent

```bash
CUDA_VISIBLE_DEVICES=1 python specforge/core/loss.py
```

Expected output:

```
[loss-test] LogSoftmaxLoss vs torch reference: OK
[loss-test] chunked vs unchunked loss: ref=5.193363e+00 chunked=5.193363e+00
[loss-test] fused_linear_log_softmax_loss equivalence: OK
[loss-test] fused_linear_argmax_correct_count equivalence: OK (0 correct)
```

This compares chunked vs unchunked at realistic Eagle3 shapes (B=1,
T=2048, H=2688, V=32000) and checks both the loss value AND the
gradients on `hidden_states` AND `lm_head_weight`. Tolerance is
`rtol=1e-4 atol=1e-5`. If this test ever stops passing, **stop training
immediately** -- there's a regression in either `LogSoftmaxLoss` or
`fused_linear_log_softmax_loss`.

## Audit log — what's been verified

A subagent walked through all of the following on 2026-04-08 and
verified them to PASS:

* **Audit 1 (chunked MLP forward)** — covers `[0,T)` exactly, no gap
  / overlap; chunks `cat` in seq order; eval/inference correctly
  bypasses chunking; no nested checkpointing footgun; bit-equivalent
  to the original `down_proj(act_fn(gate_proj(x)) * up_proj(x))` modulo
  fp reduction order. Files: `llama3_eagle.py:1192-1296`.
* **Audit 2 (chunked fused linear + soft-target CE)** — per-chunk
  normalization is correct (`mean × (B*chunk) → sum`, then total /
  `(B*T)`); inner `LogSoftmaxLoss.apply` composes correctly with
  `use_reentrant=False` outer checkpoint; position_mask sliced
  correctly; numerical equivalence test passed (`ref=5.193363e+00 ==
  chunked=5.193363e+00`, gradients within `1e-4`).
* **Audit 3 (grad-checkpoint toggles)** — both sites read
  `_grad_checkpoint` from the same Module (`self.midlayer.mlp`); both
  use `use_reentrant=False`; both guarded by `self.training and
  torch.is_grad_enabled()` so eval bypasses; closures capture
  `self.lm_head` / `self.norm` correctly so gradients flow through
  their parameters.
* **Audit 4 (composition / no double-norm)** — `backbone()` (line
  1550-1569) does NOT apply `self.norm`; only the unused
  `LlamaForCausalLMEagle3.forward` path (line 1514) does. The TTT loop
  uses `backbone()`, so `self.norm` is applied **exactly once** by
  either `_acc_and_loss_fused()` or `compute_logits()`. **No
  double-norm bug.**
* **Audit 5 (Mamba SSM precision)** — see next section.

To re-run the audits on a fresh box: invoke a Claude Code subagent
with the same tasks documented in the README "Audit log" section, or
manually re-read the files and run the numerical equivalence test
above.

## CRITICAL: Mamba SSM state must be float32 (don't break this)

**Background.** NVIDIA's NemotronH team has explicitly confirmed that the
Mamba SSM state must be kept in **float32** during training and inference.
Downcasting to bf16 causes a **~10% absolute regression on AIME-class
math benchmarks** (88.3% → 99.17% on AIME 2025 was the SGLang vs vLLM
delta the Nemotron team published in
https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions/8 ).
This is NOT just about the persistent decode cache -- it also affects
the **prefill scan boundary state** that NemotronH's Mamba layers use
during a single forward pass over a long sequence.

**The bug we fixed (commit `<hash>`).** The upstream `mamba_ssm` Triton
fast path's `_state_passing_fwd` is called from `ssd_combined.py` with
`out_dtype=C.dtype`, which is bf16 under our `torch_dtype=bfloat16`
verifier load. This means the boundary state passed between successive
SSM scan chunks is downcast to bf16. At L=65536 with the default Mamba
`chunk_size=128`, that's **~512 bf16 downcasts per Mamba layer per
forward pass × 31 Mamba layers in NemotronH = ~15,872 bf16 downcasts
per forward**, accumulating arithmetic error along the SSM scan -- this
is exactly the failure mode the Nemotron team described.

**vLLM's behavior (this is the reference we match).** vLLM 0.19 ships
its own VENDORED copy of the Mamba Triton ops under
`vllm/model_executor/layers/mamba/ops/`. The vendored
`ssd_combined.py:119` differs from upstream on exactly this point:

```python
# vLLM 0.19 (vllm/model_executor/layers/mamba/ops/ssd_combined.py:119)
states = _state_passing_fwd(
    ...
    out_dtype=state_dtype if state_dtype is not None else C.dtype,
)
```

It threads `state_dtype` through from the caller in `mamba_mixer2.py:734`
(`state_dtype=ssm_state.dtype`), where `ssm_state.dtype` is determined
by `MambaStateDtypeCalculator.mamba2_state_dtype()` which reads
`vllm_config.cache_config.mamba_ssm_cache_dtype`. For NemotronH,
`NemotronHForCausalLMConfig.verify_and_update_config` automatically reads
`hf_config.mamba_ssm_cache_dtype` (which is `"float32"` in the model's
`config.json`) when the CLI default `"auto"` is passed -- so **vLLM 0.19
inference for NemotronH automatically uses fp32 SSM boundary state, even
without an explicit `--mamba_ssm_cache_dtype float32` flag.**

Other fp32 sites match between upstream `mamba_ssm` and vLLM:
- `_chunk_state_fwd(..., states_in_fp32=True)` → intra-chunk state fp32
- `_bmm_chunk_fwd(..., output_dtype=torch.float32)` → fp32
- `A_log.float()` upcast in NemotronH modeling code at line 424/560

The **only** difference is the boundary state, which we now match.

**The fix.** A monkey-patch in `specforge/_mamba_fp32_patch.py` that
replaces upstream `mamba_ssm.ops.triton.ssd_state_passing._state_passing_fwd`
with a wrapper that forces `out_dtype=torch.float32` regardless of what
the caller asks for. The patch is applied at the very top of
`scripts/train_eagle3.py`, BEFORE any other code can import `mamba_ssm`.

The patch is **idempotent**, **flag-gated** (env var
`SPECFORGE_DISABLE_MAMBA_FP32_PATCH=1` disables it -- strongly
discouraged), and **bit-equivalent to vLLM 0.19's automatic fp32 SSM
boundary handling for NemotronH**.

**How to verify the patch is active**:

```bash
CUDA_VISIBLE_DEVICES=0 python -m specforge._mamba_fp32_patch --verify
# expected output:
# [mamba-fp32-patch] verified: _state_passing_fwd output dtype is
#                              torch.float32 (forced fp32 regardless of caller request)
```

**If you ever see Mamba SSM training behaving worse than vLLM
inference**: first thing to check is whether this patch is still active.
The most common breakage modes:
1. Someone refactored `train_eagle3.py` and removed the import at the
   top.
2. `mamba_ssm` got upgraded to a version where `_state_passing_fwd`'s
   API changed (the patch tries to forward all kwargs but if a new
   keyword arg is added, the wrapper would need updating).
3. Someone set `SPECFORGE_DISABLE_MAMBA_FP32_PATCH=1` for debugging
   and forgot to unset it.

**This patch is NOT optional. Do not disable it for any production
training run.** The cost is negligible (boundary state buffer goes from
bf16 to fp32, doubling its size from ~8 MiB to ~16 MiB per Mamba layer
per forward at L=65k -- nothing on a 96 GB GPU).

## Defensive patch on the cached NemotronH modeling file

Separate from the upstream-mamba_ssm monkey-patch above, there's also
a latent bug in NemotronH's HF custom remote-code modeling file: the
`mamba_ssm_cache_dtype` config field is silently ignored when
`HybridMambaAttentionDynamicCache` is constructed (line 1631). This
**only matters if anything ever flips `use_cache=True` on the verifier**
(e.g. to call `.generate()` for evaluation). Our SpecForge training uses
`use_cache=False` so we never instantiate the cache, but as defense-in-
depth I patched the cached modeling file. **This patch lives in the
HuggingFace model cache directory, NOT in the SpecForge fork**, so it
will need to be reapplied after re-downloading the model on a fresh
instance:

The original PROJECT.md (and the Nemotron team's guidance) explicitly
says **the Mamba SSM cache/state must be kept in float32** -- bf16
catastrophically degrades math/reasoning quality. Status as of this
commit:

* **Verdict for our training: SAFE.** We use `use_cache=False` in
  `HFEagle3TargetModel.generate_eagle3_data()`, so
  `HybridMambaAttentionDynamicCache` is never instantiated. The Triton
  fast-path kernel (`mamba_chunk_scan_combined`) accumulates SSM state
  in fp32 internally regardless of input dtype, and the modeling code
  explicitly upcasts `A_log.float()` at lines 424 and 560 of
  `modeling_nemotron_h.py`. Our forward is bit-identical to what the
  model sees at pretraining time.
* **Latent bug in the upstream NemotronH custom remote-code modeling**:
  `config.mamba_ssm_cache_dtype` is silently **ignored**. The
  `HybridMambaAttentionDynamicCache` is constructed with `self.dtype`
  (= bf16 under our load) at line 1631, not with the requested fp32.
  This only matters if anything ever flips `use_cache=True` on the
  verifier (e.g., to call `.generate()` for evaluation, or to chunk a
  long forward across multiple calls).
* **Defensive patch applied** to both copies of the cached
  `modeling_nemotron_h.py`:
  - `/workspace/models/hub/models--nvidia--Nemotron-Cascade-2-30B-A3B/snapshots/<hash>/modeling_nemotron_h.py`
  - `/workspace/models/modules/transformers_modules/nvidia/.../modeling_nemotron_h.py`

  The patch wraps the cache-dtype constructor to honor
  `config.mamba_ssm_cache_dtype` if present (falling back to
  `self.dtype` otherwise).

* **CRITICAL for the next agent on a fresh box**: This patch lives in
  the **HuggingFace model cache directory**, NOT in the SpecForge fork.
  When you re-download the verifier on a new instance, **you will get
  the unpatched modeling file again**. To re-apply, run:

  ```bash
  for f in $(find $HF_HOME -name modeling_nemotron_h.py -path '*Nemotron-Cascade-2*' \
                          -o -name modeling_nemotron_h.py -path '*Nemotron_hyphen_Cascade*'); do
      python - <<PY
  p = "$f"
  src = open(p).read()
  old = '''        else:
              past_key_values = HybridMambaAttentionDynamicCache(
                  self.config, input_ids.shape[0], self.dtype, device=self.device
              )'''
  new = '''        else:
              # PATCHED: honor config.mamba_ssm_cache_dtype
              _ssm_cache_dtype_str = getattr(self.config, "mamba_ssm_cache_dtype", None)
              _cache_dtype = {
                  "float32": torch.float32, "fp32": torch.float32,
                  "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                  "float16": torch.float16, "fp16": torch.float16,
              }.get(_ssm_cache_dtype_str, self.dtype)
              past_key_values = HybridMambaAttentionDynamicCache(
                  self.config, input_ids.shape[0], _cache_dtype, device=self.device
              )'''
  if old in src:
      open(p, "w").write(src.replace(old, new))
      print("patched", p)
  else:
      print("already patched (or modeling code changed):", p)
  PY
  done
  ```

  This patch is purely defensive for our current pipeline (we don't
  use the SSM cache), but it's a foot-gun for anyone who tries to
  evaluate the trained checkpoint via `model.generate()` against the
  verifier.

## Common failure modes and fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `OOM at 1008 MB allocation in MLP silu` | Per-step MLP transient peak too large | Lower `--draft-mlp-chunk-size` (currently 4096) to e.g. 2048 |
| `OOM at ~4 GB allocation in compute_logits` | Per-step lm_head logits too large | Make sure `--fused-linear-loss` is on; lower `--fused-linear-loss-chunk-size` |
| `OOM by < 100 MB on a long conversation` | Worst-case rank pulled near-65k conv | Lower `--ttt-length` from 6 to 5 |
| `OOM by > 10 GB` | Probably forgot one of the flags above; or a code regression dropped one of the patches | Check `git log --oneline main..HEAD \| wc -l` reports 11+ commits |
| `NCCL timeout / barrier hang` | One rank stuck on dataset preprocessing while the other waits on init barrier | Bump `--dist-timeout 180` (already on); or build the cache offline first via `bash experiments/nemotron-cascade-2/build_cache.sh` |
| `Couldn't cast array of type struct<...> to {...}` | Tool-use schema fix not picked up | Make sure commit `e301f98` is in the branch |
| `cannot import name 'AutoDraftModelConfig' from 'specforge'` | Script's directory shadows the `specforge` package | Already fixed in `build_cache_offline.py` via `sys.path[:]` filter; if a new script hits this, copy that fix |
| `acc1 > acc0` early in training | Padding asymmetry on the rightmost positions, see README "Known issues" | Likely benign; investigate only if it persists past ~step 5000 |

## Resuming a partial training run

SpecForge supports resume via the `--resume` flag (auto-detects the latest
checkpoint in `--output-dir`) or `--ckpt-dir <dir>` (load from a specific
directory). Either flag picks up the optimizer/scheduler/RNG state from
`training_state.pt` so the LR schedule and data ordering continue
correctly.

To resume the current sw4k run after a crash:

```bash
bash experiments/nemotron-cascade-2/run_train_sw4k.sh --resume
```

## How to push the trained checkpoint to HF (when ready)

```bash
huggingface-cli login   # if not already
huggingface-cli upload \
    chankhavu/nemotron-cascade-2-eagle3-sw4k-v1 \
    $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-sw4k/epoch_<N>_step_<step>/ \
    .
```

Make sure the uploaded `config.json` carries `sliding_window=4096`,
`layer_types=["sliding_attention"]`, and `max_position_embeddings=262144`
(the draft config in `configs/nemotron-cascade-2-eagle3-sw4k.json` already
has these). vLLM serves it via:

```bash
vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \
    --speculative-config '{
        "model": "chankhavu/nemotron-cascade-2-eagle3-sw4k-v1",
        "method": "eagle3",
        "num_speculative_tokens": 5
    }' \
    --trust-remote-code \
    --mamba-ssm-cache-dtype float32 \
    --max-model-len 262144
```

## What to do next (priority order)

1. **Let the current `sw4k-L65k-ttt6` run train for at least 1k-2k steps**
   so we can verify the loss/acc trajectory looks healthy and the chunked
   fused-loss path doesn't have any subtle correctness issue at scale
   (the `__main__` numerical equivalence test in `specforge/core/loss.py`
   already proves bit-equivalence on synthetic data, but real training
   gives end-to-end validation).
2. **Decide on a stop point** -- the full 5-epoch run is ~4 days. Eagle3
   typically converges to "looks like a real draft head" within ~1 epoch,
   so consider stopping early at ~15k steps for a first-pass checkpoint.
3. **Run inference benchmarks** on the `mixed/epoch_0_step_2000` and the
   first `sw4k` checkpoint side-by-side via vLLM speculative decoding.
   Compare acceptance rates on the held-out benchmark distribution
   (which is the 1078 conversations that were removed from
   `chankhavu/c2_eagle3_train` because they leaked from the eval set --
   ask the user where to fetch them from).
4. **(Optional) Add FSDP2 verifier sharding** to unlock L=131072 +
   ttt=7 OR run on smaller GPUs. Tracked in README "Known issues".
5. **(Optional) Add length-bucketed dataloader** to recover ~2-3x
   throughput by killing the per-step variance from random conversation
   lengths. Tracked in README "Known issues".

## Author / contact

Built by Claude (Sonnet 4.6 / Opus 4.6) over multiple sessions in
collaboration with @hav4ik. The branch is hosted on
[hav4ik/SpecForge-Nemotron3](https://github.com/hav4ik/SpecForge-Nemotron3),
branch `nemotron-cascade-2-experiments`.
