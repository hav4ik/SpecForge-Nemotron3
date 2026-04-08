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
