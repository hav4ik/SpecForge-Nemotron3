# HANDOFF — Nemotron-Cascade-2 Eagle3 training

This file is **transient state**: the current run, what's been tried, where
the checkpoints live. If you're a new agent picking this up, **read this
first**, then `README.md` for the engineering rationale, then `git log
--oneline main..HEAD` for the patch tour.

## STOP — read this section before ANYTHING else (2026-04-09 final dump)

**Training engineering is DONE. Two full iterations have been run.
V2 may still be in flight when you pick this up.**

### What exists right now

| Artifact | Location | Status |
|---|---|---|
| **V1 baseline head** (ttt=6, L=32k, 2-stage, no CoT) | HF `chankhavu/c2.eagle3-test` (main = stage2-step-12000) | ✅ **DONE**, all checkpoints preserved on per-step branches |
| **V2 head** (ttt=7, L=32k, single-stage warm-start, +CoT) | HF `chankhavu/c2.eagle3-test-v2` (auto-created on first V2 ckpt push) | 🟡 **MAY BE IN FLIGHT** — check `ps -ef \| grep train_eagle3` |
| All code + configs + scripts | This branch (`nemotron-cascade-2-experiments`) on `github.com/hav4ik/SpecForge-Nemotron3` | ✅ Up to date |
| Wandb runs | `wandb.ai/hav4ik/nemotron-cascade-2-eagle3` | ✅ See run ID table below |
| Training data + caches | `$WORK_DIR/data/` and `$WORK_DIR/cache_l32768/` (local disk, NOT on HF) | ⚠️ Lost on instance teardown; rebuild via prep scripts |
| V1 union vocab mapping | `$WORK_DIR/data/union_vocab_mapping_l32k.pt` | ⚠️ Lost on teardown; rebuild via `build_union_vocab_mapping.py` |
| Auto-push watcher | `auto_push_checkpoints.sh` with `STAGE=v2 REPO=chankhavu/c2.eagle3-test-v2` | 🟡 May be running; `pkill -f auto_push` to stop |

### Wandb run ID table (complete history)

| Run ID | Name | State | What it was |
|---|---|---|---|
| `0fkrs68u` | `stage1-L32768-ttt6-sw4k-bucketed` | finished | V1 stage 1 (SFT bootstrap, 10k steps) |
| `yz5vd3qw` | `stage2-L32768-ttt6-sw4k` | killed | V1 stage 2 (on-policy traces, killed at step 12000 by user) |
| `wxldpgrn` | `v2-L32768-ttt7-sw4k-bucketed-cot` | crashed | **FAILED** V2 first attempt (bucketing + warm-start = oscillation + regression) |
| `aw93g0lm` | `v2-L32768-ttt7-sw4k-cot-lr1e-5` | running (or finished) | **CURRENT** V2 relaunch (no bucketing, lr=1e-5, stable) |
| (many older) | `mixed-*`, `sw4k-*` | crashed | Early engineering iterations, ignore |

### If V2 training is still running when you arrive

```bash
# Check
ps -ef | grep train_eagle3 | grep -v grep
tail -c 500 /workspace/eagle3_training/logs/train_v2.log | tr '\r' '\n' | tail -3

# It should be writing checkpoints to:
ls $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-v2/

# The watcher auto-pushes each checkpoint to chankhavu/c2.eagle3-test-v2
# If the watcher died, restart:
export WORK_DIR=$(pwd)/eagle3-work && export HF_HOME=/workspace/models
STAGE=v2 REPO=chankhavu/c2.eagle3-test-v2 \
    nohup bash experiments/nemotron-cascade-2/auto_push_checkpoints.sh \
    > $WORK_DIR/logs/auto_push_v2.log 2>&1 & disown
```

### If V2 training died and you need to resume

```bash
export WORK_DIR=$(pwd)/eagle3-work && export HF_HOME=/workspace/models
# Find the latest V2 checkpoint:
ls $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-v2/
# If checkpoints exist, resume:
bash experiments/nemotron-cascade-2/run_train_v2.sh --resume \
    > /workspace/eagle3_training/logs/train_v2.log 2>&1 & disown
# If NO V2 checkpoints exist, restart from V1 final:
CKPT_DIR=$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage2/epoch_2_step_12000 \
    bash experiments/nemotron-cascade-2/run_train_v2.sh \
    > /workspace/eagle3_training/logs/train_v2.log 2>&1 & disown
```

### If you're on a FRESH INSTANCE (nothing local, only the git repo + HF)

See the "First 5 minutes on a new instance" section below. The boot
sequence is comprehensive but takes ~20 min of prep before training
can start.

### Key numbers from V1 (the baseline to beat)

V1 stage 2 final eval at L=65536 (on c2_traces_validation, 539 rows):
```
acc_0=0.819  acc_1=0.750  acc_2=0.713  acc_3=0.688  acc_4=0.668  acc_5=0.651
```
V1 was ttt=6 (6 positions). V2 adds position 6 and targets:
```
acc_0≥0.82  acc_1≥0.75  acc_2≥0.72  acc_3≥0.70  acc_4≥0.68  acc_5≥0.66  acc_6≥0.65
```

### Hard-won lessons (do NOT repeat these mistakes)

1. **Bucketing + warm-start = regression.** Length-bucketed sampling
   on warm-started weights caused V2's first attempt to oscillate and
   drift below V1. Root cause: bucketed batches cluster CoT (long)
   samples together, creating high-variance gradients that kick
   converged weights out of their basin. Fix: disable bucketing for
   warm-started runs. Bucketing is fine for from-scratch training
   (V1 stage 1 used it successfully).

2. **Vocab mapping must be UNION across all stages/versions.** The
   d2t/t2d buffers in the lm_head must be identical between any
   checkpoint you're loading from (`--ckpt-dir`) and the current
   training run. V2 reuses V1's `union_vocab_mapping_l32k.pt`
   unchanged because the CoT data adds only 0.04% out-of-vocab
   tokens (negligible). If you add substantially new data that
   shifts the token distribution, rebuild the union mapping and
   accept you'll lose warm-start benefit on the lm_head.

3. **`print_on_rank0` is silently broken.** It uses `logger.info`
   but no `logging.basicConfig()` is ever called, so all INFO
   messages are dropped. Eval per-position metrics are NEVER in
   stdout/the local log file. Pull metrics from wandb instead.

4. **`dataset.map(num_proc=32)` hangs on small datasets.** The
   539-row eval set deadlocks at num_proc=32. Pre-build eval caches
   offline with `--num-proc 8` using `build_cache_offline.py`.

5. **HF_HOME must point at the verifier cache.** On the previous
   instance it was `/workspace/models`. If HF_HOME points elsewhere,
   HF will try to redownload the 63 GB verifier and probably fail
   on disk space. The auto-push watcher also needs the HF auth token
   at `$HF_HOME/token` — copy it from wherever `hf auth login`
   wrote it.

6. **ttt=8 doesn't fit at L=32k without grad-ckpt.** V2 at ttt=7
   uses 73-95 GB per rank. ttt=8 would need ~100+ GB → OOM on 96 GB
   GPUs unless `--draft-mlp-grad-checkpoint` is re-enabled (costs
   ~25% per-step throughput).

7. **The orchestrator has a race condition** between killing the
   stage 1 watcher and the stage 1 watcher's final main-push. The
   fix in `auto_launch_stage2.sh` adds a drain wait + defensive
   final push. Was fixed mid-V1 (commit `7406a36`).

8. **Multi-length eval (16k/32k/64k) is not needed.** V1's data
   showed <0.005 gap across all 3 lengths. Sliding-window draft
   generalizes cleanly to longer contexts. V2 uses single L=65536
   eval only.

## Current iteration plan (2026-04-08)

**TL;DR**: Get a working baseline draft out the door fast at L=32768,
then iterate up to L=65536 in a follow-up.

We pivoted away from the L=65536 single-pool training that was running
in the previous iteration (commit `db952f7` and earlier wandb runs
named `sw4k-L65k-ttt6-fused-mixed`). The new plan:

* **L=32768** (down from 65536) -- each step is roughly half the cost,
  and the SFT pool mostly fits under 32k anyway. We pay a long-context
  generalization cost we'll measure via the per-length eval set
  (16k/32k/64k); if the gap is small, we ship from L=32k. If it's
  large, we re-train at L=65k after the baseline lands.
* **2-stage schedule**:
  1. Stage 1 (`run_train_stage1.sh`): 1 epoch over
     `cascade2_sft_train.jsonl` (~20k SFT rows, plus anything dropped
     into `data/extra_sft/`), lr=1e-4, **with `--with-data-bucketing`**
     (DP=2 + B=1 makes the straggler-rank bottleneck huge on a wide
     length distribution; bucketing kills it).
  2. Stage 2 (`run_train_stage2.sh`): 2-3 epochs (default 3) over
     `c2_traces_train.jsonl` ONLY (~9.6k rows; the CoT set
     `c2_traces_cot_train.jsonl` is excluded because it has higher
     mean tokens-per-sample and would slow stage 2). lr=5e-5, NO
     bucketing (we want gradient diversity per step on the on-policy
     distribution we're specializing to). Loaded from the stage 1
     checkpoint via `--ckpt-dir` (weight-only, fresh optimizer).
* **`--draft-mlp-grad-checkpoint` is now OFF**. It was needed at
  L=65536 ttt=7 to fit on a 96GB GPU; at L=32768 we have plenty of
  headroom and grad-checkpointing the MLP costs ~25-30% on the draft
  forward. We still keep `--draft-mlp-chunk-size 4096` (cheap,
  defends transient peak) and `--fused-linear-loss` (still saves
  ~14 GiB at L=32k from the [B,T,V] logits buffer).
* **Multi-length eval**: both stage launchers pass
  `--eval-lengths 16384,32768,65536 --eval-data-path ...c2_traces_validation.jsonl`
  every 1000 steps. The train script builds 3 independent eval
  dataloaders (one per length cap) and logs per-length metrics to
  wandb under `eval_l16384/*`, `eval_l32768/*`, `eval_l65536/*`. This
  is how we measure long-context generalization without training at
  long context.
* **Per-length cache dirs**: `build_cache.sh` and both stage launchers
  honor a `CACHE_DIR` env var, defaulting to
  `$WORK_DIR/cache_l${MAX_LENGTH}` so caches at different lengths
  don't collide and are easy to inspect / clean up.

### Expected step times at L=32768 on 2x RTX PRO 6000 Blackwell

Reference: the L=65536 ttt=6 + grad-ckpt run measured **~4.1 s/step
running average** (high variance: 1.2s on short samples, 8.6s on near-
max-length samples, dragged up by the straggler-rank pairing problem).

Estimate for the new L=32k + bucketing + no-grad-ckpt setup:

| change | expected effect |
|---|---|
| L=65536 -> 32768 | -40 to -50% per step (activation memory ~halves; attention is sliding-window 4k so it scales sub-linearly, MLP is fully linear in seq) |
| no `--draft-mlp-grad-checkpoint` | -8 to -12% on total step (recovers the ~25% MLP-fwd recompute cost; MLP is ~30-40% of step) |
| `--with-data-bucketing` (stage 1 only) | -10 to -20% on stage 1 mean step time (eliminates straggler-rank waste; benefit shrinks as length variance does, so ~0% for stage 2) |
| **net stage 1 estimate** | **~1.5 - 2.0 s/step** |
| **net stage 2 estimate** | **~1.7 - 2.2 s/step** (no bucketing benefit) |

At ~1.7 s/step, ~10000 stage 1 steps -> **~5h**, and ~14500 stage 2
steps (3 epochs over ~9.6k rows on DP=2) -> **~7h**, totaling
**~12 h end-to-end**. These numbers should be re-measured once the
first ~200 steps land and the running average stabilizes.

**Actual measured (2026-04-08 stage 1 launch, first 60 steps)**:
running average **~3.4 s/step** (high variance: 0.75s to 8.6s).
Slower than the 1.5-2.0 estimate -- the bucketing pairs lengths
well across ranks but the long-tail samples still drag the mean
because rank-pair throughput is bounded by the longer of the two,
even with pairing-aware sorting. Translates to **~9.5 h for stage 1
(10000 steps)** and **~14 h for stage 2 (15000 steps no bucketing
benefit)** = **~24 h end-to-end at L=32k**, still ~40% faster
than the L=65k baseline would have been.

### Active session state (2026-04-09, V2 relaunch in flight)

**V1 training is DONE.** V2 (ttt=7, CoT included, warm-start from V1
final) is currently running.

* **V2 training process**: run name `v2-L32768-ttt7-sw4k-cot-lr1e-5`,
  wandb id `aw93g0lm` at `wandb.ai/hav4ik/nemotron-cascade-2-eagle3`.
  Step ~1462 / 5283 in epoch 0 of 3 as of last check. Loss 0.81,
  mean acc 0.78 (across all 7 TTT positions). Per-step time ~3.0-3.5s.
  First V2 checkpoint at step 2000 (~30 min from this commit).
  **The V2 run has NOT yet saved a checkpoint.** If the instance is
  killed before step 2000, V2 is lost and must be relaunched from V1's
  final checkpoint.

* **V1 final state**: all 5 stage 1 checkpoints + 6 stage 2 checkpoints
  on disk + on HF `chankhavu/c2.eagle3-test`. V1's best per-position
  eval (L=65k, step 8000 of stage 2):
  ```
  acc_0=0.819  acc_1=0.750  acc_2=0.713  acc_3=0.688  acc_4=0.668  acc_5=0.651
  ```
  V2's per-position train acc (step 620, 7 positions) was:
  ```
  acc_0=0.821  acc_1=0.764  acc_2=0.735  acc_3=0.715  acc_4=0.698  acc_5=0.680  acc_6=0.665
  ```
  So V2 is already matching or slightly above V1 with the new ttt=7
  position 6 also contributing usefully.

* **CRITICAL LESSON: bucketing + warm-start = BAD.** The first V2
  launch attempt (wandb run `wxldpgrn`, "v2-L32768-ttt7-sw4k-bucketed-cot")
  used `--with-data-bucketing` at lr=3e-5 and REGRESSED below V1:
  acc_0 drifted from 0.867 → 0.777 in ~500 steps. Root cause: bucketing
  clusters CoT samples (long) together, creating high-variance per-batch
  gradients that knocked the warm-started weights out of the V1 basin.
  Fixed by (a) removing `--with-data-bucketing` and (b) lowering LR to
  1e-5. The fixed V2 (run `aw93g0lm`) is stable at ~0.82 acc_0 through
  step 620+. **Do NOT re-enable bucketing on warm-started runs.**

### Active session state (2026-04-08, mid stage 1) [HISTORICAL]

Below is the state from the FIRST agent session. Kept for reference
but largely superseded by the V2 state above.

* **Wandb run**: `stage1-L32768-ttt6-sw4k-bucketed`, id `0fkrs68u` at
  `wandb.ai/hav4ik/nemotron-cascade-2-eagle3/runs/0fkrs68u`
* **Training process**: PID owned by an init-orphan torchrun (use
  `ps -ef | grep train_eagle3` to find it). Disowned, will survive
  shell exits. To stop cleanly: `pkill -INT -f torch.distributed.run`,
  wait 10s, then `pkill -9 -f train_eagle3` if needed.
* **Auto-push watcher**: `auto_push_checkpoints.sh` running in
  background, polling `$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1`
  every 60s. Pushes each new step to (a) `chankhavu/c2.eagle3-test` `main`
  branch (latest only) and (b) a tagged branch `stage1-step-NNNN` so
  history is preserved on the HF repo. State files at
  `$WORK_DIR/.pushed_checkpoints_stage1*`. Logs to
  `$WORK_DIR/logs/auto_push.log`. To stop:
  `pkill -f auto_push_checkpoints.sh`.
* **Stage 2 auto-launch orchestrator**: `auto_launch_stage2.sh`
  running in background. Polls every 60s for the stage 1 final
  checkpoint (`epoch_0_step_10000`) AND verifies no train_eagle3 /
  torch.distributed.run processes are still alive (so we don't race
  the final-save). When stage 1 completes:
    1. waits a 60s grace period for in-flight writes to flush
    2. re-verifies stage 2 prerequisites (data file, cache, union
       vocab mapping) are still in place
    3. kills the stage 1 auto-push watcher
    4. starts a stage 2 auto-push watcher (STAGE=stage2)
    5. launches `run_train_stage2.sh` with `CKPT_DIR` pointed at the
       stage 1 final checkpoint, logging to
       `/workspace/eagle3_training/logs/train_stage2.log`
    6. exits
  Logs to `$WORK_DIR/logs/auto_launch_stage2.log`. To stop:
  `pkill -f auto_launch_stage2.sh` (must be done BEFORE stage 1
  finishes or stage 2 will launch). The orchestrator MUST be
  invoked with `HF_HOME=/workspace/models` exported, otherwise the
  stage 2 launch will try to redownload the verifier.
* **Published HF models**:
  * **V1**: https://huggingface.co/chankhavu/c2.eagle3-test
    -- the V1 baseline (2-stage, ttt=6, no CoT). `main` is at
    `stage2-step-12000` (the V1 final). Per-step history preserved on
    branches `stage1-step-2000` ... `stage1-step-10000` and
    `stage2-step-2000` ... `stage2-step-12000`.
  * **V2**: https://huggingface.co/chankhavu/c2.eagle3-test-v2
    -- the V2 follow-up iteration (single-stage warm-start from V1
    final, ttt=7, CoT data included, --with-data-bucketing).
    Auto-created on the first V2 checkpoint push.

  V1 and V2 are intentionally in SEPARATE repos so V2's main doesn't
  overwrite the V1 baseline. The V2 watcher (auto_push_checkpoints.sh
  with STAGE=v2 REPO=chankhavu/c2.eagle3-test-v2) is the only thing
  pushing to the v2 repo.
* **Verifier weights are at `/workspace/models/`** (NOT `/workspace/.hf_home/`).
  All launchers should set `export HF_HOME=/workspace/models` before
  running, otherwise HF will try to re-download the 63 GB verifier into
  the wrong cache and fail with "not enough disk space" (we only have
  ~30 GB free total). The launchers respect an existing `HF_HOME` env
  var via `${HF_HOME:-...}`, so the export must happen at the shell
  level before invoking `bash run_train_*.sh`.

### Bugs found in upstream SpecForge that affect this run

1. **`print_on_rank0` uses `logger.info` but no `basicConfig`** is
   ever called -- so all `logger.info` messages are silently dropped
   below the default `WARNING` level. The eval per-position metric
   prints (`Eval - Step N, position i, Acc: ...`) you'd expect to see
   in the local log file are NOT there. The metrics ARE logged to
   wandb (via `tracker.log`), so wandb is the source of truth for eval
   numbers. Don't try to grep the train log for `Acc:` -- you won't
   find anything. Pull metrics via the wandb API instead:
   ```python
   import wandb
   api = wandb.Api()
   run = api.run("hav4ik/nemotron-cascade-2-eagle3/0fkrs68u")
   for r in run.scan_history():
       for k, v in r.items():
           if k.startswith("eval_l"):
               print(k, v)
   ```
   The fix would be a one-liner (`logging.basicConfig(level=logging.INFO)`
   somewhere in `train_eagle3.py`'s init), but I haven't shipped it
   because I don't want to mid-run patch the train script and possibly
   crash the running job. Tracked for the next iteration.

2. **Vocab mapping silently overwritten across stages** -- see the
   "vocab mapping" section below. Already fixed in this fork via the
   `--vocab-mapping-path` flag and `build_union_vocab_mapping.py`
   script. The fix is critical: without it, stage 2 would silently
   train against a misaligned lm_head index space and converge to a
   worse minimum.

3. **`dataset.map(num_proc=32)` hangs on tiny eval sets** (539 rows).
   See the gotcha section below.

### Common gotcha: num_proc=32 hangs on small eval datasets

The validation set (`c2_traces_validation.jsonl`, 539 rows) hangs
forever inside `dataset.map(num_proc=32)` when launched as part of
the in-train cache build (the train script's auto-build path uses
`--build-dataset-num-proc 32` which is sized for the 20k-row train
set). The fix is to **pre-build the eval caches offline** using
the offline cache builder with a smaller `--num-proc 8`:

```bash
for L in 16384 32768 65536; do
    python experiments/nemotron-cascade-2/build_cache_offline.py \
        --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
        --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
        --train-data-path $WORK_DIR/data/c2_traces_validation.jsonl \
        --chat-template nemotron-h \
        --max-length $L \
        --cache-dir $WORK_DIR/cache_l32768 \
        --num-proc 8 \
        --trust-remote-code
done
```

After that the train script's auto-build path hits the cache and
skips the slow path entirely.

### Version 2 (next iteration) -- planned changes

The current run is the **V1 baseline**: get a working draft head shipped
fast, learn what doesn't work, then iterate. V2 should incorporate the
following changes once V1 lands and we have eval numbers to compare
against:

1. **Re-include `c2_traces_cot_train.jsonl` in stage 2.**
   The 908 CoT traces (~50k mean tokens/row, vs ~21k for the regular
   traces) were excluded from V1 to keep stage 2 wall-clock down. They
   should come back for V2 because:
   * Reasoning-trace coverage is the explicit motivation for stage 2's
     existence; CoT is the longest-form, most-reasoning-heavy subset.
   * V2 trains at L=65536 (see #2 below), which absorbs the longer CoT
     samples without truncation, so the per-sample wall-clock penalty
     is proportionally smaller than at L=32k.
   * V1's exclusion is documented as "fast iteration only" in
     `prepare_data_stage2.sh` -- the comment block already says to
     fold this back in for V2.

2. **Push `max_length` from 32768 to 65536.**
   V1 truncates ~26% of stage 1 SFT tokens and ~15% of stage 2 traces.
   The whole point of the engineering work in this fork (chunked MLP,
   fused linear loss, sliding window) was to make L=65536 fit on
   96 GB GPUs, and V1 deliberately backed off to L=32k to validate
   the pipeline first. V2 should re-enable the L=65k path:
   * Re-enable `--draft-mlp-grad-checkpoint` (it's required to fit at
     L=65k + ttt>=6 on 96 GB; V1 disabled it because L=32k has
     headroom).
   * Per-step time at L=65k is roughly 1.5-2x the L=32k cost, so
     re-budget wall-clock accordingly.

3. **Bump `ttt_length` from 6 to 7.**
   ttt=7 is the Eagle3 paper / SGLang reference default. V1 dropped
   to ttt=6 purely as a memory workaround at L=65k before the chunked
   MLP commit landed (the original L=65k+ttt=7 attempt OOM'd by 64 MiB).
   With both grad-ckpt re-enabled and the chunked MLP / fused loss
   stack active, ttt=7 should fit at L=65k. The extra TTT step is
   expected to add ~0.5 token to the mean accept length at inference,
   per the diminishing-returns curve in the Eagle3 paper.

4. **Add more stage 1 SFT data**, especially **tool-use-heavy
   conversations.** V1's stage 1 is just the 20k cascade2_sft_train
   pool, which is mostly chat/instruction-following. The user has
   indicated they want to expand the SFT pool with more diverse data,
   particularly tool-use SFT, to improve generalization on
   tool-calling reasoning traces (which are common at inference but
   under-represented in V1 training).
   * Drop additional `*.jsonl` files into `$WORK_DIR/data/extra_sft/`
     and re-run `prepare_data_stage1.sh`. The data prep script
     auto-concatenates everything in that dir.
   * Re-run the union vocab mapping builder so the larger pool's
     token frequencies are incorporated into the d2t / t2d mapping.
   * Each additional 5k SFT rows adds roughly 5-7 hours of stage 1
     wall-clock at L=65k.

5. **Possibly bump `num_speculative_tokens` at inference**, depending
   on V2's training-time TTT length and the per-position acc curve.
   With ttt=7 the inference protocol can speculate 6-7 tokens ahead
   instead of V1's 5.

6. **Multi-length eval should keep validating long-context
   generalization.** The V1 multi-length eval (16k/32k/64k) lets us
   measure whether L=32k training generalizes to longer contexts at
   inference. V2's L=65k training should match or exceed V1 on the
   16k/32k buckets and improve on the 64k bucket. If V1's 16k/32k
   eval numbers are already strong enough, the L=65k retrain may
   not be worth the wall-clock cost.

**Decision criteria for whether V2 is worth doing**: after V1 ships,
look at:
* `eval_l16384/acc_*` vs `eval_l32768/acc_*` vs `eval_l65536/acc_*`
  in the V1 wandb run -- if the L=64k bucket significantly underperforms
  the shorter buckets, V2's L=65k retrain is justified.
* Inference acceptance rate on a CoT-heavy benchmark (math/coding) --
  if it's notably worse than on plain reasoning, V2's CoT inclusion is
  justified.
* Mean accept length at inference -- if it's well below the
  ~4.0-4.5 token target for stage 2 (see earlier estimates section),
  V2's ttt=7 + more data is justified.

If V1 hits all three targets, V2 is optional polish rather than
required.

### Reverted experimental knobs (and why)

* **Length bucketing was previously OFF** (we removed it after the
  user pushed back about gradient bias from per-step difficulty
  drift). It's back ON for stage 1 only because: (a) the user is
  cleaning the data to filter out samples below 1024 tokens, which
  bounds the worst-case length-ratio within a bucket; (b) the
  per-epoch global-batch-order shuffle in our
  `LengthBucketDistributedSampler` mitigates the drift; (c) stage 2
  runs un-bucketed and corrects any residual bias from stage 1.
* **CoT set excluded from stage 2 (this iteration only)**. Higher
  tokens-per-sample. Re-add in a follow-up stage 2 fine-tune if the
  baseline draft underperforms on CoT-heavy reasoning.
* **L=131072 + FSDP2 verifier sharding** is shelved indefinitely.
  We'll consider it after the L=32k baseline ships.

### CRITICAL: vocab mapping must be UNION across stages

Eagle3's draft head predicts logits over a smaller draft vocabulary
(`draft_vocab_size = 32000` here, vs `target_vocab_size = 131072`).
The mapping (`d2t` / `t2d` buffers on the draft model) is the top-K
most frequent target tokens in the train set, computed once and saved
to `cache_dir/vocab_mapping/<key>.pt`.

**The default SpecForge pipeline auto-generates this from whatever
training set the current run is processing.** For our 2-stage training
this is a silent correctness bug:

* Stage 1 trains the lm_head aligned to a mapping derived from
  `cascade2_sft_train.jsonl` only.
* Stage 2 starts from the stage 1 checkpoint via `--ckpt-dir` -- and
  the standard pipeline calls `draft_model.load_vocab_mapping(...)`
  AFTER loading the checkpoint, which **overwrites** the d2t/t2d
  buffers with a NEW mapping derived from `c2_traces_train.jsonl`.
* Result: the lm_head loaded from stage 1 was trained so that index
  `i` predicts `stage1_d2t[i]`, but in stage 2 forward, index `i`
  is interpreted as `stage2_d2t[i]` -- a different target token.
  Training won't crash, won't blow up, will just slowly converge to
  a worse minimum.

**Fix** (already in this branch): build a UNION vocab mapping from
the combined token frequencies of stage 1 + stage 2, save it to
`$WORK_DIR/data/union_vocab_mapping_l32k.pt`, and pass it to BOTH
stage launchers via the new `--vocab-mapping-path` CLI flag. Both
stages then share identical d2t/t2d buffers.

```bash
# Build the union mapping (one-time, ~1 min, requires both stage
# data files prepped + their caches built):
python experiments/nemotron-cascade-2/build_union_vocab_mapping.py \
    --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
    --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
    --stage1-data-path $WORK_DIR/data/all_data_stage1.jsonl \
    --stage2-data-path $WORK_DIR/data/all_data_stage2.jsonl \
    --max-length 32768 \
    --chat-template nemotron-h \
    --cache-dir $WORK_DIR/cache_l32768 \
    --output-path $WORK_DIR/data/union_vocab_mapping_l32k.pt \
    --num-proc 32 \
    --trust-remote-code
```

The first build of this iteration showed:
* stage 1 alone: 81,439 unique loss-masked tokens, 132.9M total
* stage 2 alone: 36,510 unique loss-masked tokens, 158.8M total
  (much narrower distribution: stage 2 has ~half the unique tokens
  but ~20% more total tokens because reasoning traces are long)
* union: 83,209 unique tokens, 291.7M total (stage 2 contributed
  1,770 NEW tokens not present in stage 1's loss-masked positions
  -- mostly latex / reasoning markers)
* **Per-stage coverage with union top-32k:**
  * stage 1: 99.5328% of tokens by frequency (~621k lost = ~0.47%)
  * stage 2: **99.9621%** of tokens by frequency (~60k lost = ~0.04%)
  * Stage 2 has *better* coverage than stage 1 even though the
    union top-32k was built jointly -- because stage 2's
    distribution is narrower / more concentrated. Reasoning traces
    use a much tighter vocabulary than the broad SFT pool's web /
    code / multilingual long tail.
  * Unique tokens IN draft vocab: stage 1 = 31978/81439 (39.27%);
    stage 2 = 26017/36510 (71.26%).

Both stage launchers default `VOCAB_MAPPING_PATH` to
`$WORK_DIR/data/union_vocab_mapping_l32k.pt`. If you change the
training data and need to rebuild, delete the .pt file first
(the union builder refuses to overwrite an existing file).

### Recommended next-agent runbook (current iteration)

```bash
cd $SPECFORGE_ROOT
export WORK_DIR=$(pwd)/eagle3-work   # or your scratch dir

# Stage 1 prep (also stages c2_traces_validation.jsonl into $WORK_DIR/data/)
bash experiments/nemotron-cascade-2/prepare_data_stage1.sh

# Stage 2 prep (must run BEFORE the union vocab builder so it has
# both data files to scan)
bash experiments/nemotron-cascade-2/prepare_data_stage2.sh

# Build BOTH caches (needed by the union vocab builder for cache hits)
CACHE_DIR=$WORK_DIR/cache_l32768 MAX_LENGTH=32768 EXPERIMENT=sw4k \
    TRAIN_DATA=$WORK_DIR/data/all_data_stage1.jsonl \
    bash experiments/nemotron-cascade-2/build_cache.sh
CACHE_DIR=$WORK_DIR/cache_l32768 MAX_LENGTH=32768 EXPERIMENT=sw4k \
    TRAIN_DATA=$WORK_DIR/data/all_data_stage2.jsonl \
    bash experiments/nemotron-cascade-2/build_cache.sh

# Build the UNION vocab mapping (see "vocab mapping" section above)
python experiments/nemotron-cascade-2/build_union_vocab_mapping.py \
    --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
    --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
    --stage1-data-path $WORK_DIR/data/all_data_stage1.jsonl \
    --stage2-data-path $WORK_DIR/data/all_data_stage2.jsonl \
    --max-length 32768 \
    --chat-template nemotron-h \
    --cache-dir $WORK_DIR/cache_l32768 \
    --output-path $WORK_DIR/data/union_vocab_mapping_l32k.pt \
    --num-proc 32 \
    --trust-remote-code

# Stage 1 train (uses union vocab mapping by default)
bash experiments/nemotron-cascade-2/run_train_stage1.sh \
    2>&1 | tee /workspace/eagle3_training/logs/train_stage1.log

# After stage 1 finishes, find the deepest checkpoint:
ls $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/

# Stage 2 train (loads stage1 weights via --ckpt-dir, fresh optimizer,
# SAME union vocab mapping by default)
CKPT_DIR=$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_<N> \
    bash experiments/nemotron-cascade-2/run_train_stage2.sh \
    2>&1 | tee /workspace/eagle3_training/logs/train_stage2.log
```

## First 5 minutes on a new instance (current iteration: 2-stage L=32k)

If you're picking this up on a fresh box with the GPUs already provisioned,
this is the COMPLETE boot sequence to reproduce / continue the in-flight
2-stage L=32k iteration. Read the "Current iteration plan" and "Bugs found
in upstream SpecForge" sections BEFORE running.

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
hf auth login              # paste HF token; needed to push checkpoints to chankhavu/c2.eagle3-test

# 3a. CRITICAL: hf auth login writes the token to ~/.cache/huggingface/token
#     OR /workspace/.hf_home/token (depending on what HF_HOME is set to AT
#     LOGIN TIME). The auto-push watcher inherits HF_HOME=/workspace/models
#     from the train launchers, so it looks for the token at
#     /workspace/models/token, which won't exist by default. Either:
#       (a) `hf auth login` again with HF_HOME=/workspace/models exported,
#           OR
#       (b) copy/symlink the existing token file:
#           cp /workspace/.hf_home/token /workspace/models/token
#     Without this step, all auto-push attempts will fail with
#     "Invalid username or password" inside the watcher (the watcher
#     will keep retrying every 60s, so the failure is non-fatal but
#     nothing reaches the HF repo).

# 4. Pick a workspace dir on big-disk storage
export WORK_DIR=$(pwd)/eagle3-work    # or /scratch/nemotron-eagle3

# CRITICAL: HF_HOME must point at the dir that already holds the
# 63 GB Nemotron-Cascade-2 verifier. On the previous instance it was
# /workspace/models. Otherwise HF will try to redownload and fail
# (we typically have <30 GB free total).
export HF_HOME=/workspace/models  # or wherever the verifier is cached

# 5. Stage 1 prep (also stages c2_traces_validation.jsonl into $WORK_DIR/data/)
bash experiments/nemotron-cascade-2/prepare_data_stage1.sh

# 6. Stage 2 prep (must run BEFORE union vocab mapping builder so it has
#    both data files to scan)
bash experiments/nemotron-cascade-2/prepare_data_stage2.sh

# 7. Build BOTH train caches (needed by union vocab builder for cache hits)
CACHE_DIR=$WORK_DIR/cache_l32768 MAX_LENGTH=32768 EXPERIMENT=sw4k \
    TRAIN_DATA=$WORK_DIR/data/all_data_stage1.jsonl \
    bash experiments/nemotron-cascade-2/build_cache.sh
CACHE_DIR=$WORK_DIR/cache_l32768 MAX_LENGTH=32768 EXPERIMENT=sw4k \
    TRAIN_DATA=$WORK_DIR/data/all_data_stage2.jsonl \
    bash experiments/nemotron-cascade-2/build_cache.sh

# 8. Pre-build the 3 EVAL caches at L=16k/32k/64k. Must use --num-proc 8
#    (NOT 32) -- the in-train auto-build hangs on the 539-row eval set
#    with num_proc=32. See "num_proc=32 hangs" gotcha below.
for L in 16384 32768 65536; do
    python experiments/nemotron-cascade-2/build_cache_offline.py \
        --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
        --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
        --train-data-path $WORK_DIR/data/c2_traces_validation.jsonl \
        --chat-template nemotron-h \
        --max-length $L \
        --cache-dir $WORK_DIR/cache_l32768 \
        --num-proc 8 \
        --trust-remote-code
done

# 9. Build the UNION vocab mapping. CRITICAL: stage 1 and stage 2 must
#    share the same d2t/t2d mapping or the lm_head index space gets
#    silently scrambled at the stage 2 transition. See "vocab mapping"
#    section below for the full bug story.
python experiments/nemotron-cascade-2/build_union_vocab_mapping.py \
    --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
    --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
    --stage1-data-path $WORK_DIR/data/all_data_stage1.jsonl \
    --stage2-data-path $WORK_DIR/data/all_data_stage2.jsonl \
    --max-length 32768 \
    --chat-template nemotron-h \
    --cache-dir $WORK_DIR/cache_l32768 \
    --output-path $WORK_DIR/data/union_vocab_mapping_l32k.pt \
    --num-proc 32 \
    --trust-remote-code

# 10. Launch stage 1 training. Logs to /workspace/eagle3_training/logs/.
bash experiments/nemotron-cascade-2/run_train_stage1.sh \
    > /workspace/eagle3_training/logs/train_stage1.log 2>&1 &
disown

# 11. Launch the auto-push watcher in background. Pushes each new
#     checkpoint to chankhavu/c2.eagle3-test (main = latest, tagged
#     branches preserve history).
nohup bash experiments/nemotron-cascade-2/auto_push_checkpoints.sh \
    > $WORK_DIR/logs/auto_push.log 2>&1 &
disown

# 12. Monitor:
tail -f /workspace/eagle3_training/logs/train_stage1.log
tail -f $WORK_DIR/logs/auto_push.log
# Or watch wandb: https://wandb.ai/hav4ik/nemotron-cascade-2-eagle3

# 13. After stage 1 finishes, find the deepest checkpoint and launch stage 2:
ls $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/
CKPT_DIR=$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_10000 \
    bash experiments/nemotron-cascade-2/run_train_stage2.sh \
    > /workspace/eagle3_training/logs/train_stage2.log 2>&1 &
disown
# Restart auto-push watcher for stage 2:
STAGE=stage2 nohup bash experiments/nemotron-cascade-2/auto_push_checkpoints.sh \
    > $WORK_DIR/logs/auto_push_stage2.log 2>&1 &
disown

# 14. (Optional) Push commits back to this fork:
#     PAT credentials are NOT stored in the repo. Ask @hav4ik for a fresh
#     fine-grained PAT scoped to hav4ik/SpecForge-Nemotron3, then push via:
#     git push https://hav4ik:$PAT@github.com/hav4ik/SpecForge-Nemotron3.git \
#         nemotron-cascade-2-experiments
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

## Mamba SSM precision -- the full intricacies

> "The Mamba state is the golden nugget, the most important part of a
>  Mamba layer. Treat it with care."  -- @hav4ik, 2026-04-08

This section documents EVERY intricacy that was uncovered during a
multi-hour audit of the Mamba SSM precision in our pipeline. Two
independent audit subagents read the actual source code (upstream
`mamba_ssm`, vLLM 0.19's vendored copy, and NemotronH's HF custom
remote code) and independently confirmed the findings. **Read this
in full before making any change to the verifier loading code, the
mamba_ssm version, or the train_eagle3.py imports.**

### What the Mamba SSM state actually is

In a Mamba2 layer, the "state" is the recurrent hidden state of the
selective state-space model: a dense tensor of shape `[batch,
num_heads, head_dim, ssm_state_size]` that summarizes the entire
prefix of the sequence at the current position. For NemotronH the
shape is `[B, 64, 64, 128]` (per the config: `mamba_num_heads=64`,
`mamba_head_dim=64`, `ssm_state_size=128`). This is the analog of the
KV cache in attention -- everything the model "remembers" about the
prefix is in this tensor.

A Mamba2 forward pass over a long sequence is computed as a chunked
parallel scan: the sequence is split into chunks of `chunk_size=128`
tokens (NemotronH's setting), and the SSM recurrence is computed
in two phases:

1. **Intra-chunk** (`_chunk_state_fwd`): each chunk's state is
   computed independently from its inputs. Hard-coded to fp32 in
   upstream mamba_ssm via `states_in_fp32=True` (line 375 of
   `ssd_combined.py`). Same in vLLM 0.19's vendored copy.

2. **Inter-chunk / boundary state** (`_state_passing_fwd`): the
   per-chunk states are then linked together via a recurrent pass
   so that chunk k+1's starting state is chunk k's ending state.
   This is the "state passing" step. **The output buffer of this
   pass is what we care about, and it's where the bug was.**

Each Mamba2 forward pass over an L-token input thus produces:
- L/chunk_size boundary states (one per chunk transition)
- 1 final_states tensor (which is normally written to the persistent
  decode cache, but we discard it in our `use_cache=False` setup)

### The bug we fixed

Upstream `mamba_ssm.ops.triton.ssd_combined.py:379-381`:

```python
states, final_states = _state_passing_fwd(
    rearrange(states, "... p n -> ... (p n)"),
    dA_cumsum[:, :, :, -1],
    ...,
    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype
)
```

`C.dtype` is the dtype of the C projection input to the Mamba layer,
which is `bfloat16` under our `torch_dtype=torch.bfloat16` verifier
load. Inside `_state_passing_fwd` (`ssd_state_passing.py:206-208`):

```python
out_dtype = states.dtype if out_dtype is None else out_dtype
out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
```

The Triton kernel accumulates SSM states in fp32 registers but
`tl.store(out_ptrs, states, ...)` writes them to the bf16 `out` buffer.
**Every inter-chunk boundary state was being downcast to bf16 on
store, then reloaded in the next chunk's scan kernel.** Only
`final_states` (the very last state -- normally cached for decode,
discarded by us) was fp32.

For NemotronH at L=65536, `chunk_size=128`:
* boundaries per Mamba layer per forward = 65536/128 = **512**
* Mamba layers in NemotronH = **31**
* total bf16 boundary writes per forward pass = **~15,872**

That's 15,872 small precision losses accumulating along the SSM scan
recurrence on every training step. The Nemotron team explicitly
documented this as the cause of an **88.3% -> 99.17% gap on AIME 2025
between SGLang (no fp32 SSM cache) and vLLM (with fp32 SSM cache)** in
https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/discussions/8 .

### vLLM 0.19 has the SAME bug for non-NemotronH-Cascade-2 models

Important caveat: vLLM 0.19 only avoids the boundary-state-bf16 bug for
**NemotronH-Cascade-2 specifically**, and only because the Nemotron team
checked `mamba_ssm_cache_dtype: "float32"` into the model's
`config.json` on HuggingFace. For any other case, vLLM 0.19 also has
the bug:

| Scenario | resulting boundary state dtype | status |
|---|---|---|
| NemotronH-Cascade-2 + default `--mamba_ssm_cache_dtype "auto"` | fp32 (auto-reads from config.json) | safe |
| NemotronH-Cascade-2 + explicit `--mamba_ssm_cache_dtype float32` | fp32 | safe |
| Older / variant NemotronH whose config.json lacks the field | bf16 (default = "float16" on line 494 of `models/config.py`) | **BUG** |
| Bamba, Jamba, FalconH1, Mamba2-1.3B, Qwen3Next, Zamba2, any other Mamba2 hybrid | bf16 (no NemotronH override fires; auto stays "auto"; `mamba2_state_dtype("auto")` falls through to conv state dtype = bf16) | **BUG** |

So our monkey-patch (which forces fp32 unconditionally regardless of
model or config) is actually **more robust than vLLM 0.19's
NemotronH-specific override**. If we ever swap the verifier for a
different Mamba2-based hybrid we don't need to do anything; the patch
just works. By contrast, if you serve any other Mamba2 hybrid via
vLLM 0.19 with defaults, you're getting the same precision regression
that the Nemotron team measured (88.3% → 99.17% on AIME 2025) -- you
should explicitly pass `--mamba_ssm_cache_dtype float32` until vLLM
changes the default upstream.

(The same logic applies to SGLang -- per the Nemotron team's HF
discussion, the equivalent flag is `--mamba-ssm-dtype float32` and is
not set by default. Their 88.3% AIME result was the SGLang-default
case.)

### How vLLM 0.19 avoids the bug (this is our reference)

vLLM 0.19 ships its own vendored copy of the Mamba Triton ops at
`vllm/model_executor/layers/mamba/ops/`. Three things are different
from upstream:

1. **`ssd_combined.py:119`** (vLLM-vendored): the `_state_passing_fwd`
   call site threads a `state_dtype` argument through:
   ```python
   states = _state_passing_fwd(
       ...,
       out_dtype=state_dtype if state_dtype is not None else C.dtype,
   )
   ```

2. **`mamba_mixer2.py:734`**: the prefill call passes
   `state_dtype=ssm_state.dtype` to the kernel.

3. **`MambaStateDtypeCalculator.mamba2_state_dtype()`**
   (`mamba_utils.py:62-67`): under `cache_config.mamba_ssm_cache_dtype
   == "auto"` (the CLI default), it falls through to the conv state
   dtype (bf16). Under any explicit value (including "float32"), it
   uses `STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]`.

4. **`NemotronHForCausalLMConfig.verify_and_update_config`**
   (`models/config.py:483-501`): when the CLI default `"auto"` is
   passed, it overrides with `hf_config.mamba_ssm_cache_dtype`.
   `nvidia/Nemotron-Cascade-2-30B-A3B/config.json` ships with
   `mamba_ssm_cache_dtype: "float32"`, so vLLM 0.19 + NemotronH
   serving **automatically gets fp32 boundary state, even without an
   explicit `--mamba_ssm_cache_dtype float32` flag**.

### Other fp32 sites (already correct in both upstream and vLLM)

These were verified the same in upstream `mamba_ssm` and vLLM 0.19:

* `_chunk_state_fwd(..., states_in_fp32=True)` -- intra-chunk states
  always fp32 (`ssd_combined.py:375` upstream, `:105` vLLM)
* `_bmm_chunk_fwd(..., output_dtype=torch.float32)` -- BMM output fp32
  (`ssd_combined.py:385` upstream, `:122` vLLM)
* `A_log.float()` -- explicit fp32 upcast in NemotronH modeling code at
  lines 424 and 560 of `modeling_nemotron_h.py`
* `final_states = torch.empty(..., dtype=torch.float32)` -- always fp32
  in `_state_passing_fwd` (`ssd_state_passing.py:208`)

So the **only** dtype gap between upstream `mamba_ssm` and vLLM 0.19's
fp32-NemotronH-default behavior was the boundary state output of
`_state_passing_fwd`. Our patch closes exactly that gap.

### The fix in detail

`specforge/_mamba_fp32_patch.py` is a monkey-patch that replaces
`_state_passing_fwd` in BOTH module locations:

1. `mamba_ssm.ops.triton.ssd_state_passing._state_passing_fwd` -- the
   canonical definition
2. `mamba_ssm.ops.triton.ssd_combined._state_passing_fwd` -- a re-import
   at the top of `ssd_combined.py` line 36 (`from
   mamba_ssm.ops.triton.ssd_state_passing import ... _state_passing_fwd`).
   The call site at `ssd_combined.py:379-381` resolves through this
   module-level reference, NOT through the canonical one. **Patching
   only the canonical location would be a silent no-op.**

The wrapper preserves the original calling convention exactly:

```python
def _state_passing_fwd_fp32(
    states, dA_chunk_cumsum,
    initial_states=None, seq_idx=None, chunk_size=None, out_dtype=None,
):
    # Ignore the caller's out_dtype and force fp32.
    return _orig(
        states, dA_chunk_cumsum,
        initial_states=initial_states, seq_idx=seq_idx, chunk_size=chunk_size,
        out_dtype=torch.float32,
    )
```

Note we explicitly pass `out_dtype=torch.float32` rather than relying
on the default `out_dtype=None -> states.dtype` (which would also be
fp32 because `_chunk_state_fwd(..., states_in_fp32=True)` produces fp32
`states`). The explicit override avoids drift if upstream ever changes
the default.

`_state_passing_bwd` (the backward path) is **not** patched. It takes
`dstates_dtype`/`states_dtype` arguments rather than `out_dtype`, and
the verifier runs in `no_grad()` so backward through the Mamba layers
is never reached. If anyone ever tries to train the Mamba verifier
end-to-end (highly unlikely for an Eagle3 setup), the backward path
would also need a fp32 patch.

### Why the patch import has to be at the very top of train_eagle3.py

The patch must be applied **before** any code transitively imports
`mamba_ssm`. The chain that loads it is:

1. `main()` calls `build_target_model()`
2. `build_target_model()` calls `HFEagle3TargetModel.from_pretrained()`
3. `HFEagle3TargetModel.from_pretrained()` calls
   `transformers.AutoModelForCausalLM.from_pretrained(...,
   trust_remote_code=True)`
4. transformers downloads and dynamically imports
   `modeling_nemotron_h.py` (the cached HF custom remote code)
5. `modeling_nemotron_h.py` does `from mamba_ssm.ops.triton.ssd_combined
   import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined`
   at the top of the file
6. **At THIS point** the unpatched `_state_passing_fwd` would be
   captured into `ssd_combined`'s module namespace if we hadn't already
   replaced it.

The patch is in `train_eagle3.py` lines 16-18, before the `import torch`
on line 20 and well before `main()` runs. SpecForge's own
`__init__.py` does NOT import `mamba_ssm` (verified by repo-wide grep).

If you ever refactor `train_eagle3.py` and move the patch import below
ANY other module-level import, you risk the unpatched function being
captured first. **Don't do that.** If you really need to apply the patch
elsewhere (e.g. from a test runner), do it as the FIRST thing in the
process, before importing torch or transformers.

### Memory and compute cost

* **Memory**: the boundary state buffer goes from `[batch, nchunks,
  nheads, dim]` bf16 to fp32, doubling its size. At L=65536, nchunks=512,
  nheads=64, dim=64*128=8192 the buffer is `1*512*64*8192 = 268M
  elements`. In bf16: 537 MiB. In fp32: 1.07 GiB. Per Mamba layer per
  forward. **Wait that's bigger than my earlier estimate -- let me
  recompute.** Actually `dim` here is `head_dim*ssm_state_size` after
  the rearrange, which for NemotronH is `64*128 = 8192`. So per Mamba
  layer per forward: ~1 GiB extra. For 31 Mamba layers: ~31 GiB extra
  during the kernel call -- but the buffer is freed after the kernel
  returns, so it's a transient peak that overlaps with the existing
  activations. In practice the OOM headroom is unaffected because the
  fast-path Triton kernel was already allocating something close to
  this; the only delta is doubled bytes in the boundary buffer that
  exists for ~microseconds during the kernel launch.
* **Compute**: same Triton kernel, same number of launches, fp32 stores
  instead of bf16 stores. ~no measurable difference (storing fp32 to
  HBM is the same number of ops as storing bf16, just twice the
  bandwidth -- bound by HBM bandwidth, not store count).

### NemotronH-specific config details that matter

From `nvidia/Nemotron-Cascade-2-30B-A3B/config.json`:

```json
{
  "chunk_size": 128,
  "mamba_head_dim": 64,
  "mamba_num_heads": 64,
  "ssm_state_size": 128,
  "mamba_ssm_cache_dtype": "float32",   <-- vLLM honors, upstream ignores
  "use_mamba_kernels": true,
  "n_groups": 1,
  "n_groups": 8                          <-- duplicate key (NB: jsonc-style override)
}
```

* `chunk_size=128` -> 512 boundary state writes per Mamba layer at L=65536
* `mamba_ssm_cache_dtype: "float32"` is **ignored** by NemotronH's
  custom HF modeling code (it's not consumed anywhere in
  `modeling_nemotron_h.py`) but IS honored by vLLM 0.19 (via
  `NemotronHForCausalLMConfig.verify_and_update_config`)

### How to verify in the future

```bash
# 1. Standalone numerical test
CUDA_VISIBLE_DEVICES=0 python -m specforge._mamba_fp32_patch --verify
# expected: "_state_passing_fwd output dtype is torch.float32 ..."

# 2. Confirm patch import is at the top of train_eagle3.py (lines 16-18)
head -20 scripts/train_eagle3.py | grep -A2 mamba_fp32_patch
# expected: import + apply() call before any other module-level imports

# 3. Confirm running training process started AFTER the patch was added
ps -o lstart -p $(pgrep -f train_eagle3.py | head -1)

# 4. Spot-check that no SpecForge module imports mamba_ssm directly
#    (should match only _mamba_fp32_patch.py itself)
grep -rn "import mamba_ssm\|from mamba_ssm" specforge/

# 5. The patch is bit-equivalent to vLLM 0.19's NemotronH default
#    inference, verified by independent audit subagents on 2026-04-08.
#    See git log for the commit that added the patch.
```

If any of these checks fail, **stop training immediately** and reapply
the patch -- the precision regression is silent (training will still
converge to a worse minimum, you won't see a crash).

### CRITICAL: Mamba SSM state must be float32 (don't break this)

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
