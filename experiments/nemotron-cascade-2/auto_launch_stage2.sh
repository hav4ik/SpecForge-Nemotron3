#!/usr/bin/env bash
# Background orchestrator: auto-launches stage 2 training as soon as
# stage 1 finishes its final checkpoint save.
#
# Polls every POLL_INTERVAL seconds for the stage 1 final checkpoint
# (epoch_0_step_10000 by default, configurable via STAGE1_FINAL_STEP).
# When the checkpoint exists AND the stage 1 training process has
# exited, it:
#
#   1. Waits an extra grace period (default 60s) so any in-flight
#      writes flush.
#   2. Verifies the stage 2 prerequisites (data file, cache, union
#      vocab mapping) are all in place.
#   3. Optionally restarts the auto_push_checkpoints.sh watcher with
#      STAGE=stage2 (kills the stage 1 watcher first).
#   4. Launches run_train_stage2.sh with CKPT_DIR pointed at the
#      stage 1 final checkpoint.
#   5. Exits.
#
# Designed to be run in background alongside the stage 1 training:
#
#   nohup bash experiments/nemotron-cascade-2/auto_launch_stage2.sh \
#       > $WORK_DIR/logs/auto_launch_stage2.log 2>&1 &
#   disown
#
# Stop:
#
#   pkill -f auto_launch_stage2.sh
#
# Env vars:
#   WORK_DIR              path to eagle3 work tree (defaults to $(pwd)/eagle3-work)
#   HF_HOME               must be set so the verifier weights are found
#                         (typically /workspace/models on this instance)
#   STAGE1_FINAL_STEP     stage 1 final step number (default 10000)
#   POLL_INTERVAL         seconds between scans (default 60)
#   GRACE_PERIOD          seconds to wait after detecting completion
#                         before launching stage 2 (default 60)
#   AUTO_RESTART_WATCHER  1 = kill stage 1 watcher and start a stage 2
#                         watcher after launching stage 2 (default 1)
#   PUSH_REPO             HF model repo for the auto-push watcher
#                         (default chankhavu/c2.eagle3-test)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
STAGE1_FINAL_STEP=${STAGE1_FINAL_STEP:-10000}
POLL_INTERVAL=${POLL_INTERVAL:-60}
GRACE_PERIOD=${GRACE_PERIOD:-60}
AUTO_RESTART_WATCHER=${AUTO_RESTART_WATCHER:-1}
PUSH_REPO=${PUSH_REPO:-chankhavu/c2.eagle3-test}

STAGE1_FINAL_DIR="$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_${STAGE1_FINAL_STEP}"

mkdir -p "$WORK_DIR/logs"

echo "[auto-stage2] $(date -Iseconds) starting orchestrator"
echo "[auto-stage2]   WORK_DIR=$WORK_DIR"
echo "[auto-stage2]   HF_HOME=${HF_HOME:-(unset!!)}"
echo "[auto-stage2]   STAGE1_FINAL_DIR=$STAGE1_FINAL_DIR"
echo "[auto-stage2]   POLL_INTERVAL=${POLL_INTERVAL}s"
echo "[auto-stage2]   GRACE_PERIOD=${GRACE_PERIOD}s"
echo "[auto-stage2]   AUTO_RESTART_WATCHER=$AUTO_RESTART_WATCHER"

if [[ -z "${HF_HOME:-}" ]]; then
    echo "[auto-stage2] WARN: HF_HOME is not set in this watcher's env." >&2
    echo "[auto-stage2] WARN: stage 2 launch may fail to find verifier weights." >&2
    echo "[auto-stage2] WARN: set HF_HOME=/workspace/models (or wherever the" >&2
    echo "[auto-stage2] WARN: 63 GB verifier is cached) BEFORE launching this script." >&2
fi

# Sanity-check stage 2 prerequisites at startup so we fail fast if
# something is missing rather than waiting hours for stage 1 to finish.
fail_prereq=0
for required in \
    "$WORK_DIR/data/all_data_stage2.jsonl" \
    "$WORK_DIR/data/union_vocab_mapping_l32k.pt" \
    "$WORK_DIR/cache_l32768/processed_dataset" \
    ; do
    if [[ ! -e "$required" ]]; then
        echo "[auto-stage2] ERROR: missing stage 2 prerequisite: $required" >&2
        fail_prereq=1
    fi
done
if [[ "$fail_prereq" == "1" ]]; then
    echo "[auto-stage2] FATAL: stage 2 prerequisites missing; will NOT launch" >&2
    echo "[auto-stage2] (run prepare_data_stage2.sh, build_cache.sh, and" >&2
    echo "[auto-stage2]  build_union_vocab_mapping.py first, then re-launch this orchestrator)" >&2
    exit 2
fi
echo "[auto-stage2] $(date -Iseconds) stage 2 prerequisites OK"

# Poll loop.
while true; do
    if [[ -f "$STAGE1_FINAL_DIR/model.safetensors" && -f "$STAGE1_FINAL_DIR/training_state.pt" ]]; then
        # Verify the stage 1 train process has exited. We don't want to
        # launch stage 2 while stage 1 is still mid-final-save.
        n_train=$(pgrep -f "train_eagle3.*stage1" 2>/dev/null | wc -l || echo 0)
        # The pgrep above doesn't actually filter by stage1 because the
        # arg list contains the output dir which mentions stage1, but to
        # be safe we also check torch.distributed.run.
        n_torchrun=$(pgrep -f "torch.distributed.run" 2>/dev/null | wc -l || echo 0)
        if [[ "$n_train" -gt 0 || "$n_torchrun" -gt 0 ]]; then
            echo "[auto-stage2] $(date -Iseconds) final checkpoint exists but training procs still alive (train=$n_train torchrun=$n_torchrun); waiting"
            sleep "$POLL_INTERVAL"
            continue
        fi
        # Stage 1 done!
        echo "[auto-stage2] $(date -Iseconds) stage 1 done. Final checkpoint: $STAGE1_FINAL_DIR"
        echo "[auto-stage2] $(date -Iseconds) sleeping ${GRACE_PERIOD}s grace period before launching stage 2"
        sleep "$GRACE_PERIOD"

        # Re-check the prereqs in case anything was deleted in the
        # meantime (e.g. cache cleared by mistake).
        for required in \
            "$WORK_DIR/data/all_data_stage2.jsonl" \
            "$WORK_DIR/data/union_vocab_mapping_l32k.pt" \
            ; do
            if [[ ! -e "$required" ]]; then
                echo "[auto-stage2] FATAL: $required disappeared between startup and now" >&2
                exit 2
            fi
        done

        # Optionally restart the auto-push watcher to track stage 2.
        if [[ "$AUTO_RESTART_WATCHER" == "1" ]]; then
            echo "[auto-stage2] $(date -Iseconds) killing stage 1 auto-push watcher"
            pkill -f "auto_push_checkpoints.sh" || true
            sleep 2
            echo "[auto-stage2] $(date -Iseconds) starting stage 2 auto-push watcher"
            STAGE=stage2 REPO="$PUSH_REPO" \
                nohup bash "$SCRIPT_DIR/auto_push_checkpoints.sh" \
                > "$WORK_DIR/logs/auto_push_stage2.log" 2>&1 &
            disown
            echo "[auto-stage2] $(date -Iseconds) stage 2 auto-push watcher launched (logs at $WORK_DIR/logs/auto_push_stage2.log)"
        fi

        # Launch stage 2. Logs go to /workspace/eagle3_training/logs
        # which is where the user has been tailing the stage 1 log.
        # Use the system-wide path if it exists; otherwise fall back
        # to $WORK_DIR/logs.
        STAGE2_LOG_DIR="/workspace/eagle3_training/logs"
        if [[ ! -d "$STAGE2_LOG_DIR" ]]; then
            STAGE2_LOG_DIR="$WORK_DIR/logs"
        fi
        STAGE2_LOG="$STAGE2_LOG_DIR/train_stage2.log"

        echo "[auto-stage2] $(date -Iseconds) launching stage 2 with CKPT_DIR=$STAGE1_FINAL_DIR"
        echo "[auto-stage2]   stage 2 log: $STAGE2_LOG"

        cd "$SPECFORGE_ROOT"
        CKPT_DIR="$STAGE1_FINAL_DIR" \
            HF_HOME="${HF_HOME:-/workspace/models}" \
            WORK_DIR="$WORK_DIR" \
            nohup bash "$SCRIPT_DIR/run_train_stage2.sh" \
            > "$STAGE2_LOG" 2>&1 &
        disown
        STAGE2_PID=$!
        echo "[auto-stage2] $(date -Iseconds) stage 2 launched (PID=$STAGE2_PID)"
        echo "[auto-stage2] $(date -Iseconds) orchestrator exiting; monitor stage 2 via:"
        echo "[auto-stage2]   tail -f $STAGE2_LOG"
        echo "[auto-stage2]   tail -f $WORK_DIR/logs/auto_push_stage2.log"
        echo "[auto-stage2]   wandb: https://wandb.ai/hav4ik/nemotron-cascade-2-eagle3"
        exit 0
    fi
    sleep "$POLL_INTERVAL"
done
