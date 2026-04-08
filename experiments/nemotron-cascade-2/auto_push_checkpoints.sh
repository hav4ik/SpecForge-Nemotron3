#!/usr/bin/env bash
# Background watcher: pushes new SpecForge checkpoints to a HF model repo
# as soon as they land on disk. Idempotent via a small state file that
# tracks already-pushed step numbers.
#
# Designed to run alongside a long training job. Polls every 60 seconds.
# Pushes the LATEST checkpoint and ALSO each new checkpoint to a tagged
# branch (`step-NNNN`) so step-by-step history is preserved on the HF
# repo while `main` always reflects the most recent step.
#
# Usage (run in background, disowned, alongside the training process):
#
#   nohup bash experiments/nemotron-cascade-2/auto_push_checkpoints.sh \
#       > $WORK_DIR/logs/auto_push.log 2>&1 &
#   disown
#
# Stop:
#
#   pkill -f auto_push_checkpoints.sh
#
# Env vars (with defaults):
#   WORK_DIR        path to the eagle3 work tree (defaults to $(pwd)/eagle3-work)
#   REPO            target HF model repo (default chankhavu/c2.eagle3-test)
#   STAGE           which stage to watch (default stage1; pass stage2 once
#                   stage 2 starts)
#   POLL_INTERVAL   seconds between scans (default 60)
#   PUSH_TAGGED     1 = also push each step to a `step-NNNN` branch
#                   for history preservation (default 1)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SPECFORGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)

WORK_DIR=${WORK_DIR:-$(pwd)/eagle3-work}
REPO=${REPO:-chankhavu/c2.eagle3-test}
STAGE=${STAGE:-stage1}
POLL_INTERVAL=${POLL_INTERVAL:-60}
PUSH_TAGGED=${PUSH_TAGGED:-1}

CKPT_ROOT="$WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-${STAGE}"
STATE_FILE="$WORK_DIR/.pushed_checkpoints_${STAGE}"
PUSH_HELPER="$SCRIPT_DIR/_push_checkpoint.py"

mkdir -p "$WORK_DIR/logs"
touch "$STATE_FILE"

echo "[auto-push] starting watcher"
echo "[auto-push]   WORK_DIR=$WORK_DIR"
echo "[auto-push]   STAGE=$STAGE"
echo "[auto-push]   CKPT_ROOT=$CKPT_ROOT"
echo "[auto-push]   REPO=$REPO"
echo "[auto-push]   STATE_FILE=$STATE_FILE"
echo "[auto-push]   PUSH_TAGGED=$PUSH_TAGGED"
echo "[auto-push]   POLL_INTERVAL=${POLL_INTERVAL}s"

while true; do
    if [[ ! -d "$CKPT_ROOT" ]]; then
        # Training hasn't produced its first checkpoint yet -- the dir
        # doesn't exist. Wait and try again.
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Find all complete epoch_X_step_Y dirs (both model.safetensors and
    # training_state.pt present so we know SpecForge finished writing).
    # Sort by numeric step ascending, then process from lowest to highest
    # so that the LATEST checkpoint always wins on `main`.
    mapfile -t CKPTS < <(
        find "$CKPT_ROOT" -maxdepth 1 -type d -name 'epoch_*_step_*' \
        | sort -t_ -k4n
    )

    # Pass 1: per-step tagged pushes for any new checkpoint not yet in
    # the state file. Each tagged branch push is independent so
    # historical steps are preserved on the HF repo.
    if [[ "$PUSH_TAGGED" == "1" ]]; then
        for ckpt in "${CKPTS[@]:-}"; do
            [[ -z "$ckpt" ]] && continue
            bn=$(basename "$ckpt")
            if [[ ! -f "$ckpt/model.safetensors" || ! -f "$ckpt/training_state.pt" ]]; then
                continue
            fi
            if grep -qxF "$bn" "$STATE_FILE"; then
                continue
            fi
            step=$(echo "$bn" | sed -E 's/.*_step_([0-9]+)$/\1/')
            tag="${STAGE}-step-${step}"
            echo "[auto-push] $(date -Iseconds) tagged push $bn -> branch $tag"
            if python "$PUSH_HELPER" \
                --ckpt-dir "$ckpt" \
                --repo "$REPO" \
                --branch "$tag" \
                --commit-message "${STAGE} checkpoint from ${bn}"; then
                echo "$bn" >> "$STATE_FILE"
            else
                echo "[auto-push] $(date -Iseconds) WARN: tagged push for $tag failed -- will retry next scan"
            fi
        done
    fi

    # Pass 2: ALWAYS overwrite main with the LATEST (highest step) complete
    # checkpoint. This is idempotent (HF only commits when content changes)
    # and keeps `main` pointing at the most recent step regardless of
    # state-file desync.
    LATEST=""
    for ckpt in "${CKPTS[@]:-}"; do
        [[ -z "$ckpt" ]] && continue
        if [[ -f "$ckpt/model.safetensors" && -f "$ckpt/training_state.pt" ]]; then
            LATEST="$ckpt"
        fi
    done
    if [[ -n "$LATEST" ]]; then
        bn=$(basename "$LATEST")
        # Only push to main if we haven't already pushed this exact
        # checkpoint to main. We track the latest-on-main in a separate
        # tiny file.
        LATEST_MAIN_FILE="${STATE_FILE}.latest_main"
        cur_main=$(cat "$LATEST_MAIN_FILE" 2>/dev/null || echo "")
        if [[ "$bn" != "$cur_main" ]]; then
            echo "[auto-push] $(date -Iseconds) pushing $bn to $REPO main (was $cur_main)"
            if python "$PUSH_HELPER" \
                --ckpt-dir "$LATEST" \
                --repo "$REPO" \
                --branch main \
                --commit-message "${STAGE} checkpoint from ${bn}"; then
                echo "$bn" > "$LATEST_MAIN_FILE"
                echo "[auto-push] $(date -Iseconds) main is now at $bn"
            else
                echo "[auto-push] $(date -Iseconds) ERROR: main push failed -- will retry next scan"
            fi
        fi
    fi

    sleep "$POLL_INTERVAL"
done
