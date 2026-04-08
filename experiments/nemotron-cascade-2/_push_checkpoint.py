"""
Push a single SpecForge checkpoint dir to a HuggingFace model repo.

This is the workhorse called by ``auto_push_checkpoints.sh`` whenever
a new ``epoch_X_step_Y/`` checkpoint dir lands. It uploads only the
inference-relevant files (model.safetensors + config.json + the
rendered MODEL_CARD.md as README.md) and skips ``training_state.pt``
(optimizer state, useful only for ``--resume``).

Idempotent: re-uploading the same checkpoint replaces the repo
contents on the target branch (``main`` by default). For multi-step
history preservation, pass ``--branch step-NNNN`` to push to a
named revision.

Auth: requires ``hf auth login`` to have been run on the host (or
``HF_TOKEN`` env var set).

Usage:

    python experiments/nemotron-cascade-2/_push_checkpoint.py \\
        --ckpt-dir $WORK_DIR/checkpoints/nemotron-cascade-2-eagle3-stage1/epoch_0_step_4000 \\
        --repo chankhavu/c2.eagle3-test \\
        --commit-message "stage1 step 4000"
"""

import argparse
import os
import sys
import tempfile

from huggingface_hub import HfApi, create_repo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="SpecForge checkpoint dir (epoch_X_step_Y)")
    p.add_argument("--repo", required=True, help="HF repo id (e.g. chankhavu/c2.eagle3-test)")
    p.add_argument("--branch", default="main", help="Branch / revision to push to (default main)")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--private", action="store_true", help="Create the repo as private if it doesn't exist")
    args = p.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        print(f"[push] ERROR: --ckpt-dir={args.ckpt_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    required = ["model.safetensors", "config.json"]
    for f in required:
        path = os.path.join(args.ckpt_dir, f)
        if not os.path.isfile(path):
            print(f"[push] ERROR: missing required file {path}", file=sys.stderr)
            sys.exit(2)

    api = HfApi()

    # Make sure the target repo exists.
    try:
        create_repo(args.repo, exist_ok=True, repo_type="model", private=args.private)
        print(f"[push] repo {args.repo} ready")
    except Exception as e:
        print(f"[push] create_repo: {e}")

    # Make sure the target branch exists.
    if args.branch != "main":
        try:
            api.create_branch(repo_id=args.repo, branch=args.branch, exist_ok=True)
            print(f"[push] branch {args.branch} ready")
        except Exception as e:
            print(f"[push] create_branch: {e}")

    # Stage upload by symlinking inference-only files into a tmp dir.
    # MODEL_CARD.md gets renamed to README.md so HF renders it on the
    # repo page.
    with tempfile.TemporaryDirectory() as td:
        for f in required:
            os.symlink(os.path.join(args.ckpt_dir, f), os.path.join(td, f))
        mc = os.path.join(args.ckpt_dir, "MODEL_CARD.md")
        if os.path.isfile(mc):
            os.symlink(mc, os.path.join(td, "README.md"))
        else:
            print(f"[push] WARN: no MODEL_CARD.md in {args.ckpt_dir}; uploading without README")
        print(f"[push] staging files: {sorted(os.listdir(td))}")

        commit_msg = args.commit_message or f"checkpoint from {os.path.basename(args.ckpt_dir)}"
        api.upload_folder(
            folder_path=td,
            repo_id=args.repo,
            repo_type="model",
            revision=args.branch,
            commit_message=commit_msg,
        )
        print(f"[push] DONE -> https://huggingface.co/{args.repo}/tree/{args.branch}")


if __name__ == "__main__":
    main()
