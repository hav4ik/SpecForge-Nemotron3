"""
Offline cache builder for Nemotron-Cascade-2 Eagle3 training.

Tokenizes the training conversations and builds the vocab mapping using the
SAME cache_key the train_eagle3.py script will compute, so when training
launches it finds the cache and skips preprocessing entirely. Lets you
do all the CPU-bound preprocessing on a box without GPUs (or while another
job has the GPUs busy) and then start training straight into GPU work.

No GPUs / no torchrun required. Uses the patched fast loss-mask parser
from the SpecForge fork (`specforge/data/parse.py`).

Usage (defaults match the launchers in this folder):

    python experiments/nemotron-cascade-2/build_cache_offline.py \
        --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \
        --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \
        --train-data-path /path/to/all_data_shuffled.jsonl \
        --chat-template nemotron-h \
        --max-length 65536 \
        --cache-dir ./cache \
        --num-proc 32 \
        --trust-remote-code
"""

import argparse
import hashlib
import os
import sys

# Avoid the script's directory shadowing the `specforge` package as a
# namespace import (the git repo dir at `<repo>/experiments/...` would
# otherwise win over the installed editable package).
sys.path[:] = [
    p for p in sys.path
    if p not in ("", os.path.dirname(os.path.abspath(__file__)))
]

from transformers import AutoTokenizer
from datasets import Dataset

from specforge import AutoDraftModelConfig
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
)
from specforge.utils import safe_conversations_generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--draft-model-config", required=True)
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--chat-template", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--num-proc", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    print(f"[offline-cache] cache_key={cache_key}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    print(f"[offline-cache] loading raw dataset from {args.train_data_path}")
    train_dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": args.train_data_path},
    )
    print(f"[offline-cache] {len(train_dataset)} raw conversations")

    print(
        f"[offline-cache] tokenizing + computing loss masks "
        f"(num_proc={args.num_proc}, max_length={args.max_length})"
    )
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.num_proc,
    )
    print(f"[offline-cache] processed dataset cached")

    print(
        f"[offline-cache] generating vocab mapping "
        f"(target={draft_model_config.vocab_size}, "
        f"draft={draft_model_config.draft_vocab_size})"
    )
    vocab_mapping_path = generate_vocab_mapping_file(
        dataset=train_eagle3_dataset,
        target_vocab_size=draft_model_config.vocab_size,
        draft_vocab_size=draft_model_config.draft_vocab_size,
        cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
        cache_key=cache_key,
    )
    print(f"[offline-cache] vocab mapping at {vocab_mapping_path}")
    print("[offline-cache] DONE")


if __name__ == "__main__":
    main()
