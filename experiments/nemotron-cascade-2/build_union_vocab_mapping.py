"""
Build a UNION vocab mapping shared by stage 1 (SFT) and stage 2 (traces).

WHY THIS EXISTS
---------------
SpecForge's standard pipeline auto-generates the Eagle3 draft vocab
mapping (d2t / t2d, see specforge.data.preprocessing.generate_vocab_mapping_file)
from the loss-masked input_ids of WHATEVER training set is currently
being processed. The mapping is the top-K most frequent target vocab
tokens (K = draft_vocab_size, e.g. 32000).

For the 2-stage Nemotron-Cascade-2 Eagle3 training, this is a silent
correctness bug:

  * Stage 1 trains on cascade2_sft_train.jsonl. Its auto-generated
    vocab mapping is the top-K tokens of the SFT distribution.
  * Stage 2 starts from the stage 1 checkpoint via --ckpt-dir and
    trains on c2_traces_train.jsonl. The standard pipeline would
    auto-generate a NEW vocab mapping from the traces distribution
    and OVERWRITE the d2t / t2d buffers loaded from the stage 1
    checkpoint.
  * The lm_head weights loaded from stage 1 are still aligned to
    stage 1's vocab mapping. After overwrite, draft index `i` of
    the lm_head was trained to predict stage1_d2t[i] but during
    stage 2 forward, draft index `i` is interpreted as
    stage2_d2t[i] -- a completely different target token.

The result is silent: training won't crash, the loss won't blow up,
the head will just slowly converge to a worse minimum because every
optimizer step has to fight a permutation mismatch in the head it
inherited.

THE FIX
-------
Build ONE union vocab mapping from the combined token frequencies of
stage 1 + stage 2 train data, then use --vocab-mapping-path on BOTH
stage launchers to point at the union mapping. Stage 1 trains the
head aligned to the union mapping; stage 2 inherits both the head
AND the buffers from stage 1, so the alignment is preserved end to
end.

This script:

  1. Loads (or builds via cache hit) the cached eagle3 datasets for
     stage 1 and stage 2 using the same cache_key the offline
     cache builder + train script use.
  2. Counts the union of input_ids on loss_mask == 1 positions across
     both datasets.
  3. Calls process_token_dict_to_mappings() to derive d2t / t2d.
  4. Saves to $WORK_DIR/data/union_vocab_mapping.pt (path is
     configurable via --output-path).

Both stages must already have their processed_dataset/<key>.pkl
shards built (run the offline cache builder for both).

Usage:

    python experiments/nemotron-cascade-2/build_union_vocab_mapping.py \\
        --target-model-path nvidia/Nemotron-Cascade-2-30B-A3B \\
        --draft-model-config configs/nemotron-cascade-2-eagle3-sw4k.json \\
        --stage1-data-path $WORK_DIR/data/all_data_stage1.jsonl \\
        --stage2-data-path $WORK_DIR/data/all_data_stage2.jsonl \\
        --max-length 32768 \\
        --chat-template nemotron-h \\
        --cache-dir $WORK_DIR/cache_l32768 \\
        --output-path $WORK_DIR/data/union_vocab_mapping.pt \\
        --num-proc 32 \\
        --trust-remote-code
"""

import argparse
import hashlib
import os
import sys

# Avoid the script's directory shadowing the `specforge` package as a
# namespace import.
sys.path[:] = [
    p for p in sys.path
    if p not in ("", os.path.dirname(os.path.abspath(__file__)))
]

from collections import Counter

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig
from specforge.data import build_eagle3_dataset
from specforge.data.preprocessing import process_token_dict_to_mappings
from specforge.utils import safe_conversations_generator


def _build_or_load_dataset(
    *,
    train_data_path: str,
    tokenizer,
    chat_template: str,
    max_length: int,
    cache_dir: str,
    num_proc: int,
):
    """Cache-hit path: builds the eagle3 dataset using the same cache_key
    formula the offline builder + train script use, so this re-uses any
    existing processed_dataset/<key>.pkl shards instead of recomputing."""
    cache_params_string = (
        f"{train_data_path}-"
        f"{max_length}-"
        f"{chat_template}-"
        f"{tokenizer.name_or_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    print(f"[union-vocab] cache_key={cache_key} for {train_data_path}")

    raw = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": train_data_path},
    )
    print(f"[union-vocab] loaded {len(raw)} raw conversations")

    ds = build_eagle3_dataset(
        dataset=raw,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        cache_dir=os.path.join(cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=num_proc,
    )
    return ds


def _accumulate_token_counts(dataset, token_counter: Counter, label: str) -> None:
    """Counts loss-masked input_ids across the dataset into the shared
    Counter. Mirrors the logic in
    specforge.data.preprocessing.generate_vocab_mapping_file."""
    for input_ids, loss_mask in tqdm(
        zip(dataset["input_ids"], dataset["loss_mask"]),
        total=len(dataset),
        desc=f"counting tokens [{label}]",
    ):
        masked_ids = input_ids[loss_mask == 1]
        if masked_ids.numel() == 0:
            continue
        unique_ids, counts = masked_ids.unique(return_counts=True)
        token_counter.update(dict(zip(unique_ids.tolist(), counts.tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--draft-model-config", required=True)
    p.add_argument("--stage1-data-path", required=True)
    p.add_argument("--stage2-data-path", required=True)
    p.add_argument("--chat-template", required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--num-proc", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output_path):
        print(
            f"[union-vocab] output already exists at {args.output_path}; "
            f"delete it to re-build. Aborting."
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    print(
        f"[union-vocab] target_vocab={draft_model_config.vocab_size} "
        f"draft_vocab={draft_model_config.draft_vocab_size}"
    )

    stage1_ds = _build_or_load_dataset(
        train_data_path=args.stage1_data_path,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    )
    stage2_ds = _build_or_load_dataset(
        train_data_path=args.stage2_data_path,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc,
    )

    s1_counter: Counter = Counter()
    s2_counter: Counter = Counter()
    _accumulate_token_counts(stage1_ds, s1_counter, label="stage1")
    _accumulate_token_counts(stage2_ds, s2_counter, label="stage2")
    print(
        f"[union-vocab] stage1: {len(s1_counter)} unique tokens, "
        f"{sum(s1_counter.values())} total"
    )
    print(
        f"[union-vocab] stage2: {len(s2_counter)} unique tokens, "
        f"{sum(s2_counter.values())} total"
    )
    token_counter: Counter = Counter()
    token_counter.update(s1_counter)
    token_counter.update(s2_counter)
    print(
        f"[union-vocab] union: {len(token_counter)} unique tokens, "
        f"{sum(token_counter.values())} total "
        f"(stage2 contributed {len(token_counter) - len(s1_counter)} "
        f"new tokens not in stage1)"
    )

    d2t, t2d = process_token_dict_to_mappings(
        token_counter,
        draft_model_config.draft_vocab_size,
        draft_model_config.vocab_size,
    )

    # Per-stage coverage report. The union mapping is built from the
    # combined token frequencies, so the top-K is union-optimal -- but
    # the per-stage coverage may differ. Stage 2 (narrower distribution
    # like reasoning traces) typically gets better coverage than the
    # broader stage 1 SFT pool. This is the "<1% of positions are out
    # of vocab" stat that lives in HANDOFF.md and is the load-bearing
    # claim that the chosen draft_vocab_size is actually big enough.
    in_vocab_set = set(d2t.tolist())  # in target-vocab id space
    # process_token_dict_to_mappings stored d2t as offsets, not target ids;
    # reconstruct the target id set from t2d which is direct.
    in_vocab_set = set(int(i) for i, v in enumerate(t2d.tolist()) if v)

    def _coverage(counter):
        total = sum(counter.values())
        in_covered = sum(c for t, c in counter.items() if int(t) in in_vocab_set)
        n_unique = len(counter)
        n_unique_in = sum(1 for t in counter if int(t) in in_vocab_set)
        return {
            "total": total,
            "covered": in_covered,
            "lost": total - in_covered,
            "unique": n_unique,
            "unique_in": n_unique_in,
            "pct_freq": (100.0 * in_covered / total) if total > 0 else 0.0,
            "pct_unique": (100.0 * n_unique_in / n_unique) if n_unique > 0 else 0.0,
        }

    s1_cov = _coverage(s1_counter)
    s2_cov = _coverage(s2_counter)
    print()
    print("[union-vocab] === per-stage coverage with union top-K ===")
    for label, c in [("stage1", s1_cov), ("stage2", s2_cov)]:
        print(
            f"  {label}: total={c['total']:,} unique={c['unique']:,}  "
            f"covered={c['covered']:,} ({c['pct_freq']:.4f}%)  "
            f"lost={c['lost']:,} ({100.0 - c['pct_freq']:.4f}%)  "
            f"unique-in-vocab={c['unique_in']:,}/{c['unique']:,} "
            f"({c['pct_unique']:.2f}%)"
        )

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"d2t": d2t, "t2d": t2d}, args.output_path)
    print(f"[union-vocab] saved union vocab mapping to {args.output_path}")
    print(
        f"[union-vocab] d2t shape={tuple(d2t.shape)} "
        f"t2d shape={tuple(t2d.shape)} (sum True={int(t2d.sum())})"
    )
    print("[union-vocab] DONE")


if __name__ == "__main__":
    main()
