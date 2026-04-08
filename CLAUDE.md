# CLAUDE.md — entry point for the next Claude Code instance

You're picking up a downstream fork of [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)
with engineering work to train an Eagle3 speculative-decoding draft head
against [`nvidia/Nemotron-Cascade-2-30B-A3B`](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B),
a 30B-param hybrid Mamba-Transformer MoE model. **The branch you're on is
`nemotron-cascade-2-experiments`**, hosted at
https://github.com/hav4ik/SpecForge-Nemotron3.

To our knowledge this is the first Eagle3 head trained against a hybrid
Mamba-Transformer verifier, and getting it to fit at long context (L=65536,
ttt=6) on a 96 GB GPU per rank required significant engineering on top of
SpecForge. **All of that engineering is done.** What remains is monitoring
the running training, optionally pushing context to L=131072 with FSDP2
verifier sharding, and pushing the trained checkpoint to HF when ready.

## STOP — read these in order before doing anything

1. **`experiments/nemotron-cascade-2/HANDOFF.md`** — current run state,
   wandb run names, the literal "first 5 minutes on a new instance"
   runbook, common-failure-modes table, what-transfers-vs-what-doesn't
   table. **If you only read one file, read this one.**
2. **`experiments/nemotron-cascade-2/README.md`** — full reproduction
   recipe, scaling-to-more-GPUs analysis, the per-commit engineering
   tour, memory-math reference, known issues + future work.
3. **`experiments/nemotron-cascade-2/PROJECT.md`** — the original spec
   the experiment was started from, kept verbatim for traceability.
4. `git log --oneline main..HEAD` — the patch tour. Each commit message
   is a self-contained explainer with the why, the math, and file/line
   refs. Read in chronological order:
   `git log --reverse --oneline main..HEAD`.

## Critical things to know upfront

* **This is a downstream fork.** Upstream is
  https://github.com/sgl-project/SpecForge. **Do not** push commits or
  open PRs against upstream without @hav4ik's explicit go-ahead. All
  development happens on the `nemotron-cascade-2-experiments` branch of
  this fork.
* **The remote `hav4ik`** in this repo's `.git/config` points at the
  fork without an embedded PAT. You'll need a fresh fine-grained PAT
  from @hav4ik (scoped only to this repo) to push. Don't store it in
  the repo.
* **The verifier needs `trust_remote_code=True`.** Nemotron-H is loaded
  via custom HF remote-code modeling files. The HF target backend in
  SpecForge had to be patched to discover its layers under
  `model.backbone.layers` rather than `model.model.layers` -- if you
  see `Could not locate transformer layers` errors after pulling
  upstream changes, that fix is in commit `c55a87a`.
* **There's a numerical equivalence test** for the chunked fused
  linear+soft-target CE in `specforge/core/loss.py` `__main__` block.
  Run `python specforge/core/loss.py` (1 GPU, ~10 sec) to verify the
  long-context engineering didn't regress. The test compares chunked
  vs unchunked loss values AND gradients on hidden_states + lm_head_weight
  at realistic shapes (B=1, T=2048, H=2688, V=32000), `rtol=1e-4 atol=1e-5`.
* **The `experiments/` folder is the canonical recipe location.** All
  launchers and the cache builder live under
  `experiments/nemotron-cascade-2/`. Don't put new scripts in the repo
  root.
* **Wandb runs live at** `wandb.ai/hav4ik/nemotron-cascade-2-eagle3`.
  The current long-context run name pattern is `sw4k-L65k-ttt6-fused-mixed`.
  You'll need `wandb login` on the new instance.

## Working style with @hav4ik

@hav4ik values **experimental velocity over perfect engineering**. They
iterate fast and are OK with failed launches as long as the failures are
informative. Patterns observed across the previous sessions:

* **Don't ask for permission before trying things** when the user has
  already given a clear intent ("Fuck it, let's launch X" = launch X).
  Be honest about likely failure modes upfront so they can decide, but
  then execute.
* **Prefer surgical fixes** over comprehensive rewrites. The 11
  engineering commits on this branch are deliberately small and
  semantically focused -- continue that pattern.
* **Always commit + push** code changes as you go (and write detailed
  commit messages). @hav4ik reviews via the GitHub web UI, not via
  reading the local repo state.
* **Keep documentation in sync.** When you change something material,
  update README.md or HANDOFF.md. The next agent will thank you.
* **Be transparent about destructive actions.** Killing training runs,
  rebuilding caches, etc. -- name what you're killing and why. Don't
  hide behind vague status updates.
* **Avoid "here are 3 options, which do you want?"** when you have a
  clear preference. State your recommendation, mention the alternatives
  briefly, and execute unless asked otherwise.
* **The user reads logs.** When you say "training is healthy", they
  will check wandb. Make sure the claim matches reality.

## Quick orientation commands

```bash
# Where are we?
git branch --show-current     # should be: nemotron-cascade-2-experiments
git log --oneline main..HEAD | wc -l    # should be 12+ commits past main
ls experiments/nemotron-cascade-2/      # docs + launchers should be here

# Is anything training right now?
ps -ef | grep train_eagle3 | grep -v grep | wc -l
# > 0 means a run is alive (orphaned to init via `disown` is normal)

# Numerical equivalence test (proves chunked fused loss math is right)
CUDA_VISIBLE_DEVICES=0 python specforge/core/loss.py
# expect: "fused_linear_log_softmax_loss equivalence: OK"

# What's the latest training run wandb URL?
grep "View run at" $WORK_DIR/logs/train_*.log 2>/dev/null | tail -3
```

## When in doubt

Ask @hav4ik. They have full context from the previous sessions and
can disambiguate between "the experiment we already tried and it
didn't work" vs "a thing nobody has tried yet".
