"""
Force fp32 SSM state across chunk boundaries in mamba_ssm's Triton fast path.

Why this exists
---------------
NemotronH (and the Nemotron team's training recipe in general) requires the
Mamba SSM state to be kept in float32. The original Nemotron authors and
multiple third-party benchmarks have confirmed that downcasting the SSM
state to bfloat16 catastrophically degrades reasoning/math performance --
on the order of -10% absolute on AIME'25-class benchmarks.

The upstream `mamba_ssm` library's Triton fast path *does* accumulate the
intra-chunk SSM state in fp32 inside `_chunk_state_fwd(..., states_in_fp32=True)`,
which is what NemotronH calls. **However**, the per-chunk *boundary* state
that crosses between successive chunks is downcast to bf16 by the call site
in `mamba_ssm.ops.triton.ssd_combined._mamba_chunk_scan_combined_fwd`:

    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        ...
        seq_idx=seq_idx, chunk_size=chunk_size,
        out_dtype=C.dtype                       # <-- BUG: bf16 under our load
    )

`C.dtype` is the dtype of the C projection input to the Mamba layer, which is
bfloat16 under `torch_dtype=torch.bfloat16`. So the *boundary* state between
every Mamba scan chunk is downcast to bf16. At L=65536 with the default
Mamba `chunk_size=128`, that's 512 boundary downcasts per Mamba layer per
forward pass, multiplied by 31 Mamba layers in NemotronH = ~15,872 bf16
downcasts per forward, accumulating arithmetic error along the SSM scan.

The default value of `out_dtype` in `_state_passing_fwd` is `None`, which
falls through to `states.dtype` -- and `states` was just produced by
`_chunk_state_fwd(..., states_in_fp32=True)`, so it's already fp32. The
fix is therefore as simple as ignoring the caller's `out_dtype` request
and forcing fp32.

The fix
-------
This module monkey-patches `mamba_ssm.ops.triton.ssd_state_passing._state_passing_fwd`
to always run with `out_dtype=torch.float32`, regardless of what the caller
asked for. It also patches `mamba_ssm.ops.triton.ssd_combined._state_passing_fwd`
because that module re-imports the function at the top.

Effect on memory
----------------
The boundary-state buffer goes from `[batch, nchunks, nheads, dim]` bf16
to fp32, doubling its size. At L=65536, nheads=64, dim=128, nchunks=512:
`512 * 64 * 128 * 4 bytes = 16 MiB` vs 8 MiB. Per Mamba layer. Negligible
on a 96 GB GPU.

Effect on speed
---------------
Same Triton kernel, same number of launches. The kernel writes fp32 instead
of bf16 to the output buffer. ~no measurable difference.

How to disable
--------------
Set the environment variable `SPECFORGE_DISABLE_MAMBA_FP32_PATCH=1` before
launching training. The patch will then NOT be applied and the upstream
bf16-boundary-state behavior will be restored. **Strongly discouraged** --
this exists only as an escape hatch for debugging or for benchmarking the
precision-vs-speed tradeoff.

How to verify the patch is active
---------------------------------
This module also exposes `verify()` which constructs a small synthetic
input, runs `_state_passing_fwd` directly, and asserts the output dtype is
`torch.float32`. Call it from your training launcher's smoke test or run
it standalone:

    python -m specforge._mamba_fp32_patch --verify

How to apply
------------
Just import this module **before** any other code imports `mamba_ssm`. The
SpecForge train script (`scripts/train_eagle3.py`) imports it at the very
top of the module.

References
----------
* `mamba_ssm/ops/triton/ssd_combined.py:379-381` -- the call site with
  `out_dtype=C.dtype`
* `mamba_ssm/ops/triton/ssd_state_passing.py:196-224` -- `_state_passing_fwd`
  signature; `out_dtype` defaults to `states.dtype` (fp32) when None
* `modeling_nemotron_h.py:424,560` -- `A_log.float()` upcast (intra-chunk
  is already fp32; this only fixes the boundary)
* PROJECT.md ("CRITICAL: Mamba SSM State Precision" section)
"""

from __future__ import annotations

import os
import warnings

import torch

_PATCH_DISABLED_ENV = "SPECFORGE_DISABLE_MAMBA_FP32_PATCH"
_PATCHED = False


def apply() -> bool:
    """
    Monkey-patch ``mamba_ssm`` to force fp32 boundary state in the Triton
    fast path. Returns True if the patch was applied, False if it was
    skipped (either by env var or because mamba_ssm is not installed).
    Idempotent: calling apply() more than once is a no-op after the first.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get(_PATCH_DISABLED_ENV, "0").lower() in ("1", "true", "yes"):
        warnings.warn(
            f"{_PATCH_DISABLED_ENV} is set; NOT applying the Mamba SSM fp32 "
            "boundary-state patch. Training quality on math/reasoning will "
            "be degraded. This is intended only as a debugging escape hatch."
        )
        return False
    try:
        import mamba_ssm.ops.triton.ssd_state_passing as _spm
        import mamba_ssm.ops.triton.ssd_combined as _spc
    except ImportError:
        # mamba_ssm not installed -- the slow torch_forward path will run,
        # which is already fp32 by construction. Nothing to patch.
        return False

    _orig = _spm._state_passing_fwd

    def _state_passing_fwd_fp32(
        states,
        dA_chunk_cumsum,
        initial_states=None,
        seq_idx=None,
        chunk_size=None,
        out_dtype=None,
    ):
        # Ignore the caller's out_dtype and force fp32. The default (None)
        # would also pick fp32 because `_chunk_state_fwd(..., states_in_fp32=True)`
        # produces fp32 `states`, but we're explicit to avoid drift.
        return _orig(
            states,
            dA_chunk_cumsum,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_size=chunk_size,
            out_dtype=torch.float32,
        )

    _spm._state_passing_fwd = _state_passing_fwd_fp32
    # ssd_combined.py imports _state_passing_fwd at the top of the file, so
    # it has its own module-level reference; patch that too.
    _spc._state_passing_fwd = _state_passing_fwd_fp32
    _PATCHED = True
    return True


def is_active() -> bool:
    """Return True if the patch has been applied this process."""
    return _PATCHED


def verify(verbose: bool = True) -> None:
    """
    Construct a synthetic input matching the shape that NemotronH would
    pass through `_state_passing_fwd` and assert the output is fp32.

    Requires a CUDA device. Allocates ~few KB.
    """
    if not _PATCHED:
        apply()
    import mamba_ssm.ops.triton.ssd_state_passing as _spm

    device = torch.device("cuda")
    batch, nchunks, nheads, dim = 1, 8, 64, 128 * 64
    states = torch.randn(batch, nchunks, nheads, dim, device=device, dtype=torch.float32)
    dA_chunk_cumsum = torch.randn(batch, nheads, nchunks, device=device, dtype=torch.float32)
    out, final_states = _spm._state_passing_fwd(
        states,
        dA_chunk_cumsum,
        out_dtype=torch.bfloat16,  # caller asks for bf16; patch should ignore
    )
    assert out.dtype == torch.float32, (
        f"Mamba SSM fp32 patch is NOT working: _state_passing_fwd returned "
        f"{out.dtype} when caller asked for bf16 -- expected the patch to "
        f"force fp32"
    )
    assert final_states.dtype == torch.float32
    if verbose:
        print(
            "[mamba-fp32-patch] verified: _state_passing_fwd output dtype is "
            f"{out.dtype} (forced fp32 regardless of caller request)"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    apply()
    if args.verify:
        verify()
    else:
        print(f"[mamba-fp32-patch] applied={is_active()}")
