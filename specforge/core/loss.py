"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.
The idea of in-place backward pass is from Liger-Kernel.
See the original Liger-Kernel repository at https://github.com/linkedin/Liger-Kernel.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


# Reference implementation
@torch.compile(dynamic=None)
def _compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    return loss


def _calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 131072
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    # AMD GPU (ROCm)
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        num_warps //= 2

    return BLOCK_SIZE, num_warps


@triton.jit
def log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

    loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        # log-softmax: log(exp(x - max) / sum) = (x - max) - log(sum)
        normalized_logits = logits_block - m
        log_normalizer = tl.log(d)
        log_softmax_logits = normalized_logits - log_normalizer
        weighted_log_prob = target_block * log_softmax_logits
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, -loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def log_softmax_backward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad_output = grad_output * scaling_factor

    # First pass: compute sum of (target * grad_output)
    target_grad_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_grad_sum += tl.sum(tl.where(mask, target_block * grad_output, 0.0))

    # Second pass: compute log-softmax gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_block * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class LogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device)
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d)
        return loss.squeeze(1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / (B * T)
        logits = logits.contiguous().view(B * T, V)
        target = target.contiguous().view(B * T, V)
        position_mask = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_backward_kernel[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            grad_output,
            scaling_factor,
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        logits = logits.view(B, T, V)
        return logits, None, None, None, None


def fused_linear_log_softmax_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_p: torch.Tensor,
    position_mask: torch.Tensor,
    *,
    lm_head_bias: torch.Tensor | None = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Memory-efficient fused (linear + soft-target log-softmax cross-entropy)
    loss for the Eagle3 draft head.

    Computes the same value (up to floating-point error) as

        logits = F.linear(hidden_states, lm_head_weight, lm_head_bias)
        loss   = LogSoftmaxLoss.apply(logits, target_p, position_mask)

    but never materializes the full ``[B, T, V]`` logits tensor in
    memory. Instead, the sequence dimension is split into chunks of
    ``chunk_size`` positions; each chunk's `(linear, log_softmax,
    soft-target CE)` is computed inside ``torch.utils.checkpoint`` so
    that during the forward pass only one chunk's logits exist (~250 MB
    at chunk_size=4096, V=32k, bf16) and during the backward pass each
    chunk's logits are recomputed sequentially.

    The mathematics are exact: the loss is a sum over independent
    positions

        L = (1/N) * sum_t mask_t * (-sum_v target_p[t,v] * log p_draft(t,v))

    where ``N = B*T``, so chunking the outer sum is bit-equivalent to
    the unchunked computation up to floating-point reduction order.
    See the ``__main__`` block at the bottom of this file for a
    numerical equivalence test.

    At long context (L=65k, V=32k, ttt_length=7) this saves
    ``7 * (B*T*V*2 - chunk_size*V*2) = ~28 GiB`` of activation memory
    relative to the unchunked path -- the difference between fitting
    on a 96 GB GPU and OOMing.

    Args:
        hidden_states: ``[B, T, H]`` draft model output (after the
            final RMSNorm). Requires_grad=True for training.
        lm_head_weight: ``[V, H]`` draft model lm_head weight.
        target_p: ``[B, T, V]`` verifier soft-target distribution
            (already softmaxed and ``t2d``-remapped to draft vocab).
        position_mask: ``[B, T, 1]`` integer/bool mask of which
            positions contribute to the loss.
        lm_head_bias: optional ``[V]`` lm_head bias. SpecForge's
            LlamaForCausalLMEagle3 lm_head has bias=False, so this
            usually stays None.
        chunk_size: positions per chunk. 4096 is a good default --
            smaller chunks reduce per-chunk peak memory at the cost of
            more kernel launches.

    Returns:
        Scalar loss tensor (gradient-tracked w.r.t. hidden_states and
        lm_head_weight).
    """
    B, T, H = hidden_states.shape
    V = lm_head_weight.shape[0]
    if T == 0:
        return hidden_states.new_zeros((), requires_grad=True)
    chunk_size = max(1, min(chunk_size, T))

    def _chunk_loss(h_chunk, target_chunk, mask_chunk, weight, bias):
        # h_chunk:      [B, chunk, H]
        # target_chunk: [B, chunk, V]
        # mask_chunk:   [B, chunk, 1]
        logits_chunk = torch.nn.functional.linear(h_chunk, weight, bias)
        # LogSoftmaxLoss returns mean over (B*chunk). Multiply back by
        # (B*chunk) so the per-chunk values are sums; the outer loop
        # then divides by (B*T) to recover the same mean the unchunked
        # path computes.
        chunk_mean = LogSoftmaxLoss.apply(logits_chunk, target_chunk, mask_chunk)
        return chunk_mean * (h_chunk.shape[0] * h_chunk.shape[1])

    total_loss_sum = hidden_states.new_zeros(())
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_chunk = hidden_states[:, start:end, :].contiguous()
        target_chunk = target_p[:, start:end, :].contiguous()
        mask_chunk = position_mask[:, start:end, :].contiguous()

        # Wrap each chunk in checkpoint so the per-chunk logits tensor
        # is never persisted between forward and backward.
        chunk_loss_sum = torch.utils.checkpoint.checkpoint(
            _chunk_loss,
            h_chunk,
            target_chunk,
            mask_chunk,
            lm_head_weight,
            lm_head_bias,
            use_reentrant=False,
        )
        total_loss_sum = total_loss_sum + chunk_loss_sum

    return total_loss_sum / (B * T)


@torch.no_grad()
def fused_linear_argmax_correct_count(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_p: torch.Tensor,
    position_mask: torch.Tensor,
    *,
    lm_head_bias: torch.Tensor | None = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Counterpart to ``fused_linear_log_softmax_loss``: computes the
    chunked argmax-equality "accuracy correct count" without
    materializing the full ``[B, T, V]`` logits tensor.

    Returns the unreduced number of positions where
    ``argmax(linear(h)) == argmax(target_p)`` AND the position mask
    is set, as a 0-dim float tensor. The caller is responsible for
    dividing by the denominator (``loss_mask.sum()``) to get the
    accuracy.

    Runs in ``no_grad`` -- this is purely a metric.
    """
    B, T, H = hidden_states.shape
    if T == 0:
        return hidden_states.new_zeros(())
    chunk_size = max(1, min(chunk_size, T))

    target_argmax = target_p.argmax(-1)  # [B, T]  -- typically tiny
    position_mask_2d = position_mask.squeeze(-1)  # [B, T]

    correct = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_chunk = hidden_states[:, start:end, :]
        logits_chunk = torch.nn.functional.linear(h_chunk, lm_head_weight, lm_head_bias)
        argmax_chunk = logits_chunk.argmax(-1)
        del logits_chunk
        chunk_correct = (
            (argmax_chunk == target_argmax[:, start:end])
            * position_mask_2d[:, start:end]
        ).float().sum()
        correct = correct + chunk_correct

    return correct


if __name__ == "__main__":
    device = "cuda"

    # ----- Existing test: triton kernel == reference torch.compile path
    B, T, V = 1, 1024, 16000
    logits = torch.randn(B, T, V, device=device, requires_grad=True)
    logits2 = logits.clone().detach().requires_grad_(True)
    target = torch.randn(B, T, V, device=device)
    position_mask = torch.randint(0, 2, (B, T, 1), dtype=torch.bool, device=device)
    position_mask = torch.ones((B, T, 1), dtype=torch.bool, device=device)
    output1 = LogSoftmaxLoss.apply(logits, target, position_mask)
    output2 = _compute_loss(logits2, target, position_mask)
    torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
    output1.backward()
    output2.backward()
    torch.testing.assert_close(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)
    print("[loss-test] LogSoftmaxLoss vs torch reference: OK")

    # ----- New test: chunked fused linear loss == unchunked path
    # Realistic-ish shapes for the Eagle3 draft head.
    torch.manual_seed(0)
    B, T, H, V = 1, 2048, 2688, 32000
    chunk_size = 512  # use a small chunk so multiple chunks exercise the loop

    hidden = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1
    weight = torch.randn(V, H, device=device, dtype=torch.float32) * 0.05
    target_logits = torch.randn(B, T, V, device=device, dtype=torch.float32) * 0.1
    target_p = torch.softmax(target_logits, dim=-1)
    pos_mask = torch.randint(0, 2, (B, T, 1), dtype=torch.long, device=device)

    # ---- Reference: unchunked compute
    hidden_ref = hidden.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    logits_ref = torch.nn.functional.linear(hidden_ref, weight_ref)
    loss_ref = LogSoftmaxLoss.apply(logits_ref, target_p, pos_mask)
    loss_ref.backward()

    # ---- Chunked path
    hidden_ck = hidden.clone().detach().requires_grad_(True)
    weight_ck = weight.clone().detach().requires_grad_(True)
    loss_ck = fused_linear_log_softmax_loss(
        hidden_ck,
        weight_ck,
        target_p,
        pos_mask,
        chunk_size=chunk_size,
    )
    loss_ck.backward()

    print(
        f"[loss-test] chunked vs unchunked loss: ref={loss_ref.item():.6e} "
        f"chunked={loss_ck.item():.6e}"
    )
    torch.testing.assert_close(loss_ref, loss_ck, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hidden_ref.grad, hidden_ck.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight_ref.grad, weight_ck.grad, rtol=1e-4, atol=1e-5)
    print("[loss-test] fused_linear_log_softmax_loss equivalence: OK")

    # ---- Sanity test for the argmax counter
    correct_chunked = fused_linear_argmax_correct_count(
        hidden, weight, target_p, pos_mask, chunk_size=chunk_size
    )
    with torch.no_grad():
        logits_full = torch.nn.functional.linear(hidden, weight)
        correct_ref = (
            (logits_full.argmax(-1) == target_p.argmax(-1)) * pos_mask.squeeze(-1)
        ).float().sum()
    torch.testing.assert_close(correct_chunked, correct_ref, rtol=0, atol=0)
    print(
        f"[loss-test] fused_linear_argmax_correct_count equivalence: "
        f"OK ({int(correct_ref.item())} correct)"
    )
