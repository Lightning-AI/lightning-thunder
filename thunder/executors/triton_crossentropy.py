import math

import torch

import thunder.torch as ltorch

# TODO Make a "triton_utils" and create a function that replaces this
#   by checking for availability + a minimum version required for Triton
from lightning_utilities.core.imports import package_available

TRITON_AVAILABLE = package_available("triton")
assert TRITON_AVAILABLE, f"Attempting to import a Triton executor but Triton is not available"

import triton
import triton.language as tl

# Temporarily borrowed from https://github.com/openai/triton
FORWARD_NUM_STAGES = 1


@triton.autotune(
    configs=[
        # fmt: off
        triton.Config({'BLOCK': 1024}, num_stages=FORWARD_NUM_STAGES, num_warps=1),
        triton.Config({'BLOCK': 2048}, num_stages=FORWARD_NUM_STAGES, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_stages=FORWARD_NUM_STAGES, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_stages=FORWARD_NUM_STAGES, num_warps=16),
        triton.Config({'BLOCK': 16384}, num_stages=FORWARD_NUM_STAGES, num_warps=16),
        # fmt: on
    ],
    key=[
        "N",
        "CLASS_INDICES",
        "log_size_logits",
    ],
)
@triton.jit
def _forward(
    LOGITS,
    PROBS,
    IDX,
    LOSS,
    weight,
    N,
    WEIGHT_BUFFER,
    smoothing_factor,
    log_size_logits,
    WEIGHTS: tl.constexpr,
    CLASS_INDICES: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float("inf")
    l_prev = 0.0
    m_prev = m_prev.to(LOGITS.dtype.element_ty)
    l_prev = l_prev.to(LOGITS.dtype.element_ty)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(
            logit_ptrs,
            mask=cols < N - (start_n * BLOCK),
            other=-float("inf"),
        )

        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    output_ptrs = PROBS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    if not CLASS_INDICES:
        sum_total = 0.0
        weights_total = 0.0
        sum_total = sum_total.to(LOGITS.dtype.element_ty)
        weights_total = weights_total.to(LOGITS.dtype.element_ty)
        idx_ptr = IDX + row * N + cols
        if WEIGHTS:
            weight_ptr = weight + cols

    l_prev_log = tl.log(l_prev)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(
            logit_ptrs,
            mask=cols < N - start_n * BLOCK,
            other=l_prev_log + m_prev,
        )
        if not CLASS_INDICES:
            idx = tl.load(idx_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
            if WEIGHTS:
                weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
                full_weights_val = weights_val * full_weights_val
            weights_total += tl.sum(full_weights_val, 0)

        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max

        if not CLASS_INDICES:
            log_softmax *= full_weights_val
            sum_total += tl.sum(log_softmax, 0)
        # Store it back

        tl.store(
            WRIT_PROBS,
            log_softmax,
            mask=cols < N - start_n * BLOCK,
        )
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        if not CLASS_INDICES:
            idx_ptr += BLOCK
            if WEIGHTS:
                weight_ptr += BLOCK

    if CLASS_INDICES:
        idx = tl.load(IDX + row)
        use_class = 0.0
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
        READ_PROBS = PROBS + row * N + idx
        tl.debug_barrier()
        # write-back loss
        probs = tl.load(READ_PROBS)
        probs = probs * (1.0 - use_class)
        if WEIGHTS:
            weight_ptr = weight + idx
            weights_val = tl.load(weight_ptr)
            probs = weights_val * probs
    else:
        # Need to load all the indices and smooth it.
        tl.store(WEIGHT_BUFFER + row, weights_total)
        probs = sum_total

    tl.store(LOSS + row, probs)


@triton.autotune(
    configs=[
        # fmt: off
        triton.Config({'BLOCK': 1024}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK': 2048}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_stages=1, num_warps=16),
        triton.Config({'BLOCK': 16384}, num_stages=1, num_warps=16),
        # fmt: on
    ],
    key=[
        "N",
        "CLASS_INDICES",
        "log_size_logits",
    ],
)
@triton.jit
def _backward(
    PROBS,
    IDX,
    DPROBS,
    dprob_stride,
    DIN,
    weight,
    N,
    WEIGHT_BUFFER,
    smoothing_factor,
    log_size_logits,
    WEIGHTS: tl.constexpr,
    CLASS_INDICES: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    PROBS = PROBS + row * N
    # pointers to probs
    probs_start = PROBS + cols + BLOCK * start_n
    # for start_n in range(0, tl.cdiv(N, BLOCK)):  # need to change this
    probs = -tl.load(
        probs_start,
        mask=cols < N - (start_n * BLOCK),
        other=float("inf"),
    )
    DIN = DIN + row * N + cols + BLOCK * start_n
    dout = tl.load(DPROBS + row * dprob_stride)
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    if CLASS_INDICES:
        probs = tl.exp(probs.to(tl.float32))
        idx = tl.load(IDX + row)
        delta = ((start_n * BLOCK) + cols) == idx
        # write result in-place in PROBS
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
            dout = dout * (1 - use_class)
        din = (probs - delta) * dout
        if WEIGHTS:
            weight_ptr = weight + idx
            weights_val = tl.load(weight_ptr)
            din = weights_val * din
    else:
        weights_total = tl.load(WEIGHT_BUFFER + row)
        idx = tl.load(
            IDX + row * N + cols + BLOCK * start_n,
            mask=cols < N - start_n * BLOCK,
            other=0.0,
        )
        full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
        if WEIGHTS:
            weight_ptr = weight + cols + BLOCK * start_n
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        probs = probs / full_weights_val
        probs = tl.exp(probs.to(tl.float32))
        weighted_probs = probs * weights_total
        weighted_probs_per_class = weighted_probs - full_weights_val
        din = (weighted_probs_per_class) * dout

    tl.store(DIN, din.to(DIN.dtype.element_ty), mask=cols + BLOCK * start_n < N)


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, indices, weight, ignore_index, reduction, label_smoothing):
        # make sure we can use triton
        # assert (
        #     indices.dtype == torch.int64
        # ), "Indices are expected to be of type long."
        assert weight is None or (len(weight.shape) == 1 and weight.shape[0] == logits.shape[-1])
        # make kernel
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        # run the kernel
        result = torch.empty((logits.shape[0],), dtype=dtype, device=device)
        # result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
        weights_buffer = torch.empty_like(result)
        grid = lambda opt: (logits.numel() // n_cols,)
        log_size_logits = int(math.log(math.prod(logits.shape) / n_cols))
        _forward[grid](
            logits,
            neg_logprobs,
            indices,
            result,
            weight,
            n_cols,
            weights_buffer,
            label_smoothing,
            log_size_logits,
            WEIGHTS=(weight is not None),
            CLASS_INDICES=(indices.dtype == torch.int64),
            IGNORE_INDEX=ignore_index,
        )
        # save for backward
        ctx.save_for_backward(neg_logprobs, indices, weights_buffer)
        ctx.WEIGHT = weight
        ctx.label_smoothing = label_smoothing
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        if reduction == "none":
            return result
        elif reduction == "sum":
            return result.sum(dim=0)
        elif reduction == "mean":
            if indices.dtype == torch.int64:
                denom = (indices != ignore_index).float()
                if weight is not None:
                    class_weights = weight[indices]
                    denom *= class_weights
                denom = denom.sum()
            else:
                denom = indices.shape[0]
            ctx.denom = denom
            return result.sum(dim=0) / denom

    @staticmethod
    def backward(ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        # load saved tensors
        reduction = ctx.reduction
        if reduction == "mean" or reduction == "sum":
            dneg_logprobs = dneg_logprobs.expand(1)
        neg_logprobs, indices, weights_buffer = ctx.saved_tensors
        din = torch.empty_like(neg_logprobs)
        weight = ctx.WEIGHT
        # run the kernel
        # neg_logprobs will be modified in place to become our gradient:
        n_cols = neg_logprobs.shape[-1]
        grid = lambda opt: (
            neg_logprobs.numel() // n_cols,
            triton.cdiv(n_cols, opt["BLOCK"]),
        )
        log_size_logits = int(math.log(math.prod(neg_logprobs.shape) / n_cols))
        _backward[grid](
            neg_logprobs,
            indices,
            dneg_logprobs,
            dneg_logprobs.stride(0),
            din,
            weight,
            n_cols,
            weights_buffer,
            ctx.label_smoothing,
            log_size_logits,
            WEIGHTS=(weight is not None),
            CLASS_INDICES=(indices.dtype == torch.int64),
            IGNORE_INDEX=ctx.ignore_index,
        )
        if ctx.reduction == "mean":
            din /= ctx.denom
        return din, None, None, None, None, None


def cross_entropy(
    a,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
):
    r"""
    Returns the Cross Entropy loss of a. If the target is class indcies
    then the ignore_index argument is applicable, while the label_smoothing argument
    is not.  On the other hand, if the target is class probabilites, then the
    label_smoothing argument is applicable, while the ignore_index argument is not.

    Args:
        a: Tensor of shape (B, N)
            where B is the batch dim and N is the number of classes
        target: Int Tensor of shape (B,), min = 0, max = N-1 or
            Float Tensor of shape (B, N), rows sum to 1.0
            Int tensor of class labels.
        weight: Optional, Float Tensor of shape (N,)
            Weight to scale each class
        ignore_index: Int, which class label should be ignored
        reduction: String: ['none', 'sum', 'mean']
        label_smoothing: Float between 0 and 1
    """
    return CrossEntropy.apply(a, target, weight, ignore_index, reduction, label_smoothing)


# TODO: What is correct handling of ignore_index?
def cross_entropy_impl(
    a,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    loss = cross_entropy(a, target, weight, ignore_index, reduction, label_smoothing)

    return loss


def cross_entropy_checker(
    a,
    /,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
) -> bool:
    if triton is None:
        return False

    torch_dtype = ltorch.to_torch_dtype(a.dtype)
    if torch_dtype not in (torch.float32, torch.float64):
        return False

    # These arguments are deprecated and not supported
    if size_average is not None or reduce is not None:
        return False

    # We only support reduction of "sum", "mean" or "none"
    if reduction not in ["sum", "mean", "none"]:
        return False

    if len(a.shape) != 2:
        return False

    return True


_op_to_xentropy = {
    "torch.nn.functional.cross_entropy": ("triton_cross_entropy", cross_entropy_checker, cross_entropy_impl),
}


def register_triton_entropyex(*, add_to_default_executors: bool = True) -> None:
    from thunder.executors import add_operator_executor

    return add_operator_executor(
        "triton_crossentropy", _op_to_xentropy, add_to_default_executors=add_to_default_executors
    )


def deregister_triton_entropyex() -> None:
    from thunder.executors import remove_operator_executor

    return remove_operator_executor("triton_crossentropy")
