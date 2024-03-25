import math
from enum import Enum

import torch

from thunder.extend import OperatorExecutor, register_executor
from thunder.executors import triton_utils

# Requires triton 2.1 or greater
min_triton_version = "2.1"

triton_version: None | str = triton_utils.triton_version()
TRITON_AVAILABLE: bool = triton_utils.is_triton_version_at_least(min_triton_version)
assert (
    TRITON_AVAILABLE
), f"Trying to import a Triton executor, but it requires Triton version {min_triton_version} or greater, and the current Triton version is {triton_version}"

triton_ex: OperatorExecutor = OperatorExecutor("triton", version=triton_version)
register_executor(triton_ex)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# Temporarily borrowed from https://github.com/openai/triton
FORWARD_NUM_STAGES = 1


class TritonDtype(Enum):
    kFP16 = 0
    kBF16 = 1
    kFP32 = 2
    kFP64 = 3


_TORCH2DTYPE = {
    torch.float16: TritonDtype.kFP16,
    torch.bfloat16: TritonDtype.kBF16,
    torch.float32: TritonDtype.kFP32,
    torch.float64: TritonDtype.kFP64,
}
_DTYPE2TRITON = {
    TritonDtype.kFP16: tl.float16,
    TritonDtype.kBF16: tl.bfloat16,
    TritonDtype.kFP32: tl.float32,
    TritonDtype.kFP64: tl.float64,
}


@triton.jit
def _class_indices_forward(
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
    LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float("inf")
    l_prev = 0.0
    m_prev = m_prev.to(buffer_dtype)
    l_prev = l_prev.to(buffer_dtype)

    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(
            logit_ptrs,
            mask=cols < N - (start_n * BLOCK),
            other=-float("inf"),
        ).to(buffer_dtype)

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
    if LABEL_SMOOTHING:
        sum_total = 0.0
        sum_total = sum_total.to(buffer_dtype)
        weights_total = 0.0
        weights_total = weights_total.to(buffer_dtype)
        if WEIGHTS:
            weight_ptr = weight + cols

    l_prev_log = tl.log(l_prev)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(
            logit_ptrs,
            mask=cols < N - start_n * BLOCK,
            other=l_prev_log + m_prev,
        ).to(buffer_dtype)
        if LABEL_SMOOTHING and WEIGHTS:
            full_weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            weights_total += tl.sum(full_weights_val, 0)

        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max

        if LABEL_SMOOTHING and WEIGHTS:
            log_softmax *= full_weights_val

        if LABEL_SMOOTHING:
            sum_total += tl.sum(log_softmax, 0)
        # Store it back

        tl.store(
            WRIT_PROBS,
            log_softmax,
            mask=cols < N - start_n * BLOCK,
        )
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        if LABEL_SMOOTHING and WEIGHTS:
            weight_ptr += BLOCK

    idx = tl.load(IDX + row)
    use_class = 0.0
    if IGNORE_INDEX >= 0:
        use_class = idx == IGNORE_INDEX
    READ_PROBS = PROBS + row * N + idx
    tl.debug_barrier()
    # write-back loss
    probs = tl.load(READ_PROBS)
    if WEIGHTS and not LABEL_SMOOTHING:
        weight_ptr = weight + idx
        weights_val = tl.load(weight_ptr)
        probs = weights_val * probs
    if LABEL_SMOOTHING:
        tl.store(WEIGHT_BUFFER + row, weights_total)
        probs = (1 - smoothing_factor) * probs + smoothing_factor * (sum_total) / N
    probs = probs * (1.0 - use_class)

    tl.store(LOSS + row, probs)


@triton.jit
def _class_probs_forward(
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
    LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float("inf")
    l_prev = 0.0
    m_prev = m_prev.to(buffer_dtype)
    l_prev = l_prev.to(buffer_dtype)

    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(
            logit_ptrs,
            mask=cols < N - (start_n * BLOCK),
            other=-float("inf"),
        ).to(buffer_dtype)

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

    sum_total = 0.0
    weights_total = 0.0
    sum_total = sum_total.to(buffer_dtype)
    weights_total = weights_total.to(buffer_dtype)
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
        ).to(buffer_dtype)
        idx = tl.load(idx_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
        if WEIGHTS:
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        else:
            full_weights_val = tl.where(cols < N - start_n * BLOCK, full_weights_val, 0.0)
        weights_total += tl.sum(full_weights_val, 0)

        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max

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
        idx_ptr += BLOCK
        if WEIGHTS:
            weight_ptr += BLOCK

    tl.store(WEIGHT_BUFFER + row, weights_total)
    probs = sum_total

    tl.store(LOSS + row, probs)


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
        "BUFFER_DTYPE",
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
    LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if CLASS_INDICES:
        _class_indices_forward(
            LOGITS,
            PROBS,
            IDX,
            LOSS,
            weight,
            N,
            WEIGHT_BUFFER,
            smoothing_factor,
            log_size_logits,
            WEIGHTS,
            CLASS_INDICES,
            LABEL_SMOOTHING,
            IGNORE_INDEX,
            BUFFER_DTYPE,
            BLOCK,
        )
    else:
        _class_probs_forward(
            LOGITS,
            PROBS,
            IDX,
            LOSS,
            weight,
            N,
            WEIGHT_BUFFER,
            smoothing_factor,
            log_size_logits,
            WEIGHTS,
            CLASS_INDICES,
            LABEL_SMOOTHING,
            IGNORE_INDEX,
            BUFFER_DTYPE,
            BLOCK,
        )


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
        "BUFFER_DTYPE",
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
    LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
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
    ).to(buffer_dtype)
    DIN = DIN + row * N + cols + BLOCK * start_n
    dout = tl.load(DPROBS + row * dprob_stride).to(buffer_dtype)
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    if CLASS_INDICES:
        idx = tl.load(IDX + row)
        delta = ((start_n * BLOCK) + cols) == idx
        # write result in-place in PROBS
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
            dout = dout * (1 - use_class)
        if LABEL_SMOOTHING:
            if WEIGHTS:
                weight_ptr = weight + cols + BLOCK * start_n
                full_weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0).to(buffer_dtype)
                weights_val = tl.load(weight + idx)
                probs = probs / full_weights_val
            probs = tl.exp(probs)
            if WEIGHTS:
                weights_total = tl.load(WEIGHT_BUFFER + row)
                numerator_contrib = weights_val * (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = ((weights_total * probs) - (full_weights_val)) * smoothing_factor / N
            else:
                numerator_contrib = (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = (smoothing_factor * probs) - (smoothing_factor / N)

            din = (numerator_contrib + mean_contrib) * dout

        else:
            probs = tl.exp(probs)
            din = (probs - delta) * dout
            if WEIGHTS:
                weight_ptr = weight + idx
                weights_val = tl.load(weight_ptr)
                din = weights_val * din
    else:
        idx = tl.load(
            IDX + row * N + cols + BLOCK * start_n,
            mask=cols < N - start_n * BLOCK,
            other=0.0,
        ).to(buffer_dtype)
        full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
        weights_total = tl.load(WEIGHT_BUFFER + row)
        if WEIGHTS:
            weight_ptr = weight + cols + BLOCK * start_n
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0).to(buffer_dtype)
            full_weights_val = weights_val * full_weights_val
        probs = probs / full_weights_val
        probs = tl.exp(probs.to(buffer_dtype))
        weighted_probs = probs * weights_total
        weighted_probs_per_class = weighted_probs - full_weights_val
        din = (weighted_probs_per_class) * dout

    tl.store(DIN, din.to(DIN.dtype.element_ty), mask=cols + BLOCK * start_n < N)


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        indices,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
    ):
        buffer_dtype = None
        # make sure we can use triton
        # assert (
        #     indices.dtype == torch.int64
        # ), "Indices are expected to be of type long."
        assert weight is None or (len(weight.shape) == 1 and weight.shape[0] == logits.shape[-1])
        # make kernel
        if buffer_dtype is None:
            if logits.dtype in [torch.bfloat16, torch.float16]:
                buffer_dtype = torch.float32
            else:
                buffer_dtype = logits.dtype
        buffer_dtype_enum = _TORCH2DTYPE[buffer_dtype]
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        # run the kernel
        result = torch.empty((logits.shape[0],), dtype=dtype, device=device)
        # result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=buffer_dtype, device=device)
        weights_buffer = torch.empty_like(result, dtype=buffer_dtype)
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
            LABEL_SMOOTHING=(label_smoothing > 0.0),
            IGNORE_INDEX=ignore_index,
            BUFFER_DTYPE=buffer_dtype_enum,
        )
        # save for backward
        ctx.save_for_backward(neg_logprobs, indices, weights_buffer)
        ctx.WEIGHT = weight
        ctx.label_smoothing = label_smoothing
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.buffer_dtype = buffer_dtype_enum
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
            return (result.sum(dim=0) / denom).to(dtype)

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
        buffer_dtype = ctx.buffer_dtype
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
            LABEL_SMOOTHING=(ctx.label_smoothing > 0.0),
            IGNORE_INDEX=ctx.ignore_index,
            BUFFER_DTYPE=buffer_dtype,
        )
        if ctx.reduction == "mean":
            din /= ctx.denom
        return din, None, None, None, None, None, None


def cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
):
    r"""
    Returns the Cross Entropy loss of input. If the target is class indcies
    then the ignore_index argument is applicable, while the label_smoothing argument
    is not.  On the other hand, if the target is class probabilites, then the
    label_smoothing argument is applicable, while the ignore_index argument is not.

    Args:
        input: Tensor of shape (B, N)
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
    return CrossEntropy.apply(
        input,
        target,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
    )


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
    if torch_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
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


import thunder.torch as ltorch

ce = triton_ex.register_operator("triton_crossentropy", like=ltorch.cross_entropy, fn=cross_entropy_impl)
triton_ex.register_implementation(ltorch.cross_entropy, ce, checker=cross_entropy_checker)
