from functools import partial
from typing import Any

import pytest
import torch
from looseversion import LooseVersion
from torch.testing import assert_close

import thunder
import thunder.core.devices as devices
from thunder import dtypes
from thunder.core.transforms import vjp
from thunder.core.utils import flatten_func
from thunder.tests.framework import instantiate, NOTHING, ops, requiresCUDA, run_snippet, TorchExecutor
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.opinfos import get_opinfo, OpInfo
from thunder.tests.test_grad import _make_differentiable_wrapper

cudnn = pytest.importorskip("cudnn")
from thunder.executors.cudnn_layernormex import cudnn_layernorm_ex
from thunder.executors.cudnnex import cudnn_ex, cudnn_version


# These reference inputs are currently used by cudnnex
def grad_scaled_dot_product_attention_reference_generator(op, device, dtype, requires_grad, **kwargs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    from thunder.executors.sdpaex import SpdaBackend
    from thunder.tests.opinfos import SampleInput

    # TODO: cudnnex seems to produce large mismatches against reference when tensor initialized from the wider default range of [-9,9]
    # See issue "cuDNN SDPA backward might return NaNs for inputs with absolute
    # value more than certain threshold"
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-0.5, high=0.5)

    n_head = 2
    N = 8  # batch size
    L = 640  # query's sequence length
    S = 80  # key/value's sequence length
    E = 128  # query/key's embedding size
    Ev = 64  # value's embedding size

    # 4-dim (multiheaded) causal cases
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=True)

    # Same sequence length and embedding size for Q, K and V, a common use case.
    yield SampleInput(make(N, n_head, L, E), make(N, n_head, L, E), make(N, n_head, L, E), None, is_causal=True)

    # Non-contiguous input tensor case
    nq = make(N, n_head, E, L).permute(0, 1, 3, 2)
    nk = make(N, n_head, E, S).permute(0, 1, 3, 2)
    nv = make(N, n_head, Ev, S).permute(0, 1, 3, 2)
    yield SampleInput(nq, nk, nv, None, dropout_p=0.0, is_causal=False)

    # Test the scale factor which was added in torch 2.1
    if LooseVersion(torch.__version__) >= LooseVersion("2.1.0"):
        q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
        yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=False, scale=0.123)

    # TODO: cudnnex only support of grad_attn_mask with batch dim 1 and both sequence lenghts divisible by 64. Release 9.0.1 will relax this constraint.
    # Additive attn_mask
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    additive_attn_mask = make((1, n_head, L, S), dtype=q.dtype).tril()
    yield SampleInput(q, k, v, additive_attn_mask, is_causal=False)

    # Boolean attn_mask
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    bool_attn_mask = make((1, n_head, L, S), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    yield SampleInput(q, k, v, bool_attn_mask, is_causal=False)


grad_sdpa_cudnn_opinfo = OpInfo(
    thunder.torch.scaled_dot_product_attention,
    name="grad_forward_scaled_dot_product_attention",
    sample_input_generator=None,
    reference_input_generator=grad_scaled_dot_product_attention_reference_generator,
    torch_reference=torch.nn.functional.scaled_dot_product_attention,
    # RuntimeError: Only half & bf16 supported at the moment
    dtypes=(
        thunder.dtypes.float16,
        thunder.dtypes.bfloat16,
    ),
    devicetypes=(devices.DeviceType.CUDA,),
)


@requiresCUDA
def test_cudnn_sdpa():
    # expect sdpa to fail for 8.9.2 and below
    if cudnn.backend_version() <= 8902:
        pytest.xfail("Only interleaved layout is supported pre 8.9.2.")

    dev: torch.device = thunder.core.devices.to_torch_device("cuda:0")
    cuda_major: int
    cuda_minor: int
    cuda_major, cuda_minor = torch.cuda.get_device_capability(dev)
    if cuda_major < 8:
        pytest.xfail("cuDNN SDPA uses flash attention, which requires Ampere+")

    for dtype in (thunder.float16, thunder.bfloat16):
        b, h, s_q, s_kv, d_q, d_v = 8, 8, 256, 256, 64, 64
        shape_Q = (b, h, s_q, d_q)
        shape_K = (b, h, s_kv, d_q)
        shape_V = (b, h, s_q, d_v)

        query = 1 * (torch.randn(shape_Q, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        key = 2 * (torch.randn(shape_K, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        value = 3 * (torch.randn(shape_V, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        is_causal = True
        attn_mask = None

        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, attn_mask=attn_mask
        )

        def test(query, key, value, is_causal=False, attn_mask=None):
            return thunder.torch.scaled_dot_product_attention(
                query, key, value, is_causal=is_causal, attn_mask=attn_mask
            )

        ctest = thunder.jit(test, executors=[cudnn_ex])
        actual = ctest(query, key, value, is_causal=is_causal, attn_mask=attn_mask)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
        last_trace = thunder.last_traces(ctest)[-1]
        assert any(bsym.sym.name == "cudnn_sdpa_fwd" for bsym in last_trace.bound_symbols)


# NOTE These wrappers are necessary because preprocessing cannot compile the function directly
def layer_norm_wrapper(*args, **kwargs):
    return thunder.torch.layer_norm(*args, **kwargs)


def spda_wrapper(*args, **kwargs):
    return thunder.torch.scaled_dot_product_attention(*args, **kwargs)


op_name_to_fn = {
    "layer_norm": layer_norm_wrapper,
    "scaled_dot_product_attention": spda_wrapper,
    "grad_forward_scaled_dot_product_attention": spda_wrapper,
}


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    # TODO: Pass a custom_comparator which has higher tol for bf16 cases
    assert_close(thunder_result, torch_result, equal_nan=True, atol=0.0625, rtol=5e-2)


# TODO Make it easier for executors to write tests like this, including writing them out-of-tree
# TODO The executor passed below is just a "dummy" that actually gets ignored -- we should provide
#   a way to use decorators like @ops without a particular executor
@ops(
    (
        get_opinfo("layer_norm"),
        get_opinfo("scaled_dot_product_attention"),
        grad_sdpa_cudnn_opinfo,
    ),
    supported_devicetypes=(devices.DeviceType.CUDA,),
    supported_dtypes=(dtypes.float16, dtypes.bfloat16),
    supported_executors=(TorchExecutor,),
)
def test_cudnn_vs_torch_consistency(op, device, dtype, *_):
    if cudnn.backend_version() < 8905:  # todo: could be more specific, just for some cases?
        pytest.xfail("s_kv not a multiple of 64 required cudnn version atleast 8.9.5")

    # expect layer_norm to fail for 8.9.3 and below
    if op.name == "layer_norm":
        if cudnn.backend_version() <= 8903:
            pytest.xfail("Only fp32 weight/bias supported pre 8.9.3.")

    # expect sdpa to fail for 8.9.2 and below
    if op.name == "scaled_dot_product_attention":
        if cudnn.backend_version() <= 8902:
            pytest.xfail("Only interleaved layout is supported pre 8.9.2.")

    for sample in op.reference_inputs(device, dtype, requires_grad=False):
        cfn = thunder.jit(op_name_to_fn[op.name], executors=[cudnn_ex, cudnn_layernorm_ex])

        result = run_snippet(
            snippet_torch_consistency,
            op,
            device,
            dtype,
            cfn,
            op.torch_reference,
            sample,
        )
        if result is not None:
            return result


@pytest.mark.skipif(cudnn_version() < 8905, reason="cuDNN is required to be at least `8.9.5`")
@pytest.mark.parametrize("may_cat_grad_qkv", (True, False), ids=("may-cat-grad-qkv", "never-cat-grad-qkv"))
@pytest.mark.parametrize("dtype", grad_sdpa_cudnn_opinfo.dtypes(), ids=tuple(map(str, grad_sdpa_cudnn_opinfo.dtypes())))
def test_vjp_correctness_cudnn_sdpa(dtype, may_cat_grad_qkv):
    for sample in grad_sdpa_cudnn_opinfo.reference_inputs("cuda", dtype, requires_grad=True):
        # Enforce tensor arguments are contiguous for torch reference
        contiguous_args = list(map(lambda a: a.contiguous() if isinstance(a, torch.Tensor) else a, sample.args))

        # query, key, value
        grad_inputs = list(contiguous_args[:3])
        if (attn_mask := sample.args[3]) is not None:
            if attn_mask.requires_grad:
                grad_inputs.append(attn_mask)
            # TODO(#2470): With cudnn frontend 1.1 and A100, this test hits
            # RuntimeError when `attn_mask` is provided: `[cudnn_frontend]
            # Error: No execution plans built successfully`.
            continue

        # Compute vjp result using PyTorch
        expect_out = grad_sdpa_cudnn_opinfo.torch_reference(*contiguous_args, **sample.kwargs)
        v = make_tensor_like(expect_out)
        expected_grad = torch.autograd.grad(expect_out, grad_inputs, v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(grad_sdpa_cudnn_opinfo.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)

        cfoo = thunder.compile(
            vjp(filtered_op),
            disable_torch_autograd_support=True,
            disable_preprocessing=True,
            executors_list=[cudnn_ex],
            cudnn_sdpa_bwd_may_cat_grad_qkv=may_cat_grad_qkv,
        )

        actual_out, actual_grad = cfoo(filtered_args, (v,))

        torch.testing.assert_close(actual_out, expect_out, atol=1e-2, rtol=1e-2)
        # compare gradients of query, key, value, and attn_mask
        for eg, ag in zip(expected_grad, actual_grad):
            torch.testing.assert_close(eg, ag, atol=2e-1, rtol=2e-2)
