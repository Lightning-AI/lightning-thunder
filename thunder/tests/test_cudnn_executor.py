from functools import partial

import pytest
import torch
from looseversion import LooseVersion
from torch.testing import assert_close

import thunder
import thunder.core.devices as devices
from thunder import dtypes
from thunder.core.transforms import vjp
from thunder.core.utils import flatten_func
from thunder.tests.framework import ops, requiresCUDA, run_snippet, TorchExecutor
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.opinfos import get_opinfo, OpInfo, SampleInput
from thunder.tests.test_grad import _make_differentiable_wrapper

cudnn = pytest.importorskip("cudnn")
from thunder.executors.cudnn_layernormex import cudnn_layernorm_ex
from thunder.executors.cudnnex import cudnn_ex


def _maybe_xfail() -> None:
    dev: torch.device = thunder.core.devices.to_torch_device("cuda:0")
    cuda_major: int
    cuda_major, _ = torch.cuda.get_device_capability(dev)
    if cuda_major < 8:
        pytest.xfail("cuDNN SDPA uses flash attention, which requires Ampere+")


# These reference inputs are currently used by cudnnex
def grad_scaled_dot_product_attention_reference_generator(op, device, dtype, requires_grad, **kwargs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
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
        yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=False, scale=0.125)

    # Additive attn_mask
    # with different broadcasting patterns
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    additive_attn_mask = make((1, 1, L, S), dtype=q.dtype, requires_grad=requires_grad)
    yield SampleInput(q, k, v, additive_attn_mask, is_causal=False)
    additive_attn_mask = make((1, n_head, L, S), dtype=q.dtype, requires_grad=requires_grad)
    yield SampleInput(q, k, v, additive_attn_mask, is_causal=False)
    additive_attn_mask = make((N, n_head, L, S), dtype=q.dtype, requires_grad=requires_grad)
    yield SampleInput(q, k, v, additive_attn_mask, is_causal=False)

    # Boolean attn_mask
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    bool_attn_mask = make((1, n_head, L, S), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    yield SampleInput(q, k, v, bool_attn_mask, is_causal=False)

    # non-contiguous with stride 0 cases
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    q_broadcast = torch.as_strided(q, size=q.shape, stride=(0, 0, E, 1))
    k_broadcast = torch.as_strided(k, size=k.shape, stride=(0, 0, E, 1))
    v_broadcast = torch.as_strided(v, size=v.shape, stride=(0, 0, Ev, 1))
    yield SampleInput(q_broadcast, k_broadcast, v_broadcast, None, dropout_p=0.0, is_causal=True)

    # Additional dimension test cases for different GPU architectures and cuDNN versions
    b, h, s_q, s_kv = 2, 4, 64, 64

    # Standard dimensions - should work on all GPUs (d_q <= 128, d_kv <= 128)
    standard_dims = [
        (64, 64),  # standard small
        (32, 32),  # divisible by 8 small
        (96, 96),  # divisible by 8 mid
        (120, 88),  # divisible by 8 asymmetric
    ]
    for d_q, d_kv in standard_dims:
        q = make(b, h, s_q, d_q)
        k = make(b, h, s_kv, d_q)
        v = make(b, h, s_kv, d_kv)
        yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=True)

    # Larger dimensions - only supported on Hopper (SM90) with cuDNN 9.x
    hopper_dims = [
        (192, 192),  # hopper 192
        (256, 256),  # hopper max
        (256, 128),  # hopper asymmetric
    ]
    for d_q, d_kv in hopper_dims:
        q = make(b, h, s_q, d_q)
        k = make(b, h, s_kv, d_q)
        v = make(b, h, s_kv, d_kv)
        yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=True)

    # DeepSeek-style dimensions (d_q=192, d_kv=128) - Hopper 9.11+ or Blackwell 9.11+
    d_q, d_kv = 192, 128
    q = make(b, h, s_q, d_q)
    k = make(b, h, s_kv, d_q)
    v = make(b, h, s_kv, d_kv)
    yield SampleInput(q, k, v, None, dropout_p=0.0, is_causal=True)


def _should_skip_sdpa_sample(sample) -> str | None:
    """Return a skip reason if the SDPA sample dimensions are not supported on current GPU/cuDNN, else None."""
    q, k, v = sample.args[:3]
    d_q = q.shape[-1]
    d_kv = v.shape[-1]

    cudnn_version = cudnn.backend_version()
    cc_major = torch.cuda.get_device_capability()[0]

    # Standard dimensions (d_q <= 128, d_kv <= 128) - should work on all GPUs
    if d_q <= 128 and d_kv <= 128:
        return None

    # For dimensions > 128, need cuDNN 9.x
    if cudnn_version < 90000:
        return f"cuDNN 9.x required for dimensions > 128 (d_q={d_q}, d_kv={d_kv})"

    # DeepSeek case (d_q=192, d_kv=128)
    if d_q == 192 and d_kv == 128:
        if cudnn_version < 91100:
            return "cuDNN 9.11+ required for DeepSeek dimensions (d_q=192, d_kv=128)"
        if cc_major not in (9, 10):
            return "Hopper (SM90) or Blackwell (SM100) required for DeepSeek dimensions"
        return None

    # Larger dimensions (128 < d <= 256) - only Hopper
    if d_q > 128 or d_kv > 128:
        if cc_major != 9:
            return f"Hopper GPU (SM90) required for dimensions > 128 (d_q={d_q}, d_kv={d_kv})"
        if d_q > 256 or d_kv > 256:
            return f"Dimensions exceed Hopper max of 256 (d_q={d_q}, d_kv={d_kv})"
        return None

    return None


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
    _maybe_xfail()

    # expect sdpa to fail for 8.9.2 and below
    if cudnn.backend_version() <= 8902:
        pytest.xfail("Only interleaved layout is supported pre 8.9.2.")

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


@requiresCUDA
def test_cudnn_sdpa_NumberProxy_dropout():
    _maybe_xfail()

    # expect sdpa to fail for 8.9.2 and below
    if cudnn.backend_version() <= 8902:
        pytest.xfail("Only interleaved layout is supported pre 8.9.2.")
    shape = (4, 8, 16, 16)
    q = torch.rand(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.rand(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.rand(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    def func(q, k, v, dropout_p):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    cf = thunder.jit(func, cache="symbolic values", executors=[cudnn_ex])
    cf(q, k, v, dropout_p=0.5)
    last_trace = thunder.last_traces(cf)[-1]
    assert any(bsym.sym.name == "cudnn_sdpa_fwd" for bsym in last_trace.bound_symbols)


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
    _maybe_xfail()

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
    cfn = thunder.jit(op.op, executors=[cudnn_ex, cudnn_layernorm_ex])

    for sample in op.reference_inputs(device, dtype, requires_grad=False):
        # Skip SDPA samples with unsupported dimensions for current GPU/cuDNN
        if op.name == "grad_forward_scaled_dot_product_attention":
            if _should_skip_sdpa_sample(sample):
                continue

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


dtypes_for_grad_sdpa_cudnn = sorted(grad_sdpa_cudnn_opinfo.dtypes(), key=lambda x: str(x))


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < LooseVersion("8.9.5"),
    reason="cuDNN is required to be at least `8.9.5`",
)
@pytest.mark.parametrize("may_cat_grad_qkv", (True, False), ids=("may-cat-grad-qkv", "never-cat-grad-qkv"))
@pytest.mark.parametrize("dtype", dtypes_for_grad_sdpa_cudnn, ids=tuple(map(str, dtypes_for_grad_sdpa_cudnn)))
def test_vjp_correctness_cudnn_sdpa(dtype, may_cat_grad_qkv):
    from thunder.common import CompileData
    from thunder.core.compile_data import compile_data_and_stats

    _maybe_xfail()

    for sample in grad_sdpa_cudnn_opinfo.reference_inputs("cuda", dtype, requires_grad=True):
        # Skip samples with unsupported dimensions for current GPU/cuDNN
        if _should_skip_sdpa_sample(sample):
            continue

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

        cd = CompileData(
            fn=vjp(filtered_op),
            executors_list=[cudnn_ex],
            disable_preprocessing=True,
        )
        with compile_data_and_stats(cd, None):
            initial_trace = thunder.trace()(vjp(filtered_op), filtered_args, (v,))

        from thunder.executors.cudnn_sdpa import cudnn_sdpa_fwd, cudnn_sdpa_bwd

        # This is a workaround for the issue with python_ctx replacing symbols
        # with their "call_ctx" values which are not traceable and accept only
        # regular torch tensors
        initial_trace.python_ctx = lambda: {
            "cudnn_sdpa_fwd": cudnn_sdpa_fwd,
            "cudnn_sdpa_bwd": cudnn_sdpa_bwd,
        }

        cfoo = thunder.jit(
            initial_trace.python_callable(),
            disable_torch_autograd=True,
            executors=[cudnn_ex],
            cudnn_sdpa_bwd_may_cat_grad_qkv=may_cat_grad_qkv,
        )

        actual_out, actual_grad = cfoo(filtered_args, (v,))

        torch.testing.assert_close(actual_out, expect_out, atol=1e-2, rtol=1e-2)
        # compare gradients of query, key, value, and attn_mask
        for eg, ag in zip(expected_grad, actual_grad):
            torch.testing.assert_close(eg, ag, atol=2e-1, rtol=2e-2)


@requiresCUDA
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("float16", "bfloat16"))
def test_vjp_correctness_cudnn_layer_norm(dtype):
    from thunder.common import CompileData
    from thunder.core.compile_data import compile_data_and_stats
    from thunder.executors.cudnn_layernormex import layer_norm_aug_fwd, layer_norm_bwd
    from thunder.tests.opinfos import layer_norm_opinfo

    # Skip if cuDNN is not available or doesn't support the dtype
    if cudnn_layernorm_ex is None:
        pytest.skip("cuDNN LayerNorm executor not available")

    test_cases = [
        SampleInput(
            make_tensor((512, 1024), device="cuda", dtype=dtype, requires_grad=True),
            (1024,),
            make_tensor((1024,), device="cuda", dtype=dtype, requires_grad=True),
            make_tensor((1024,), device="cuda", dtype=dtype, requires_grad=True),
            1e-5,
        ),
    ]

    for sample in test_cases:
        # Enforce tensor arguments are contiguous for torch reference
        contiguous_args = list(map(lambda a: a.contiguous() if isinstance(a, torch.Tensor) else a, sample.args))
        grad_inputs = [contiguous_args[0], contiguous_args[2], contiguous_args[3]]

        expect_out = layer_norm_opinfo.torch_reference(*contiguous_args, **sample.kwargs)
        v = make_tensor_like(expect_out)
        expected_grad = torch.autograd.grad(expect_out, grad_inputs, v)

        flat_op, flat_args, spec = flatten_func(layer_norm_opinfo.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)

        cd = CompileData(
            fn=vjp(filtered_op),
            executors_list=[cudnn_layernorm_ex],
            disable_preprocessing=True,
        )
        with compile_data_and_stats(cd, None):
            initial_trace = thunder.trace()(vjp(filtered_op), filtered_args, (v,))

        initial_trace.python_ctx = lambda: {
            "cudnn_layernorm_aug_fwd": layer_norm_aug_fwd,
            "cudnn_layernorm_bwd": layer_norm_bwd,
        }
        cfoo = thunder.jit(
            initial_trace.python_callable(),
            disable_torch_autograd=True,
            executors=[cudnn_layernorm_ex],
        )

        actual_out, actual_grad = cfoo(filtered_args, (v,))
        torch.testing.assert_close(actual_out, expect_out, atol=1e-2, rtol=1e-2)
        for eg, ag in zip(expected_grad, actual_grad):
            torch.testing.assert_close(eg, ag, atol=1e-2, rtol=1e-2)


def test_compute_row_major_strides():
    from thunder.executors.cudnn_sdpa import _compute_row_major_strides

    # Test case 1: 2D tensor
    shape = (3, 4)
    expected_strides = (4, 1)
    assert _compute_row_major_strides(shape) == expected_strides

    # Test case 2: 3D tensor
    shape = (2, 3, 4)
    expected_strides = (12, 4, 1)
    assert _compute_row_major_strides(shape) == expected_strides

    # Test case 3: 1D tensor
    shape = (5,)
    expected_strides = (1,)
    assert _compute_row_major_strides(shape) == expected_strides

    # Test case 4: 4D tensor
    shape = (2, 3, 4, 5)
    expected_strides = (60, 20, 5, 1)
    assert _compute_row_major_strides(shape) == expected_strides

    # Test case 5: Empty shape
    shape = ()
    expected_strides = ()
    assert _compute_row_major_strides(shape) == expected_strides
