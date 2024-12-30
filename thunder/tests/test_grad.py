from collections.abc import Sequence
import itertools
from functools import partial
from typing import Any

# NOTE: Dependency on fdm and NumPy is temporary.
# We will remove it once we have a native way to compute numerical derivatives.
import fdm
import numpy as np
import pytest
import torch

import thunder
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
import thunder.clang as clang

from thunder import torch as ltorch
from thunder.core.dtypes import is_exact_dtype, to_dtype as thunder_dtype
from thunder.core.pytree import tree_map, tree_flatten
from thunder.core.transforms import vjp, grad, check_bsym_for_vjp
from thunder.core.utils import flatten_func, is_cpu_scalar_tensor
from thunder.tests.framework import (
    instantiate,
    NOTHING,
    ops,
    run_snippet,
    assert_closer,
    IN_CI,
    requiresCUDA,
    version_between,
)
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.opinfos import opinfos, push_away_from_singularities, tensor_creation_ops, get_opinfo

# TODO: Move this to thunder.tests.opinfos
op_skip = {
    # See issue "Support closures of torch.Tensor"
    # TODO: AttributeError: 'Tensor' object has no attribute 'true_dtype'
    "masked_fill",
    # TODO: RuntimeError: Expected index=tensor([2, 3, 2, 0, 3, 1, 0, 2],
    # device='cuda:0', dtype=torch.int32) to be a TensorProxy!
    "index_select",
    # Finite difference approximation doesn't work for this function
    "embedding",
    "index_put",
    "batch_norm",
    "type_as",
}

if not torch.cuda.is_available():
    # Requires CUDA runtime to be available (fails on CPU only runtimes).
    op_skip.add("cuda")

# Don't rely on the generated list of supported ops.
# TODO: modify the generated list to support composite ops
vjp_op_force = {
    "abs",  # There's no clang.abs or prims.abs OpInfo, only torch.abs
    "amax",
    "amin",
    "cat",
    "softmax",
    "to",
    "linear",
    "matmul",
    "var",
    "var_mean",
    "interpolate",
    "prod",
    "repeat",
    "split",
    "stack",
    "cumsum",
    "mse_loss",
    "adaptive_avg_pool2d",
    "max_pool2d",
}


def _is_exact_dtype(torch_dtype):
    """Check if the given torch.dtype is an exact dtype.

    Args:
        torch_dtype (torch.dtype): The torch dtype to check.

    Returns:
        bool: True if the given torch.dtype is an exact dtype, False otherwise.
    """
    return is_exact_dtype(thunder_dtype(torch_dtype))


def _generate_supported_op_list(checker):
    """Generate a list of operators that is supported by the given checker.

    Args:
        checker (callable): A function that takes an operator info object and returns True if the operator
            satisfies the condition.

    Returns:
        generator: A generator of operator info objects that support vjp.
    """
    from thunder.core.transforms import trace_interpreter_skip_list

    for opinfo in opinfos:
        if opinfo not in tensor_creation_ops and opinfo.name not in op_skip:
            if opinfo.dtypes().intersection({dtypes.float64}) == set():
                continue
            samples = iter(opinfo.sample_inputs("cpu", dtypes.float64, requires_grad=True))
            while (sample := next(samples, None)) is not None:
                trc = thunder.trace()(opinfo.op, *sample.args, **sample.kwargs)
                all_skipped = all(s.sym.id in trace_interpreter_skip_list for s in trc.bound_symbols)
                if all_skipped:
                    continue
                all_supported = all(checker(s) for s in trc.bound_symbols)
                if all_supported:
                    yield opinfo.name


supported_vjp_ops = set(_generate_supported_op_list(check_bsym_for_vjp)).union(vjp_op_force)


def _to_numpy(x):
    """Convert a torch.Tensor or a numpy.ndarray to a numpy.ndarray.

    Args:
        x (torch.Tensor or numpy.ndarray): The input tensor.

    Returns:
        numpy.ndarray: The output array.

    Raises:
        ValueError: If the input is not a torch.Tensor or a numpy.ndarray.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise ValueError(f"_to_numpy: Unsupported type {type(x)}")


def _from_numpy(x, like):
    """Convert a numpy.ndarray to a torch.Tensor.

    Args:
        x (torch.Tensor or numpy.ndarray or numpy.float64): The input tensor.
        like (torch.Tensor): The tensor to use as a reference for the device and dtype.

    Returns:
        torch.Tensor: The output tensor.

    Raises:
        ValueError: If the input is not a torch.Tensor, a numpy.ndarray or a numpy.float64.
    """
    assert isinstance(like, torch.Tensor), f"_from_numpy: Unsupported type of the second argument {type(like)}"
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=like.device)
    if isinstance(x, torch.Tensor) or isinstance(x, np.float64):
        return torch.tensor(x, device=like.device, dtype=like.dtype)
    raise ValueError(f"_from_numpy: Unsupported type of the first argument {type(x)}")


def numerical_jvp(f):
    """Compute the numerical Jacobian-vector product of a function.

    It's a wrapper around fdm.jvp that converts the inputs and outputs to numpy.ndarray.
    It's meant to be used for testing of transforms.vjp.

    Args:
        f (callable): The function to differentiate.

    Returns:
        callable: The Jacobian-vector product function.
    """

    def jvp(primals, tangents):
        assert isinstance(primals, Sequence)
        assert isinstance(tangents, Sequence)
        assert len(primals) == len(tangents)
        np_primals, np_tangents = tree_map(_to_numpy, (primals, tangents))
        out_primals = f(*primals)

        multiple_outputs = True
        if not isinstance(out_primals, Sequence):
            out_primals = (out_primals,)
            multiple_outputs = False

        def ff(*args):
            out = f(*args)
            if not multiple_outputs:
                return (out,)
            return out

        np_out_primals = tree_map(_to_numpy, out_primals)
        np_out_tangents = tuple(np.zeros_like(o, dtype=np.float64) for o in np_out_primals)
        for j, out_tangent in enumerate(np_out_tangents):
            # Skip computing the jth output tangent if the jth output is 0-sized.
            if out_tangent.size == 0:
                continue
            for i in range(len(primals)):
                if _is_exact_dtype(primals[i].dtype):
                    # It doesn't contribute to the Jacobian-vector product.
                    continue

                # fdm only supports single input single output functions
                # Create a function that only varies the `i`th argument.
                def f_i(x):
                    x = _from_numpy(x, like=primals[i])
                    out = ff(*(primals[:i] + (x,) + primals[i + 1 :]))[j]
                    return _to_numpy(out)

                out_tangent += fdm.jvp(f_i, np_tangents[i])(np_primals[i])
        out_tangents = tree_map(lambda x: _from_numpy(x, like=out_primals[0]), np_out_tangents)
        if not multiple_outputs:
            return out_primals[0], out_tangents[0]
        return out_primals, out_tangents

    return jvp


def _replace_none_with_zero(x, y):
    """Replace None with torch.tensor(0.0) to avoid errors when computing the dot product.

    Args:
        x (list): The first list of tensors.
        y (list): The second list of tensors.

    Returns:
        tuple: The two lists of tensors.
    """
    x = list(x)
    y = list(y)
    assert x[0] is not None or y[0] is not None, "Both x and y are None"
    for i, (a, b) in enumerate(zip(x, y)):
        if a is None or b is None:
            device = x[i].device if x[i] is not None else y[i].device
            x[i] = torch.tensor(0.0, device=device, dtype=torch.float64)
            y[i] = torch.tensor(0.0, device=device, dtype=torch.float64)

    return x, y


# If one tensor is a CPU scalar tensor and the other is on CUDA, move the scalar tensor to CUDA
# Then do the ravel and dot operation
def _tensor_dot(x, y):
    if is_cpu_scalar_tensor(x) and y.is_cuda:
        x = x.cuda()
    elif is_cpu_scalar_tensor(y) and x.is_cuda:
        y = y.cuda()
    return torch.dot(x.ravel().type(torch.float64), y.ravel().type(torch.float64))


def _dot(x, y):
    """Compute the dot product of two lists of tensors.

    Args:
        x (list): The first list of tensors.
        y (list): The second list of tensors.

    Returns:
        torch.Tensor: The dot product.
    """
    x, y = _replace_none_with_zero(x, y)
    assert all(
        isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) for a, b in zip(x, y)
    ), "Not all elements are torch.Tensor"
    return sum([_tensor_dot(a, b) for a, b in zip(x, y)])


def check_vjp(f, *primals, comp, executor="torch", set_compile_data: bool = False, prologue_required: bool = False):
    """Check that the vector-Jacobian product of a function is correct.

    Args:
        f (callable): The function to differentiate.
        *primals (torch.Tensor): The input tensors.
        executor (str): The executor to use. Defaults to "torch".
        atol (float): Absolute tolerance. Defaults to None.
        rtol (float): Relative tolerance. Defaults to None.

    Raises:
        AssertionError: If the vector-Jacobian product is not correct.
    """
    # Let f be a function from vectors of size n to vectors of size m.
    # Its Jacobian is a matrix J of size m x n.
    # Represent by J^* the conjugate transpose (adjoint) of J.
    # J^* is a matrix of size n x m.
    # For any vector v of size m, J^* v is a vector of size n.
    # For any vector u of size n, J u is a vector of size m.
    # The dot product of J^* v and u is the same as the dot product of v and J u.
    # This function checks that the dot product of J^* v and u is the same as the dot product of v and J u.
    # 〈J u, v〉 == 〈u, J* v〉
    # Since u and v can be arbitrary, we take u = rand_like(primals), and v = rand_like(f(primals)).
    # We compute J u using numerical_jvp, and J* v using Thunder's vjp. That way we check correctness of Thunder's vjp.
    # Using finite differences we can compute J u, but we can't compute J* v, without computing full J, which is expensive.

    make = partial(make_tensor_like, low=0, high=1)

    u = tree_map(make, primals)

    # dirty little trick for speed: skip the prologue, however, the prologue is required when
    # there are non-differentiable kwargs
    jf = executor.make_callable(f, disable_torch_autograd=True)

    # if there are things the prologue passes to the epilogue, we need the prologue
    # this happens e.g. if the function returns inputs
    prologue_trc = thunder.compile_data(jf).get_computation_and_inputs(*primals)[0].prologue_traces[-1]
    prologue_required = prologue_required or prologue_trc.output[1]  # non-empty prologue_to_epilogue

    if prologue_required:
        comp_f = jf
    else:
        comp_f = thunder.compile_data(jf).get_computation_and_inputs(*primals)[0].computation_fn

    outs_p, J_u = numerical_jvp(comp_f)(primals, u)

    multiple_results = isinstance(outs_p, Sequence)

    v = tree_map(make, outs_p)
    if set_compile_data:
        with thunder.core.compile_data.compile_data_and_stats(thunder.compile_data(jf), None):
            initial_trace_vjp_f = thunder.trace()(vjp(f), primals, v)
    else:
        initial_trace_vjp_f = thunder.trace()(vjp(f), primals, v)
    _, J_star_v = executor.make_callable(initial_trace_vjp_f.python_callable(), disable_torch_autograd=True)(primals, v)

    if not multiple_results:
        v = (v,)
        J_u = (J_u,)

    J_u_v = _dot(J_u, v)
    u_J_star_v = _dot(u, J_star_v)
    if J_u_v.isnan().any():
        # TODO: find a better way to handle NaNs in finite differences
        return  # skip this sample
    comp(J_u_v, u_J_star_v)


def _is_differentiable(x):
    """Check if a tensor is allowed to be sent as an argument to a differentiable function.

    Args:
        x (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is differentiable, False otherwise.
    """
    if isinstance(x, torch.Tensor):
        # Allow passing through bool and integer tensors
        # Their gradient is None
        if _is_exact_dtype(x.dtype):
            return True
        return x.requires_grad
    # NOTE: we skip testing Python numbers for now
    # because internally fp32 may be used for computations with PyTorch
    # leading to numerical differences
    return False


def _make_differentiable_wrapper(func, args):
    """Make a wrapper for a function that takes a subset of differentiable arguments.

    Args:
        func (callable): The function to wrap.
        args (tuple): The arguments to the function.

    Returns:
        tuple: A tuple containing the wrapper and the filtered arguments.
    """
    differentiable_args_idx = tuple(i for i, arg in enumerate(args) if _is_differentiable(arg))

    def wrapper(*differentiable_args):
        args_iter = iter(differentiable_args)
        full_args = [next(args_iter) if i in differentiable_args_idx else arg for i, arg in enumerate(args)]
        return func(*full_args)

    filtered_args = tuple(arg for i, arg in enumerate(args) if i in differentiable_args_idx)
    return wrapper, filtered_args


def snippet_vjp_correctness(func, args, comp, executor, set_compile_data, prologue_required):
    check_vjp(
        func,
        *args,
        comp=comp,
        executor=executor,
        set_compile_data=set_compile_data,
        prologue_required=prologue_required,
    )


# TODO Use the given comparator
# TODO(crcrpar): Reason special-casing `adaptive_avg_pool2d` -- https://github.com/Lightning-AI/lightning-thunder/issues/1178
# With the slight revert for the mentioned issue, the VJP rule for `adaptive_avg_pool2d` is unavailable
# unless compile data is available, as it's registered to `TorchExecutor.implmap` but not to
# `thunder.core.transforms.augmented_forward_impls`.
@ops((op for op in opinfos if op.name in supported_vjp_ops), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness(op, device, dtype, executor, comp):
    at_least_one_differentiable_input = False
    eps = 1e-2
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # Here we convert any args in the sample to thunder args (specifically
        # to thunder dtypes). It was necessary to add this for the
        # convert_element_type tests, which have a non-differentiable argument
        # `dtype`. That argument is typically provided as a `torch.dtype`. In
        # the lines below, we strip non-differentiable arguments before
        # evaluating the op, which means stripped arguments do not undergo the
        # usual conversions in thunder.__init__._make_proxies(). The
        # sample.thunder() line below attempts to approximate those conversions
        # for non-differentiable arguments like dtypes so that the test will
        # execute properly.
        sample = sample.thunder()  # converts torch.dtype to thunder.dtype

        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)

        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        if len(filtered_args) == 0:
            continue
        if (singularity_fn := op.singularity_fn_producer(sample)) is not None:
            filtered_args = [push_away_from_singularities(arg, singularity_fn, eps) for arg in filtered_args]
        at_least_one_differentiable_input = True
        result = run_snippet(
            snippet_vjp_correctness,
            op,
            device,
            dtype,
            filtered_op,
            filtered_args,
            comp,
            executor,
            "adaptive_avg_pool2d" in op.name,
            len(sample.kwargs) != 0,
        )
        if result is not None:
            return result

    if not at_least_one_differentiable_input:
        raise pytest.skip("No differentiable inputs found")


# Embedding is a special case because its Jacobian product can't be approximated
# with finite differences
@ops((op for op in opinfos if op.name == "embedding"), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_embedding_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected = torch.autograd.grad(out, sample.args[1], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        initial_trace = thunder.trace()(vjp(filtered_op), filtered_args, (v,))
        actual_out, (gindices, gweight) = executor.make_callable(
            initial_trace.python_callable(), disable_torch_autograd=True
        )(filtered_args, (v,))
        assert gindices is None, "gindices should be None"
        comp(gweight, expected[0])
        comp(actual_out, out)


@ops((op for op in opinfos if op.name == "type_as"), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_type_as_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected = torch.autograd.grad(out, sample.args[0], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        initial_trace = thunder.trace()(vjp(flat_op), filtered_args, (v,))
        actual_out = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            filtered_args, (v,)
        )
        comp(actual_out[1][0], expected[0])
        comp(actual_out[0], out)


@ops(
    (get_opinfo("batch_norm"),),
    supported_dtypes=(dtypes.float64,),
)
def test_vjp_correctness_batch_norm_manual(op, device, dtype, executor, comp):
    from thunder.tests.framework import nvFuserTestExecutor

    if type(executor) is nvFuserTestExecutor and dtype is dtypes.float64:
        pytest.skip("nvFuser issue #1964")

    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # Compute vjp result using PyTorch
        weight = sample.args[3]
        bias = sample.args[4]
        # Torch fails with "RuntimeError: tensor does not have a device"
        if weight is None and bias is not None:
            continue
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        grad_inputs = [x for x in (sample.args[0], weight, bias) if x is not None]
        expected = torch.autograd.grad(out, grad_inputs, v)
        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, actual_grad = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (v,)
        )
        actual_grad = [
            x
            for x, grad_input in zip(actual_grad, sample.args[:5])
            if grad_input is not None and grad_input.requires_grad
        ]

        comp = partial(comp, equal_nan=True)
        comp(actual_out, out)
        assert len(actual_grad) == len(expected)
        for actual, expect in zip(actual_grad, expected):
            comp(actual, expect)


# Testing with finite differences has flaky accuracy fails
@ops((op for op in opinfos if op.name == "index_put"), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_index_put_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # skip the test cases when indices > 1D or indices are bool
        # values.requires_grad is used here just as a way to distinguish unsupported cases
        if not sample.args[2].requires_grad:
            continue

        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        # args: a, indices, values, accumulate
        grad_inputs = [sample.args[0], sample.args[2]]
        expected = torch.autograd.grad(out, grad_inputs, v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, actual_grad = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (v,)
        )
        comp(actual_out, out)
        comp(actual_grad[0], expected[0])
        comp(actual_grad[-2], expected[1])


# NOTE Scaled_Dot_Product_Efficient_Attention_Backward does not support fp64 dtypes
# RuntimeError: Only fp32, half & bf16 supported at the moment
@ops(
    (get_opinfo("grad_forward_scaled_dot_product_attention"),),
    supported_dtypes=(dtypes.float16, dtypes.bfloat16),
    supported_devicetypes=(devices.DeviceType.CUDA,),
)
def test_vjp_correctness_sdpa_manual(op, device, dtype, executor, comp):
    from thunder.common import CompileData
    from thunder.core.compile_data import compile_data_and_stats

    if version_between(torch.__version__, min_ver="2.5.0a0", max_ver="2.6.0a99"):
        raise pytest.skip(
            "https://github.com/Lightning-AI/lightning-thunder/issues/703",
        )

    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        from thunder.executors.sdpaex import sdpa_ex

        # Enforce tensor arguments are contiguous for torch reference
        contiguous_args = list(map(lambda a: a.contiguous() if isinstance(a, torch.Tensor) else a, sample.args))

        # query, key, value
        grad_inputs = list(contiguous_args[:3])
        if (attn_mask := sample.args[3]) is not None and attn_mask.requires_grad:
            grad_inputs.append(attn_mask)

        # Compute vjp result using PyTorch
        expect_out = op.torch_reference(*contiguous_args, **sample.kwargs)
        v = make_tensor_like(expect_out)
        expected_grad = torch.autograd.grad(expect_out, grad_inputs, v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        cd = CompileData(
            fn=vjp(filtered_op),
            executors_list=[sdpa_ex, *executor.executors_list()],
            disable_preprocessing=True,
        )
        with compile_data_and_stats(cd, None):
            initial_trace = thunder.trace()(vjp(filtered_op), filtered_args, (v,))

        from thunder.executors.sdpaex import sdpea_gradfwd, sdpea_bwd, sdpfa_gradfwd, sdpfa_bwd

        # This is a workaround for the issue with python_ctx replacing symbols
        # with their "call_ctx" values which are not traceable and accept only
        # regular torch tensors
        initial_trace.python_ctx = lambda: {
            "sdpaex_grad_forward_scaled_dot_product_efficient_attention": sdpea_gradfwd,
            "sdpaex_scaled_dot_product_efficient_attention_backward": sdpea_bwd,
            "sdpafx_grad_forward_scaled_dot_product_efficient_attention": sdpfa_gradfwd,
            "sdpafx_scaled_dot_product_efficient_attention_backward": sdpfa_bwd,
        }
        actual_out, actual_grad = thunder.jit(
            initial_trace.python_callable(),
            disable_torch_autograd=True,
            executors=[sdpa_ex, *executor.executors_list()],
        )(filtered_args, (v,))
        comp(actual_out, expect_out)

        # compare gradients of query, key, value, and attn_mask
        for eg, ag in zip(expected_grad, actual_grad):
            comp(eg, ag)


@ops((get_opinfo("zeta"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_zeta_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True, no_rhs_numbers=True):
        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected_grad = torch.autograd.grad(out, sample.args[1], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, (grad_lhs, grad_rhs) = executor.make_callable(
            initial_trace.python_callable(), disable_torch_autograd=True
        )(flat_args, (v,))
        assert grad_lhs is None, "grad_lhs should be None"
        comp(actual_out, out, equal_nan=True)
        comp(grad_rhs, expected_grad[0], equal_nan=True)


@ops((get_opinfo("item"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_torch_item_manual(op, device, dtype, executor, comp):
    from thunder.torch import item

    for sample in op.sample_inputs(device, dtype, requires_grad=True, no_rhs_numbers=True):
        out = op.torch_reference(*sample.args, **sample.kwargs)
        flat_op, flat_args, spec = flatten_func(item, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (None,))
        actual_out, (grad_in,) = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (None,)
        )
        assert grad_in is None, "grad_in should be None"
        comp(actual_out, out, equal_nan=True)


@ops((get_opinfo("nll_loss"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_nll_loss_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True, no_rhs_numbers=True):
        # Traced backwards function does not follow PyTorch nll_loss behavior with zero element tensors
        if sample.args[0].numel() == 0:
            continue

        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected_grad = torch.autograd.grad(out, sample.args[0], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, grad_out = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (v,)
        )

        comp(actual_out, out)
        comp(grad_out[0], expected_grad[0])


@ops((get_opinfo("cross_entropy"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_cross_entropy_manual(op, device, dtype, executor, comp):
    for sample in op.sample_inputs(device, dtype, requires_grad=True, no_rhs_numbers=True):
        # Traced backwards function does not follow PyTorch cross_entropy behavior with zero element tensors
        if sample.args[0].numel() == 0:
            continue

        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected_grad = torch.autograd.grad(out, sample.args[0], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, grad_out = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (v,)
        )

        comp(actual_out, out)
        comp(grad_out[0], expected_grad[0])


@ops((get_opinfo("einsum"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_einsum_manual(op, device, dtype, executor, comp):
    from thunder.tests.framework import nvFuserTestExecutor

    if type(executor) is nvFuserTestExecutor and dtype is dtypes.float64:
        pytest.skip("nvFuser issue #1645")

    for sample in op.sample_inputs(device, dtype, requires_grad=True, no_rhs_numbers=True):
        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected_grads = torch.autograd.grad(out, sample.args[1:], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        initial_trace = thunder.trace()(vjp(flat_op), flat_args, (v,))
        actual_out, grads_out = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(
            flat_args, (v,)
        )

        comp(actual_out, out)
        assert len(expected_grads) == len(grads_out) - 1
        for torch_grad, thunder_grad in zip(expected_grads, grads_out[1:]):
            comp(torch_grad, thunder_grad)


# TODO Extend requires_grad so that tensors produced from thunder.jit functions requires_grad
#   and have their autograd functions set properly
# Tests that we track the requires_grad property properly
@instantiate(dtypes=(dtypes.float32,))
def test_requires_grad(executor, device, dtype):
    import thunder.torch as ltorch

    torch_dtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2, 2), device=device, dtype=torch_dtype, requires_grad=False)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype, requires_grad=False)

    ag = make_tensor((2, 2), device=device, dtype=torch_dtype, requires_grad=True)

    def foo(a, b):
        c = a + b
        return c.requires_grad

    cfoo = executor.make_callable(foo)

    # Tests that when neither inputs requires grad, the result of the addition doesn't, either
    result = cfoo(a, b)
    assert result is False

    # Tests that when one input requires grad, the result of the addition requires grad, too
    result = cfoo(ag, b)
    assert result is True

    def bar(a, b):
        c = ltorch.cat((a, b))
        return c.requires_grad

    cbar = executor.make_callable(bar)

    # Tests that when neither inputs requires grad, the result of the cat doesn't, either
    result = cbar(a, b)
    assert result is False

    # Tests that when one input requires grad, the result of the cat requires grad, too
    result = cbar(ag, b)
    assert result is True


@instantiate(
    dtypes=NOTHING,
)
def test_convert_element_type_with_float(executor, device, _):
    # Verifies the fix for "grad transform hits error: AttributeError: 'float'
    # object has no attribute 'dtype'"
    from thunder.core.transforms import value_and_grad

    a = make_tensor([5], dtype=torch.float32, device=device)

    @value_and_grad
    def fn(t0):
        return t0 / 2

    initial_trace = thunder.trace()(fn, a)
    out, (grad,) = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)(a)
    torch.testing.assert_close(out, a / 2)
    torch.testing.assert_close(grad, torch.ones_like(a) / 2)


@instantiate(
    dtypes=NOTHING,
)
def test_multiple_output_vjp(executor, device, _):
    from thunder.core.prims import cos, make_prim, sin
    from thunder.core.transforms import register_augmented_forward, register_backward, vjp

    def sincos_meta(x):
        return sin(x), cos(x)

    sincos = make_prim("sincos", "sincos", meta=sincos_meta)

    @register_augmented_forward("sincos")
    def sincos_vjp_rule(x):
        out = sincos(x)
        saved = out
        return out, saved

    @register_backward("sincos")
    def sincos_backward(sin_x, cos_x, g1, g2):
        return g1 * cos_x, g2 * -sin_x

    def func(x):
        return sincos(x)

    x = torch.tensor(1.0)
    v = torch.tensor(1.0)

    # Let's check that we get the correct error if we don't pass the right number of cotangents
    with pytest.raises(RuntimeError, match="Expected cotangents to be a sequence of length 2"):
        initial_trace = thunder.trace()(vjp(func), (x,), (v,))

    # The "vjp" function defined above is incorrect, let's check that we get the correct error
    with pytest.raises(RuntimeError, match="Backward for sincos returned 2 values, but expected at most 1"):
        initial_trace = thunder.trace()(vjp(func), (x,), (v, v))

    # Let's define a correct sincos_backward function
    @register_backward("sincos")
    def sincos_backward(sin_x, cos_x, g1, g2):
        return g1 * cos_x + g2 * -sin_x

    # It's not possible to teach Thunder about the PyTorch implementation of sincos
    # The following doesn't work because the PyTorch executor generates
    # a string of code with something like "out1, out2 = <lambda>(input)"
    # ops_to_torch_ops_map["sincos"] = lambda x: (torch.sin(x), torch.cos(x))
    # Therefore here we'll just check that the trace is correct
    trace = thunder.trace()(vjp(func), (x,), (v, v))
    # Length of outputs should be two
    assert len(trace.output) == 2
    # Length of the first output should be two
    assert len(trace.output[0]) == 2
    # Length of the second output should match the length of primal args
    assert len(trace.output[1]) == len(trace.args[0])
    # The fifth symbol is sincos
    assert trace.bound_symbols[4].sym.name == "sincos"
    # The first output should be from sincos
    assert trace.output[0] == trace.bound_symbols[4].output


# TODO: see issue
# "thunder/tests/test_grad.py::test_torch_autograd_saved_tensors_memory_release
# is flaky"
@pytest.mark.xfail(strict=False, reason="This test is flaky")
@requiresCUDA
def test_torch_autograd_saved_tensors_memory_release():
    # This test checks that the saved tensors are released during compiled
    # backward function execution. It's a regression test for the memory leak.

    from thunder.core.prims import make_prim
    from thunder.core.transforms import register_augmented_forward, register_backward
    from thunder.core.proxies import TensorProxy
    from thunder.core import codeutils

    def noop_meta(x):
        return TensorProxy(like=x)

    def noop_printer(bsym, out_printables, arg_printables, kwarg_printables):
        result_str = f"{codeutils.prettyprint(out_printables, literals_as_underscores=True)} = "
        arg_string = ", ".join(codeutils.prettyprint(x, literals_allowed=False) for x in arg_printables)
        return result_str + f"{arg_string}.clone()"

    noop = make_prim(
        "noop",
        "noop",
        meta=noop_meta,
        python_printer=noop_printer,
        python_impl=lambda x: x,
    )

    def noop_backward_meta(x, g):
        return TensorProxy(like=g)

    def noop_backward_printer(bsym, out_printables, arg_printables, kwarg_printables):
        result_str = f"{codeutils.prettyprint(out_printables, literals_as_underscores=True)} = "
        return result_str + "torch.tensor(torch.cuda.memory_allocated())"

    noop_backward = make_prim(
        "noop_backward",
        "noop_backward",
        meta=noop_backward_meta,
        python_printer=noop_backward_printer,
        python_impl=lambda x, g: g,
    )

    @register_augmented_forward("noop")
    def noop_vjp_rule(x):
        out = noop(x)
        saved = (out,)
        return out, saved

    @register_backward("noop")
    def noop_backward_rule(out, g):
        return noop_backward(out, g)

    def func(x):
        x = x + 0
        for i in range(10):
            x = noop(x)
        return x

    cfunc = thunder.jit(func, executors=[thunder.executors.torchex.ex])

    initial_allocated = torch.cuda.memory_allocated()

    x = torch.tensor(1e20, device="cuda", requires_grad=True)
    v = torch.tensor(1e20, device="cuda")

    fw_out = cfunc(x)
    intermediate_allocated = torch.cuda.memory_allocated()
    fw_out.backward(v)
    final_allocated = torch.cuda.memory_allocated()

    assert int(x.grad.item() - initial_allocated) == 2048
    assert intermediate_allocated - initial_allocated == 6144
    assert final_allocated - initial_allocated == 2048


@instantiate(
    dtypes=NOTHING,
)
def test_make_aug_forward_and_backward(executor, device, _):
    from thunder.core.vjp_utils import make_aug_forward_and_backward
    from thunder.core.prims import mul

    def fun(a, b):
        return mul(a, b)

    @executor.make_callable
    def expected_aug_fw(a, b):
        return fun(a, b), (a, b)

    @executor.make_callable
    def fun_bw(a, b, g):
        return {"a": g * b, "b": g * a}

    x = torch.tensor(2.0, device=device)
    y = torch.tensor(3.0, device=device)
    v = torch.tensor(1.5, device=device)

    trace = thunder.trace()(fun, x, y)
    mul_bsym = trace.bound_symbols[2]
    assert mul_bsym.sym.name == "mul"

    aug_fw, bw = make_aug_forward_and_backward(mul_bsym)
    aug_fw = executor.make_callable(aug_fw)
    actual_aug_fw, actual_saved = aug_fw(x, y)
    expected_aug_fw, expected_saved = expected_aug_fw(x, y)
    torch.testing.assert_close(actual_aug_fw, expected_aug_fw)

    bw = executor.make_callable(bw)
    actual_bw = bw(*actual_saved, v)
    expected_bw = fun_bw(*expected_saved, v)
    torch.testing.assert_close(actual_bw, expected_bw)


@instantiate(
    dtypes=NOTHING,
)
def test_make_aug_forward_and_backward_var_mean(executor, device, _):
    # This test checks that the split of the joint forward/backward function for
    # var_mean correctly puts the forward part into the augmented forward
    # function and the backward part into the backward function without
    # overlapping symbols.
    from thunder.core.vjp_utils import make_aug_forward_and_backward
    from thunder.core.prims import var_mean

    def fun(a):
        return var_mean(a, (0,), correction=1)

    x = torch.tensor((2, 2), device=device, dtype=torch.float32)

    trace = thunder.trace()(fun, x)
    var_mean_bsym = trace.bound_symbols[-2]
    assert var_mean_bsym.sym.name == "var_mean"

    aug_fw, bw = make_aug_forward_and_backward(var_mean_bsym)
    aug_fw = executor.make_callable(aug_fw)
    out, saved = aug_fw(x, (0,), correction=1)
    bw = executor.make_callable(bw)
    _ = bw(*saved, *out)
    bw_trace = thunder.last_traces(bw)[0]
    assert "var_mean" not in (s.sym.name for s in bw_trace.bound_symbols)


def test_no_duplicate_backward_registered():
    from thunder.core.transforms import backward_impls, _grad_fn_map

    same_keys = set(_grad_fn_map.keys()).intersection(set(backward_impls.keys()))
    assert not same_keys, f"Duplicate keys: {same_keys}"


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_function(executor, device, _):
    from thunder.clang import cos, sin
    import thunder.torch as ltorch

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    func = thunder.jit(func, executors=executor.executors_list(), disable_torch_autograd=False)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b, c: func(a, b, c=c), (a, b, c))


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_function_single_input(executor, device, _):
    from thunder.clang import sin

    def func(a):
        return sin(a)

    func = thunder.jit(func, executors=executor.executors_list(), disable_torch_autograd=False)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (a,))


@instantiate(
    dtypes=(dtypes.float32,),
)
def test_torch_autograd_crazy_collections_in_and_out(executor, device, dtype):
    # Borrowed from `test_crazy_collections_in_and_out`.
    def foo(a, b, c, *, ka, kb, kc):
        d = {
            5: 2,
            7: 9,
            "a": [a, b],
            "b": {"a": a, "b": b, "c": [b, (a, c)]},
            "x": (a, [a, a, a], (b, (a, a, c, b))),
        }

        e = a["a"]["a"] + b[0]
        f = c[1]["c"] + b[1]
        g = e + f
        h = f + ka + kb
        # NOTE The following computation is intentionally unused
        i = ka + ka  # noqa
        j = kc[0] + kc[1]

        d["j"] = j

        return (
            a,
            (g,),
            (((j,),),),
            g,
            g,
            b,
            e,
            d["j"],
            (f, d, c, (d,), c, {"a": a, 5: f, "b": h}),
            (5,),
            (),
            (a,),
            [5, a, (b,), (), {}],
            {},
        )

    cfoo = thunder.jit(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype, requires_grad=True)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype, requires_grad=True)
    c = make_tensor((2, 2), device=device, dtype=tdtype, requires_grad=True)

    args = ({"a": {"a": a}}, (b, c), (3, {"c": c}))
    kwargs = {"ka": b, "kb": 3.0, "kc": (a, 2)}
    thunder_result = cfoo(*args, **kwargs)
    torch_result = foo(*args, **kwargs)
    torch.testing.assert_close(thunder_result, torch_result)

    flat_thunder_result, _ = tree_flatten(thunder_result)
    sum(flat_thunder_result).sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_module(executor, device, _):
    l = torch.nn.Linear(3, 4, bias=False, device=device)
    a = make_tensor((2, 3), device=device, dtype=torch.float32, requires_grad=True)
    g = make_tensor((2, 4), device=device, dtype=torch.float32)

    for cache_mode in ("constant values", "same input"):
        lc = executor.make_callable(
            l,
            disable_torch_autograd=False,
            cache_mode=cache_mode,
        )
        lc.zero_grad()
        a.grad = None
        out = lc(a)
        out.backward(g)
        l_grad = l.weight.grad
        torch.testing.assert_close(l_grad, g.mT @ a)
        torch.testing.assert_close(a.grad, g @ l.weight)


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_module_get_compile_stats(executor, device, _):
    from thunder.core.trace import TraceCtx
    from thunder import compile_stats

    l = torch.nn.Linear(3, 4, bias=False, device=device)
    a = make_tensor((2, 3), device=device, dtype=torch.float32, requires_grad=True)
    g = make_tensor((2, 4), device=device, dtype=torch.float32)

    lc = thunder.jit(
        l,
    )
    lc.zero_grad()
    a.grad = None
    out = lc(a)
    out.backward(g)

    compile_stats = compile_stats(lc)
    forward_traces = compile_stats.last_traces
    backward_traces = compile_stats.last_backward_traces
    assert isinstance(forward_traces, list)
    assert len(forward_traces) >= 1
    assert isinstance(backward_traces, list)
    assert len(backward_traces) >= 1
    fw_traces = thunder.last_traces(lc)
    bw_traces = thunder.last_backward_traces(lc)
    assert fw_traces == forward_traces
    assert bw_traces == backward_traces


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_function_with_kwargs_static_caching(executor, device, _):
    def func(a, b):
        return a - b

    func = thunder.jit(func, executors=executor.executors_list(), disable_torch_autograd=False)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)

    # First call func(a, b) to populate the cache
    func(a, b)
    torch.testing.assert_close(func(a, b), a - b)
    torch.testing.assert_close(func(b=a, a=b), b - a)
    assert torch.autograd.gradcheck(lambda a, b: func(b=a, a=b), (a, b))


@instantiate(
    dtypes=NOTHING,
)
def test_forward_and_backward_from_trace(executor, device, _):
    from thunder import trace
    from thunder.clang import cos, sin
    import thunder.torch as ltorch
    from thunder.core.transforms import forward_and_backward_from_trace, value_and_grad
    from thunder.core.transform_common import wrap_return_value_together_with_arguments

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float64, requires_grad=True)
    initial_trace = trace(inline_trace=False)(func, a, b, c=c)
    wrapped_trace = wrap_return_value_together_with_arguments(initial_trace)
    fw_trace, bw_trace = forward_and_backward_from_trace(wrapped_trace)
    fw = executor.make_callable(fw_trace)
    bw = executor.make_callable(bw_trace)
    fw_out, saved_for_backward = fw(a, b, c=c)

    initial_trace = trace()(value_and_grad(func), a, b, c=c)
    expected_vjp_func = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)

    expected_fw_out, expected_grads = expected_vjp_func(a, b, c=c)
    torch.testing.assert_close(fw_out["output"], expected_fw_out)

    output_grads = tree_map(lambda x: torch.ones_like(x), fw_out["output"])
    bw_out = bw(saved_for_backward, output_grads)
    torch.testing.assert_close(bw_out, expected_grads)


@instantiate(
    dtypes=NOTHING,
)
def test_update_forward_with_new_saved_for_backward_numberproxy(executor, device, _):

    def foo(t, ab):
        return t * ab * 0.5

    jfoo = thunder.jit(foo, cache="symbolic values")

    t = make_tensor((5, 3), device=device, dtype=torch.float32)
    t_ref = t.detach()
    t.requires_grad_()
    t_ref.requires_grad_()

    out = jfoo(t, 1.5)
    out_ref = foo(t_ref, 1.5)
    torch.testing.assert_close(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(t.grad, t_ref.grad)

    t.grad = None
    t_ref.grad = None

    out = jfoo(t, 2.7)
    out_ref = foo(t_ref, 2.7)
    torch.testing.assert_close(out, out_ref)

    out.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(t.grad, t_ref.grad)


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_redundant_casts(executor, device, _):
    # There was a bug where we would eliminate the redundant casts in forward
    # but backward wasn't updated with the new proxies. This test ensures that
    # we don't regress.
    from thunder.core.prims import convert_element_type
    import thunder.torch as ltorch

    def func(a, b, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return ltorch.sin(e) + ltorch.cos(e)

    func = thunder.jit(func, executors=executor.executors_list(), disable_torch_autograd=False)

    a = make_tensor((2, 3), device=device, dtype=torch.float16, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float16, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float16, requires_grad=True)

    # This would fail if we didn't update the backward with the new proxies
    func(a, b, c).sum().backward()


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_optional_args(executor, device, _):
    # Test that we can define a function with optional arguments
    # and that we can call it with or without those arguments
    import thunder.torch as ltorch

    @executor.make_callable
    def func(a, b, c=None):
        return ltorch.sin(a) + ltorch.cos(b)

    a = make_tensor((2, 3), device=device, dtype=torch.float16, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float16, requires_grad=True)
    func(a, b).sum().backward()
    func(a, b, object()).sum().backward()


@instantiate(
    dtypes=NOTHING,
)
def test_backward_none_propagation(executor, device, _):
    import thunder.torch as ltorch
    from thunder.core.transforms import vjp

    @vjp
    def func(a):
        return ltorch.split(a, 1)

    a = make_tensor((2, 4), device=device, dtype=torch.float16)
    initial_trace = thunder.trace()(func, (a,), (None, None))
    func = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
    result = func((a,), (None, None))
    assert result[1][0] is None


#
# Phantom grad tests
#
# TODO Jax consistency testing (slice and slice_in_dim don't have torch references)
# TODO Double-backward testing
# TODO Add more module tests


def snippet_phantom_grad_vs_torch_consistency(op, torch_op, sample, comp, singularity_fn):
    if singularity_fn:
        sample = sample.remove_singularities(singularity_fn, 1e-2)

    args, kwargs = sample.args, sample.kwargs

    def is_output_differentiable(x):
        # grad_fn is set only if one of the input `requires_grad=True`
        # and the op is differentiable.
        # Example:
        # >>> x = torch.ones(3, requires_grad=True)
        # >>> y = torch.ones(3, requires_grad=False)
        # >>> (x + x).grad_fn  # <AddBackward0 object at 0x7f0502edcf40>
        # >>> (y + y).grad_fn  # None
        # >>> (y + x).grad_fn  # <AddBackward0 object at 0x7f0502e21060>
        # >>> (x < 1).grad_fn  # None (non-differentiable op)
        # Op with differentiable and non-differentiable outputs.
        # >>> torch.topk(x, k=2)
        # torch.return_types.topk(
        # values=tensor([1., 1.], grad_fn=<TopkBackward0>),
        # indices=tensor([0, 1]))
        # >>> torch.topk(torch.ones(3, requires_grad=False), k=2)
        # torch.return_types.topk(
        # values=tensor([1., 1.]),
        # indices=tensor([0, 1]))
        return x.grad_fn is not None

    def filter_differentiable_outputs(outputs):
        if isinstance(outputs, torch.Tensor):
            # Otherwise `filter` below will
            # iterate over the Tensor data.
            outputs = [outputs]

        return list(filter(is_output_differentiable, outputs))

    # Computes PyTorch (competition) result
    torch_flats, spec = tree_flatten((args, kwargs))
    torch_result = torch_op(*args, **kwargs)
    torch_result = filter_differentiable_outputs(torch_result)
    if torch_result == []:
        raise RuntimeError(
            f"phantom_grad: Expected atleast 1 differentiable output. If {op.name} is non-differentiable, set op.supports_grad=False."
        )

    grads = []
    assert isinstance(torch_result, torch.Tensor) or isinstance(
        torch_result, Sequence
    ), "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
    if isinstance(torch_result, Sequence):
        for x in torch_result:
            assert isinstance(
                x, torch.Tensor
            ), "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
            if is_output_differentiable(x):
                grads.append(torch.ones_like(x))
    else:
        if is_output_differentiable(torch_result):
            grads = [torch.ones_like(torch_result)]

    torch_tensors_requiring_grad = tuple(f for f in torch_flats if isinstance(f, torch.Tensor) and f.requires_grad)
    torch_grad_result = torch.autograd.grad(torch_result, torch_tensors_requiring_grad, grads)

    # Computes reference result (upcasting floats to double)
    def upcast_tensors(x: Any) -> Any:
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            requires_grad = x.requires_grad
            return x.to(torch.double).detach().requires_grad_(requires_grad)

        return x

    reference_args = tree_map(upcast_tensors, args)
    reference_kwargs = tree_map(upcast_tensors, kwargs)
    reference_flats, spec = tree_flatten((reference_args, reference_kwargs))
    reference_tensors_requiring_grad = tuple(
        f for f in reference_flats if isinstance(f, torch.Tensor) and f.requires_grad
    )
    reference_result = torch_op(*reference_args, **reference_kwargs)
    reference_result = filter_differentiable_outputs(reference_result)
    reference_grad_result = torch.autograd.grad(reference_result, reference_tensors_requiring_grad, grads)

    # Computes thunder result
    grad_op = grad(op)
    thunder_flat_grads = grad_op(*sample.args, **sample.kwargs)

    assert_closer(
        reference=reference_grad_result, candidate=thunder_flat_grads, competitor=torch_grad_result, comparator=comp
    )


@ops(
    tuple(op for op in opinfos if op.supports_grad and op.torch_reference is not None),
    supported_dtypes=dtypes.float_math_dtypes,
)
def test_phantom_grad_vs_torch_consistency(op, device: str, dtype: dtypes.dtype, executor, comp):
    if dtypes.is_complex_dtype(dtype):
        pytest.skip("Skipping complex operator tests in CI for speed")
    if torch.device(device).type == "cuda" and dtype is dtypes.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("Your CUDA device does not support bfloat16")

    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        comp = sample.comp if sample.comp is not None else comp

        result = run_snippet(
            snippet_phantom_grad_vs_torch_consistency,
            op,
            device,
            dtype,
            executor.make_callable(op.op),
            op.torch_reference,
            sample,
            lambda a, b, **kwargs: comp(a, b, equal_nan=True, **kwargs),
            op.singularity_fn_producer(sample),
        )

        # See [NOTE] dynamo reset
        if any("torchcompile" in ex.name for ex in executor.executors_list()):
            torch._dynamo.reset()

        if result is not None:
            return result


from torch.testing import assert_close
from thunder.core.transforms import populate_grads, clear_grads, extract_grads, put_grad, put_grads, get_grad


@instantiate(dtypes=(thunder.float32,))
def test_phantom_grad_unpack(executor, device: str, dtype: dtypes.dtype):
    # Tests tuple unpacking
    def foo(tup):
        a, b = tup
        return a * b

    cfoo = thunder.jit(foo)
    cfoo_grad = grad(cfoo)

    a = torch.randn((2, 2), requires_grad=True)
    b = torch.randn((2, 2), requires_grad=True)

    a_grad, b_grad = cfoo_grad((a, b))

    assert_close(a_grad, b)
    assert_close(b_grad, a)

    # Tests dict unpacking
    def bar(d):
        a, b = d["a"], d["b"]
        return a * b

    cbar = thunder.jit(bar)
    cbar_grad = grad(cbar)

    a_grad, b_grad = cbar_grad({"a": a, "b": b})

    assert_close(a_grad, b)
    assert_close(b_grad, a)


@instantiate(dtypes=(thunder.float32,))
def test_phantom_grad_multiple_outputs(executor, device: str, dtype: dtypes.dtype):
    pass


@instantiate(dtypes=(thunder.float32,))
def test_populate_grads_mlp(executor, device, dtype):
    from thunder.benchmarks import NanoGPTMLPBenchmark, NanoGPTConfig

    # NOTE Currently setting dropout to zero for reproducibility, other settings taken from gpt2 config
    config = NanoGPTConfig(dropout=0, n_layer=12, n_head=12, n_embd=768)

    bench = NanoGPTMLPBenchmark(config=config, requires_grad=True, device=device, dtype=dtype)
    model = bench.fn()
    (x,), kwargs = bench.make_batch()

    result = model(x)
    result.backward(torch.ones_like(result))
    torch_grads = extract_grads(model)

    clear_grads(model)

    tom = executor.make_callable(model)
    tom_grad = grad(tom)
    thunder_grads = tom_grad(x)

    populate_grads(thunder_grads, tom, args=(x,))
    thunder_grads = extract_grads(tom)

    assert_close(torch_grads, thunder_grads, atol=1e-3, rtol=1e-5)


@instantiate(dtypes=(thunder.float32,))
def test_populate_grads_csa(executor, device, dtype):
    from thunder.benchmarks import NanoGPTCSABenchmark, NanoGPTConfig

    # NOTE Currently setting dropout to zero for reproducibility, other settings taken from gpt2 config
    config = NanoGPTConfig(dropout=0, n_layer=12, n_head=12, n_embd=768)

    bench = NanoGPTCSABenchmark(config=config, requires_grad=True, device=device, dtype=dtype)
    model = bench.fn()
    (x,), kwargs = bench.make_batch()

    result = model(x)
    result.backward(torch.ones_like(result))
    torch_grads = extract_grads(model)

    clear_grads(model)

    tom = executor.make_callable(model)
    tom_grad = grad(tom)
    thunder_grads = tom_grad(x)

    populate_grads(thunder_grads, tom, args=[x])
    thunder_grads = extract_grads(tom)

    assert_close(torch_grads, thunder_grads, atol=1e-2, rtol=1e-2)


@instantiate(dtypes=(thunder.float32,))
def test_populate_grads_block(executor, device, dtype):
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig

    # NOTE Currently setting dropout to zero for reproducibility, other settings taken from gpt2 config
    config = NanoGPTConfig(dropout=0, n_layer=12, n_head=12, n_embd=768)

    bench = NanoGPTBlockBenchmark(config=config, requires_grad=True, device=device, dtype=dtype)
    model = bench.fn()
    (x,), kwargs = bench.make_batch()

    result = model(x)
    result.backward(torch.ones_like(result))
    torch_grads = extract_grads(model)

    clear_grads(model)

    tom = executor.make_callable(model)
    tom_grad = grad(tom)
    thunder_grads = tom_grad(x)

    populate_grads(thunder_grads, tom, args=[x])
    thunder_grads = extract_grads(tom)

    assert_close(torch_grads, thunder_grads, atol=1e-2, rtol=1e-2)


@instantiate(dtypes=(thunder.float32,))
def test_populate_grads_nanogpt(executor, device, dtype):
    import sys

    if sys.platform == "win32":
        pytest.skip(
            "This test crashes its worked on Windows when run using pytest distributed (Windows fatal exception: access violation)"
        )
    if IN_CI and torch.device(device).type == "cpu":
        pytest.skip("Skipping the CPU version of this test in CI because it's very slow")

    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig

    # NOTE Currently setting dropout to zero for reproducibility
    config = NanoGPTConfig(dropout=0, n_layer=2, n_head=1, n_embd=64)

    bench = NanoGPTBenchmark(config=config, requires_grad=True, device=device, dtype=dtype)
    model = bench.fn()
    (x, targets), kwargs = bench.make_batch()

    logits, loss = model(x, targets)
    torch.autograd.backward((logits, loss), (torch.ones_like(logits), torch.ones_like(loss)))
    torch_grads = extract_grads(model)

    clear_grads(model)

    tom = executor.make_callable(model)

    tom_grad = grad(tom)
    thunder_grads = tom_grad(x, targets)

    populate_grads(thunder_grads, tom, args=[x, targets])
    thunder_grads = extract_grads(tom)

    assert_close(torch_grads, thunder_grads, atol=1e-2, rtol=1e-2)


def test_too_few_results_from_backward():
    # this tests the error message, previously we checked the condition too late and hit an
    # opaque error

    global myadd
    from thunder.core.prims import make_prim
    from thunder.core.transforms import register_augmented_forward, register_backward
    from thunder.core.proxies import TensorProxy
    from thunder.core import codeutils

    def myadd_meta(a, b):
        return TensorProxy(like=a)

    myadd = make_prim(
        "myadd",
        "myadd",
        meta=myadd_meta,
    )

    myex = thunder.extend.OperatorExecutor("myex", version="0.1")
    thunder.extend.register_executor(myex)
    myadd_op = myex.register_operator("myadd", like=myadd_meta, fn=lambda a, b: a + b)

    @register_augmented_forward("myadd")
    def myadd_augmented_fw(a, b):
        out = myadd(a, b)
        saved = (out,)
        return out, saved

    @register_backward("myadd")
    def myadd_backward(out, g):
        return g

    def func(a, b):
        return myadd(a, b)

    cfunc = thunder.jit(func, executors=[myex, thunder.executors.torchex.ex])

    a = torch.tensor(1.0, requires_grad=False)
    b = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(RuntimeError, match=r"Backward for myadd returned 1 value\(s\), but expected 2"):
        fw_out = cfunc(a, b)

    thunder.extend.deregister_executor(myex)


def test_make_forward_backward_symbol_caching_with_executor():
    # See issue : https://github.com/Lightning-AI/lightning-thunder/issues/230
    ex_1 = thunder.extend.OperatorExecutor("ex_1")

    def sin_meta(a):
        return thunder.TensorProxy(like=a)

    call_cnt = 0

    def sin_impl(a):
        nonlocal call_cnt
        call_cnt += 1
        if call_cnt > 1:
            raise RuntimeError("This symbol shouldn't have been called more than once.")
        return torch.sin(a)

    ex_1_sin = ex_1.register_operator("ex_1_sin", meta=sin_meta, fn=sin_impl)

    def ex_1_sin_grad(a):
        c = ex_1_sin(a)
        g = get_grad(c)
        put_grad(a, g * thunder.torch.cos(a))
        return c

    ex_1.register_implementation(thunder.prims.sin, ex_1_sin, grad_transform=ex_1_sin_grad)

    def foo(a):
        return thunder.prims.sin(a)

    a = torch.randn(3, 3, requires_grad=True)

    # This should call the implementation from ex_1
    thunder.jit(foo, executors=[ex_1])(a)

    # This should call the core implementation.
    thunder.jit(foo)(a)


def test_grad_transform_saved_for_backward_proxy():
    from thunder.core.proxies import Proxy

    def foo(a, c):
        return a * c

    a = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)
    c = 2.0

    dynamic_jit = thunder.jit(foo, cache="symbolic values")
    static_jit = thunder.jit(foo)

    out = dynamic_jit(a, c)
    torch.autograd.backward(out, torch.rand_like(out), retain_graph=True)
    dynamic_trace = thunder.last_backward_traces(dynamic_jit)[-1]
    # dynamic trace should save `c` as proxy for backward
    assert any(map(lambda x: isinstance(x, Proxy), tree_flatten(dynamic_trace.args[0])[0]))

    out = static_jit(a, c)
    torch.autograd.backward(out, torch.rand_like(out), retain_graph=True)
    static_trace = thunder.last_backward_traces(static_jit)[-1]
    # static trace should bake `c` as scalar number, so it won't show up in backward as proxy
    assert not any(map(lambda x: isinstance(x, Proxy), tree_flatten(static_trace.args[0])[0]))


def test_get_saved_for_backward_tensors():
    from thunder.core.vjp_utils import get_saved_for_backward_tensors

    def func(a, b):
        return a * b

    a = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)
    b = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)

    jfunc = thunder.jit(func)

    _ = jfunc(a, b)
    execution_trace = thunder.last_traces(jfunc)[-1]
    saved_for_backward = get_saved_for_backward_tensors(execution_trace)
    assert "a" in map(lambda x: x.name, saved_for_backward)
    assert "b" in map(lambda x: x.name, saved_for_backward)


def test_get_saved_for_backward_tensors_error():
    from thunder.core.vjp_utils import get_saved_for_backward_tensors

    def func(a, b):
        return a * b

    a = make_tensor((2, 2), device="cpu", dtype=torch.float32)
    b = make_tensor((2, 2), device="cpu", dtype=torch.float32)

    jfunc = thunder.jit(func, disable_torch_autograd=True)

    _ = jfunc(a, b)
    execution_trace = thunder.last_traces(jfunc)[-1]
    with pytest.raises(RuntimeError, match="The trace must be generated by Thunder's automatic differentiation"):
        get_saved_for_backward_tensors(execution_trace)


def test_torch_checkpoint():
    import torch.utils.checkpoint
    import torch._higher_order_ops.wrap

    def fn_to_checkpoint(x, y):
        return x.sin().cos().exp().mul(y)

    checkpoint_fns = (
        thunder.torch.checkpoint,
        partial(torch.utils.checkpoint.checkpoint, use_reentrant=False),
        torch.ops.higher_order.tag_activation_checkpoint,
    )

    for checkpoint_fn in checkpoint_fns:

        def f(x, y):
            return checkpoint_fn(fn_to_checkpoint, x, y)

        x = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)
        y = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)
        jf = thunder.jit(f)
        out = jf(x, y)

        # With activation checkpointing, we are saving only the original input.
        # The intermediate values are recomputed during backward pass.
        assert len(out.grad_fn.saved_tensors) == 2
        # We detach the saved tensors (which returns a new Python tensor backed by same storage)
        assert out.grad_fn.saved_tensors[0].data_ptr() == x.data_ptr()
        assert out.grad_fn.saved_tensors[1].data_ptr() == y.data_ptr()

        g = torch.ones_like(out)
        out.backward(g)

        x_ref = x.detach().requires_grad_()
        y_ref = y.detach().requires_grad_()
        out_ref = fn_to_checkpoint(x_ref, y_ref)
        out_ref.backward(g)
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(y.grad, y_ref.grad)


def test_inconsistent_output_length_grad_transform():
    from thunder.extend import OperatorExecutor
    from thunder.core.proxies import AnyProxy, TensorProxy
    from thunder.core.transforms import get_grad, put_grad

    my_ex = OperatorExecutor("my_ex")

    forward_op = my_ex.register_operator(
        "forward_op", meta=lambda x: (TensorProxy(like=x), AnyProxy(object())), fn=lambda x: (x, x.shape)
    )

    backward_op = my_ex.register_operator(
        "backward_op", meta=lambda saved_meta, g: TensorProxy(like=g), fn=lambda saved_meta, g: g
    )

    def forward_op_grad(x):
        out, meta = forward_op(x)
        g = get_grad(out)
        g_o = backward_op(meta, g)
        put_grad(x, g_o)
        return out

    my_ex.register_implementation(forward_op, forward_op, grad_transform=forward_op_grad)

    def f(x):
        return forward_op(x)

    jf = thunder.jit(f, executors=[my_ex])
    a = make_tensor((2, 2), device="cpu", dtype=torch.float32, requires_grad=True)

    with pytest.raises(
        RuntimeError,
        match="number of outputs of the original forward function must be the same as the number of primal outputs",
    ):
        _ = jf(a)


@pytest.mark.parametrize("device", ("cuda", "cpu"))
@requiresCUDA
def test_grad_softmax_dtype(device):
    def forward(x):
        topk, _idxs = x.topk(2)
        return topk.softmax(dim=1, dtype=torch.float)

    jforward = thunder.jit(forward)

    x = torch.randn([8, 2], dtype=torch.bfloat16, device=device, requires_grad=True)

    actual = jforward(x)
    expected = forward(x)
    torch.testing.assert_close(actual, expected)

    grad_o = torch.randn_like(actual)

    actual_grad = torch.autograd.grad(actual, x, grad_o)
    expected_grad = torch.autograd.grad(expected, x, grad_o)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ("cuda", "cpu"))
def test_grad_split_unused_output(device):
    # Test to verify that the grad rule for split is
    # correct even if few of the outputs are unused
    # (leading to `grad=None` for them).

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    def forward(x):
        x_1, x_2, x_3 = torch.split(x, 2)
        return x_1

    jforward = thunder.jit(forward)

    x = torch.randn([5, 2], dtype=torch.bfloat16, device=device, requires_grad=True)

    actual = jforward(x)
    expected = forward(x)
    torch.testing.assert_close(actual, expected)

    grad_o = torch.randn_like(actual)

    actual_grad = torch.autograd.grad(actual, x, grad_o)
    expected_grad = torch.autograd.grad(expected, x, grad_o)
    torch.testing.assert_close(actual_grad, expected_grad)


@instantiate(
    dtypes=NOTHING,
)
def test_adhoc_executor_grad(executor, device, _):
    import torch
    import thunder

    x = torch.ones(2, device=device, requires_grad=True)

    class Sin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sin(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * torch.cos(x) * 200

    def func(x):
        return Sin.apply(x)

    cfunc = thunder.jit(func)
    actual = cfunc(x)
    (actual_gr,) = torch.autograd.grad(actual.sum(), x)
    expected = func(x)
    (expected_gr,) = torch.autograd.grad(expected.sum(), x)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gr, expected_gr)
