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
from thunder.core.transforms import jvp, vjp, grad, check_bsym_for_vjp
from thunder.core.utils import flatten_func
from thunder.tests.framework import instantiate, NOTHING, ops, run_snippet, assert_closer, IN_CI, requiresCUDA
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
}

# Don't rely on the generated list of supported ops.
# TODO: modify the generated list to support composite ops
vjp_op_force = {
    "abs",  # There's no clang.abs or prims.abs OpInfo, only torch.abs
    "amax",
    "amin",
    "cat",
    "cross_entropy",
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
    from thunder.core.transforms import transform_skip_list

    for opinfo in opinfos:
        if opinfo not in tensor_creation_ops and opinfo.name not in op_skip:
            if opinfo.dtypes().intersection({dtypes.float64}) == set():
                continue
            samples = iter(opinfo.sample_inputs("cpu", dtypes.float64, requires_grad=True))
            while (sample := next(samples, None)) is not None:
                trc = thunder.trace()(opinfo.op, *sample.args, **sample.kwargs)
                all_skipped = all(s.sym.id in transform_skip_list for s in trc.bound_symbols)
                if all_skipped:
                    continue
                all_supported = all(checker(s) for s in trc.bound_symbols)
                if all_supported:
                    yield opinfo.name


def _jvp_symbol_checker(symbol):
    from thunder.core.transforms import jvp_impls
    from thunder.core.transforms import transform_skip_list

    return symbol.sym.id in jvp_impls or symbol.sym.id in transform_skip_list


supported_vjp_ops = set(_generate_supported_op_list(check_bsym_for_vjp)).union(vjp_op_force)
supported_jvp_ops = set(_generate_supported_op_list(_jvp_symbol_checker))


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
    It's meant to be used for testing of transforms.jvp and transforms.vjp.

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


def check_jvp(f, *primals, executor, atol=None, rtol=None):
    """Check that the Jacobian-vector product of a function is correct.

    Args:
        f (callable): The function to differentiate.
        *primals (torch.Tensor): The input tensors.
        executor (str): The executor to use. Defaults to "torch".
        atol (float): Absolute tolerance. Defaults to None.
        rtol (float): Relative tolerance. Defaults to None.

    Raises:
        AssertionError: If the Jacobian-vector product is not correct.
    """
    tangents = tree_map(make_tensor_like, primals)
    actual_p, actual_t = executor.make_callable_legacy(jvp(f))(primals, tangents)
    expected_p, expected_t = numerical_jvp(executor.make_callable_legacy(f))(primals, tangents)
    torch.testing.assert_close(expected_p, actual_p, atol=atol, rtol=rtol)
    torch.testing.assert_close(expected_t, actual_t, atol=atol, rtol=rtol)


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
    device = x[0].device if x[0] is not None else y[0].device
    zero = torch.tensor(0.0, device=device, dtype=torch.float64)
    for i, (a, b) in enumerate(zip(x, y)):
        if a is None or b is None:
            x[i] = zero
            y[i] = zero
    return x, y


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
    return sum([torch.dot(a.ravel().type(torch.float64), b.ravel().type(torch.float64)) for a, b in zip(x, y)])


def check_vjp(f, *primals, executor="torch", atol=1e-5, rtol=1.3e-6):
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
    # The adjoint property is J^* J = I, where J^* is the conjugate transpose (adjoint) of J.
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
    outs_p, J_u = numerical_jvp(executor.make_callable_legacy(f, disable_torch_autograd_support=True))(primals, u)

    multiple_results = isinstance(outs_p, Sequence)

    v = tree_map(make, outs_p)
    _, J_star_v = executor.make_callable_legacy(vjp(f), disable_torch_autograd_support=True)(primals, v)

    if not multiple_results:
        v = (v,)
        J_u = (J_u,)

    J_u_v = _dot(J_u, v)
    u_J_star_v = _dot(u, J_star_v)
    if J_u_v.isnan().any():
        # TODO: find a better way to handle NaNs in finite differences
        return  # skip this sample
    torch.testing.assert_close(J_u_v, u_J_star_v, atol=atol, rtol=rtol, check_device=False)


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


def snippet_jvp_correctness(func, args, executor):
    check_jvp(func, *args, executor=executor)


# TODO Use the given comparator
@ops((op for op in opinfos if op.name in supported_jvp_ops), supported_dtypes=(dtypes.float64,))
def test_jvp_correctness(op, device, dtype, executor, comp):
    at_least_one_differentiable_input = False
    eps = 1e-2
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        if len(filtered_args) == 0:
            continue
        if op.singularity_fn is not None:
            filtered_args = [push_away_from_singularities(arg, op.singularity_fn, eps) for arg in filtered_args]
        at_least_one_differentiable_input = True
        result = run_snippet(
            snippet_jvp_correctness,
            op,
            device,
            dtype,
            filtered_op,
            filtered_args,
            executor,
        )
        if result is not None:
            return result

    if not at_least_one_differentiable_input:
        raise pytest.skip("No differentiable inputs found")


def snippet_vjp_correctness(func, args, executor):
    check_vjp(func, *args, executor=executor)


# TODO Use the given comparator
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
        if op.singularity_fn is not None:
            filtered_args = [push_away_from_singularities(arg, op.singularity_fn, eps) for arg in filtered_args]
        at_least_one_differentiable_input = True
        result = run_snippet(
            snippet_vjp_correctness,
            op,
            device,
            dtype,
            filtered_op,
            filtered_args,
            executor,
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
        actual_out, (gindices, gweight) = executor.make_callable_legacy(
            vjp(filtered_op), disable_torch_autograd_support=True
        )(filtered_args, (v,))
        assert gindices is None, "gindices should be None"
        comp(gweight, expected[0])
        comp(actual_out, out)


@ops((get_opinfo("batch_norm"),), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness_batch_norm_manual(op, device, dtype, executor, comp):
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
        actual_out, actual_grad = executor.make_callable_legacy(vjp(flat_op), disable_torch_autograd_support=True)(
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
        actual_out, actual_grad = executor.make_callable_legacy(vjp(flat_op), disable_torch_autograd_support=True)(
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
        actual_out, actual_grad = thunder.compile(
            vjp(filtered_op),
            disable_torch_autograd_support=True,
            disable_preprocessing=True,
            executors_list=[sdpa_ex, *executor.executors_list()],
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
        actual_out, (grad_lhs, grad_rhs) = executor.make_callable_legacy(
            vjp(flat_op), disable_torch_autograd_support=True
        )(flat_args, (v,))
        assert grad_lhs is None, "grad_lhs should be None"
        comp(actual_out, out, equal_nan=True)
        comp(grad_rhs, expected_grad[0], equal_nan=True)


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
        actual_out, grad_out = executor.make_callable_legacy(vjp(flat_op), disable_torch_autograd_support=True)(
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
        actual_out, grads_out = executor.make_callable_legacy(vjp(flat_op), disable_torch_autograd_support=True)(
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

    out, (grad,) = executor.make_callable_legacy(fn)(a)
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
        out, (g,) = executor.make_callable_legacy(vjp(func))((x,), (v,))

    # The "vjp" function defined above is incorrect, let's check that we get the correct error
    with pytest.raises(RuntimeError, match="Backward for sincos returned 2 values, but expected at most 1"):
        out, (g,) = executor.make_callable_legacy(vjp(func))((x,), (v, v))

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
    from thunder.executors.torch_autograd import thunder_backward
    from thunder.common import CompileData
    from thunder.clang import cos, sin
    import thunder.torch as ltorch

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    compile_data = CompileData(fn=func, disable_preprocessing=True, executors_list=executor.executors_list())
    func = thunder_backward(compile_data=compile_data)(func)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b, c: func(a, b, c=c), (a, b, c))


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_function_single_input(executor, device, _):
    from thunder.clang import sin
    from thunder.executors.torch_autograd import thunder_backward
    from thunder.common import CompileData

    def func(a):
        return sin(a)

    compile_data = CompileData(fn=func, disable_preprocessing=True, executors_list=executor.executors_list())
    func = thunder_backward(compile_data=compile_data)(func)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (a,))


@instantiate(
    dtypes=(dtypes.float32,),
)
def test_torch_autograd_crazy_collections_in_and_out(executor, device, dtype):
    from thunder.executors.torch_autograd import thunder_backward

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
    from thunder.executors.torch_autograd import thunder_backward
    from thunder.common import CompileData

    def func(a, b):
        return a - b

    compile_data = CompileData(fn=func, disable_preprocessing=True, executors_list=executor.executors_list())
    func = thunder_backward(compile_data=compile_data)(func)

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

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    expected_vjp_func = executor.make_callable_legacy(value_and_grad(func))

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float64, requires_grad=True)
    trace = trace(inline_trace=False)(func, a, b, c=c)
    fw_trace, bw_trace = forward_and_backward_from_trace(trace)
    fw = executor.make_callable(fw_trace)
    bw = executor.make_callable(bw_trace)
    fw_out, saved_for_backward = fw(a, b, c=c)
    expected_fw_out, expected_grads = expected_vjp_func(a, b, c=c)
    torch.testing.assert_close(fw_out, expected_fw_out)

    output_grads = tree_map(lambda x: torch.ones_like(x), fw_out)
    bw_out = bw(saved_for_backward, output_grads)
    torch.testing.assert_close(bw_out, expected_grads)


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_redundant_casts(executor, device, _):
    # There was a bug where we would eliminate the redundant casts in forward
    # but backward wasn't updated with the new proxies. This test ensures that
    # we don't regress.
    from thunder.core.prims import convert_element_type
    from thunder.executors.torch_autograd import thunder_backward
    from thunder.common import CompileData
    import thunder.torch as ltorch

    def func(a, b, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return ltorch.sin(e) + ltorch.cos(e)

    compile_data = CompileData(fn=func, disable_preprocessing=True, executors_list=executor.executors_list())
    func = thunder_backward(compile_data=compile_data)(func)

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

    @executor.make_callable_legacy
    @vjp
    def func(a):
        return ltorch.split(a, 1)

    a = make_tensor((2, 4), device=device, dtype=torch.float16)
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

    # Computes PyTorch (competition) result
    torch_flats, spec = tree_flatten((args, kwargs))
    torch_result = torch_op(*args, **kwargs)

    grads = []
    assert isinstance(torch_result, torch.Tensor) or isinstance(
        torch_result, Sequence
    ), "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
    if isinstance(torch_result, Sequence):
        for x in torch_result:
            assert isinstance(
                x, torch.Tensor
            ), "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
            grads.append(torch.ones_like(x))
    else:
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
    reference_grad_result = torch.autograd.grad(reference_result, reference_tensors_requiring_grad, grads)

    # Computes thunder result
    grad_op = grad(op)
    thunder_flat_grads = grad_op(*sample.args, **sample.kwargs)

    assert_closer(
        reference=reference_grad_result, candidate=thunder_flat_grads, competitor=torch_grad_result, comparator=comp
    )


@ops(
    tuple(op for op in opinfos if op.supports_grad and op.torch_reference is not None),
    supported_dtypes=(dtypes.floating,),
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
            executor.make_callable_legacy(op.op),
            op.torch_reference,
            sample,
            lambda a, b, **kwargs: comp(a, b, equal_nan=True, **kwargs),
            op.singularity_fn,
        )
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

    cfoo = thunder.compile(foo, disable_preprocessing=True)
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

    cbar = thunder.compile(bar, disable_preprocessing=True)
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
    loss.backward()
    torch_grads = extract_grads(model)

    clear_grads(model)

    tom = executor.make_callable(model)

    def grad_specifier(out) -> None:
        logits, loss = out
        put_grad(loss, ltorch.ones_like(loss))

    tom_grad = grad(tom, grad_specifier=grad_specifier)
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
