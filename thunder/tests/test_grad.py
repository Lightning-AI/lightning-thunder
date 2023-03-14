from typing import Sequence

# NOTE: Dependency on fdm and NumPy is temporary.
# We will remove it once we have a native way to compute numerical derivatives.
import fdm
import numpy as np
import pytest
import torch

import thunder.core.dtypes as dtypes

from thunder import make_trace, make_traced
from thunder.core.pytree import tree_map
from thunder.core.transforms import jvp, vjp
from thunder.core.utils import flatten_func
from thunder.tests.framework import ops, run_snippet
from thunder.tests.make_tensor import make_tensor_like
from thunder.tests.opinfos import opinfos, tensor_creation_ops, push_away_from_singularities

boolean_ops = {
    "eq",
    "ne",
    "ge",
    "gt",
    "le",
    "lt",
    "isfinite",
}

# TODO: Move this to thunder.tests.opinfos
op_skip = {
    # TODO: thunder.make_traced doesn't work with closures with torch.Tensor arguments
    # See https://github.com/Lightning-AI/lightning-thunder/issues/226
    # TODO: AttributeError: 'Tensor' object has no attribute 'true_dtype'
    "masked_fill",
    # TODO: RuntimeError: Unexpected type <class 'torch.Tensor'> for pred
    "where",
    # TODO: RuntimeError: Expected index=tensor([2, 3, 2, 0, 3, 1, 0, 2],
    # device='cuda:0', dtype=torch.int32) to be a TensorProxy!
    "index_select",
}


def _generate_supported_op_list(checker):
    """Generate a list of operators that is supported by the given checker.

    Args:
        checker (callable): A function that takes an operator info object and returns True if the operator
            satisfies the condition.

    Returns:
        generator: A generator of operator info objects that support vjp.
    """
    for opinfo in opinfos:
        if opinfo not in tensor_creation_ops and opinfo.name not in op_skip | boolean_ops:
            if opinfo.dtypes().intersection({dtypes.float64}) == set():
                continue
            samples = iter(opinfo.sample_inputs("cpu", dtypes.float64, requires_grad=True))
            sample = next(samples)
            # TODO: generalize for other executors
            trace = make_trace(opinfo.op, executor="torch")(*sample.args, **sample.kwargs)
            all_supported = all(checker(s) for s in trace.symbols)
            if all_supported:
                yield opinfo.name


def _vjp_symbol_checker(symbol):
    from thunder.core.transforms import vjp_impls

    return symbol.op in vjp_impls


def _jvp_symbol_checker(symbol):
    from thunder.core.transforms import jvp_impls

    return symbol.op in jvp_impls


supported_vjp_ops = set(_generate_supported_op_list(_vjp_symbol_checker))
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
        np_out_tangents = tuple(np.zeros_like(o) for o in np_out_primals)
        for j, out_tangent in enumerate(np_out_tangents):
            for i in range(len(primals)):
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


def check_jvp(f, *primals, executor="torch", atol=None, rtol=None):
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
    actual_p, actual_t = make_traced(jvp(f), executor=executor)(primals, tangents)
    expected_p, expected_t = numerical_jvp(make_traced(f, executor=executor))(primals, tangents)
    torch.testing.assert_close(expected_p, actual_p, atol=atol, rtol=rtol)
    torch.testing.assert_close(expected_t, actual_t, atol=atol, rtol=rtol)


def _dot(x, y):
    """Compute the dot product of two lists of tensors.

    Args:
        x (list): The first list of tensors.
        y (list): The second list of tensors.

    Returns:
        torch.Tensor: The dot product.
    """
    assert all(
        isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) for a, b in zip(x, y)
    ), "Not all elements are torch.Tensor"
    return sum([torch.dot(a.ravel(), b.ravel()) for a, b in zip(x, y)])


def check_vjp(f, *primals, executor="torch", atol=None, rtol=None):
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
    # Since u and v can be arbitrary, we take u = randn_like(primals), and v = randn_like(f(primals)).
    # We compute J u using numerical_jvp, and J* v using Thunder's vjp. That way we check correctness of Thunder's vjp.
    # Using finite differences we can compute J u, but we can't compute J* v, without computing full J, which is expensive.

    u = tree_map(make_tensor_like, primals)
    outs_p, J_u = numerical_jvp(make_traced(f, executor=executor))(primals, u)
    multiple_results = isinstance(outs_p, Sequence)

    v = tree_map(make_tensor_like, outs_p)
    _, J_star_v = make_traced(vjp(f), executor=executor)(primals, v)

    if not multiple_results:
        v = (v,)
        J_u = (J_u,)

    J_u_v = _dot(J_u, v)
    u_J_star_v = _dot(u, J_star_v)
    if J_u_v.isnan().any():
        # TODO: find a better way to handle NaNs in finite differences
        return  # skip this sample
    torch.testing.assert_close(J_u_v, u_J_star_v, atol=atol, rtol=rtol)


def _is_differentiable(x):
    """Check if a tensor is differentiable.

    Args:
        x (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is differentiable, False otherwise.
    """
    if isinstance(x, torch.Tensor):
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


@ops((op for op in opinfos if op.name in supported_jvp_ops), supported_dtypes=(dtypes.float64,))
def test_jvp_correctness(op, device, dtype, executor):
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


@ops((op for op in opinfos if op.name in supported_vjp_ops), supported_dtypes=(dtypes.float64,))
def test_vjp_correctness(op, device, dtype, executor):
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
