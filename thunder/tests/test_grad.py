from collections.abc import Sequence

# NOTE: Dependency on fdm and NumPy is temporary.
# We will remove it once we have a native way to compute numerical derivatives.
import fdm
import numpy as np
import pytest
import torch

import thunder.core.dtypes as dtypes

from thunder import trace as construct_trace
from thunder.core.dtypes import is_exact_dtype
from thunder.core.pytree import tree_map
from thunder.core.transforms import jvp, vjp, inline
from thunder.core.utils import flatten_func
from thunder.torch import to_thunder_dtype as thunder_dtype
from thunder.tests.framework import instantiate, NOTHING, ops, run_snippet
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.opinfos import opinfos, push_away_from_singularities, tensor_creation_ops

# TODO: Move this to thunder.tests.opinfos
op_skip = {
    # See https://github.com/Lightning-AI/lightning-thunder/issues/226
    # TODO: AttributeError: 'Tensor' object has no attribute 'true_dtype'
    "masked_fill",
    # TODO: RuntimeError: Expected index=tensor([2, 3, 2, 0, 3, 1, 0, 2],
    # device='cuda:0', dtype=torch.int32) to be a TensorProxy!
    "index_select",
    # Finite difference approximation doesn't work for this function
    "embedding",
}

# Don't rely on the generated list of supported ops.
# TODO: modify the generated list to support composite ops
vjp_op_force = {
    "cross_entropy",
    "softmax",
    "linear",
    "matmul",
    "var_mean",
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
    for opinfo in opinfos:
        if opinfo not in tensor_creation_ops and opinfo.name not in op_skip:
            if opinfo.dtypes().intersection({dtypes.float64}) == set():
                continue
            samples = iter(opinfo.sample_inputs("cpu", dtypes.float64, requires_grad=True))
            sample = next(samples)
            trace = construct_trace(opinfo.op, *sample.args, **sample.kwargs)
            all_supported = all(checker(s) for s in trace.bound_symbols)
            if all_supported:
                yield opinfo.name


def _vjp_symbol_checker(symbol):
    from thunder.core.transforms import augmented_forward_impls, backward_impls
    from thunder.core.transforms import transform_skip_list

    return (symbol.sym.id in augmented_forward_impls and symbol.sym.id in backward_impls) or (
        symbol.sym.id in transform_skip_list
    )


def _jvp_symbol_checker(symbol):
    from thunder.core.transforms import jvp_impls
    from thunder.core.transforms import transform_skip_list

    return symbol.sym.id in jvp_impls or symbol.sym.id in transform_skip_list


supported_vjp_ops = set(_generate_supported_op_list(_vjp_symbol_checker)).union(vjp_op_force)
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
    actual_p, actual_t = executor.make_callable(inline(jvp(f)))(primals, tangents)
    expected_p, expected_t = numerical_jvp(executor.make_callable(f))(primals, tangents)
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
    # Since u and v can be arbitrary, we take u = randn_like(primals), and v = randn_like(f(primals)).
    # We compute J u using numerical_jvp, and J* v using Thunder's vjp. That way we check correctness of Thunder's vjp.
    # Using finite differences we can compute J u, but we can't compute J* v, without computing full J, which is expensive.

    u = tree_map(make_tensor_like, primals)
    outs_p, J_u = numerical_jvp(executor.make_callable(f))(primals, u)

    multiple_results = isinstance(outs_p, Sequence)

    v = tree_map(make_tensor_like, outs_p)
    _, J_star_v = executor.make_callable(inline(vjp(f)))(primals, v)

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
def test_vjp_correctness_embedding_manual(op, device, dtype, executor):
    for sample in op.sample_inputs(device, dtype, requires_grad=True):
        # Compute vjp result using PyTorch
        out = op.torch_reference(*sample.args, **sample.kwargs)
        v = make_tensor_like(out)
        expected = torch.autograd.grad(out, sample.args[1], v)

        # Compute vjp result using Thunder
        flat_op, flat_args, spec = flatten_func(op.op, sample.args, sample.kwargs)
        filtered_op, filtered_args = _make_differentiable_wrapper(flat_op, flat_args)
        actual_out, (gindices, gweight) = executor.make_callable(inline(vjp(filtered_op)))(filtered_args, (v,))
        assert gindices is None, "gindices should be None"
        torch.testing.assert_close(gweight, expected[0])
        torch.testing.assert_close(actual_out, out)


# TODO Extend requires_grad so that tensors produced from lightning.compile functions requires_grad
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
    # Verifies a fix for https://github.com/Lightning-AI/lightning-thunder/issues/537
    from thunder.core.transforms import inline, value_and_grad

    a = make_tensor([5], dtype=torch.float32, device=device)

    @inline
    @value_and_grad
    def fn(t0):
        return t0 / 2

    out, (grad,) = executor.make_callable(fn)(a)
    torch.testing.assert_close(out, a / 2)
    torch.testing.assert_close(grad, torch.ones_like(a) / 2)


@instantiate(
    dtypes=NOTHING,
)
def test_multiple_output_vjp(executor, device, _):
    from thunder.core.prims import cos, make_prim, sin
    from thunder.core.transforms import inline, register_augmented_forward, register_backward, vjp

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
        out, (g,) = executor.make_callable(vjp(func))((x,), (v,))

    # The "vjp" function defined above is incorrect, let's check that we get the correct error
    with pytest.raises(RuntimeError, match="Pullback for sincos returned 2 values, but expected 1"):
        out, (g,) = executor.make_callable(vjp(func))((x,), (v, v))

    # Let's define a correct sincos_backward function
    @register_backward("sincos")
    def sincos_backward(sin_x, cos_x, g1, g2):
        return g1 * cos_x + g2 * -sin_x

    # It's not possible to teach Thunder about the PyTorch implementation of sincos
    # The following doesn't work because the PyTorch executor generates
    # a string of code with something like "out1, out2 = <lambda>(input)"
    # ops_to_torch_ops_map["sincos"] = lambda x: (torch.sin(x), torch.cos(x))
    # Therefore here we'll just check that the trace is correct
    trace = construct_trace(inline(vjp(func)), (x,), (v, v))
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


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_function(executor, device, _):
    from thunder.clang import cos, sin
    from thunder.executors.torchex import thunder_backward
    import thunder.torch as ltorch

    @thunder_backward(executors_list=executor.executors_list())
    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor((3,), device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b, c: func(a, b, c=c), (a, b, c))


@instantiate(
    dtypes=NOTHING,
)
def test_torch_autograd_module(executor, device, _):
    l = torch.nn.Linear(3, 4, bias=False, device=device)
    a = make_tensor((2, 3), device=device, dtype=torch.float32, requires_grad=True)
    g = make_tensor((2, 4), device=device, dtype=torch.float32)

    for use_static_caching in (True, None):
        lc = executor.make_callable(
            l,
            disable_preprocessing=False,
            use_generated_backward=True,
            use_static_caching=use_static_caching,
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
def test_torch_autograd_function_with_kwargs_static_caching(executor, device, _):
    from thunder.executors.torchex import thunder_backward

    @thunder_backward(
        executors_list=executor.executors_list(),
        use_static_caching=True,
    )
    def func(a, b):
        return a - b

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)

    # First call func(a, b) to populate the cache
    func(a, b)
    torch.testing.assert_close(func(a, b), a - b)
    torch.testing.assert_close(func(b=a, a=b), b - a)
    assert torch.autograd.gradcheck(lambda a, b: func(b=a, a=b), (a, b))
