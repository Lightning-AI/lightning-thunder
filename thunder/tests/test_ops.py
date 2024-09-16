from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.testing import assert_close

import thunder
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_map
from thunder.tests.framework import ops, run_snippet, requiresJAX
from thunder.tests.opinfos import OpInfo, SampleInput, opinfos
import thunder.tests.bf16

#
# Generic test templates for all operators
#


# NOTE err_msg_match=None will match any error message
def snippet_errors(op, sample, ex_type, err_msg_match=None):
    with pytest.raises(ex_type, match=err_msg_match):
        op(*sample.args, **sample.kwargs)


@ops(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op, device, dtype, executor, comp):
    for sample, ex_type, err_msg in op.error_inputs(device):
        result = run_snippet(snippet_errors, op, device, None, executor.make_callable(op.op), sample, ex_type, err_msg)
        if result is not None:
            return result


# Snippets run a single test using a single sample
# TODO: should snippets be able to access the original opinfo? -- No?
# TODO: revisit atol/rtol, maybe be more selective about which ops need a more permissive check
def snippet_torch_consistency(op: OpInfo, torch_op, sample: SampleInput, comp: Callable):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    # TODO Review how thunder.jit returns Exception information
    if isinstance(thunder_result, Exception):
        raise thunder_result

    comp(thunder_result, torch_result)


# TODO consider structuring tests like this to be autogenerated
#   using a snippet and an "extractor" that constructs the args and kwargs for the snippet
# TODO The name of this test is misleading as it may test operators from a variety of languages,
#   maybe we should cut it up so developers can test just torch operators or just core lang operators
# TODO Extend this test with some reproducible randomness (maybe using hypothesis)
# TODO Remove the atol and rtol defaults and rely on the given comparator to set them
@ops(tuple(op for op in opinfos if op.torch_reference is not None))
def test_core_vs_torch_consistency(op, device: str, dtype: dtypes.dtype, executor, comp):
    if dtypes.is_complex_dtype(dtype):
        pytest.skip("Skipping complex operator tests in CI for speed")
    if (
        torch.device(device).type == "cuda"
        and dtype is dtypes.bfloat16
        and not thunder.tests.bf16.device_supports_bf16(device)
    ):
        pytest.skip("Your CUDA device does not support bfloat16")

    for sample in op.sample_inputs(device, dtype):
        comp = sample.comp if sample.comp is not None else comp

        tfn: Callable
        tfn = thunder.jit(
            op.op,
            executors=executor.executors_list(),
            cache="no caching",
            disable_torch_autograd=True,
        )

        result = run_snippet(
            snippet_torch_consistency,
            op,
            device,
            dtype,
            tfn,
            op.torch_reference,
            sample,
            lambda a, b: comp(a, b, equal_nan=True),
        )

        # See [NOTE] dynamo reset
        if any("torchcompile" in ex.name for ex in executor.executors_list()):
            torch._dynamo.reset()

        if result is not None:
            return result


def snippet_jax_consistency(op, jax_op, sample, comp):
    import jax.numpy as jnp

    jax_sample = sample.jax()

    thunder_result = op(*sample.args, **sample.kwargs)
    jax_result = jax_op(*jax_sample.args, **jax_sample.kwargs)

    # NOTE This strange unpacking is to handle NumPy's and JAX's sometimes odd
    #   number vs. array representation. In particular, NumPy can mimic
    #   Python numbers, but `asarray` doesn't understand this mimicry
    def convert_to_torch(x):
        if not isinstance(x, jnp.ndarray):
            return x

        np_array = np.array(x)
        if np_array.shape == ():
            return torch.tensor(np_array.item(), device=thunder_result.device)
        else:
            return torch.asarray(np_array, device=thunder_result.device)

    jax_result = tree_map(convert_to_torch, jax_result)

    comp(thunder_result, jax_result)


# TODO Consider structuring tests like this to be autogenerated
#   using a snippet and an "extractor" that constructs the args and kwargs for the snippet
# TODO Extend this test with some reproducible randomness (maybe using hypothesis)
@ops(tuple(op for op in opinfos if op.jax_reference is not None))
@requiresJAX
def test_core_vs_jax_consistency(op, device: str, dtype: dtypes.dtype, executor, comp):
    if dtypes.is_complex_dtype(dtype):
        pytest.skip("Skipping complex operator tests in CI for speed")
    if dtype is dtypes.complex32:
        pytest.skip("jax doesn't support complex32!")
    if dtype is dtypes.bfloat16:
        pytest.skip("jax bfloat16 support is spotty (at least on CPU)")

    for sample in op.sample_inputs(device, dtype):
        comp = sample.comp if sample.comp is not None else comp

        result = run_snippet(
            snippet_jax_consistency,
            op,
            device,
            dtype,
            executor.make_callable(op.op),
            op.jax_reference,
            sample,
            # NOTE: dtype is not checked because jax will translate
            # int64, float64, and complex128 to int32, float32 and complex64
            lambda a, b: comp(a, b, equal_nan=True, check_dtype=False),
        )
        if result is not None:
            return result


def snippet_numpy_consistency(op: OpInfo, np_op, sample: SampleInput, comp: Callable):
    np_sample = sample.numpy()

    thunder_result = op(*sample.args, **sample.kwargs)
    np_result = np_op(*np_sample.args, **np_sample.kwargs)

    # Converts NumPy results to PyTorch.
    # NOTE This assumes PyTorch will return tensors where NumPy is aggressive about returning `np.number` objects.
    def convert_to_torch(x):
        if not isinstance(x, (np.ndarray, np.number, np.bool_)):
            return x

        if isinstance(x, (np.number, np.bool_)):
            return torch.tensor(x, device=thunder_result.device)
        elif x.shape == ():
            return torch.tensor(x.item(), device=thunder_result.device)
        else:
            return torch.asarray(x, device=thunder_result.device)

    np_result = tree_map(convert_to_torch, np_result)

    comp(thunder_result, np_result)


@ops(tuple(op for op in opinfos if op.numpy_reference is not None))
def test_core_vs_numpy_consistency(op: OpInfo, device: str, dtype: dtypes.dtype, executor, comp):
    if dtypes.is_complex_dtype(dtype):
        pytest.skip("Skipping complex operator tests in CI for speed")
    if dtype == dtypes.complex32:
        pytest.skip("NumPy does not support complex32")
    if dtype == dtypes.bfloat16:
        pytest.skip("NumPy does not support bfloat16")

    for sample in op.sample_inputs(device, dtype):
        comp = sample.comp if sample.comp is not None else comp

        result = run_snippet(
            snippet_numpy_consistency,
            op,
            device,
            dtype,
            executor.make_callable(op.op),
            op.numpy_reference,
            sample,
            # NOTE dtype is intentionally not checked because NumPy sometimes has slight dtype variances
            lambda a, b: comp(a, b, equal_nan=True, check_dtype=False),
        )
        if result is not None:
            return result


def test_notimplemented_interpolate_align():
    def foo():
        t0 = torch.randn((22, 288, 15, 20), dtype=torch.float16)
        t1 = torch.nn.functional.interpolate(t0, scale_factor=2.0, mode="nearest", align_corners=True)
        return t1

    with pytest.raises(NotImplementedError, match="not yet support"):
        tfoo = thunder.jit(foo)
        tfoo()


def test_notimplemented_interpolate_recompute_scale():
    def foo():
        t0 = torch.randn((22, 288, 15, 20), dtype=torch.float16)
        t1 = torch.nn.functional.interpolate(t0, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        return t1

    with pytest.raises(NotImplementedError, match="not yet support"):
        tfoo = thunder.jit(foo)
        tfoo()


def test_notimplemented_interpolate_antialias():
    def foo():
        t0 = torch.randn((22, 288, 15, 20), dtype=torch.float16)
        t1 = torch.nn.functional.interpolate(
            t0,
            scale_factor=2.0,
            mode="nearest",
            antialias=True,
        )
        return t1

    with pytest.raises(NotImplementedError, match="not yet support"):
        tfoo = thunder.jit(foo)
        tfoo()


def test_setitem():
    def fn(a):
        a[:3] = 2
        return a * 2

    a_ref = torch.ones(5)
    out_ref = fn(a_ref)
    a = torch.ones(5)
    jf = thunder.jit(fn)
    out = jf(a)
    assert_close(a, a_ref)
    assert_close(out, out_ref)

def test_exponential():
    def fn(a):
        return a.exponential_(1)

    size = 1000
    seed = 1234
    with torch.device("cuda"):
        a_ref = torch.ones(size)
        torch.manual_seed(seed)
        a_ref = fn(a_ref)

        a = torch.ones(size)
        torch.manual_seed(seed)

        # nbfuser fuses prims.uniform, which is used by our exponential resulting in differing numerics.
        jf = thunder.jit(fn, executors={})
        a = jf(a)

        assert_close(a, a_ref)

