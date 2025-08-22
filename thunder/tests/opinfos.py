from collections import namedtuple
from collections.abc import Callable, Generator, Iterable, Sequence
from functools import partial, wraps
import itertools
import math
from numbers import Number
import operator
from typing import Any

import numpy as np
import pytest
import random

# TODO: make this import conditional on Torch being available and querying if should test with torch
import torch
from looseversion import LooseVersion
from torch.testing import assert_close

import thunder.clang as clang
import thunder.core.devices as devices
import thunder.core.dtypes as datatypes
from thunder.core.dtypes import to_dtype, to_torch_dtype
import thunder.core.prims as prims
from thunder.core.pytree import tree_map
from thunder.core.symbol import Symbol
import thunder.executors as executors
from thunder.tests.framework import _all_devicetypes, JAX_AVAILABLE, custom_comparator, IS_WINDOWS
from thunder.tests.make_tensor import make_tensor, make_tensor_like
import thunder.tests.bf16
import thunder.torch as ltorch

#
# Helpful constants and utility functions
#

# TODO This is a hack to support comparisons like nvfuser_version > LooseVersion("0.0.3") even when
#   nvfuser_version is None. A better approach would probably be to create a helper function
#   nvfuser_atleast(X) which handles nvfuser_version being None properly
nvfuser_version: LooseVersion = (
    LooseVersion(executors.get_nvfuser_executor().version) if executors.nvfuser_available() else LooseVersion("0.0.0")
)

# Useful when specifying the domain of an operation
# NOTE: Big enough such that -1 + eps != -1 in bfloat16
# TODO: improve domain specification to allow intervals to be open or closed at the left and right
#   Today, the domain is assumed to be closed on the left and open on the right, that is: [x, y)
eps = 1e-2


# NOTE This wrapper is necessary because prims cannot be compiled directly as they are not callable
# TODO Review if this is still necessary
def prims_wrapper(prim):
    def fn_(*args, **kwargs):
        return prim(*args, **kwargs)

    return fn_


def round_remainder(x, y):
    return x - torch.round(x / y) * y


# Randomly select a fraction of the elements in a tensor and set them to specified value
def replace_random_percentage(a: torch.Tensor, value: Number, percentage: float) -> torch.Tensor:
    flat = torch.flatten(a.detach().clone())
    num_values_to_replace: int = math.floor(flat.numel() * percentage)
    choice_np = np.random.choice(np.arange(0, flat.numel()), (num_values_to_replace,), replace=False)
    choice = torch.asarray(choice_np, device=a.device)
    flat[choice] = value
    return flat.reshape(a.shape).requires_grad_(a.requires_grad)


def make_number(**kwargs):
    v = make_tensor((), device="cpu", **kwargs).item()
    return v


# Returns a noncontiguous tensor (with the same shape and values as t)
# The noncontiguous tensor is constructed such that elements in the innermost
#   dimension are separated by zeros or (whenever possible) nans
# TODO: consider more complicated noncontiguity schemes
def noncontiguous_like(t):
    # Short-circuits if t is already noncontiguous
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that won't be accessed
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad_(t.requires_grad)
    return result


_torch_to_numpy_dtype_map = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

_torch_to_jax_dtype_map = None
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    _torch_to_jax_dtype_map = {
        torch.bool: jnp.bool_,
        torch.uint8: jnp.uint8,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.complex64: jnp.complex64,
        torch.complex128: jnp.complex128,
    }


class TorchTensorComp:
    """
    This class provides a very simple wrapper over torch.testing.assert_close,
    and is used as a default comparator for per-SampleInput comparisons.
    """

    __slots__ = [
        "kwargs",
    ]

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, a, b, **kwargs):
        # Call assert_close with parameters defined as a union
        # of `kwargs` and `self.kwargs` with the preference
        # given to `self.kwargs`.
        assert_close(a, b, **(kwargs | self.kwargs))


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
        "comp",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.comp = None

    def __repr__(self):
        return f"[SampleInput args={self.args} kwargs={self.kwargs}]"

    def set_comparator(self, comp):
        self.comp = comp
        return self

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        args, kwargs = tree_map(to_noncontiguous, self.args), tree_map(to_noncontiguous, self.kwargs)
        return SampleInput(*args, **kwargs).set_comparator(self.comp)

    def to(self, dtype: torch.dtype):
        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype)

            return x

        args, kwargs = tree_map(_to, self.args), tree_map(_to, self.kwargs)
        return SampleInput(*args, **kwargs)

    def remove_singularities(self, op, eps):
        singularity_fn = op.singularity_fn_producer(self)
        if singularity_fn is None:
            return self

        def _push_away_from_singularities(x, dist_fn, eps):
            """This function takes a tensor and moves individual values away
            from singularities in `eps` increments, until they are further than
            `eps` away from them. The `dist_fn` returns the (signed)
            distance from `x` to the nearest singularity."""
            x_dist = dist_fn(x)
            x_ = torch.where((x_dist >= 0) & (x_dist < eps), x + eps, x)
            return torch.where((x_dist < 0) & (x_dist > -eps), x_ - eps, x_)

        def _remove_singularities(x):
            if isinstance(x, torch.Tensor) and datatypes.is_float_dtype(datatypes.to_dtype(x)):
                return _push_away_from_singularities(x, singularity_fn, eps)

            return x

        args, kwargs = tree_map(_remove_singularities, self.args), tree_map(_remove_singularities, self.kwargs)
        return SampleInput(*args, **kwargs)

    # NOTE This conversion is always to a jax cpu array, we could consider
    #   converting to a jax gpu array, although we would probably have to update
    #   how we're installing jax in ci
    def jax(self):
        def to_jax(t):
            if isinstance(t, torch.Tensor):
                return jnp.array(t.cpu().numpy())
            if isinstance(t, torch.dtype):
                return _torch_to_jax_dtype_map[t]

            return t

        args, kwargs = tree_map(to_jax, self.args), tree_map(to_jax, self.kwargs)
        return SampleInput(*args, **kwargs).set_comparator(self.comp)

    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.cpu().numpy()
            if isinstance(t, torch.dtype):
                return _torch_to_numpy_dtype_map[t]

            return t

        args, kwargs = tree_map(to_numpy, self.args), tree_map(to_numpy, self.kwargs)
        return SampleInput(*args, **kwargs).set_comparator(self.comp)

    def thunder(self):
        def to_thunder(t):
            if isinstance(t, torch.dtype):
                return to_dtype(t)
            return t

        args, kwargs = tree_map(to_thunder, self.args), tree_map(to_thunder, self.kwargs)
        return SampleInput(*args, **kwargs).set_comparator(self.comp)


class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given decorator when testing an operator.

    Any test that matches all provided arguments will be decorated. The decorator will only be applied if the active_if
    argument is True.
    """

    __slots__ = [
        "decorator",
        "test_template_name",
        "executors",
        "devicetypes",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorator: Any,
        test_template_name: None | str = None,
        *,
        executors: None | Iterable[str] = None,
        devicetypes: None | Sequence[devices.DeviceType] = None,
        dtypes=None,
        active_if: bool = True,
    ):
        self.decorator = decorator
        self.test_template_name = test_template_name
        self.executors: None | set[str] = {ex.lower() for ex in executors} if executors is not None else None
        self.devicetypes: None | Sequence[devices.DeviceType] = devicetypes

        if devicetypes is not None:
            for x in devicetypes:
                assert isinstance(x, devices.DeviceType), (
                    f"Found non-devicetype {x} when initializing a DecorateInfo's devicetypes"
                )

        self.dtypes = None if dtypes is None else datatypes.resolve_dtypes(dtypes)
        self.active_if = active_if

    def is_active(
        self, test_template_name, executor, device_or_devicetype: str | devices.Device | devices.DeviceType, dtype
    ):
        # Acquires devicetype
        devicetype_: devices.DeviceType
        if isinstance(device_or_devicetype, str):
            devicetype_ = devices.to_device(device_or_devicetype).devicetype
        elif isinstance(device_or_devicetype, devices.Device):
            devicetype_ = device_or_devicetype.devicetype
        else:
            assert False, f"Unknown device or devicetype {device_or_devicetype}, expect a string, device, or devicetype"

        executor_match = self.executors is None or executor.name.lower() in self.executors
        test_name_match = self.test_template_name is None or self.test_template_name == test_template_name
        devicetype_match = self.devicetypes is None or devicetype_ in self.devicetypes
        dtype_match = self.dtypes is None or dtype in self.dtypes

        return self.active_if and executor_match and test_name_match and devicetype_match and dtype_match


Domain = namedtuple("Domain", "low high")


class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op: Symbol | Callable,
        *,
        name: str | None = None,
        devicetypes: Sequence[devices.DeviceType] | None = None,
        dtypes=None,
        supports_grad: bool = False,
        sample_input_generator,
        reference_input_generator=None,
        error_input_generator=None,
        benchmark_generator=None,
        method_variant=None,
        operator_variant=None,
        torch_reference=None,
        numpy_reference=None,
        jax_reference=None,
        test_directives=(),
        domain=(None, None),
        singularity_fn=None,
        singularity_fn_producer=None,
        test_torch_compile_executor=False,
        instantiate_complex_tests=False,
    ):
        self.op = op

        # Acquires or infers the name of the operation
        name_: str
        if name is not None:
            name_ = name
        elif isinstance(op, Symbol):
            name_ = op.name
        else:
            assert isinstance(op, Callable)
            name_ = op.__name__
        self.name = name_

        self._devicetypes = devicetypes if devicetypes is not None else _all_devicetypes()

        # Validates devicetypes
        for devtyp in self._devicetypes:
            assert isinstance(devtyp, devices.DeviceType), "OpInfo devicetypes must be DeviceTypes"

        self._dtypes = dtypes if dtypes is not None else (datatypes.exact, datatypes.inexact)
        self.supports_grad = supports_grad
        self.sample_input_generator = sample_input_generator
        self.reference_input_generator = reference_input_generator
        self.error_input_generator = error_input_generator
        self.benchmark_generator = benchmark_generator
        self.method_variant = method_variant
        self.operator_variant = operator_variant
        self.torch_reference = torch_reference
        self.numpy_reference = numpy_reference
        self.jax_reference = jax_reference
        self.test_directives = test_directives
        self.domain = Domain(*domain)
        self.singularity_fn = singularity_fn
        # singularity_fn_producers are expected to produce a singularity_fn based on a given SampleInput.
        # This can be useful in cases when the definition of the op function invovles kwargs of the input.
        self.singularity_fn_producer = (
            (lambda _: singularity_fn) if singularity_fn_producer is None else singularity_fn_producer
        )
        self.test_torch_compile_executor = test_torch_compile_executor
        self.instantiate_complex_tests = instantiate_complex_tests

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    # TODO Maybe allow sample input generation not using torch?
    # NOTE Today all sample inputs are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def sample_inputs(
        self, device: str | devices.Device, dtype: datatypes.dtype, *, requires_grad: bool = False, **kwargs
    ) -> Generator:
        torch_dtype = to_torch_dtype(dtype)
        return self.sample_input_generator(self, device, torch_dtype, requires_grad, **kwargs)

    def reference_inputs(
        self, device: str | devices.Device, dtype: datatypes.dtype, *, requires_grad: bool = False, **kwargs
    ) -> Generator:
        torch_dtype = to_torch_dtype(dtype)
        return self.reference_input_generator(self, device, torch_dtype, requires_grad, **kwargs)

    def error_inputs(self, device: devices.Device, **kwargs):
        return self.error_input_generator(self, device, **kwargs)

    # NOTE Today all benchmarks are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def benchmarks(self, device: devices.Device, dtype: datatypes.dtype, *, requires_grad: bool = False, **kwargs):
        torch_dtype = to_torch_dtype(dtype)
        return self.benchmark_generator(self, device, dtype, requires_grad, **kwargs)

    def devicetypes(self):
        return set(self._devicetypes)

    # TODO Add per-device dtype support
    def dtypes(self, devicetype: devices.DeviceType = None):
        if devicetype is not None:
            raise NotImplementedError

        return datatypes.resolve_dtypes(self._dtypes)

    def test_decorators(self, test_name, executor, devicetype: devices.DeviceType, dtype: datatypes.dtype):
        return [d.decorator for d in self.test_directives if d.is_active(test_name, executor, devicetype, dtype)]


opinfos: list[OpInfo] = []


def list_opinfos() -> None:
    for opinfo in opinfos:
        print(f"{opinfo.name}")


# Acquires an OpInfo by name
def get_opinfo(name: str) -> OpInfo:
    for opinfo in opinfos:
        if opinfo.name == name:
            return opinfo

    raise RuntimeError(f"Failed to find OpInfo {name}")


#
# Elementwise Unary OpInfos
#

# TODOA Create elementwise unary OpInfo subclass and maybe auto add to list
elementwise_unary_ops = []


# TODO Add small value, large value, and extremal-valued samples
def elementwise_unary_generator(
    op,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    *,
    supports_numbers: bool = True,
    small=False,
    **kwargs,
):
    low = None if op.domain.low is None else max(-9, op.domain.low)
    high = None if op.domain.high is None else min(9, op.domain.high)
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, low=low, high=high, requires_grad=requires_grad, **kwargs
    )

    shapes = (
        # TODO: restore size zero cases
        # (0, 2, 1),
        # (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (4, 2, 4, 5),
    )

    if not small:
        shapes += (
            (1024, 1024),
            (64, 64, 64),
        )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))

    # Noncontiguous inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape, noncontiguous=True))

    # Arbitrarily strided inputs
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        a = a.detach().requires_grad_(requires_grad)
        yield SampleInput(a)


def elementwise_unary_benchmarks(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # name x shape
    cases = (
        ("8x8", (8, 8)),
        ("64x64", (64, 64)),
        ("1024x1024", (1024, 1024)),
    )

    for name, shape in cases:
        yield name, SampleInput(make_arg(shape))


class ElementwiseOpInfo(OpInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ElementwiseUnaryOpInfo(ElementwiseOpInfo):
    def __init__(
        self,
        *args,
        sample_input_generator=elementwise_unary_generator,
        benchmark_generator=elementwise_unary_benchmarks,
        **kwargs,
    ):
        super().__init__(
            *args,
            sample_input_generator=sample_input_generator,
            benchmark_generator=elementwise_unary_benchmarks,
            **kwargs,
        )

        elementwise_unary_ops.append(self)


# NOTE: many PyTorch operations don't accept numbers as inputs,
#   so this helper wraps and unwraps numbers
def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x, **kwargs):
        if isinstance(x, torch.Tensor):
            return op(x, **kwargs)

        return op(torch.tensor(x), **kwargs).item()

    return _fn


#
# Tensor Property OpInfos
#
tensor_properties: list[OpInfo] = []

is_complex_opinfo = OpInfo(
    ltorch.is_complex,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.is_complex),
    dtypes=(datatypes.all_dtypes),
    instantiate_complex_tests=True,
)

tensor_properties.append(is_complex_opinfo)


def _is_cuda_torch(x: torch.Tensor) -> bool:
    return x.is_cuda


is_cuda_opinfo = OpInfo(
    _is_cuda_torch,
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_is_cuda_torch,
    dtypes=(datatypes.all_dtypes),
)

tensor_properties.append(is_cuda_opinfo)


def _is_nested_torch(x: torch.Tensor) -> bool:
    return x.is_nested


is_nested_opinfo = OpInfo(
    _is_nested_torch,
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_is_nested_torch,
    dtypes=(datatypes.all_dtypes),
)

tensor_properties.append(is_nested_opinfo)


def numel_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (0,),
        (4, 2, 0),
        (2, 2),
    )

    for shape in cases:
        yield SampleInput(make(shape))


numel_opinfo = OpInfo(
    ltorch.numel,
    dtypes=(datatypes.floating,),
    sample_input_generator=numel_sample_generator,
    torch_reference=torch.numel,
)
tensor_properties.append(numel_opinfo)


def size_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make_t = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        (),
        (1,),
        (
            2,
            2,
        ),
        (0, 2, 1),
        (2, 0, 1),
    )

    for shape in shapes:
        t = make_t(shape)
        yield SampleInput(t)

        for d in range(len(shape)):
            yield SampleInput(t, d)


size_opinfo = OpInfo(
    ltorch.size,
    sample_input_generator=size_sample_generator,
    torch_reference=torch.Tensor.size,
)

tensor_properties.append(size_opinfo)


opinfos.extend(tensor_properties)


# NOTE: slightly different from generic _elementwise_unary_torch helper
#   because this returns the input when given an unsigned type
@wraps(torch.abs)
def _abs_torch(x: torch.Tensor | Number):
    if isinstance(x, torch.Tensor):
        if datatypes.is_unsigned_dtype(to_dtype(x.dtype)):
            return x
        return torch.abs(x)

    # Handles numbers
    assert isinstance(x, Number)
    if datatypes.is_unsigned_dtype(type(x)):
        return x
    return torch.abs(torch.tensor(x)).item()


abs_opinfo = ElementwiseUnaryOpInfo(
    ltorch.abs,
    supports_grad=True,
    torch_reference=_abs_torch,
    singularity_fn=lambda x: torch.where(x == 0, 1.0, x),
    test_directives=(
        # complex32 cpu abs is sometimes flaky in CI
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.skip(reason="PyTorch doesn't support abs for bool on cpu"),
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Ref - https://github.com/Lightning-AI/lightning-thunder/issues/2363
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            dtypes=(datatypes.float64,),
            executors=("nvfuser",),
        ),
    ),
)

logical_not_opinfo = OpInfo(
    clang.logical_not,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.logical_not),
)
elementwise_unary_ops.append(logical_not_opinfo)

acos_opinfo = OpInfo(
    ltorch.acos,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.acos),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 acos
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(acos_opinfo)

acosh_opinfo = OpInfo(
    ltorch.acosh,
    domain=(1, math.inf),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.acosh),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 acosh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < LooseVersion("0.0.3"),
        ),
    ),
)
elementwise_unary_ops.append(acosh_opinfo)

asin_opinfo = OpInfo(
    clang.asin,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.asin),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 asin
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO: RuntimeError: Unexpected operator type sqrt in d4 = sqrt(double(0.33680657142871817));
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
elementwise_unary_ops.append(asin_opinfo)

asinh_opinfo = OpInfo(
    clang.asinh,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.asinh),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 asinh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < LooseVersion("0.0.3"),
        ),
        # Sets (slightly) more permissive atol and rtol precisions for complex64
        #   vs. assert_close's default atol=1e-5 and rtol=1.3e-6
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-4, rtol=1.3e-6)),
            dtypes=(datatypes.complex64,),
        ),
    ),
)
elementwise_unary_ops.append(asinh_opinfo)

atan_opinfo = OpInfo(
    clang.atan,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.atan),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 atan
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(atan_opinfo)

atanh_opinfo = OpInfo(
    clang.atanh,
    domain=(-1 + eps, 1 - eps),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.atanh),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 atanh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(atanh_opinfo)

bitwise_not_opinfo = OpInfo(
    clang.bitwise_not,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.bitwise_not),
)
elementwise_unary_ops.append(bitwise_not_opinfo)

ceil_opinfo = OpInfo(
    clang.ceil,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.ceil),
    test_directives=(
        # Torch doesn't support bool ceil
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # Torch doesn't support cpu float16 ceil
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch didn't support ceil on exact types before 1.13
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < "1.13",
        ),
    ),
)
elementwise_unary_ops.append(ceil_opinfo)

cos_opinfo = OpInfo(
    clang.cos,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.cos),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 cos
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(cos_opinfo)

cosh_opinfo = OpInfo(
    clang.cosh,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.cosh),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 cosh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(cosh_opinfo)

# digamma is defined for all complex numbers EXCEPT negative integers and zero
digamma_opinfo = OpInfo(
    clang.digamma,
    # NOTE: Restrict domain to avoid singularities because of issue
    # "OpInfos do not use singularity_fn to produce "more stable" samples."
    domain=(eps, math.inf),
    # NOTE: digamma returns NaN for all negative integers. It returns -Inf when x = 0.
    singularity_fn=lambda x: torch.where(x > 0, x, (x - torch.round(x))),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.digamma),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 digamma prior to v2.1
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < LooseVersion("2.1.0"),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch"),
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < LooseVersion("2.1.0"),
        ),
        # NOTE Neither Torch nor NvFuser supports bfloat16 digamma
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # NOTE Torch doesn't support complex digamma
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch"),
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(digamma_opinfo)

erf_opinfo = OpInfo(
    clang.erf,
    supports_grad=True,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.erf),
    test_directives=(
        # Torch doesn't support CPU float16 erf
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support complex erf
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(erf_opinfo)

erfc_opinfo = OpInfo(
    clang.erfc,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.erfc),
    test_directives=(
        # Torch doesn't support CPU float16 erfc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support complex erfc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(erfc_opinfo)

erfcinv_opinfo = OpInfo(
    clang.erfcinv,
    dtypes=(datatypes.floating,),
    # erfcinv is only defined for x in [0, 2]
    # We use [0.3, 0.7] to avoid the stability issues because we're using
    # erfinv(1 - x) as the reference that is less accurate and less stable than
    # erfcinv
    # TODO Use a better reference (SciPy or pyerf)
    domain=(0.3, 0.7),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(lambda x: torch.erfinv(1 - x)),
    test_directives=(
        # Torch doesn't support CUDA bfloat16 erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # The Torch executor doesn't run erfcinv (since torch doesn't have the operation)
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch"),
        ),
        # Torch doesn't support CPU float16 erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch is not accurate enough with native bfloat16
        # torch.erfinv(1 - bfloat16) is far from torch.erfinv(1 - bfloat16.float())
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support complex erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
    ),
)
elementwise_unary_ops.append(erfcinv_opinfo)

erfinv_opinfo = OpInfo(
    clang.erfinv,
    domain=(-1, 1),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.erfinv),
    test_directives=(
        # Torch doesn't support CUDA bfloat16 erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # Torch doesn't support CPU float16 erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support complex erfinv
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
    ),
)
elementwise_unary_ops.append(erfinv_opinfo)

exp_opinfo = OpInfo(
    clang.exp,
    supports_grad=True,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.exp),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 exp
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO: this test fails (slightly out of tolerance) on CI machines
        #   Maybe restrict the test to A100 and H100 cards?
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float64,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
elementwise_unary_ops.append(exp_opinfo)

exp2_opinfo = OpInfo(
    clang.exp2,
    supports_grad=True,
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.exp2),
    test_directives=(
        # Torch doesn't support complex exp2
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
    ),
)
elementwise_unary_ops.append(exp2_opinfo)

expm1_opinfo = OpInfo(
    clang.expm1,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.expm1),
    test_directives=(
        # Torch doesn't support CPU float16 expm1
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support complex expm1
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(expm1_opinfo)

floor_opinfo = OpInfo(
    clang.floor,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.floor),
    test_directives=(
        # Torch doesn't support bool floor
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # Torch doesn't support cpu float16 floor
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch didn't support floor on exact types before 1.13
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < "1.13",
        ),
    ),
)
elementwise_unary_ops.append(floor_opinfo)

# TODO: test_vjp_correctness fails for float64 inputs of larger shapes
# https://github.com/Lightning-AI/lightning-thunder/issues/1991
frexp_opinfo = OpInfo(
    clang.frexp,
    supports_grad=True,
    dtypes=(datatypes.floating,),
    sample_input_generator=partial(elementwise_unary_generator, small=True),
    torch_reference=_elementwise_unary_torch(torch.frexp),
    test_directives=(
        # AssertionError: Scalars are not close!
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("torch", "nvfuser"),
            dtypes=(datatypes.float64,),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
        ),
    ),
)
elementwise_unary_ops.append(frexp_opinfo)

isfinite_opinfo = OpInfo(
    clang.isfinite,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.isfinite),
    test_directives=(
        # Torch preserves the uint8 dtype
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.uint8,),
        ),
    ),
)
elementwise_unary_ops.append(isfinite_opinfo)

isinf_opinfo = OpInfo(
    ltorch.isinf,
    dtypes=(datatypes.all_dtypes),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.isinf),
)
elementwise_unary_ops.append(isinf_opinfo)

isnan_opinfo = OpInfo(
    clang.isnan,
    dtypes=(datatypes.all_dtypes),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.isnan),
)
elementwise_unary_ops.append(isnan_opinfo)

# TODO The domain of rsqrt should be (0, math.inf), but too small values of rsqrt
#   can cause numerical issues in lower precision (like float16 overflowing)
#   We should think about how best to address this
rsqrt_opinfo = OpInfo(
    clang.rsqrt,
    domain=(0.1, math.inf),
    singularity_fn=torch.abs,
    supports_grad=True,
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.rsqrt),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 or complex32 tanh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Used to be an nvFuser bug here; TODO explore removing this xfail
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        # NOTE: low-precision types are too different
        # TODO: verify that thunder is weakly more accurate or reduce precision required in these cases
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32, datatypes.bfloat16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(rsqrt_opinfo)


sigmoid_opinfo = OpInfo(
    clang.sigmoid,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sigmoid),
    test_directives=(
        # torch.sigmoid is not implemented for CPU float16 or complex32
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=(datatypes.float16, datatypes.complex32),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            devicetypes=(devices.DeviceType.CUDA,),
            dtypes=(
                # reciprocal_cuda for ComplexHalf is not implemented in torch
                datatypes.complex32,
            ),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            dtypes=(
                # sometimes fails due to tight tolerances (passes with rtol=1e-4)
                datatypes.complex64,
            ),
        ),
        # test tols are too tight for these half precision tests
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
        ),
        # TODO Investigate this failure due to a significant numeric difference
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
            dtypes=(datatypes.complex64,),
        ),
    ),
)
elementwise_unary_ops.append(sigmoid_opinfo)

sign_opinfo = OpInfo(
    clang.sign,
    sample_input_generator=elementwise_unary_generator,
    singularity_fn=lambda x: x,
    torch_reference=_elementwise_unary_torch(torch.sgn),
    test_directives=(
        # TODO nvFuser needs support for complex sign
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvfuser",),
        ),
        # The finite difference method used in test_vjp_correctness has numerical
        # issues with constant derivative 0.
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(sign_opinfo)

# NOTE signbit is not defined for complex types
signbit_opinfo = OpInfo(
    clang.signbit,
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.signbit),
)
elementwise_unary_ops.append(signbit_opinfo)


silu_opinfo = OpInfo(
    ltorch.silu,
    dtypes=(datatypes.floating,),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.silu),
    test_directives=(
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
        # NOTE: Torch doesn't support CPU float16 silu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # test tols are too tight for these half precision tests
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
        ),
    ),
)
elementwise_unary_ops.append(silu_opinfo)

sin_opinfo = OpInfo(
    clang.sin,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sin),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 or complex32 sin
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(sin_opinfo)

sinh_opinfo = OpInfo(
    clang.sinh,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sinh),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 or complex32 sinh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(sinh_opinfo)

# TODO: refine domain vs. complex domain
sqrt_opinfo = OpInfo(
    clang.sqrt,
    domain=(0, math.inf),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sqrt),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 or complex32 sqrt
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(sqrt_opinfo)


tan_opinfo = OpInfo(
    clang.tan,
    singularity_fn=lambda x: round_remainder(x, torch.pi / 2),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.tan),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvfuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 or complex32 tan
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(tan_opinfo)

tanh_opinfo = OpInfo(
    clang.tanh,
    supports_grad=True,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.tanh),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvfuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 or complex32 tanh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(tanh_opinfo)

# lgamma is defined for all complex numbers EXCEPT negative integers and zero
lgamma_opinfo = OpInfo(
    clang.lgamma,
    domain=(eps, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.lgamma),
    test_directives=(
        # NOTE: Torch doesn't support CPU float16 lgamma
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support CUDA bfloat16 lgamma
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # Torch doesn't support complex lgamma
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(lgamma_opinfo)

log_opinfo = OpInfo(
    clang.log,
    domain=(0, math.inf),
    supports_grad=True,
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.log),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvfuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 or complex32 log
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
elementwise_unary_ops.append(log_opinfo)

log10_opinfo = OpInfo(
    clang.log10,
    domain=(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.log10),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvfuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 log10
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # NOTE: Torch doesn't support complex32 log10
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
    ),
)
elementwise_unary_ops.append(log10_opinfo)

# TODO: need a way to specify that lhs of the domain is open
log1p_opinfo = OpInfo(
    clang.log1p,
    domain=(-1 + eps, math.inf),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.log1p),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
            dtypes=(datatypes.complexfloating,),
        ),
        # NOTE: Torch has an issue: https://github.com/pytorch/pytorch/issues/94333
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # NOTE: Torch doesn't support CPU float16 log1p
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # NOTE: Torch doesn't support complex32 log1p
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # PyTorch didn't support CPU complex log1p before 2.0
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < "2.0",
        ),
    ),
)
elementwise_unary_ops.append(log1p_opinfo)

log2_opinfo = OpInfo(
    clang.log2,
    domain=(0, math.inf),
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.log2),
    test_directives=(
        # TODO investigate nvFuser's implementation here; for complex datatypes
        # nvFuser's tanh might be inaccurate, causing numerical mismatches, but
        # also this concern is potentially stale in 03/2024.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvfuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 log2
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # NOTE: Torch doesn't support complex32 log2
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
    ),
)
elementwise_unary_ops.append(log2_opinfo)

neg_opinfo = OpInfo(
    clang.neg,
    supports_grad=True,
    dtypes=set(datatypes.all_dtypes) - set(datatypes.boolean_dtypes),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.neg),
    instantiate_complex_tests=True,
)
elementwise_unary_ops.append(neg_opinfo)

ndtri_opinfo = OpInfo(
    clang.ndtri,
    # ndtri is well-define in (0, 1), and returns nan outside.
    # Currently, the vjp tests never fail when nan's are encountered,
    # so we force the (0, 1) domain to prioritize the grad corretness now.
    # TODO: change that once vjp can handle nan's.
    domain=(0, 1),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.special.ndtri),
    test_directives=(
        # Torch doesn't support bfloat16 and float16 ndtri
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
        ),
        # Torch doesn't support complex ndtri
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(ndtri_opinfo)

reciprocal_opinfo = OpInfo(
    clang.reciprocal,
    domain=(0 + eps, math.inf),
    singularity_fn=lambda x: x,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.reciprocal),
    test_directives=(
        # Torch doesn't support complex32 reciprocal
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
    ),
)
elementwise_unary_ops.append(reciprocal_opinfo)


def get_elementwise_unary_with_alpha_generator():
    kwargs_list = [{}, {"alpha": -1.0}, {"alpha": 0.5}]
    return get_elementwise_unary_with_kwargs_generator(kwargs_list)


def get_elementwise_unary_with_kwargs_generator(kwargs_list):
    def gen(op, device, dtype, requires_grad):
        samples = elementwise_unary_generator(op, device, dtype, requires_grad)
        for kwargs, sample in itertools.product(kwargs_list, samples):
            yield SampleInput(*sample.args, **kwargs, **sample.kwargs)

    return gen


celu_opinfo = OpInfo(
    ltorch.celu,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_alpha_generator(),
    torch_reference=_elementwise_unary_torch(torch.celu),
    singularity_fn=lambda x: x,
    test_directives=(),
)
elementwise_unary_ops.append(celu_opinfo)


elu_opinfo = OpInfo(
    ltorch.elu,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_alpha_generator(),
    torch_reference=torch.nn.functional.elu,
    # fdm.jvp, which is used in test_vjp_correctness, behaves badly on (-1e-6, 1e-6) for this function
    singularity_fn=lambda x: x,
    test_directives=(),
)
elementwise_unary_ops.append(elu_opinfo)


leaky_relu_opinfo = OpInfo(
    ltorch.leaky_relu,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_kwargs_generator([{}, {"negative_slope": 0.5}]),
    torch_reference=torch.nn.functional.leaky_relu,
    # fdm.jvp, which is used in test_vjp_correctness, behaves badly on (-1e-6, 1e-6) for this function
    singularity_fn=lambda x: x,
    test_directives=(),
)
elementwise_unary_ops.append(leaky_relu_opinfo)


logsigmoid_opinfo = OpInfo(
    ltorch.logsigmoid,
    dtypes=(datatypes.floating,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.nn.functional.logsigmoid,
    domain=(-1, 1),
    test_directives=(),
)
elementwise_unary_ops.append(logsigmoid_opinfo)


mish_opinfo = OpInfo(
    ltorch.mish,
    dtypes=(datatypes.floating,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.nn.functional.mish,
    test_directives=(),
)
elementwise_unary_ops.append(mish_opinfo)


def prelu_generator(op, device, dtype, requires_grad):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        ((), ()),
        ((11,), ()),
        ((11,), (1,)),
        ((4, 3), ()),
        ((4, 3), (3,)),
        ((4, 2, 4, 5), (1,)),
        ((4, 2, 4, 5), (2,)),
    )

    for shape, weight in shapes:
        yield SampleInput(make_arg(shape), make_arg(weight))


prelu_opinfo = OpInfo(
    ltorch.prelu,
    dtypes=(datatypes.inexact,),
    sample_input_generator=prelu_generator,
    torch_reference=torch.nn.functional.prelu,
    singularity_fn=lambda x: x,
    test_directives=(),
)
elementwise_unary_ops.append(prelu_opinfo)


relu_opinfo = OpInfo(
    ltorch.relu,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.relu),
    test_directives=(
        # PyTorch does not support bool and complex types
        # for both the CPU and CUDA relu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8, datatypes.complexfloating),
        ),
        # PyTorch does not support CPU Half relu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO: we might have a tolerance issue here with relu6.
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(relu_opinfo)


rrelu_opinfo = OpInfo(
    ltorch.rrelu,
    dtypes=(datatypes.inexact,),
    sample_input_generator=get_elementwise_unary_with_kwargs_generator([{}, {"lower": 0.1, "upper": 0.8}]),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.rrelu),
    singularity_fn=lambda x: x,
    test_directives=(),
)
elementwise_unary_ops.append(rrelu_opinfo)


relu6_opinfo = OpInfo(
    ltorch.relu6,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.nn.functional.relu6),
    test_directives=(
        # PyTorch does not support bool for both CPU and CUDA relu6
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # TODO: we might have a tolerance issue here with relu6.
        DecorateInfo(
            pytest.mark.xfail(strict=False),
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(relu6_opinfo)


hardsigmoid_opinfo = OpInfo(
    ltorch.hardsigmoid,
    dtypes=(datatypes.floating,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.nn.functional.hardsigmoid),
    singularity_fn=lambda a: torch.where(a > 0, a - 3, a + 3),
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-3, rtol=1e-1)),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
    ),
)
elementwise_unary_ops.append(hardsigmoid_opinfo)


# fdm.jvp, which is used in test_vjp_correctness, behaves badly at jump discontinuties of the partial derviatives
def shrink_singularity_fn_producer(sample: SampleInput):
    lambd = sample.kwargs.get("lambd", 0.5)
    return lambda a: torch.where(a >= 0, a - lambd, a + lambd)


hardshrink_opinfo = OpInfo(
    ltorch.hardshrink,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_kwargs_generator([{}, {"lambd": 0.25}, {"lambd": -0.1}]),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.hardshrink),
    singularity_fn_producer=shrink_singularity_fn_producer,
    test_directives=(),
)
elementwise_unary_ops.append(hardshrink_opinfo)


def soft_plus_singularity_fn_producer(sample):
    beta = sample.kwargs.get("beta", 1.0)
    threshold = sample.kwargs.get("threshold", 20.0)
    return lambda a: a * beta - threshold


softplus_opinfo = OpInfo(
    ltorch.softplus,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_kwargs_generator(
        [{}, {"beta": 0.5}, {"beta": 2.0, "threshold": 10.0}]
    ),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.softplus),
    singularity_fn_producer=soft_plus_singularity_fn_producer,
    test_directives=(),
)
elementwise_unary_ops.append(softplus_opinfo)


softshrink_opinfo = OpInfo(
    ltorch.softshrink,
    dtypes=(datatypes.floating,),
    sample_input_generator=get_elementwise_unary_with_kwargs_generator([{}, {"lambd": 0.25}, {"lambd": 0.1}]),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.softshrink),
    singularity_fn_producer=shrink_singularity_fn_producer,
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-4, rtol=1e-2)),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
    ),
)
elementwise_unary_ops.append(softshrink_opinfo)


softsign_opinfo = OpInfo(
    ltorch.softsign,
    dtypes=(datatypes.inexact,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.nn.functional.softsign),
    singularity_fn=lambda x: x,
    test_directives=(
        # Ref - https://github.com/Lightning-AI/lightning-thunder/issues/2363
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=5e-4, rtol=5e-4)),
            "test_vjp_correctness",
            dtypes=(datatypes.float64,),
            executors=("nvfuser",),
        ),
    ),
)
elementwise_unary_ops.append(softsign_opinfo)


hardswish_opinfo = OpInfo(
    ltorch.hardswish,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.nn.functional.hardswish),
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support CPU Half hardswish
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO: we might have a tolerance issue here with hardsiwsh, a function of relu6
        DecorateInfo(
            pytest.mark.xfail(strict=False),
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(hardswish_opinfo)


def hardtanh_singularity_fn_producer(sample):
    min_val = sample.kwargs.get("min_val", -1.0)
    max_val = sample.kwargs.get("max_val", 1.0)
    mid_point = (min_val + max_val) / 2
    return lambda a: torch.where(a >= mid_point, a - max_val, a - min_val)


hardtanh_opinfo = OpInfo(
    ltorch.hardtanh,
    sample_input_generator=get_elementwise_unary_with_kwargs_generator(
        [{}, {"min_val": 0.5}, {"max_val": 0}, {"min_val": -1.5, "max_val": 2}]
    ),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.hardtanh),
    dtypes=(datatypes.floating,),
    singularity_fn_producer=hardtanh_singularity_fn_producer,
    test_directives=(
        # test_vjp_correctess compares exact derivatives to finite differences,
        # and there are numerical issues for finite differences of (piecewise) constant functions
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(hardtanh_opinfo)


selu_opinfo = OpInfo(
    ltorch.selu,
    dtypes=(datatypes.floating,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.selu),
    test_directives=(
        # Some versions of PyTorch do not support CPU float16 selu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=(datatypes.float16,),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
        # TODO: we might have a tolerance issue here with relu6.
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(selu_opinfo)


tanhshrink_opinfo = OpInfo(
    ltorch.tanhshrink,
    dtypes=(datatypes.inexact,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.nn.functional.tanhshrink),
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 tanhshrink
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
            executors=("nvfuser",),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
    ),
)
elementwise_unary_ops.append(tanhshrink_opinfo)


def threshold_singularity_fn_producer(sample):
    threshold = sample.kwargs.get("threshold")
    return lambda a: a - threshold


threshold_opinfo = OpInfo(
    ltorch.threshold,
    sample_input_generator=get_elementwise_unary_with_kwargs_generator(
        [{"threshold": 0.5, "value": 0.0}, {"threshold": 0.0, "value": 5.0}]
    ),
    torch_reference=torch.nn.functional.threshold,
    dtypes=(datatypes.floating,),
    singularity_fn_producer=threshold_singularity_fn_producer,
    test_directives=(
        # test_vjp_correctess compares exact derivatives to finite differences,
        # and there are numerical issues for finite differences of (piecewise) constant functions
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_unary_ops.append(threshold_opinfo)


round_opinfo = OpInfo(
    clang.round,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.round),
    test_directives=(
        # Torch doesn't support CPU float16 and bool round
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bool8),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Torch doesn't support CUDA bool round
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
elementwise_unary_ops.append(round_opinfo)

trunc_opinfo = OpInfo(
    clang.trunc,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.trunc),
    test_directives=(
        # Torch doesn't support bool trunc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # Torch doesn't support cpu float16 trunc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch didn't support trunc on exact types before 1.13
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
            devicetypes=(devices.DeviceType.CPU,),
            active_if=LooseVersion(torch.__version__) < "1.13",
        ),
        # TODO: nvFuser does not define an integer trunc() and thus compilation
        # fails. They should probably map integer trunc() to an identity op.
        # Until they do, this test won't work for integer types.
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
            dtypes=(datatypes.int32, datatypes.int64),
        ),
    ),
)
elementwise_unary_ops.append(trunc_opinfo)


real_opinfo = OpInfo(
    clang.real,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.real),
    instantiate_complex_tests=True,
    test_directives=(),
)
elementwise_unary_ops.append(real_opinfo)


def imag_error_generator(op, device, **kwargs):
    dtypes = [torch.float32, torch.int64]
    cases = (
        (),
        (5),
        (4, 4),
    )

    err_msg = "imag is not implemented for tensors with non-complex dtypes"
    for dtype, shape in itertools.product(dtypes, cases):
        make = partial(make_tensor, device=device, dtype=dtype)
        yield (SampleInput(make(shape)), RuntimeError, err_msg)


imag_opinfo = OpInfo(
    clang.imag,
    dtypes=(datatypes.complexfloating,),
    sample_input_generator=elementwise_unary_generator,
    error_input_generator=imag_error_generator,
    torch_reference=_elementwise_unary_torch(torch.imag),
    instantiate_complex_tests=True,
    test_directives=(),
)
elementwise_unary_ops.append(imag_opinfo)


clone_opinfo = OpInfo(
    ltorch.clone,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.clone),
    test_directives=(),
)
elementwise_unary_ops.append(clone_opinfo)


square_opinfo = OpInfo(
    ltorch.square,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.square),
)
elementwise_unary_ops.append(square_opinfo)


# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)


#
# Elementwise Binary OpInfos
#

# TODO Create elementwise binary OpInfo subclass and maybe auto add to list
elementwise_binary_ops = []


# TODO Extend this generator
# Generates sample inputs compatible with the elementwise binary primitives
def elementwise_binary_prims_generator(op, device, dtype, requires_grad, **kwargs):
    a = make_tensor((4, 4), device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)
    b = make_tensor((4, 4), device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)

    yield SampleInput(a, b)

    # Tests the inputs are a CPU scalar tensor and a CUDA tensor
    a = make_tensor((4, 4), device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)
    b = make_tensor((), device="cpu", dtype=dtype, requires_grad=requires_grad, **kwargs)

    yield SampleInput(a, b)

    a = make_tensor((), device="cpu", dtype=dtype, requires_grad=requires_grad, **kwargs)
    b = make_tensor((4, 4), device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)

    yield SampleInput(a, b)


# TODO Extend this generator
def elementwise_binary_generator(
    op,
    device,
    dtype,
    requires_grad,
    *,
    no_rhs_numbers: bool = False,
    no_weak_dtypes: bool = False,
    no_rhs_negative_numbers: bool = False,
    **kwargs,
):
    yield from elementwise_binary_prims_generator(op, device, dtype, requires_grad, **kwargs)

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)

    # Tests broadcasting
    a = make((4, 4), **kwargs)
    b = make((4, 1), **kwargs)
    yield SampleInput(a, b)

    if not no_rhs_numbers:
        # Tests tensor x number
        c = make((2, 2), **kwargs)
        d = number(**kwargs)
        if no_rhs_negative_numbers:
            d = abs(d)
        yield SampleInput(c, d)

    if not no_weak_dtypes:
        # Test tensor x scalar tensor with a different dtype
        # We first convert the dtype to its base and then use
        # the table to get the reference dtype.
        base_tdtype = type(datatypes._torch_to_thunder_dtype_map[dtype])

        weak_dtype_table = {
            datatypes.signedinteger: torch.int64,
            datatypes.unsignedinteger: torch.uint8,
            datatypes.floating: torch.float64,
            datatypes.complexfloating: torch.complex64,
            datatypes.bool_: torch.int64,
        }

        e = make((4, 4), **kwargs)
        f = make((), **kwargs, dtype=weak_dtype_table[base_tdtype])

        sample = SampleInput(e, f)

        yield sample


# TODO: update dtypes with Thunder dtypes (when they exist)
add_opinfo = OpInfo(
    clang.add,
    supports_grad=True,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.add,
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
elementwise_binary_ops.append(add_opinfo)

atan2_opinfo = OpInfo(
    clang.atan2,
    dtypes=(datatypes.floating,),
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    # NOTE If x == 0 and y == 0, then atan2(x, y) is undefined.
    singularity_fn=lambda x: x,
    torch_reference=torch.atan2,
    test_directives=(
        # RuntimeError: "atan2_cpu" not implemented for 'Half'
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=(datatypes.float16,),
        ),
    ),
)
elementwise_binary_ops.append(atan2_opinfo)


# NOTE: nvfuser does not currently support uint8, int8, or int16
bitwise_and_opinfo = OpInfo(
    clang.bitwise_and,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.bitwise_and,
)
elementwise_binary_ops.append(bitwise_and_opinfo)

bitwise_or_opinfo = OpInfo(
    clang.bitwise_or,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.bitwise_or,
)
elementwise_binary_ops.append(bitwise_or_opinfo)

bitwise_xor_opinfo = OpInfo(
    clang.bitwise_xor,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.bitwise_xor,
)
elementwise_binary_ops.append(bitwise_xor_opinfo)

# NOTE copysign is not defined for complex numbers
copysign_opinfo = OpInfo(
    clang.copysign,
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.copysign,
    test_directives=(
        # See issue: "flaky test:
        # test_vjp_correctness_copysign_torch_cuda_float64 is flaky"
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
    ),
)
elementwise_binary_ops.append(copysign_opinfo)

# For grad test stability it's better to use wider range of values
elementwise_comparison_generator = partial(elementwise_binary_generator, low=-1000, high=1000)

eq_opinfo = OpInfo(
    clang.eq,
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.eq,
    test_directives=(),
)
elementwise_binary_ops.append(eq_opinfo)


# TODO (mruberry) For some reason the pytest decorators weren't working properly with
#   floor_divide -- I implemented this custom skip as a workaround
def skip(_):
    def fn_(*args, **kwargs):
        pytest.skip("Skipped!")

    return fn_


# NOTE floor division is not defined for complex numbers
floor_divide_opinfo = OpInfo(
    clang.floor_divide,
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=partial(elementwise_binary_generator, exclude_zero=True),
    torch_reference=torch.floor_divide,
    test_directives=(
        # TODO FIXME
        # nvfuser's division operation is true division, so the dtypes are wrong
        DecorateInfo(
            skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
            executors=("nvfuser",),
        ),
        # TODO FIXME Connect to nvfuser's trunc division correctly
        DecorateInfo(
            skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float32,),
            executors=("nvfuser",),
        ),
        # TODO FIXME AssertionError: Tensor-likes are not close!
        DecorateInfo(
            skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            executors=("torch",),
        ),
        # PyTorch doesn't support boolean floor division
        DecorateInfo(
            skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
    ),
)
elementwise_binary_ops.append(floor_divide_opinfo)


def fmod_sample_input_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    a = make((4, 4), **kwargs)
    b = make((4, 4), **kwargs)
    if dtype.is_floating_point or dtype.is_complex:
        # Fmod is differentiable only in the neighborhood where trunc(x / y) is constant.
        # We force x + dx never cross trunc(x) to avoid the finite-differences issues when testing the grad.
        with torch.no_grad():
            a[torch.fmod(a, b) > b - eps] -= eps
            b[torch.fmod(a, b) < eps] += eps

    yield SampleInput(a, b)


fmod_opinfo = OpInfo(
    clang.fmod,
    sample_input_generator=partial(fmod_sample_input_generator, exclude_zero=True, low=1),
    torch_reference=torch.fmod,
    test_directives=(
        # torch doesn't support bool or complex fmod
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8, datatypes.complexfloating)
        ),
        # bfloat16 computation is too inconsistent
        # TODO: improve bfloat16 testing to allow more accurate computations and/or looser
        #   bfloat16 tolerances
        DecorateInfo(pytest.mark.skip, "test_core_vs_torch_consistency", dtypes=(datatypes.bfloat16,)),
    ),
)
elementwise_binary_ops.append(fmod_opinfo)

ge_opinfo = OpInfo(
    clang.ge,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.ge,
    test_directives=(),
)
elementwise_binary_ops.append(ge_opinfo)

gt_opinfo = OpInfo(
    clang.gt,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.gt,
)
elementwise_binary_ops.append(gt_opinfo)

# NOTE Comparing to the reference implementation is important because torch.logical_and
#   doesn't support complexhalf inputs on CPU or CUDA devices
# NOTE Unfortunately, refs.logical_and does not support RHS numbers
logical_and_opinfo = OpInfo(
    clang.logical_and,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch._refs.logical_and,
)
elementwise_binary_ops.append(logical_and_opinfo)

logical_or_opinfo = OpInfo(
    clang.logical_or,
    dtypes=(datatypes.all_dtypes),
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch._refs.logical_or,
)
elementwise_binary_ops.append(logical_or_opinfo)

logical_xor_opinfo = OpInfo(
    clang.logical_xor,
    dtypes=(datatypes.all_dtypes),
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch._refs.logical_xor,
)
elementwise_binary_ops.append(logical_xor_opinfo)


def ldexp_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    a = make((4, 4), **kwargs)
    b = make((4, 4), **kwargs)
    c = make((4, 1), **kwargs)

    yield SampleInput(a, b)
    yield SampleInput(a, c)


def ldexp_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    if torch.cuda.is_available():
        a = make((4, 4), device="cuda")
        b = make((4, 4), device="cpu")

        err_msg = "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu"
        yield (SampleInput(a, b), RuntimeError, err_msg)


ldexp_opinfo = OpInfo(
    ltorch.ldexp,
    supports_grad=True,
    dtypes=(datatypes.all_dtypes),
    sample_input_generator=ldexp_sample_generator,
    error_input_generator=ldexp_error_generator,
    torch_reference=torch.ldexp,
)
elementwise_binary_ops.append(ldexp_opinfo)

le_opinfo = OpInfo(
    clang.le,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.le,
    test_directives=(),
)
elementwise_binary_ops.append(le_opinfo)

lt_opinfo = OpInfo(
    clang.lt,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.lt,
    test_directives=(),
)
elementwise_binary_ops.append(lt_opinfo)


def min_max_singularity_fn_producer(sample):
    a, b = sample.args
    if a.shape == b.shape or b.shape == ():
        return lambda x: x - b if x is a else make_tensor_like(x, low=1)
    return lambda x: x - a if x is b else make_tensor_like(x, low=1)


maximum_opinfo = OpInfo(
    clang.maximum,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch.maximum,
    supports_grad=True,
    singularity_fn_producer=min_max_singularity_fn_producer,
)
elementwise_binary_ops.append(maximum_opinfo)

minimum_opinfo = OpInfo(
    clang.minimum,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch.minimum,
    singularity_fn_producer=min_max_singularity_fn_producer,
)
elementwise_binary_ops.append(minimum_opinfo)

mul_opinfo = OpInfo(
    clang.mul,
    supports_grad=True,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.mul,
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
elementwise_binary_ops.append(mul_opinfo)

ne_opinfo = OpInfo(
    clang.ne,
    # NOTE: comparison is only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.ne,
    test_directives=(),
)
elementwise_binary_ops.append(ne_opinfo)

# NOTE torch.nextafter doens't support RHS numbers
nextafter_opinfo = OpInfo(
    clang.nextafter,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch.nextafter,
    numpy_reference=np.nextafter,
    # NOTE: nextafter is supported by PyTorch only for bfloat16, float32,
    # and float64 arguments (after normal promotion rules) and by NVFuser
    # only for float32, and float64 arguments (after normal promotion rules).
    dtypes=(datatypes.floating,),
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            dtypes=(datatypes.float16, datatypes.bfloat16),
        ),
        # TODO There was an issue with nextafter in PyTorch that should now be
        # resolved; re-enable this and test.
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # TODO There was an issue with nextafter in PyTorch that should now be
        # resolved; re-enable this and test.
        DecorateInfo(
            pytest.mark.skip,
            executors=("torch",),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.7",
        ),
    ),
)
elementwise_binary_ops.append(nextafter_opinfo)


def polygamma_sample_input_generator(op, device, dtype, requires_grad, *, no_rhs_numbers: bool = False, **kwargs):
    rhs_generator = elementwise_unary_generator(op, device, dtype, requires_grad, exclude_zero=True, small=True)
    # NOTE Polygamma grows very fast because of factorial term; Limit lhs values to avoid extremal values.
    lhs_generator = [0, 2, 4]  # range(5)

    for n, rhs_arg in itertools.product(lhs_generator, rhs_generator):
        yield SampleInput(n, rhs_arg.args[0])


polygamma_opinfo = OpInfo(
    ltorch.polygamma,
    # NOTE: Restrict domain to avoid singularities. See issue "OpInfos do not
    # use singularity_fn to produce "more stable" samples"
    # NOTE: polygamma returns NaN, -Inf, or Inf for all negative integers.
    domain=(eps, math.inf),
    sample_input_generator=polygamma_sample_input_generator,
    torch_reference=torch.polygamma,
    test_directives=(
        # NOTE Torch does not support complex, fp16, or bf16 polygamma
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating, datatypes.bfloat16, datatypes.float16),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch"),
            dtypes=(datatypes.complexfloating, datatypes.bfloat16, datatypes.float16),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-6, rtol=1e-6)),
            "test_vjp_correctness",
        ),
    ),
)
elementwise_binary_ops.append(polygamma_opinfo)


def pow_sample_input_generator(op, device, dtype, requires_grad, *, no_rhs_numbers: bool = False, **kwargs):
    default_generator = partial(elementwise_binary_generator, no_rhs_numbers=True)
    yield from default_generator(op, device, dtype, requires_grad, **kwargs)

    # For backward of pow, we need to make sure that when the base is zero, the
    # backward result is not nan.
    yield (
        SampleInput(
            torch.tensor([0.0], device=device, dtype=dtype, requires_grad=requires_grad),
            3.0,
        )
    )


# TODO pow can accept RHS numbers, but not negative RHS numbers when the LHS is an exact tensor
#   we could extend the kwargs on elementwise_binary_generator to account for this when
#   generating the RHS values
pow_opinfo = OpInfo(
    clang.pow,
    sample_input_generator=pow_sample_input_generator,
    torch_reference=None if LooseVersion(torch.__version__) < "1.13" else torch._refs.pow,
    test_directives=(
        # NOTE: PyTorch doesn't support bool pow
        DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),
        # NOTE: PyTorch doesn't support cpu complex32 pow, and doesn't seem to promote it properly
        # NOTE: The CUDA version of this test also fails occasionally -- maybe switch to torch reference?
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # TODO For complex numbers we have some numerical consistency issues.
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvfuser,"),
            dtypes=(datatypes.complex64, datatypes.complex128),
        ),
        # NOTE: PyTorch fails with RuntimeError: "reciprocal_cuda" not implemented for 'Long' occasionally when the exponent is CPU scalar tensor
        # e.g.: x=torch.tensor([[ 6,  5,  1, -8],], device='cuda:0');y=torch.tensor(-1);torch.pow(x,y)
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.int8, datatypes.int16, datatypes.int32, datatypes.int64),
        ),
    ),
)
elementwise_binary_ops.append(pow_opinfo)

remainder_prim_opinfo = OpInfo(
    prims_wrapper(prims.remainder),
    name="prims_remainder",
    dtypes=(datatypes.all_dtypes - datatypes.low_precision_dtypes),
    sample_input_generator=partial(elementwise_binary_prims_generator, exclude_zero=True),
    torch_reference=torch.remainder,
    jax_reference=jax.numpy.remainder if JAX_AVAILABLE else None,
    test_directives=(
        # torch doesn't support bool or complex remainder.
        # torch_reference is inaccurate since it computes in the lower precision dtype.
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8, datatypes.float16, datatypes.bfloat16, datatypes.complexfloating),
        ),
        # Torch executor doesn't support bool or complex remainder
        DecorateInfo(pytest.mark.xfail, dtypes=(datatypes.bool8, datatypes.complexfloating), executors=("torch",)),
        # JAX doesn't support complex remainder
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_jax_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_binary_ops.append(remainder_prim_opinfo)

remainder_torch_opinfo = OpInfo(
    ltorch.remainder,
    sample_input_generator=partial(elementwise_binary_generator, exclude_zero=True),
    torch_reference=torch.remainder,
    test_directives=(
        # torch doesn't support bool or complex remainder.
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8, datatypes.complexfloating)
        ),
        # low precision tests are flaky due to numerical accuracy in CI
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=datatypes.low_precision_dtypes,
        ),
    ),
)
elementwise_binary_ops.append(remainder_torch_opinfo)

sub_opinfo = OpInfo(
    clang.sub,
    supports_grad=True,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.sub,
    test_directives=(
        # torch doesn't support bool sub
        DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),
    ),
)
elementwise_binary_ops.append(sub_opinfo)

true_divide_opinfo = OpInfo(
    clang.true_divide,
    supports_grad=True,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.true_divide,
    singularity_fn=torch.abs,  # true divide has a singularity where the denominator is zero
    test_directives=(
        # torch cpu doesn't support complex32 div
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # torch doesn't support bool true_divide
        DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),
        # See issue "broadcast_in_dim: The size of contiguity must equal to the
        # number of non-broadcasting IterDomains"
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
        # This test sometimes fails in CI
        #   Absolute difference: 12.348295743693598 (up to 1e-05 allowed)
        #   Relative difference: 6.48032003869975e-05 (up to 1.3e-06 allowed)
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("torch",),
            dtypes=(datatypes.float64,),
        ),
        # TODO Investigate this grad difference
        DecorateInfo(
            pytest.mark.skip,
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-3, rtol=1e-3)),
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.float32,),
        ),
    ),
)
elementwise_binary_ops.append(true_divide_opinfo)

zeta_opinfo = OpInfo(
    clang.zeta,
    # NOTE The domain for x is any value greater than 1.
    # y's domain is any value greater than 0.
    domain=(1.0 + eps, math.inf),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.special.zeta,
    test_directives=(
        # NOTE Torch does not support zeta with complex, fp16 or bf16 dtypes
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating, datatypes.bfloat16, datatypes.float16),
        ),
        # NOTE Skip Nvfuser executor because of segmentation fault
        # See https://github.com/NVIDIA/Fuser/issues/922 for more details.
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
            dtypes=(datatypes.exact,),
        ),
        # NOTE See issues 1095 and 1104
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            dtypes=(datatypes.float64,),
        ),
        # NOTE zeta(x, y) returns NaN when x < 1
        DecorateInfo(
            custom_comparator(partial(assert_close, equal_nan=True)),
        ),
    ),
)
elementwise_binary_ops.append(zeta_opinfo)


def div_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, exclude_zero=True)
    number = partial(make_number, dtype=dtype, exclude_zero=True)
    shapes = [((4, 2, 3), (4, 2, 3)), ((4, 2, 3), (2, 3)), ((4, 2, 3), (2, 1))]
    for shape_a, shape_b in shapes:
        for rounding_mode in (None, "trunc", "floor"):
            # numerator, denominator, rounding_mode
            yield SampleInput(make(shape_a), make(shape_b), rounding_mode=rounding_mode)
            yield SampleInput(make(shape_a), number(), rounding_mode=rounding_mode)
            yield SampleInput(number(), make(shape_a), rounding_mode=rounding_mode)


div_opinfo = OpInfo(
    ltorch.div,
    sample_input_generator=div_sample_generator,
    dtypes=(datatypes.exact, datatypes.floating),
    torch_reference=torch.div,
    test_directives=(
        # NOTE: PyTorch doesn't support boolean division
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        # NOTE: bfloat16 and float16 is skipped
        # See: https://github.com/Lightning-AI/lightning-thunder/issues/1724
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
            dtypes=(datatypes.bool8, datatypes.bfloat16, datatypes.float16),
        ),
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
    ),
)
elementwise_binary_ops.append(div_opinfo)


bitwise_left_shift_opinfo = OpInfo(
    ltorch.bitwise_left_shift,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_negative_numbers=True),
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger),
    torch_reference=torch.bitwise_left_shift,
)
elementwise_binary_ops.append(bitwise_left_shift_opinfo)


bitwise_right_shift_opinfo = OpInfo(
    ltorch.bitwise_right_shift,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_negative_numbers=True),
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger),
    torch_reference=torch.bitwise_right_shift,
)
elementwise_binary_ops.append(bitwise_right_shift_opinfo)

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_binary_ops)


def addcmul_addcdiv_sample_generator(op, device, dtype, requires_grad, **kwargs):
    S = 4
    cases = (
        ((S,), (S,), (S,)),
        ((S, S), (S, S), (S, S)),
        ((S, 1), (1, S), (S, 1)),
    )
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)
    for s0, s1, s2 in cases:
        yield SampleInput(make(s0), make(s1), make(s2))
        val = number(**kwargs)
        yield SampleInput(make(s0), make(s1), make(s2), value=val)


addcmul_opinfo = OpInfo(
    ltorch.addcmul,
    sample_input_generator=addcmul_addcdiv_sample_generator,
    torch_reference=torch.addcmul,
    dtypes=(datatypes.exact, datatypes.inexact),
    test_directives=(
        # torch.addcmul doesn't support bool8
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # torch.addcmul doesn't support complex32 on CUDA
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # complex32 has flaky accuracy fails
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            dtypes=(datatypes.complex32,),
        ),
    ),
)
opinfos.append(addcmul_opinfo)


addcdiv_opinfo = OpInfo(
    ltorch.addcdiv,
    sample_input_generator=addcmul_addcdiv_sample_generator,
    torch_reference=torch.addcdiv,
    dtypes=(datatypes.exact, datatypes.inexact),
    test_directives=(
        # torch.addcdiv doesn't support half on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("torch",),
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=(datatypes.float16,),
        ),
        # torch.addcdiv doesn't support complex32
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # torch.addcdiv doesn't support Integer division
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
        ),
        # See issue "flaky test:
        # test_vjp_correctness_addcdiv_nvfuser_cuda_float64"
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
    ),
)
opinfos.append(addcdiv_opinfo)


def lerp_sample_generator(op, device, dtype, requires_grad, **kwargs):
    S = 4
    # start_shape, end_shape, weight_shape
    cases = (
        ((), (), ()),
        ((S,), (S,), (S,)),
        ((S, S), (S, S), (S, S)),
        ((S, 1), (1, S), (S, 1)),
    )
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)
    for start_shape, end_shape, weight_shape in cases:
        # Generates two cases, one with a tensor weight using the shape from the case, one with a number weight
        yield SampleInput(make(start_shape, **kwargs), make(end_shape, **kwargs), make(weight_shape, **kwargs))
        number_weight = number(**kwargs)
        yield SampleInput(make(start_shape, **kwargs), make(end_shape, **kwargs), number_weight)


lerp_opinfo = OpInfo(
    ltorch.lerp,
    sample_input_generator=lerp_sample_generator,
    torch_reference=torch.lerp,
    dtypes=(datatypes.inexact,),
    test_directives=(),
)
opinfos.append(lerp_opinfo)


#
# Conditional and masking operations
#
conditional_and_mask_ops = []


# TODO: add number tensors for value
# TODO: error inputs
def masked_fill_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)

    # pred_shape, a_shape, value
    cases = (
        ((2, 1, 2), (1, 2, 2), number()),
        ((4, 6), (6, 4, 6), number()),
        ((3,), (3,), number()),
    )

    for pred_shape, a_shape, value in cases:
        pred, a = make(pred_shape, dtype=torch.bool, requires_grad=False), make(a_shape)
        yield SampleInput(a, pred, value)


masked_fill_opinfo = OpInfo(
    ltorch.masked_fill,
    supports_grad=True,
    sample_input_generator=masked_fill_sample_generator,
    torch_reference=torch.masked_fill,
)
conditional_and_mask_ops.append(masked_fill_opinfo)


def where_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # pred_shape, a_shape, b_shape
    # NOTE: shapes must be broadcastable
    cases = (
        ((5,), (5,), (5,)),
        ((2, 1, 2), (1, 2, 2), (2, 2, 1)),
    )

    # NOTE: pred must have a boolean dtype
    for pred_shape, a_shape, b_shape in cases:
        pred, a, b = make(pred_shape, dtype=torch.bool, requires_grad=False), make(a_shape), make(b_shape)
        yield SampleInput(pred, a, b)

    # NOTE: requires_grad needs tensor inputs on non-pred.
    if not requires_grad:
        # generate scalar inputs
        dtypes = [float, int, bool, complex]

        for dtype in dtypes:
            pred = make([2, 3], dtype=torch.bool, requires_grad=False)
            a = dtype(1.0)
            b = dtype(0.0)
            yield SampleInput(pred, a, b)


def where_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    err_msg = r"torch.where\(\) does not support only specifying a condition"
    yield (
        SampleInput(
            make(
                5,
            )
        ),
        NotImplementedError,
        err_msg,
    )
    yield (SampleInput(make(2, 1, 2)), NotImplementedError, err_msg)


where_opinfo = OpInfo(
    ltorch.where,
    supports_grad=True,
    sample_input_generator=where_sample_generator,
    error_input_generator=where_error_generator,
    torch_reference=torch.where,
)
conditional_and_mask_ops.append(where_opinfo)


def nan_to_num_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    a = make((4, 4), dtype=dtype, requires_grad=requires_grad)
    double_max = torch.finfo(torch.float64).max
    double_min = torch.finfo(torch.float64).min
    nan = float("nan")
    inf = float("inf")
    if dtype.is_floating_point:
        a = torch.tensor((0, nan, inf, -inf))
    elif dtype.is_complex:
        a = torch.tensor(
            (
                complex(0, 0),
                complex(nan, nan),
                complex(inf, -inf),
                complex(nan, 0),
                complex(nan, inf),
                complex(inf, 0),
                complex(0, inf),
                complex(-inf, 0),
                complex(nan, 5),
                complex(inf, 3),
            )
        )
    # input tensor, nan, posinf, neginf
    cases = (
        (a, None, None, None),
        (a, None, 1.0, None),
        (a, None, None, 1.0),
        (a, None, 1.0, 0.0),
        (a, 1, None, None),
        (a, 1, 1.0, None),
        (a, 1, None, 0.0),
        (a, 1, 1.0, 0.0),
        (a, None, double_max, 0.0),
        (a, 1, double_min, 0.0),
        (a, None, 1.0, double_min),
        (a, 1, 1.0, double_max),
        (a, double_max, None, double_max),
        (a, double_min, double_max, None),
    )

    for a, nan, posinf, neginf in cases:
        yield SampleInput(a, nan, posinf, neginf)


def nan_to_num_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    err_msg = "out is not None which is currently unsupported"
    yield (
        SampleInput(
            make(
                5,
            ),
            None,
            None,
            None,
            make(
                5,
            ),
        ),
        NotImplementedError,
        err_msg,
    )


nan_to_num_opinfo = OpInfo(
    ltorch.nan_to_num,
    sample_input_generator=nan_to_num_sample_generator,
    error_input_generator=nan_to_num_error_generator,
    torch_reference=torch.nan_to_num,
)
conditional_and_mask_ops.append(nan_to_num_opinfo)


def clamp_sample_generator(op, device, dtype, requires_grad, **kwargs):
    cases = (
        ((5,), (5,), (5,)),
        ((3, 1, 2), (1, 2, 2), (3, 2, 1)),
    )
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)

    for a_shape, min_shape, max_shape in cases:
        yield SampleInput(make(a_shape), make(min_shape), make(max_shape))
        # Generates sample inputs with None and number min/max values
        for min_vals, max_vals in itertools.product((min_shape, None, number()), (max_shape, None, number())):
            # skip the type combination Torch doesn't support
            if not isinstance(min_vals, type(max_vals)) or (min_vals is None and max_vals is None):
                continue
            if isinstance(min_vals, tuple):
                min_vals = make(min_vals)
            if isinstance(max_vals, tuple):
                max_vals = make(max_vals)
            yield SampleInput(make(a_shape), min_vals, max_vals)

    # Generates samples contain nan and inf
    if dtype.is_floating_point:
        # Generates extremal value samples
        inf, neginf, nan = float("inf"), float("-inf"), float("nan")
        values = (inf, neginf, nan, 1.0)

        # Constructs all ternary combinations of values
        avals, bvals, cvals = [], [], []
        for a, b, c in itertools.product(values, values, values):
            avals.append(a)
            bvals.append(b)
            cvals.append(c)

        tensor = partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)

        # Generates extremal value combinations, including numbers
        for min_vals, max_vals in itertools.product(
            (bvals, None, inf, neginf, nan, 1.0), (cvals, None, inf, neginf, nan, 1.0)
        ):
            # skip the type combination Torch doesn't support
            if not isinstance(min_vals, type(max_vals)) or (min_vals is None and max_vals is None):
                continue
            if isinstance(min_vals, list):
                min_vals = tensor(min_vals)

            if isinstance(max_vals, list):
                max_vals = tensor(max_vals)

            yield SampleInput(tensor(avals), min_vals, max_vals)


clamp_opinfo = OpInfo(
    ltorch.clamp,
    sample_input_generator=clamp_sample_generator,
    torch_reference=torch.clamp,
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
    test_directives=(
        # see issue "test_vjp_correctness_clamp_nvfuser_cuda_float64 is flaky"
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
conditional_and_mask_ops.append(clamp_opinfo)


def tril_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_diagonal = partial(make_number, dtype=torch.long)

    # shape
    cases = (
        (3, 5, 2),
        (1, 0, 7, 9),
        (6, 6),
        (9, 1),
        (1, 11),
        (7, 2, 3, 5),
    )

    for shape in cases:
        yield SampleInput(make(shape), make_diagonal())


tril_opinfo = OpInfo(
    ltorch.tril,
    sample_input_generator=tril_sample_generator,
    torch_reference=torch.tril,
    test_directives=(
        # Not all PyTorch versions support complex32 tril
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # PyTorch 2.0 doesn't support CUDA bfloat16 tril
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CUDA,),
            dtypes=(datatypes.bfloat16,),
            active_if=(LooseVersion(torch.__version__) < "2.1"),
        ),
    ),
)
conditional_and_mask_ops.append(tril_opinfo)

triu_opinfo = OpInfo(
    ltorch.triu,
    sample_input_generator=tril_sample_generator,
    torch_reference=torch.triu,
    test_directives=(
        # Not all PyTorch versions support complex32 triu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # PyTorch 2.0 doesn't support CUDA bfloat16 triu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CUDA,),
            dtypes=(datatypes.bfloat16,),
            active_if=(LooseVersion(torch.__version__) < "2.1"),
        ),
    ),
)

conditional_and_mask_ops.append(triu_opinfo)

# Puts all elementwise ternary opinfos into the "opinfos" list
opinfos.extend(conditional_and_mask_ops)

#
# Data movement ops
#
data_movement_ops = []

# NOTE The following opinfo for convert_element_type is commented out because each language
#   needs its own dtype conversion (torch -> jax, and torch -> thunder conversions, in particular)
# def convert_element_type_sample_generator(op, device, dtype, requires_grad, **kwargs):
#     a = make_tensor((2, 3, 4), device=device, dtype=dtype, requires_grad=requires_grad)

#     # TODO Add more source and target dtype pairs
#     yield SampleInput(a, torch.float32)


# convert_element_type_opinfo = OpInfo(
#     prims.convert_element_type,
#     sample_input_generator=convert_element_type_sample_generator,
#     torch_reference=torch.Tensor.to,
#     jax_reference=jax.lax.convert_element_type if JAX_AVAILABLE else None,
#     test_directives=(
#         # These usually pass but tols are still too tight to perform these tests
#         DecorateInfo(
#             pytest.mark.skip,
#             "test_vjp_correctness",
#         ),
#     ),
# )
# data_movement_ops.append(convert_element_type_opinfo)


def type_sample_generator_tensor(op, device, dtype, requires_grad, **kwargs):
    # dtype is not None
    # expected to return tensor

    _torch_dtype_to_old_torch_typestring_map = {
        torch.float32: "FloatTensor",
        torch.float64: "DoubleTensor",
        torch.float16: "HalfTensor",
        torch.bfloat16: "BFloat16Tensor",
        torch.uint8: "ByteTensor",
        torch.int8: "CharTensor",
        torch.int16: "ShortTensor",
        torch.int32: "IntTensor",
        torch.long: "LongTensor",
        torch.bool: "BoolTensor",
    }

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    to_dtype = torch.complex128 if dtype.is_complex else torch.float64

    yield SampleInput(make(4, 4, device=device), dtype)
    yield SampleInput(make(4, 4, device=device), dtype=to_dtype)
    # below can be deleted if we don't support strings
    yield SampleInput(make(4, 4, device=device), f"torch.{_torch_dtype_to_old_torch_typestring_map[to_dtype]}")

    # Explictly pass device
    if torch.device(device).type == "cuda":
        yield SampleInput(make(4, 4, device=device), f"torch.cuda.{_torch_dtype_to_old_torch_typestring_map[to_dtype]}")


# kind of redundant?
def type_error_generator_tensor(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    err_msg = r"type\(\): `non_blocking==True` is currently not supported."
    yield SampleInput(make(3, 3), dtype, non_blocking=True), RuntimeError, err_msg


type_opinfo_tensor = OpInfo(
    ltorch.torch_type,
    sample_input_generator=type_sample_generator_tensor,
    error_input_generator=type_error_generator_tensor,
    torch_reference=torch.Tensor.type,
)

data_movement_ops.append(type_opinfo_tensor)


def type_sample_generator_str(op, device, dtype, requires_grad, **kwargs):
    # dtype is None
    # expected to return string
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(make(4, 4))


# kind of redundant?
def type_error_generator_str(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    err_msg = r"type\(\): `non_blocking==True` is currently not supported."
    yield SampleInput(make(3, 3), non_blocking=True), RuntimeError, err_msg


# comparing strings (NOTE: assert_close does not support string comparison)
def string_compare(actual, expected, **kwargs):
    assert actual == expected


type_opinfo_str = OpInfo(
    ltorch.torch_type,
    sample_input_generator=type_sample_generator_str,
    error_input_generator=type_error_generator_str,
    torch_reference=torch.Tensor.type,
    test_directives=(
        DecorateInfo(
            custom_comparator(string_compare),
            "test_core_vs_torch_consistency",
        ),
    ),
)

data_movement_ops.append(type_opinfo_str)


def to_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    to_dtype = torch.complex128 if dtype.is_complex else torch.float64

    # None
    yield SampleInput(make(4, 4))

    # None - copy
    yield SampleInput(make(4, 4), copy=True)

    # device
    yield SampleInput(make(4, 4), device)
    yield SampleInput(make(4, 4), "cpu")
    yield SampleInput(make(4, 4), "cpu", dtype=to_dtype)

    # device - copy
    yield SampleInput(make(4, 4), device, copy=True)
    yield SampleInput(make(4, 4), "cpu", copy=True)
    yield SampleInput(make(4, 4), "cpu", dtype=to_dtype, copy=True)

    # dtype
    yield SampleInput(make(4, 4), dtype)

    # dtype - copy
    yield SampleInput(make(4, 4), dtype, copy=True)

    # device and dtype
    yield SampleInput(make(4, 4), device, dtype)
    yield SampleInput(make(4, 4), "cpu", to_dtype)

    # device and dtype - copy
    yield SampleInput(make(4, 4), device, dtype, copy=True)
    yield SampleInput(make(4, 4), "cpu", to_dtype, copy=True)

    # tensor
    yield SampleInput(make(4, 4), make(2, 2))
    yield SampleInput(make(4, 4), make(2, 2, device="cpu", dtype=to_dtype))

    # tensor - copy
    yield SampleInput(make(4, 4), make(2, 2), copy=True)
    yield SampleInput(make(4, 4), make(2, 2, device="cpu", dtype=to_dtype), copy=True)


to_opinfo = OpInfo(
    ltorch.to,
    sample_input_generator=to_sample_generator,
    torch_reference=torch.Tensor.to,
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, check_device=False)),
            "test_vjp_correctness",
        ),
    ),
)
data_movement_ops.append(to_opinfo)


def view_with_dtype_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    for dst_dtype in {torch.float32, torch.bfloat16, torch.float64} - {dtype}:
        yield SampleInput(make((8, 8)), dtype)


view_with_dtype_opinfo = OpInfo(
    ltorch.view,
    sample_input_generator=view_with_dtype_sample_generator,
    torch_reference=torch.Tensor.view,
)
data_movement_ops.append(view_with_dtype_opinfo)


def cuda_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # None
    # cpu -> cuda
    # default cuda -> default cuda (no-op)
    yield SampleInput(make(4, 4))

    # Explictly pass device
    # default cuda -> default cuda
    if torch.device(device).type == "cuda":
        yield SampleInput(make(4, 4), device)


def cuda_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device="cpu", dtype=torch.float, requires_grad=False)

    err_msg = "`non_blocking==True` is currently not supported."
    yield SampleInput(make(3, 3), non_blocking=True), RuntimeError, err_msg

    err_msg = "Invalid device cpu, must be cuda device"
    yield SampleInput(make(3, 3), device="cpu"), RuntimeError, err_msg


cuda_opinfo = OpInfo(
    ltorch.cuda,
    sample_input_generator=cuda_sample_generator,
    error_input_generator=cuda_error_generator,
    torch_reference=torch.Tensor.cuda,
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            active_if=not torch.cuda.is_available(),
        ),
        DecorateInfo(
            custom_comparator(lambda a, b, **kwargs: assert_close(a, b.to(a.device), atol=1e-5, rtol=1e-5)),
            "test_vjp_correctness",
        ),
    ),
)
data_movement_ops.append(cuda_opinfo)


def type_as_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, low=-1, high=1, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        (),
        (0,),
        (2,),
        (1, 2),
        (1, 2, 3),
    )

    for a_shape, b_shape in itertools.product(shapes, shapes):
        yield SampleInput(make(a_shape), make(b_shape))
        yield SampleInput(make(a_shape), make(b_shape, dtype=torch.float32))
        # Tests when inputs from different devices
        yield SampleInput(make(a_shape), make(b_shape, device="cpu"))


type_as_sample = OpInfo(
    ltorch.type_as,
    sample_input_generator=type_as_sample_generator,
    torch_reference=torch.Tensor.type_as,
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-5, rtol=1e-5)),
            "test_vjp_correctness",
        ),
    ),
)
data_movement_ops.append(type_as_sample)


def long_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input shape
    shapes = (
        (),
        (0,),
        (2,),
        (1, 2),
        (1, 2, 3),
    )

    for shape in shapes:
        yield SampleInput(make(shape))


long_opinfo = OpInfo(
    ltorch.long,
    sample_input_generator=long_sample_generator,
    torch_reference=torch.Tensor.long,
)

opinfos.extend(data_movement_ops)

#
# Shape ops
#
shape_ops = []


# TODO: these samples could be improved
def broadcast_in_dim_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # The first 5 test cases below are taken from JAX's broadcast_in_dim tests
    #   https://github.com/google/jax/blob/main/tests/lax_test.py#L1171

    # inshape, outshape, dims
    cases = (
        ([5], [5], [0]),
        ([2], [2, 2], [0]),
        ([2], [2, 2], [1]),
        ([2], [2, 3], [0]),
        ([], [2, 3], []),
        ([1], [2, 3], [1]),
        ((4, 6, 3, 1), (5, 4, 7, 6, 3, 6, 6), (1, 3, 4, 5)),
    )

    for inshape, outshape, dims in cases:
        a = make(inshape)
        yield SampleInput(a, outshape, dims)


def broadcast_in_dim_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    # inshape, outshape, dims, ex_info, err_msg_match or None for universal match
    # NOTE: all these tests xfail, so err_msg_match
    # is not yet specified.
    cases = (
        # broadcast dimensions must be strictly ascending
        ((2, 2), (2, 2), (1, 0), RuntimeError, None),
        # broadcast dimensions must have the same length as a.ndim
        ((3, 2, 2), (3, 2, 2), (0, 1), RuntimeError, None),
        ((3, 2, 2), (3, 2, 2), (0, 1, 2, 3), RuntimeError, None),
        # Invalid outshape
        ((3, 2, 2), (6, 2, 2), (0, 1, 2), RuntimeError, None),
        ((3, 2, 2), (3, 1, 2), (0, 1, 2), RuntimeError, None),
    )

    for inshape, outshape, dims, ex_info, err_msg_match in cases:
        a = make(inshape)
        yield SampleInput(a, outshape, dims), ex_info, err_msg_match


broadcast_in_dim_opinfo = OpInfo(
    prims.broadcast_in_dim,
    sample_input_generator=broadcast_in_dim_sample_generator,
    error_input_generator=broadcast_in_dim_error_generator,
    jax_reference=jax.lax.broadcast_in_dim if JAX_AVAILABLE else None,
    test_directives=(
        # AttributeError("module 'thunder.torch' has no attribute 'gt'"
        DecorateInfo(
            pytest.mark.xfail,
            "test_errors",
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(broadcast_in_dim_opinfo)


def cat_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shapes, dim
    cases = [
        ([(3,)], 0),  # single tensor provided
        # 1D
        ([(2,), (3,)], 0),
        ([(1,), (1,)], 0),
        ([(2,), (4,)], 0),
        ([(1,), (2,), (3,)], 0),
        ([(0,), (2,)], 0),
        ([(0,), (2,)], -1),
        ([(2, 3), (2, 4)], 1),
        ([(2, 1), (2, 1)], 1),
        ([(2, 3), (2, 4), (2, 5)], 1),
    ]

    for shapes, dim in cases:
        yield SampleInput(*[make(s) for s in shapes], dim=dim)

    # Tests concatenating with a tensor broadcast along the concatenation dimension
    a = make((5,))
    b = make((1,)).expand((5,))
    yield SampleInput(a, b, dim=0)


def cat_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shapes, dim, exception type, error message match or None for universal match
    cases = [
        ([], 0, RuntimeError, "expects a non-empty list of tensors"),
        ([(2,), (2,)], 1, IndexError, "Expected dimension in inclusive range of -1 and 0"),  # pos dim
        ([(1,), (2,)], -2, IndexError, "Expected dimension in inclusive range of -1 and 0"),  # neg dim
        ([(2,), (2, 3)], 0, RuntimeError, "Attempted to concatenate tensors of different dimension: got 1 and 2"),
        ([(2, 3), (4, 5)], 0, RuntimeError, "Sizes of tensors must match except in dimension"),
    ]

    for shapes, dim, exc_type, err_msg_match in cases:
        yield SampleInput(*[make(s) for s in shapes], dim=dim), exc_type, err_msg_match


# nvfuserex_impl.to_descriptors refuses to take a **nested** list of tensors,
# reporting `ValueError: unrecognized type in arguments: <class 'list'>`.
# `cat_wrapper` is created to work around that.
def cat_wrapper(*args, dim):
    return ltorch.cat(args, dim=dim)


cat_opinfo = OpInfo(
    cat_wrapper,
    supports_grad=True,
    sample_input_generator=cat_sample_generator,
    error_input_generator=cat_error_generator,
    torch_reference=lambda *args, dim: torch.cat(args, dim=dim),
    test_torch_compile_executor=True,
    test_directives=(
        # There's a bug in torch.compile + torch.cat for empty tensors in 2.1.0
        DecorateInfo(
            pytest.mark.xfail(strict=True),
            "test_core_vs_torch_consistency",
            active_if=(LooseVersion(torch.__version__) < "2.2.0"),
            executors=("torchcompile",),
        ),
        DecorateInfo(
            pytest.mark.xfail(strict=True),
            "test_vjp_correctness",
            active_if=(LooseVersion(torch.__version__) < "2.2.0"),
            executors=("torchcompile",),
        ),
        DecorateInfo(
            pytest.mark.xfail(strict=True),
            active_if=(nvfuser_version < "0.1.7"),
            executors=("nvFuser",),
        ),
    ),
)
shape_ops.append(cat_opinfo)


def diagonal_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Input shape, offset, dim1, dim2
    cases = (
        ((4, 7, 9), 0, 0, 1),
        ((1, 2, 0, 3), -1, 0, -1),
        ((4, 7), 1, -2, -1),
        ((4, 3, 5, 7), 2, 1, 2),
    )

    for shape, offset, dim1, dim2 in cases:
        yield SampleInput(make(shape), offset, dim1, dim2)


diagonal_opinfo = OpInfo(
    ltorch.diagonal,
    sample_input_generator=diagonal_sample_generator,
    torch_reference=torch.diagonal,
    test_directives=(
        # thunder.torch.diagonal meta function is not correctly implemented for
        # input case ((1, 2, 0, 3), -1, 0, -1)
        DecorateInfo(pytest.mark.xfail(strict=True), "test_vjp_correctness"),
        # See: [Fix runtime-trace shape/dtype/device mismatch]
        # In https://github.com/Lightning-AI/lightning-thunder/pull/2069,
        # torch-consistency test checks the parity of shape/dtype/device
        # between runtime and trace. This needs to be a temporary decorator
        # and we are working on resolving the mismatches.
        DecorateInfo(
            pytest.mark.xfail(strict=True),
            "test_core_vs_torch_consistency",
        ),
    ),
)
shape_ops.append(diagonal_opinfo)


def expand_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Input shape, arg shape
    cases = (
        ((), ()),  # Scalar identity
        ((), (3, 4, 5)),  # Broadcast scalar tensor, adding dims
        ((0,), (0,)),  # Zero dim tensor identity
        ((0,), (-1,)),  # Scalar wildcard
        ((1, 0), (1, 0)),  # Nonleading zero dim
        # Empty output cases
        ((1, 0), (0, -1)),
        ((1, 0), (0, 0)),  # Empty input (one broadcast, one zero)
        ((1, 1), (0, 0)),  # Non-empty fully broadcast input
        ((1, 3), (1, 1, 3)),  # Add dim
        ((1, 1), (1, 2)),  # Broadcast trailing dim
        ((1, 1), (2, 1)),  # Broadcast leading dim
        ((2, 2), (-1, 2)),  # Wildcard leading dim
        ((1, 1), (1, 2, -1)),  # Broadcast trailing dim, wildcard, add dim
        ((1, 1), (3, -1, -1)),  # Broadcast leading dim, wildcard, add dim
    )

    # Yield both, to make sure the arguments get packed/unpacked correctly.
    # expand(tensor, args...) and expand(tensor, (args...)) should be equivalent.
    for ishape, argshape in cases:
        yield SampleInput(make(ishape), argshape)
        if argshape != ():
            yield SampleInput(make(ishape), *argshape)


def expand_error_generator(op, device, *, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # Input shape, arg shape, exception type, error message match or None for universal match
    cases = [
        ((0,), (1,), RuntimeError, "attempting to expand a dimension of length 0"),
        ((0,), (2,), RuntimeError, "attempting to expand a dimension of length 0"),
        # TODO: Bug - "Found invalid length [IntegerProxy name=i0, value=-1]"
        # is at least confusing, and might be wrong.
        ((1,), (-1, 2), RuntimeError, None),  # Expand nonexisting dim
        ((2, 2), (2, 4), RuntimeError, "attempting to expand a dimension of length 2"),
        # TODO: Bug - "Found invalid length [IntegerProxy name=i0, value=-1]"
        # is at least confusing, and might be wrong.
        ((1, 1), (-1, 3, -1), RuntimeError, None),  # Leading wildcard, expand, add dim with trailing wildcard
    ]

    for ishape, argshape, exc_type, err_msg_match in cases:
        yield SampleInput(make(ishape), argshape), exc_type, err_msg_match
        yield SampleInput(make(ishape), *argshape), exc_type, err_msg_match


expand_opinfo = OpInfo(
    ltorch.expand,
    sample_input_generator=expand_sample_generator,
    error_input_generator=expand_error_generator,
    torch_reference=torch.Tensor.expand,
    test_directives=(
        # vjp not yet implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
    ),
)
shape_ops.append(expand_opinfo)


def expand_as_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Input shape, output shape
    cases = (
        ((), ()),  # Scalar identity
        ((), (3, 4, 5)),  # Broadcast scalar tensor, adding dims
        ((0,), (0,)),  # Zero dim tensor identity
        ((1, 0), (1, 0)),  # Nonleading zero dim
        ((1, 0), (0, 0)),  # Empty input (one broadcast, one zero)
        ((1, 1), (0, 0)),  # Non-empty fully broadcast input
        ((1, 3), (1, 1, 3)),  # Add dim
        ((1, 1), (1, 2)),  # Broadcast trailing dim
        ((1, 1), (2, 1)),  # Broadcast leading dim
    )

    for ishape, oshape in cases:
        yield SampleInput(make(ishape), make(oshape))


def expand_as_error_generator(op, device, *, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # Input shape, output shape, exception type, error message match or None for universal match
    cases = [
        ((0,), (1,), RuntimeError, "attempting to expand a dimension of length 0"),
        ((1,), (), RuntimeError, "expand: the requested shape has too few dimensions!"),
        ((0,), (2,), RuntimeError, "attempting to expand a dimension of length 0"),
        ((2, 2), (2, 4), RuntimeError, "attempting to expand a dimension of length 2"),
    ]

    for ishape, oshape, exc_type, err_msg_match in cases:
        yield SampleInput(make(ishape), make(oshape)), exc_type, err_msg_match


expand_as_opinfo = OpInfo(
    ltorch.expand_as,
    sample_input_generator=expand_as_sample_generator,
    error_input_generator=expand_as_error_generator,
    torch_reference=torch.Tensor.expand_as,
    test_directives=(
        # vjp not yet implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
    ),
)
shape_ops.append(expand_as_opinfo)


def flatten_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shape, start_dim, end_dim
    cases = (
        ((1, 3), 0, -1),
        ((2), 0, -1),
        ((3, 7, 4, 2, 5, 2), 1, 3),
        ((3, 7, 4, 2, 5, 2), 2, 5),
        ((), 0, 0),
    )

    for shape, start_dim, end_dim in cases:
        yield SampleInput(make(shape), start_dim, end_dim)


flatten_opinfo = OpInfo(
    ltorch.flatten,
    sample_input_generator=flatten_sample_generator,
    torch_reference=torch.flatten,
)
shape_ops.append(flatten_opinfo)


def unbind_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shape, dim
    cases = (
        ((1, 3), 0),
        ((1, 3), -1),
        ((2), 0),
        ((3, 7, 4, 2, 5, 2), 1),
        ((3, 7, 4, 2, 5, 2), 3),
    )

    for shape, dim in cases:
        yield SampleInput(make(shape), dim)


unbind_opinfo = OpInfo(
    ltorch.unbind,
    sample_input_generator=unbind_sample_generator,
    torch_reference=torch.unbind,
)
shape_ops.append(unbind_opinfo)


def unfold_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((), 0, 1, 3),
        ((), -1, 0, 5),
        ((0,), 0, 0, 1),
        ((8,), 0, 2, 1),
        ((6, 2), 0, 2, 2),
    )

    for shape, dim, size, step in cases:
        yield SampleInput(make(shape), dim, size, step)


def unfold_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    cases = (
        ((), 0, 2, 1, RuntimeError, "Maximum size for tensor at dimension 0 is 1 but size is 2"),
        ((0,), 0, 0, -1, RuntimeError, "Step is -1 but must be > 0"),
        ((8,), 1, 2, 1, IndexError, r"Dimension out of range \(expected to be in range of \[-1, 0\], but got 1\)"),
        ((8,), 0, -5, 1, RuntimeError, "Size is -5 but must be >= 0"),
        ((8,), 0, 10, 1, RuntimeError, "Maximum size for tensor at dimension 0 is 8 but size is 10"),
    )

    for shape, dim, size, step, err_type, err_msg in cases:
        yield SampleInput(make(shape), dim, size, step), err_type, err_msg


unfold_opinfo = OpInfo(
    clang.unfold,
    sample_input_generator=unfold_sample_generator,
    error_input_generator=unfold_error_generator,
    torch_reference=torch.Tensor.unfold,
    test_directives=(
        # See [Fix runtime-trace shape/dtype/device mismatch]
        DecorateInfo(
            pytest.mark.xfail(strict=True),
            "test_core_vs_torch_consistency",
        ),
    ),
)

shape_ops.append(unfold_opinfo)


def flip_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    for sample in elementwise_unary_generator(op, device, dtype, requires_grad, **kwargs):
        a, *_ = sample.args

        # No flip
        yield SampleInput(a, ())

        # Flip everything
        yield SampleInput(make(a.shape), tuple(range(a.ndim)))

        # Flip last dim, even for scalars.
        yield SampleInput(make(a.shape), (-1,))

        # 0-1D cases are exhausted above.
        # Now we can perform more interesting manipulations with ndim > 1
        if a.ndim > 1:
            # pos/neg-only dims
            for dims in (tuple(range(-a.ndim, 0)), tuple(range(a.ndim))):
                # Flip everything but a single dimension
                for i, _ in enumerate(dims):
                    yield SampleInput(make(a.shape), dims[:i] + dims[i + 1 :])


def flip_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device)

    # Cases with scalar inputs
    yield (SampleInput(make(), (1, -1)), RuntimeError, "Expected dims(.*?) of length 1")
    yield (SampleInput(make(), (-2,)), RuntimeError, r"Expected dim(.*?) in range \[-1, 0\]")
    yield (SampleInput(make(), (0.0,)), RuntimeError, "Expected dim(.*?) to be a sequence of integers")

    # Cases with non-scalar inputs
    yield (
        SampleInput(make(1, 1), (1, -1)),
        RuntimeError,
        # 1 and -1 are the same dimesion
        "Duplicate value in list of dimensions",
    )
    yield (SampleInput(make(1, 1), (3,)), IndexError, "Dimension out of range")


flip_opinfo = OpInfo(
    ltorch.flip,
    sample_input_generator=flip_sample_generator,
    error_input_generator=flip_error_generator,
    torch_reference=torch.flip,
)
shape_ops.append(flip_opinfo)


def getitem_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # NOTE PyTorch does not allow negative steps
    # a.shape, key
    basic_indexing_cases = (
        # Fully specified slicing
        ((5, 5), (slice(1, 3, 1), slice(2, 4, 2))),
        ((11, 23), (slice(4, 9, 6), slice(3, 21, 4))),
        ((11, 23), (slice(4, 9, 33), slice(3, 21, 1))),
        # NOTE: PyTorch allows start > stop and will return a 0 length dim
        ((5, 3), (slice(3, 1), slice(1, 2))),
        # NOTE: NumPy allows slicing beyond the end of a dimension
        ((5, 3), (slice(6, 7), slice(0, 2))),
        ((5, 3), (slice(6, 2), slice(0, 2))),
        ((5, 3), (slice(1, 9), slice(0, 2))),
        # Inferred start
        ((5, 3), (slice(None, 9), slice(0, 2))),
        # Inferred end
        ((5, 3), (slice(2, None), slice(0, 2))),
        # Inferred start and end
        ((5, 3), (slice(None, None), slice(0, 2))),
        # Negative start and stop
        ((5, 3), (slice(-3, -1), slice(0, -2))),
        ((5, 3), (slice(-4, -1), slice(-1, -2))),
        # Partially specified slicing
        ((5, 3), (slice(-4, -1),)),
        ((5, 3), slice(-4, -1)),
        # Slicing and numbers
        # TODO: grad failure for the commented cases below! FIX IT!
        # ((1, 5, 3), (0, slice(2, 3), 2)),
        # ((1, 5, 3), (-1, slice(2, 3), -2)),
        # All numbers
        # ((1, 5, 3), (-1, 3, -2)),
        # Ellipses
        ((1, 5, 3), (..., slice(1, 2))),
        ((1, 5, 3), (0, ..., slice(1, 2))),
        ((1, 5, 3), ...),
        # Newaxis/None
        # NOTE: nvfuser cannot handle more than 8 dims.
        ((1, 5, 3), (None, 0, None, 2, ..., None, None)),
        ((1, 5, 3), (None, None, 0, None, 2, ..., None, None)),
        ((1, 5, 3), None),
        # Addtl. cases
        # NOTE: nvfuser cannot handle more than 8 dims.
        ((7, 9, 5), (slice(2, 6, 2), None, ..., slice(3, 7), None, 2, None)),
        ((11, 7, 9, 5), (None, slice(2, 6, 2), None, ..., slice(3, 7), None, 2, None)),
        # Basic cases + advanced indexing that dispatches to basic indexing
        ((5,), ([-1],)),
        ((5,), (..., [-1])),
        ((5,), ([-1], ...)),
        ((5, 5), (slice(1, 3, 1), [-1])),
        ((5, 5), (slice(1, 3, 1), [-3])),
        ((2, 2, 2), (slice(None, None), (-1,), slice(None, None))),
        ((2, 2), (..., [-1])),
        # check performance optimization regarding slice_prim
        ((1, 5, 3), (slice(0, 2), slice(0, 5), slice(0, 4))),
        ((4, 5, 3), (slice(0, 2, 2), slice(0, 5, 3), slice(0, 4, 2))),
        # This sample shows inconsistencies between PyTorch and Numpy
        # >>> t = torch.rand(2, 2, 2)
        # >>> n = np.random.rand(2, 2, 2)
        # >>> t.shape; n.shape
        # torch.Size([2, 2, 2])
        # (2, 2, 2)
        # >>> t.shape
        # torch.Size([2, 2, 2])
        # >>> n.shape
        # (2, 2, 2)
        # >>> t[0, ..., [-1]].shape
        # torch.Size([2, 1])
        # >>> n[0, ..., [-1]].shape
        # (1, 2)
        # ((2, 2, 2), (0, ..., [-1])),
        # TODO: grad failure for the commented cases below! FIX IT!
        # ((1, 5, 3), ([-1], None, None, 0, None, 2, ..., None, None)),
        # ((1, 5, 3), (None, [-1], None, 0, None, 2, ..., None, None)),
        # ((1, 5, 3), (None, None, 0, None, 2, ..., [-1], None, None)),
        # ((1, 5, 3), (None, None, 0, [-1], None, 2, ..., None, None)),
    )

    for shape, key in basic_indexing_cases:
        a = make(shape)
        yield SampleInput(a, key)

    def make_idx(dim_length: int, indices: int):
        return make_tensor((indices,), low=-dim_length, high=dim_length, device=device, dtype=torch.int64)

    # NOTE Advanced indexing currently supports:
    #   - a 0D or 1D integer tensor
    #   - a series of one or more 0D or 1D integer tensors containing at most one ellipsis as the first sequence element and at least one sequence element

    # 1D tensor cases
    a = make((5, 4, 7))
    idx = make_idx(5, 12)
    yield SampleInput(a, idx)

    # Empty idx
    a = make((5, 4, 7))
    idx = make_idx(5, 0)
    yield SampleInput(a, idx)

    # 0D tensor
    a = make((5, 4, 7))
    idx = make_idx(5, 1).squeeze()
    yield SampleInput(a, idx)

    # Tensor with no elements
    a = make((5, 0, 7))
    idx = make_idx(5, 9)
    yield SampleInput(a, idx)

    # n-dimensional tensor advanced indexing cases
    def make_nd_idx(dim_length: int, indices: int, ndim: int):
        shape = (indices,) * ndim
        return make_tensor(shape, low=-dim_length, high=dim_length, device=device, dtype=torch.int64)

    # 2D tensor index
    a = make((5, 4, 7))
    idx = make_nd_idx(5, 3, 2)  # shape (3, 3)
    yield SampleInput(a, idx)

    # 3D tensor index
    a = make((5, 4, 7, 3))
    idx = make_nd_idx(5, 2, 3)  # shape (2, 2, 2)
    yield SampleInput(a, idx)

    # Broadcasting n-dim indices
    a = make((5, 4, 7, 3))
    idx1 = make_nd_idx(5, 2, 2)  # shape (2, 2)
    idx2 = make_nd_idx(4, 1, 2)  # shape (1, 2), will broadcast
    yield SampleInput(a, (idx1, idx2))

    # Sequence cases

    # Fully specified
    a = make((5, 4, 7))
    idx0 = make_idx(5, 9)
    idx1 = make_idx(4, 1).squeeze()
    idx2 = make_idx(7, 9)
    yield SampleInput(a, (idx0, idx1, idx2))

    # Partially specified
    a = make((5, 4, 7))
    idx0 = make_idx(5, 3)
    idx1 = make_idx(4, 3)
    yield SampleInput(a, (idx0, idx1))

    a = make((5, 4, 7, 2, 1))
    idx0 = make_idx(5, 3)
    idx1 = make_idx(4, 3)
    yield SampleInput(a, (idx0, idx1))

    a = make((5, 4, 7, 2, 1))
    idx0 = make_idx(5, 4)
    idx1 = make_idx(4, 4)
    idx2 = make_idx(7, 1)
    yield SampleInput(a, (idx0, idx1, idx2))

    a = make((5, 4, 7, 2, 1))
    idx0 = make_idx(5, 4)
    idx1 = make_idx(4, 4)
    idx2 = make_idx(7, 1).squeeze()
    yield SampleInput(a, (idx0, idx1, idx2))

    # Ellipsis
    a = make((5, 2, 9, 4, 7))
    idx0 = make_idx(4, 8)
    idx1 = make_idx(7, 1)
    yield SampleInput(a, (Ellipsis, idx0, idx1))

    # Ellipsis indexing into a tensor with no elements
    a = make((5, 0, 9, 4, 7))
    idx0 = make_idx(7, 9)
    yield SampleInput(a, (Ellipsis, idx0))

    # list indexing
    a = make((5, 4, 7))
    yield SampleInput(a, ([1, 2]))

    # list indexing into a tensor with no elements
    a = make((5, 0, 7))
    yield SampleInput(a, [1, 2])

    # list indexing with tensor indexing
    a = make((5, 4, 7))
    idx = make_idx(5, 12)
    yield SampleInput(a, (idx, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    # list indexing with Ellipsis
    a = make((5, 2, 9, 4, 7))
    idx0 = make_idx(4, 8)
    idx1 = make_idx(7, 1)
    yield SampleInput(a, (Ellipsis, idx0, [0]))

    # basic indexing and advanced indexing together
    a = make((5, 5, 7))
    yield SampleInput(a, (None, [1, 2]))
    a = make((5, 5))
    yield SampleInput(a, (slice(1, 4), [1, 2]))
    yield SampleInput(a, (slice(None, None, None), [1, 2]))
    a = make((5, 5))
    yield SampleInput(a, (1, [1, 2]))

    # Ellipsis, basic, and advanced indexing
    a = make((5, 5, 7))
    yield SampleInput(a, (Ellipsis, None, [1, 2]))
    a = make((5, 5, 9, 5))
    yield SampleInput(a, (Ellipsis, slice(1, 4), None, [1, 2]))
    a = make((5, 5))
    yield SampleInput(a, (Ellipsis, 1, [1, 2]))

    # mixed order basic and advanced indexing
    a = make((5, 5, 7))
    yield SampleInput(a, ([1, 2], None))
    a = make((5, 5))
    yield SampleInput(a, ([1, 2], slice(1, 4)))
    yield SampleInput(a, ([1, 2], slice(None, None, None)))
    a = make((5, 5))
    yield SampleInput(a, ([1, 2], 1))

    # Additional n-dimensional and None index test cases

    # None with n-dimensional advanced indexing
    a = make((5, 5, 7))
    idx = make_nd_idx(5, 2, 2)  # shape (2, 2)
    yield SampleInput(a, (None, idx))  # None before n-dim index
    yield SampleInput(a, (idx, None))  # None after n-dim index

    # Multiple None with advanced indexing
    a = make((5, 5, 7))
    yield SampleInput(a, (None, [1, 2], None))
    yield SampleInput(a, (None, None, [1, 2]))

    # None with multiple advanced indices
    a = make((5, 5, 7, 3))
    idx1 = make_idx(5, 3)
    idx2 = make_idx(5, 3)
    yield SampleInput(a, (None, idx1, idx2))
    yield SampleInput(a, (idx1, None, idx2))
    yield SampleInput(a, (idx1, idx2, None))

    # None with n-dimensional indices
    a = make((10, 10, 10))
    idx = make_nd_idx(10, 3, 3)  # shape (3, 3, 3)
    yield SampleInput(a, (None, idx, None))

    # Complex mixed indexing
    a = make((5, 5, 7, 3))
    yield SampleInput(a, (None, slice(1, 4), [1, 2], None))
    yield SampleInput(a, (slice(1, 4), None, [1, 2]))

    # None with list indexing
    a = make((5, 5, 7))
    yield SampleInput(a, (None, [0, 2, 4]))
    yield SampleInput(a, ([0, 2, 4], None, None))

    # Edge case: all None except one advanced index
    a = make((5, 5, 7))
    yield SampleInput(a, (None, None, [1]))

    # Broadcasting with None
    a = make((5, 5, 7))
    idx1 = make_idx(5, 1)  # shape (1,)
    idx2 = make_idx(5, 3)  # shape (3,)
    yield SampleInput(a, (None, idx1, idx2))  # Will broadcast to (1, 3)


# NOTE getitem intentionally defines 3 references, since advanced indexing is probably
#   the most complicated operation that any framework implements, and there's a good chance
#   that PyTorch, NumPy, and JAX have inconsistent behavior that we'd like to detect
getitem_opinfo = OpInfo(
    operator.getitem,
    supports_grad=True,
    sample_input_generator=getitem_sample_generator,
    torch_reference=operator.getitem,
    jax_reference=operator.getitem,
    numpy_reference=operator.getitem,
    test_directives=(
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < LooseVersion("0.1.4"),
        ),
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness", active_if=IS_WINDOWS),
        DecorateInfo(pytest.mark.xfail, "test_phantom_grad_vs_torch_consistency", active_if=IS_WINDOWS),
        # TypeError: Using a non-tuple sequence for multidimensional indexing is not allowed; use `arr[array(seq)]`
        # instead of `arr[seq]`. See https://github.com/google/jax/issues/4564 for more information.
        DecorateInfo(pytest.mark.xfail, "test_core_vs_jax_consistency"),
        # TODO: https://github.com/Lightning-AI/lightning-thunder/issues/841
        # check_slice_value(p0, slice(1, 3, 1)) in prologue trace fails
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
    ),
)
shape_ops.append(getitem_opinfo)


def movedim_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape src dst
    cases = (
        ((2, 3, 4, 5), (0, -1), (2, 0)),
        ((), (-1,), (0,)),
        ((1, 3, 0, 2), (0, 1, 2, 3), (0, 1, 2, 3)),
        ((1, 3, 0, 2), (0, 1, 2, 3), (3, 2, 0, 1)),
        ((2, 7, 3), 1, -1),
        ((2, 7, 3), (0, 1), (0, 1)),
    )

    for shape, src, dst in cases:
        yield SampleInput(make(shape), src, dst)


movedim_opinfo = OpInfo(
    ltorch.movedim,
    sample_input_generator=movedim_sample_generator,
    torch_reference=torch.movedim,
)
shape_ops.append(movedim_opinfo)


def pad_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, padding_config
    cases = (
        # 1-numel cases
        ((1, 1), ((0, 2, 0), (0, 2, 0))),
        # Other generic cases
        ((1, 3), ((0, 0, 0), (0, 0, 0))),
        ((3, 7, 5), ((-2, 1, 0), (1, 3, 0), (-1, 2, 0))),
        ((2, 2), ((1, 1, 1), (-1, 2, 0))),
        ((2, 0, 3), ((1, 0, 0), (1, 1, 2), (0, 0, 0))),
        ((7, 5), ((0, 0, 3), (-6, 2, 1))),
        ((3, 2, 5), ((-2, 1, 0), (1, -1, 0), (-1, 3, 1))),
        # Versions of above examples but with padding between elements set to 0
        ((2, 2), ((1, 1, 0), (-1, 2, 0))),
        ((2, 0, 3), ((1, 0, 0), (1, 1, 0), (0, 0, 0))),
        # See issue "PyTorch pad prim lowering handles out-of-bands negative padding incorrectly"
        # ((7, 5), ((0, 0, 0), (-6, 2, 0))),
        ((5, 7), ((0, 0, 0), (-6, 2, 0))),
        ((3, 2, 5), ((-2, 1, 0), (1, -1, 0), (-1, 3, 0))),  # negative pad in all 3 dims
    )

    for shape, padding_config in cases:
        yield SampleInput(make(shape), make_number(dtype=dtype), padding_config)


# NOTE: jax is very strict about tensor dtype vs number type, necessitating this helper
def _jax_pad(a, padding_value, padding_config):
    padding_value = jax.lax.convert_element_type(padding_value, a.dtype)
    return jax.lax.pad(a, padding_value, padding_config)


pad_opinfo = OpInfo(
    prims.pad,
    sample_input_generator=pad_sample_generator,
    jax_reference=_jax_pad if JAX_AVAILABLE else None,
    test_directives=(
        # PyTorch's pad doesn't support complex padding values
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch",),
            dtypes=(datatypes.complexfloating,),
        ),
        # See issue "pad+nvFuser: wrong results when applied to 1-numel inputs"
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
        ),
        # See issue "pad+nvFuser: wrong results when applied to 1-numel inputs"
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(pad_opinfo)


def pad_torch_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, padding_config
    cases = (
        ((1,), (0, 2)),
        ((1, 3), (0, 0, 0, 0)),
        ((2, 2), (1, 1, 1, 2)),
        ((2, 0, 3), (1, 0, 1, 1, 0, 0)),
        ((2, 0, 3), (1, 0, 1, 1)),
        ((5, 2, 1), ()),
        ((3, 3, 4, 2), (-1, -1)),
        ((3, 3, 4, 2), (-1, 0, 5, 2)),
        ((3, 3, 4, 4), (-1, -1, 5, 2)),
        ((3, 3, 4, 2), (1, 2, -3, 2)),
        ((3, 3, 4, 2), (1, 1, -2, -1)),
    )

    for shape, padding_config in cases:
        yield SampleInput(make(shape), padding_config, "constant", make_number(dtype=dtype))

    # The `value` parameter of the pad op is unceremoniously cast to the type of the
    # tensor being padded. Yield some tests with explicitly-different data types.
    yield SampleInput(make((2, 3)), pad=(1, 2), value=6.4)
    yield SampleInput(make((2,)), pad=(1, 2), value=1)


def pad_torch_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shape, padding_config, error type, error message
    cases = (
        ((), (1, 1), RuntimeError, "Padding length should be less than or equal to two times the input dimension"),
        ((), (1,), RuntimeError, "Padding length must be divisible by 2"),
        (
            (1,),
            (1, 1, 1, 1),
            RuntimeError,
            "Padding length should be less than or equal to two times the input dimension",
        ),
    )

    for shape, padding_config, err_type, err_msg in cases:
        yield SampleInput(make(shape), padding_config, "constant", make_number(dtype=dtype)), err_type, err_msg


pad_torch_opinfo = OpInfo(
    ltorch.pad,
    name="torch_pad",
    sample_input_generator=pad_torch_sample_generator,
    error_input_generator=pad_torch_error_generator,
    torch_reference=torch.nn.functional.pad,
    test_directives=(
        # PyTorch's pad doesn't support complex padding values
        DecorateInfo(
            pytest.mark.xfail,
            executors=("torch",),
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
shape_ops.append(pad_torch_opinfo)


# TODO: only remove these cases when the executor is nvfuser
# TODO: zero-dim cases had a bug, now fixed; re-enable.
# FIXME: tensors with no elements are skipped because of no nvfuser support
def reshape_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # tensor shape, shape
    cases = (
        ((4, 2), (2, -1, 2)),
        # ((), (-1,)),  # neg index, empty
        ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1)),  # neg index
    )

    reversible_cases = (
        ((4,), (4,)),  # no-op
        ((2, 2), (2, 2)),  # no-op
        ((1, 2, 1), (1, 2, 1)),  # no-op
        ((2, 2, 2), (4, 2)),
        ((125,), (25, 5)),
        ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1)),
        ((16, 32), (2, 4, 1, 4, 4, 1, 4)),
        ((16, 12), (12, 16)),
        ((1, 16, 12), (12, 16)),
        ((1, 5, 1, 5), (25, 1)),
        ((2, 4, 2), (4, 4)),
        ((1, 4), (1, 1, 2, 1, 2)),
        ((3, 5, 7), (7, 5, 3)),
        # ((1,), ()),  # empty
        # ((5, 0, 2, 3), (5, 0, 2, 3)),
        # ((2, 1, 0, 3, 1), (5, 0)),
        # ((1,), ()),  # empty
        ((4, 5, 6), (4, 5, 6, 1, 1, 1)),
        # ((), (1, 1, 1, 1)),  # empty
        # ((), ()),
    )

    for tensor_shape, shape in cases:
        yield SampleInput(make(tensor_shape), shape)

    for shape0, shape1 in reversible_cases:
        yield SampleInput(make(shape0), shape1)
        yield SampleInput(make(shape1), shape0)


reshape_opinfo = OpInfo(
    clang.reshape,
    supports_grad=True,
    sample_input_generator=reshape_sample_generator,
    torch_reference=torch.reshape,
)
shape_ops.append(reshape_opinfo)


def unflatten_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim, unflatten_shape
    cases = (
        ((4,), 0, (4,)),  # no-op
        ((2, 2), -1, (2,)),  # no-op
        ((1, 2, 1), 1, (2,)),  # no-op
        ((4, 2), 0, (2, 2)),
        ((125, 3), 0, (25, 5)),
        ((25, 25), -1, (1, 5, 5)),
        ((16, 32), -1, (2, 4, 1, 4)),
        ((4, 5, 6), -1, (6, 1, 1, 1)),
        ((5, 125, 5), 1, (5, 5, 5)),
        ((4, 12, 6), 1, (2, 2, 3)),
        ((12, 2), 0, (-1, 6)),
        ((4, 12, 6), 1, (12, -1)),
        ((4, 12, 6), 1, (6, -1)),
    )

    for tensor_shape, dim, shape in cases:
        yield SampleInput(make(tensor_shape), dim, shape)


def unflatten_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device)
    input_tensor = make(4, 4)
    yield (SampleInput(input_tensor, 0, ()), RuntimeError, r"unflatten\(\) sizes must be non-empty")

    err_msg = r"Attempting to reshape a.shape=(.*?) to shape=(.*?), but a.numel=.* is different from the number of elements in shape, .*"
    yield (SampleInput(input_tensor, 1, (2, 3)), RuntimeError, err_msg)

    err_msg = r"Trying to reshape, but can't infer how to reshape (.*?) to (.*?)"
    yield (SampleInput(input_tensor, 0, (-1, 3)), RuntimeError, err_msg)

    dim = 3
    yield (
        SampleInput(input_tensor, dim, (2, 2)),
        IndexError,
        rf"Dimension out of range \(expected to be in range of \[{-len(input_tensor.shape)}, {len(input_tensor.shape) - 1}\], but got {dim}\)",
    )


unflatten_opinfo = OpInfo(
    ltorch.unflatten,
    sample_input_generator=unflatten_sample_generator,
    error_input_generator=unflatten_error_generator,
    torch_reference=torch.unflatten,
)

shape_ops.append(unflatten_opinfo)


def view_as_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Input shape, output shape
    cases = (
        ((4,), (4,)),  # no-op
        ((2, 2), (2, 2)),  # no-op
        ((1, 2, 1), (1, 2, 1)),  # no-op
        ((2, 2, 2), (4, 2)),
        ((125,), (25, 5)),
        ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1)),
        ((16, 32), (2, 4, 1, 4, 4, 1, 4)),
        ((16, 12), (12, 16)),
        ((1, 16, 12), (12, 16)),
        ((1, 5, 1, 5), (25, 1)),
        ((2, 4, 2), (4, 4)),
        ((1, 4), (1, 1, 2, 1, 2)),
        ((3, 5, 7), (7, 5, 3)),
        ((1,), ()),  # empty
        ((5, 0, 2, 3), (5, 0, 2, 3)),
        ((2, 1, 0, 3, 1), (5, 0)),
        ((1,), ()),  # empty
        ((4, 5, 6), (4, 5, 6, 1, 1, 1)),
        ((), (1, 1, 1, 1)),  # empty
        ((), ()),
    )

    for ishape, oshape in cases:
        yield SampleInput(make(ishape), make(oshape))


view_as_opinfo = OpInfo(
    ltorch.view_as,
    sample_input_generator=view_as_sample_generator,
    torch_reference=torch.Tensor.view_as,
)
shape_ops.append(view_as_opinfo)


def repeat_interleave_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(make(), repeats=2)
    yield SampleInput(make(2, 3, 4), repeats=2)
    yield SampleInput(make(2, 3, 4), repeats=2, dim=1)


repeat_interleave_opinfo = OpInfo(
    ltorch.repeat_interleave,
    sample_input_generator=repeat_interleave_sample_generator,
    torch_reference=torch.Tensor.repeat_interleave,
)
shape_ops.append(repeat_interleave_opinfo)


def repeat_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    tensor_shapes = (
        (),
        (0,),
        (2,),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
        (0, 0, 0),
        (2, 0, 3),
        (2, 1, 3),
        (1, 1, 1),
    )

    for shape in tensor_shapes:
        # all-zeros repeat with out.ndim == input.ndim
        repeat_shape = (0,) * len(shape)
        yield SampleInput(make(shape), repeat_shape)

        # all-zeros repeat with out.ndim == input.ndim + 1
        repeat_shape = (0,) + repeat_shape
        yield SampleInput(make(shape), repeat_shape)

        # repeat with out.ndim == input.ndim
        repeat_shape = tuple(range(1, len(shape) + 1))
        yield SampleInput(make(shape), repeat_shape)

        # repeat with out.ndim == input.ndim + 1
        repeat_shape = (2,) + repeat_shape
        yield SampleInput(make(shape), repeat_shape)

        # repeat with out.ndim == input.ndim + 2
        repeat_shape = (2,) + repeat_shape
        yield SampleInput(make(shape), repeat_shape)


repeat_opinfo = OpInfo(
    ltorch.repeat,
    sample_input_generator=repeat_sample_generator,
    torch_reference=torch.Tensor.repeat,
)
shape_ops.append(repeat_opinfo)


def slice_in_dim_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, start_index, limit_index, stride, dim
    cases = (
        ((3, 2), 0, 3, 2, 0),
        ((4, 6, 7), 1, 3, 2, 1),
        ((4, 6, 7), 0, -1, 3, 2),
    )

    for shape, start_idx, limit_idx, stride, dim in cases:
        a = make(shape)
        yield SampleInput(a, start_idx, limit_idx, stride, dim)


slice_in_dim = OpInfo(
    clang.slice_in_dim,
    supports_grad=True,
    sample_input_generator=slice_in_dim_sample_generator,
    jax_reference=jax.lax.slice_in_dim if JAX_AVAILABLE else None,
    test_directives=(
        # TODO: nvfuser executor didn't support pad correctly, but now it should.
        # Test and re-enable.
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(slice_in_dim)


# See issue "Slice prim samples need strides and slicing beyond tensor
# boundaries"
def slice_prim_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, start_indices, end_indices
    cases = (
        ((5, 7, 8), (1, 0, 3), (2, 6, 8)),
        ((3,), (1,), (2,)),
    )

    for shape, start_indices, end_indices in cases:
        a = make(shape)
        yield SampleInput(a, start_indices, end_indices)


slice_prim_opinfo = OpInfo(
    prims.slice_prim,
    name="slice_prim",
    supports_grad=True,
    sample_input_generator=slice_prim_sample_generator,
    jax_reference=jax.lax.slice if JAX_AVAILABLE else None,
    test_directives=(
        # TODO: nvfuser executor didn't support pad correctly, but now it should.
        # Test and re-enable.
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(slice_prim_opinfo)


def select_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim, index
    cases = (
        ((1, 3, 3), 1, 1),
        # Extreme dims
        ((4, 5, 6), 0, 2),
        ((4, 5, 6), 2, 0),
        ((4, 5, 6), -1, 1),
        ((4, 5, 6), -3, 1),
        # Extreme indices
        ((4, 5, 6), 0, 0),
        ((4, 5, 6), 0, 3),
        ((4, 5, 6), 1, -5),
        ((4, 5, 6), 1, -1),
    )

    for shape, dim, index in cases:
        yield SampleInput(make(shape), dim, index)


def select_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device)
    yield (SampleInput(make(), 0, 0), RuntimeError, r"select\(\) cannot be applied to a 0-dim tensor.")
    msg = r"out of range for tensor of size \(1, 2, 3\) at dimension"
    yield (SampleInput(make((1, 2, 3)), 1, 2), RuntimeError, msg)
    msg = r"out of range for tensor of size \(1, 2, 3\) at dimension"
    yield (SampleInput(make((1, 2, 3)), 1, -3), RuntimeError, msg)


select_opinfo = OpInfo(
    ltorch.select,
    sample_input_generator=select_sample_generator,
    error_input_generator=select_error_generator,
    torch_reference=torch.select,
)
shape_ops.append(select_opinfo)


def split_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, size_or_sections, dim
    cases = (
        ((4, 6, 7), 2, 0),
        ((4, 6, 7), 3, 0),
        ((4, 6, 7), 3, -1),
        ((4, 6, 7), 9, 1),
        ((4, 6, 7), (1, 2, 1, 2), 1),
        # See issue "nvFuser split test failure"
        # ((4, 6, 7), (3, 1, 2, 0, 0, 1), -1),
        ((4, 4, 12), 4, 2),
    )

    for shape, size_or_sections, dim in cases:
        yield SampleInput(make(shape), size_or_sections, dim)


def split_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device)
    msg = r"is zero then the length of the split dimension \(4\) must also be zero"
    yield (SampleInput(make(4, 5, 6), 0, 0), RuntimeError, msg)


split_opinfo = OpInfo(
    ltorch.split,
    sample_input_generator=split_sample_generator,
    error_input_generator=split_error_generator,
    torch_reference=torch.split,
)
shape_ops.append(split_opinfo)


def chunk_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, chunks size, dim
    cases = (
        # Case - any size, chunk=1.
        ((0,), 1, 0),
        ((1,), 1, 0),
        ((0, 2, 3), 1, 0),
        ((2, 2, 3), 1, 0),
        # Case - shape[dim] == 0, any chunk.
        ((0,), 3, 0),
        ((0, 2, 3), 1, 0),
        ((0, 2, 3), 4, 0),
        # Exampes from the PyTorch's docs.
        ((11,), 6, 0),
        ((12,), 6, 0),
        ((13,), 6, 0),
        # Examples taken from split_sample_generator
        ((4, 6, 7), 2, 0),
        ((4, 6, 7), 3, 0),
        ((4, 6, 7), 3, -1),
        ((4, 6, 7), 9, 1),
        ((4, 4, 12), 4, 2),
        # Just interesting cases when you get not what you expected
        ((9, 2), 4, 0),  # Expects 3 chunks, not 4. See the impl code for more details.
    )

    for shape, size_or_sections, dim in cases:
        yield SampleInput(make(shape), size_or_sections, dim)


def chunk_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device)
    yield (SampleInput(make(), 1), RuntimeError, "must be at least 1-dimensional")
    yield (SampleInput(make((0,)), -1), RuntimeError, "chunks(.*?) must be greater than 0")
    yield (SampleInput(make((0,)), 0), RuntimeError, "chunks(.*?) must be greater than 0")
    # NOTE: maybe not needed as it tests dim canonization.
    yield (SampleInput(make((0,)), 1, dim=1), IndexError, "Dimension out of range")


chunk_opinfo = OpInfo(
    ltorch.chunk,
    sample_input_generator=chunk_sample_generator,
    error_input_generator=chunk_error_generator,
    torch_reference=torch.chunk,
)
shape_ops.append(chunk_opinfo)


def squeeze_torch_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # a.shape, dim
    cases = (
        ((1, 2, 1, 1, 3, 1), None),
        ((), None),
        ((1, 1, 1), None),
        ((1, 2, 1, 1, 3, 1), 0),
        ((1, 2, 1, 1, 3, 1), 2),
        ((1, 2, 1, 1, 3, 1), 5),
        ((1, 2, 1, 1, 3, 1), (2, 3)),
        ((1, 1, 1), (0, 1, 2)),
        ((2, 2, 1), (0,)),
        ((2, 2, 1), (0, 1)),
        ((2, 2, 1), (0, 1, 2)),
    )

    for shape, dim in cases:
        a = make(shape)
        yield SampleInput(a, dim)


def torch_squeeze_helper(a, dim):
    # TODO: dim as a sequence is only supported on PyTorch 2.0 and greater
    if isinstance(dim, Sequence):
        for dim in sorted(dim, reverse=True):
            a = a.squeeze(dim)
        return a

    if dim is None:
        return torch.squeeze(a)

    # dim is a number
    return torch.squeeze(a, dim)


squeeze_torch_opinfo = OpInfo(
    ltorch.squeeze,
    name="squeeze_torch",
    supports_grad=True,
    sample_input_generator=squeeze_torch_sample_generator,
    torch_reference=torch_squeeze_helper,
)
shape_ops.append(squeeze_torch_opinfo)


def squeeze_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # a.shape, dim
    cases = (
        ((1, 2, 1, 1, 3, 1), (2, 3)),
        ((1, 1, 1), (0, 1, 2)),
    )

    for shape, dim in cases:
        a = make(shape)
        yield SampleInput(a, dim)


squeeze_opinfo = OpInfo(
    clang.squeeze,
    supports_grad=True,
    sample_input_generator=squeeze_sample_generator,
    jax_reference=jax.lax.squeeze if JAX_AVAILABLE else None,
)
shape_ops.append(squeeze_opinfo)


def stack_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shapes, dim
    cases = [
        ([(3,)], 0),  # single tensor provided
        # 1D
        ([(3,), (3,)], 0),
        ([(4,), (4,)], 0),
        ([(3,), (3,)], 1),
        ([(3,), (3,)], -1),
        ([(2, 3), (2, 3)], 1),
        ([(2, 3), (2, 3), (2, 3)], 1),
    ]

    for shapes, dim in cases:
        yield SampleInput(*[make(s) for s in shapes], dim=dim)


def stack_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shapes, dim, exception type, error message match or None for universal match
    cases = [
        ([], 0, RuntimeError, "list of tensors cannot be empty"),
        ([(2,), (3,)], 1, RuntimeError, "tensors must be of the same shape"),
        ([(2,), (2,)], -3, IndexError, "Dimension out of range"),
        ([(2,), (2,)], 4, IndexError, "Dimension out of range"),
        # TODO: BUG - differing dimensions is not captured.
        ([(2,), (2, 3)], 0, RuntimeError, None),
        # TODO: BUG - same shape but dim is not captured.
        ([(2, 3), (4, 5)], 0, RuntimeError, None),
    ]

    for shapes, dim, exc_type, err_msg_match in cases:
        yield SampleInput(*[make(s) for s in shapes], dim=dim), exc_type, err_msg_match


# `stack_wrapper` is created for the same reason as `cat_wrapper.
def stack_wrapper(*args, dim):
    return ltorch.stack(args, dim=dim)


stack_opinfo = OpInfo(
    stack_wrapper,
    sample_input_generator=stack_sample_generator,
    error_input_generator=stack_error_generator,
    torch_reference=lambda *args, dim: torch.stack(args, dim=dim),
)
shape_ops.append(stack_opinfo)


def tensor_split_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, indices_or_sections, dim
    cases = (
        ((4, 6, 7), 2, 1),
        ((4, 6, 7), 2, 2),
        ((4, 6, 7), 3, 0),
        ((4, 6, 7), 5, -1),
        ((4, 6, 7), (0, 1), 1),
        ((4, 6, 7), (1, 5, 6), 2),
        ((4, 6, 7), (1, 5, 9, 9), 2),
        ((4, 6, 7), (1, 5, 6, 7), 2),
        ((4, 6, 7), (0, 0, 1, 1, 2), -2),
    )

    for shape, indices_or_sections, dim in cases:
        yield SampleInput(make(shape), indices_or_sections, dim)


tensor_split_opinfo = OpInfo(
    ltorch.tensor_split,
    sample_input_generator=tensor_split_sample_generator,
    torch_reference=torch.tensor_split,
    test_directives=(
        # TODO: nvfuser executor didn't support pad correctly, but now it should.
        # Test and re-enable.
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(tensor_split_opinfo)


def transpose_torch_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # tensor shape, dim0, dim1
    cases = (
        ((2, 12, 1024, 64), 1, 2),
        ((4, 3, 2), 0, -1),
        ((4, 3, 2), 0, -2),
        ((4, 3, 2), 1, 2),
        ((1, 2), 0, -1),
        ((5,), 0, 0),
    )

    for shape, dim0, dim1 in cases:
        yield SampleInput(make(shape), dim0, dim1)


transpose_torch_opinfo = OpInfo(
    ltorch.transpose,
    name="torch_transpose",
    supports_grad=True,
    sample_input_generator=transpose_torch_sample_generator,
    torch_reference=torch.transpose,
)
shape_ops.append(transpose_torch_opinfo)


def transpose_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, perm
    cases = (
        ((4, 7, 8), (0, 1, 2)),
        ((4, 7, 8), (1, 2, 0)),
        ((4, 7, 8), (2, 1, 0)),
        ((4, 7, 8), (0, 2, 1)),
        ((4, 7, 8), (0, -1, 1)),
        ((4, 7), (1, 0)),
        ((3, 4, 7, 8), (0, 1, 2, 3)),
        ((3, 4, 7, 8), (0, 2, 3, 1)),
        ((3, 4, 7, 8), (1, 3, 2, 0)),
        ((2, 3, 4, 7, 8), (0, 1, 2, 3, 4)),
        ((2, 3, 4, 7, 8), (0, 2, 3, 4, 1)),
        ((2, 3, 4, 7, 8), (1, 4, 3, 0, 2)),
    )

    for shape, perm in cases:
        yield SampleInput(make(shape), perm)


transpose_opinfo = OpInfo(
    clang.transpose,
    supports_grad=True,
    sample_input_generator=transpose_sample_generator,
    torch_reference=torch.permute,
)
shape_ops.append(transpose_opinfo)


def permute_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, perm
    cases = (
        ((2, 3, 4), (0, 1, 2)),  # no-op
        ((2, 3, 4), (1, 2, 0)),
        ((2, 3, 4), (2, 1, 0)),
        ((2, 3, 4), (0, 2, 1)),
        ((2, 3, 4), (0, -1, 1)),
        ((4, 7), (1, 0)),
        ((3,), (0,)),  # no-op
    )

    # NOTE These cases are tuple-only because *() becomes no arguments to varargs
    # shape, perm
    tuple_only_cases = (((), ()),)

    for shape, perm in cases:
        # Tests specifying the permutation as a tuple
        yield SampleInput(make(shape), perm)
        # Tests specifying the permutation as varargs
        yield SampleInput(make(shape), *perm)

    for shape, perm in tuple_only_cases:
        yield SampleInput(make(shape), perm)


def permute_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # Checks that a len(permutation) != rank(tensor) throws an error
    t = make(2, 3, 4)
    yield (
        SampleInput(t, (0, 1)),
        RuntimeError,
        r"Expected the length \(2\) of the permutation(.*?) to be the number of dimensions \(3\)",
    )


# NOTE This reference is required because torch.permute requires the
#   permutation be specified as a tuple, while torch.Tensor.permute
#   allows it to be specified as a tuple or varargs.
def torch_permute_reference(a, *dims):
    return a.permute(*dims)


permute_opinfo = OpInfo(
    ltorch.permute,
    supports_grad=True,
    sample_input_generator=permute_sample_generator,
    error_input_generator=permute_error_generator,
    torch_reference=torch_permute_reference,
)
shape_ops.append(permute_opinfo)


def t_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape
    cases = (
        (),
        (1),
        (4),
        (4, 5),
    )

    for shape in cases:
        yield SampleInput(make(shape))


def t_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shape, error type, error message
    cases = (
        ((4, 5, 6), RuntimeError, r"t\(\) expects a tensor with <= 2 dimensions, but self is 3D"),
        (
            (4, 5, 6, 7),
            RuntimeError,
            r"t\(\) expects a tensor with <= 2 dimensions, but self is 4D",
        ),
    )

    for shape, err_type, err_msg in cases:
        yield SampleInput(make(shape)), err_type, err_msg


t_opinfo = OpInfo(
    ltorch.t,
    sample_input_generator=t_sample_generator,
    error_input_generator=t_error_generator,
    torch_reference=lambda x: torch.Tensor.t(x),
)
shape_ops.append(t_opinfo)


def reverse_dims_T_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape
    cases = (
        (),
        (1),
        (4),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 7),
    )

    for shape in cases:
        yield SampleInput(make(shape))


reverse_dims_T_opinfo = OpInfo(
    ltorch.reverse_dims_T,
    sample_input_generator=reverse_dims_T_sample_generator,
    torch_reference=lambda x: x.T,
)
shape_ops.append(reverse_dims_T_opinfo)


def matrix_transpose_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape
    cases = (
        (),
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 2),
    )

    for shape in cases:
        yield SampleInput(make(shape))


def matrix_transpose_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # shape, error type, error message
    cases = (((3), RuntimeError, "tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor."),)

    for shape, err_type, err_msg in cases:
        yield SampleInput(make(shape)), err_type, err_msg


transpose_opinfo = OpInfo(
    clang.matrix_transpose,
    sample_input_generator=matrix_transpose_sample_generator,
    error_input_generator=matrix_transpose_error_generator,
    torch_reference=lambda x: x.mT,
)
shape_ops.append(transpose_opinfo)


# a.shape, dim, b.shape
take_cases = (
    ((4, 2, 3), 0, (8,)),
    ((4, 2, 3), 1, (7,)),
    ((4, 2, 3), 2, (2,)),
    ((4, 2, 3), -1, (2,)),
    ((4,), 0, (8,)),
    ((4,), 0, (1,)),
    ((4, 1), 0, (3,)),
    ((4, 1), 1, (5,)),
    ((1, 0, 3), 0, (8,)),
    ((4, 2, 3), 0, (0,)),
    ((4, 2, 3), 1, (0,)),
    ((4, 2, 3), 2, (0,)),
    ((4, 2, 3), 0, ()),
    ((4, 2, 3), 1, ()),
    ((4, 2, 3), 2, ()),
)


def take_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, requires_grad=False)

    for shape_a, dim, shape_b in take_cases:
        for index_dtype in [torch.int, torch.long]:
            a = make(shape_a)
            b = make_index(shape_b, low=0, high=shape_a[dim], dtype=index_dtype)
            yield SampleInput(a, b, dim)


def torch_index_select_wrapper(a, b, dim):
    return torch.index_select(a, dim, b)


# TODO: mapping jax.lax.gather for testing
take_opinfo = OpInfo(
    clang.take,
    supports_grad=True,
    sample_input_generator=take_sample_generator,
    torch_reference=torch_index_select_wrapper,
)
shape_ops.append(take_opinfo)


def index_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, requires_grad=False)
    # Not sure if we need to consider higher order gradient, marking requires_grad as False
    make_source = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    for shape_a, dim, shape_b in take_cases:
        for index_dtype in [torch.int, torch.long]:
            canonicalized_dim = dim if dim >= 0 else dim + len(shape_a)
            shape_source = list(shape_a)
            shape_source[canonicalized_dim] = shape_b[0] if len(shape_b) != 0 else 1
            a = make(shape_a)
            b = make_index(shape_b, low=0, high=shape_a[dim], dtype=index_dtype)
            c = make_source(shape_source)
            yield SampleInput(a, index=b, source=c, dim=dim)


index_add_opinfo = OpInfo(
    ltorch.index_add,
    sample_input_generator=index_add_sample_generator,
    torch_reference=torch.index_add,
)
shape_ops.append(index_add_opinfo)


def index_copy_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample in index_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
        # Only int64 index for now
        if sample.kwargs["index"].dtype is torch.int32:
            continue

        dim = sample.kwargs["dim"]
        canonicalized_dim = dim if dim >= 0 else dim + sample.args[0].ndim

        # Only unique indices make the op differentiable
        index = sample.kwargs["index"]
        index = index.unique()

        source = sample.kwargs["source"]
        source = torch.narrow(source, canonicalized_dim, 0, len(index)).detach().clone().requires_grad_(requires_grad)

        yield SampleInput(sample.args[0], index=index, source=source, dim=dim)


index_copy_opinfo = OpInfo(
    ltorch.index_copy,
    supports_grad=True,
    sample_input_generator=index_copy_sample_generator,
    torch_reference=torch.index_copy,
)
shape_ops.append(index_copy_opinfo)


# NOTE: index_put uses getitem in backward which currently doesn't support indices>1D and bool indices
# Cases with indices>1D are only tested for forward
# vjp test is disabled by setting values.requires_grad=False
def index_put_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # a.shape, [index.shape], values.shape, accumulate
    cases_0D_1Dindices = (
        ((4,), [()], (), False),
        ((4, 2, 3), [(), (), ()], (), True),
        ((0,), [(0,)], (0,), False),
        ((4, 2, 3), [(), (), ()], (), False),
        ((4, 2, 3), [(3,), (3,), (3,)], (3,), False),
        ((4, 2, 3), [(2,), (2,)], (), True),
        ((4, 2, 3), [(4,), (), ()], (4,), True),
        ((4, 2, 3), [(4,), (1,), ()], (1,), True),
        ((4, 2, 3), [(), (2,), ()], (1,), False),
        ((4, 2, 3), [(2,), (2,)], (1,), True),
        ((4, 2, 3), [3], (1,), False),
        ((4, 2, 3), [(0,)], (2, 3), False),
        ((4, 2, 3), [(0,), (0,)], (1,), False),
        ((4,), [(2,)], (1,), True),
    )

    cases = (
        ((4, 2, 3), [(2, 2), (2, 2), (2, 2)], (2,), False),
        ((4, 2, 3), [(2, 2), (2, 1)], (1,), True),
        ((4, 2, 3), [(2, 2), (), (2, 1)], (2, 1), True),
        ((4, 2, 3), [(4, 2)], (1, 1), False),
        ((4,), [(2, 2)], (), True),
    )
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, requires_grad=False)
    make_values = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Cases with indices>1D are only tested for forward
    # vjp test is disabled by setting values.requires_grad=False
    for shape_a, index_shapes, shape_vals, accumulate in cases:
        for index_dtype in [torch.int, torch.long]:
            indices = []
            a = make(shape_a)
            vals = make_values(shape_vals)
            for i, index_shape in enumerate(index_shapes):
                indices.append(make_index(index_shape, low=-shape_a[i], high=shape_a[i], dtype=index_dtype))
            yield SampleInput(a, indices, vals, accumulate)

    # Tested for both forward and vjp tests
    for shape_a, index_shapes, shape_vals, accumulate in cases_0D_1Dindices:
        for index_dtype in [torch.int, torch.long]:
            indices = []
            a = make(shape_a)
            vals = make(shape_vals)
            for i, index_shape in enumerate(index_shapes):
                indices.append(make_index(index_shape, low=-shape_a[i], high=shape_a[i], dtype=index_dtype))
            yield SampleInput(a, indices, vals, accumulate)


index_put_opinfo = OpInfo(
    ltorch.index_put,
    sample_input_generator=index_put_sample_generator,
    torch_reference=torch.index_put,
)
shape_ops.append(index_put_opinfo)


# a.shape, dim, b.shape
take_along_axis_cases = (
    ((4, 2, 3), 0, (8, 2, 3)),
    ((4, 2, 3), 1, (4, 1, 3)),
    ((4, 2, 3), 2, (4, 2, 5)),
    ((4, 2, 3), -1, (4, 2, 5)),
    ((4,), 0, (8,)),
    ((4,), 0, (1,)),
    ((4, 1), 0, (3, 1)),
    ((4, 1), 1, (4, 5)),
    # broadcasting is supported by numpy.take_along_axis
    ((4, 2, 3), 2, (1, 2, 7)),
    ((4, 2, 3), -1, (1, 2, 7)),
)


def take_along_axis_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # torch.take_along_dim expects index to be long but not int
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    for shape_a, dim, shape_b in take_along_axis_cases:
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, b, dim)


take_along_axis_opinfo = OpInfo(
    clang.take_along_axis,
    supports_grad=True,
    sample_input_generator=take_along_axis_sample_generator,
    torch_reference=torch.take_along_dim,
    # Torch doesn't support complex half on take_along_dim
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # Support for take_along_axis was added in nvfuser v0.0.10
        DecorateInfo(
            pytest.mark.skip,
            executors=("nvfuser"),
            active_if=nvfuser_version < LooseVersion("0.0.10"),
        ),
    ),
)
shape_ops.append(take_along_axis_opinfo)


def gather_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # torch.gather expects index to be long but not int
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    for shape_a, dim, shape_b in take_along_axis_cases:
        canonicalized_dim = dim if dim >= 0 else dim + len(shape_a)
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, index=b, dim=dim)

    # Note that gather doesn't have the broadcast requirement, it only requires
    # 1. a.shape[i]      >= index.shape[i] for i != dim
    #
    # a.shape, dim, index.shape
    scatter_add_cases = (
        ((4, 5, 3), 0, (3, 2, 3)),
        ((4, 5, 3), 1, (3, 5, 2)),
        ((4, 5, 3), 2, (3, 2, 8)),
    )
    for shape_a, dim, shape_b in scatter_add_cases:
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        yield SampleInput(a, index=b, dim=dim)
        yield SampleInput(input=a, index=b, dim=dim)


gather_opinfo = OpInfo(
    ltorch.gather,
    supports_grad=True,
    sample_input_generator=gather_sample_generator,
    torch_reference=torch.gather,
)
shape_ops.append(gather_opinfo)


def scatter_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # torch.scatter_add expects index to be long but not int
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_source = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # NOTE The value gradient is only correct when src.shape == index.shape.
    # For gradient testing, we use the index shape for the source tensor.
    # See https://github.com/pytorch/pytorch/issues/27614#issuecomment-564648819
    for shape_a, dim, shape_b in take_along_axis_cases:
        canonicalized_dim = dim if dim >= 0 else dim + len(shape_a)
        if requires_grad:
            shape_source = shape_b
        else:
            shape_source = list(shape_a)
            shape_source[canonicalized_dim] = shape_b[canonicalized_dim]
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        c = make_source(shape_source)
        yield SampleInput(a, index=b, src=c, dim=dim)

    # Questionable use case. Do we want to support these?!
    # Note that scatter_add doesn't have the broadcast requirement, it only requires
    # 1. a.shape[i]      >= index.shape[i] for i != dim
    # 2. source.shape[i] >= index.shape[i] for all i
    #
    # a.shape, dim, index.shape, source.shape
    scatter_add_cases = (
        ((4, 5, 3), 0, (3, 2, 3), (4, 3, 9)),
        ((4, 5, 3), 1, (3, 5, 2), (3, 8, 8)),
        ((4, 5, 3), 2, (3, 2, 8), (5, 8, 8)),
    )
    for shape_a, dim, shape_b, shape_source in scatter_add_cases:
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        c = make_source(shape_b if requires_grad else shape_source)
        yield SampleInput(a, index=b, src=c, dim=dim)


def scatter_add_error_generator(op, device, dtype=torch.float32, requires_grad=True, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # torch.scatter_add expects index to be long but not int
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_source = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Gradient definition for src tensor is valid only if src.shape == index.shape
    # NOTE The scatter_add prim renames src variable to value.
    shape_a = (4, 5, 3)
    dim = 0
    shape_b = (3, 2, 3)
    shape_source = (4, 3, 9)

    a = make(shape_a)
    b = make_index(shape_b, low=0, high=shape_a[dim])
    c = make_source(shape_source)
    yield (
        SampleInput(a, index=b, src=c, dim=dim),
        RuntimeError,
        "The gradient for the value Tensor is implemented only when value.shape == index.shape. value shape is (.*?) while index shape is (.*?).",
    )


scatter_add_opinfo = OpInfo(
    ltorch.scatter_add,
    supports_grad=True,
    sample_input_generator=scatter_add_sample_generator,
    error_input_generator=scatter_add_error_generator,
    torch_reference=torch.scatter_add,
    test_directives=(
        # Torch doesn't support complex half on scatter_add
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-1, rtol=1e-1)),
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
shape_ops.append(scatter_add_opinfo)


def scatter_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if not requires_grad:
        # If not requires_grad, we allow repeated indices
        for sample in scatter_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
            # Scatter is non-deterministic with repeated indices.
            # The easiest way to make it deterministic is to make `src` a scalar tensor.
            src = sample.kwargs["src"]
            src = torch.ones_like(src)
            sample.kwargs["src"] = src
            yield sample
    else:
        # In this case repeated indices will break finite difference checks
        # for the grad wrt `src` since it, just like in PyTorch, is implemented
        # with the use of `gather`.
        # To address that, we pick samples with a.shape == index.shape == src.shape.
        # Moreover, if we define
        # I = {(i_1, ..., i_{dim - 1}, index[i_1, ..., i_n], i_{dim + 1}, ..., i_n)}, then,
        # to avoid issues with repated indices, I has to define the set of all valid
        # indices (n-dim tuples) into `a`.
        # We choose `index` such that
        # index[..., 0:a.shape[dim], ...] = randperm(a.shape[dim]).
        # It is not hard to see that such `index` turns I into the set with the desired properties.
        for sample in scatter_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
            dim = sample.kwargs["dim"]
            a = sample.args[0]
            dim_canon = a.ndim + dim if dim < 0 else dim

            scatter_dim_len = a.shape[dim]

            n_reps_before = 1
            for d in a.shape[:dim_canon]:
                n_reps_before *= d

            n_reps_after = 1
            for d in a.shape[dim_canon + 1 :]:
                n_reps_after *= d

            new_idx = torch.zeros((n_reps_before, scatter_dim_len, n_reps_after), dtype=torch.long, device=device)
            for before_dim in range(n_reps_before):
                for after_dim in range(n_reps_after):
                    new_idx[before_dim, :, after_dim] = torch.randperm(scatter_dim_len, device=device)
            new_idx = new_idx.reshape(a.shape)

            # NOTE: setting `src` = `a` turns `scatter` into a "randperm"-kind operation
            src = a.detach().clone().requires_grad_(requires_grad)

            yield SampleInput(a, dim, new_idx, src)

    for scalar_src in (1, 1.0):
        for sample in scatter_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
            # PyTorch uses `src` for Tensor inputs, and `value` for scalar inputs
            del sample.kwargs["src"]
            sample.kwargs["value"] = scalar_src
            yield sample


scatter_opinfo = OpInfo(
    ltorch.scatter,
    supports_grad=True,
    sample_input_generator=scatter_sample_generator,
    torch_reference=torch.scatter,
)
shape_ops.append(scatter_opinfo)


def unsqueeze_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # a.shape, dims
    cases = (
        ((4, 2), (0, 1, 4)),
        ((2, 1, 3), ()),
        ((2, 1, 3), (-1,)),
        ((2, 1, 3), (-1, 1, 2, -2)),
        ((), (0, -1)),
        ((2, 2), (1,)),
    )

    for shape, dims in cases:
        a = make(shape)
        yield SampleInput(a, dims)


unsqueeze_opinfo = OpInfo(
    clang.unsqueeze,
    supports_grad=True,
    sample_input_generator=unsqueeze_sample_generator,
    jax_reference=jax.lax.expand_dims if JAX_AVAILABLE else None,
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvfuser",),
        ),
    ),
)
shape_ops.append(unsqueeze_opinfo)

opinfos.extend(shape_ops)

#
# Reduction OpInfos
#
reduction_ops = []


def all_tensor_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input shape, dim, keepdim
    dim_cases = (
        ((4, 4), None, False),
        ((4, 4), None, True),
        ((2, 3), 0, True),
        ((2, 3, 4), (1, 2), False),
        ((2, 3, 4), (1, 2), True),
        ((2, 3, 4), (-1, 1), False),
        ((2, 3, 4), (-1, 1), True),
    )

    for input_shape, dim, keepdim in dim_cases:
        yield SampleInput(make(input_shape), dim, keepdim)


def all_tensor_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    err_msg = r"Dimension out of range \(expected to be in range of \[.*?\], but got .*\)"
    yield (
        SampleInput(make(5, 1, 2, 3), 4),
        IndexError,
        err_msg,
    )


all_tensor_opinfo = OpInfo(
    ltorch.all_tensor,
    sample_input_generator=all_tensor_sample_generator,
    error_input_generator=all_tensor_error_generator,
    torch_reference=torch.all,
)

reduction_ops.append(all_tensor_opinfo)


any_tensor_opinfo = OpInfo(
    ltorch.any_tensor,
    sample_input_generator=all_tensor_sample_generator,
    torch_reference=torch.any,
)

reduction_ops.append(any_tensor_opinfo)


# TODO: increase reduction samples and refacort amax and sum generators
def amax_amin_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # For grad test stability it's better to use wider range of values
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-1000, high=1000)

    # shape, dim, keepdim
    cases = (
        ((4, 4), None, False),
        ((8, 1, 6), (1,), True),
        ((8, 7, 5, 1), (0, 1), False),
    )

    for shape, dim, keepdim in cases:
        yield (SampleInput(make(shape), dim, keepdim))


amax_opinfo = OpInfo(
    ltorch.amax,
    supports_grad=True,
    sample_input_generator=amax_amin_sample_generator,
    torch_reference=torch.amax,
    # NOTE Complex numbers are unordered
    dtypes=(datatypes.exact, datatypes.floating),
)
reduction_ops.append(amax_opinfo)


amin_opinfo = OpInfo(
    ltorch.amin,
    sample_input_generator=amax_amin_sample_generator,
    torch_reference=torch.amin,
    # Complex numbers are unordered
    dtypes=(datatypes.exact, datatypes.floating),
)
reduction_ops.append(amin_opinfo)


def max_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # For grad test stability it's better to use wider range of values
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-1000, high=1000)

    def make_with_extremal_value(shape, extremal, percentage=0.5):
        return replace_random_percentage(make(shape), extremal, percentage=percentage)

    # NOTE: Gradient Computation with multiple max values
    # Currently, if there are multiple `max` values
    # `torch` eager - gradients max(dim) propagates gradient only to a single index in the source tensor
    # `thunder` - gradients are distributed evenly.
    # Also, we need unique values when grad check with finite differences.
    # So, we use the function below to create tensor with unique values.
    def make_t(shape):
        if dtype.is_floating_point and requires_grad:
            # Use `linspace` to create tensor with unique values.
            numel = math.prod(shape)
            inp_t = torch.linspace(
                -1000, 1000, steps=numel, dtype=dtype, device=device, requires_grad=requires_grad
            ).view(shape)
            return inp_t

        # If we are not computing gradients,
        # it is ok to have repeated values.
        return make(shape)

    # shape, dim, keepdim
    cases = (
        ((2, 2, 3), 1, True),
        ((2, 3, 1), 0, False),
    )

    for shape, dim, keepdim in cases:
        # overload: torch_max(a: TensorLike, /) -> TensorLike
        # This overload corresponds to taking the max over the flattened tensor.
        yield SampleInput(make_t(shape))

        if not requires_grad and dtype.is_floating_point:
            # See NOTE: Gradient Computation with multiple max values
            # Thus we don't pass these inputs to grad tests
            yield SampleInput(make_with_extremal_value(shape, float("nan")))
            yield SampleInput(make_with_extremal_value(shape, float("inf")))

        # overload: torch_max(a: TensorLike, b: TensorLike, /) -> TensorLike
        # This overload corresponds to taking the elementwise max between tensors `a` and `b`.
        yield SampleInput(make(shape), make(shape))

        if dtype is not torch.bool:  # argmax is not supported on `bool`
            # overload: torch_max(a: TensorLike, /, dim: int | tuple[int], keepdim: bool = False) -> TensorLike, TensorLike
            # This overload corresponds to taking the max along the specified dimension `dim`.
            # It returns first occurence of the maximum value along the dimension and it's corresponding index.
            # NOTE: When same values are present, the first occurence of the `value` and corresponding index is returned
            yield SampleInput(make_t(shape), dim)
            yield SampleInput(make_t(shape), dim, keepdim)

            if not requires_grad and dtype.is_floating_point:
                # See NOTE: Gradient Computation with multiple max values
                # Thus we don't pass these inputs to grad tests
                yield SampleInput(make_with_extremal_value(shape, float("nan")), dim)
                yield SampleInput(make_with_extremal_value(shape, float("inf")), dim)


def max_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float, low=-1000, high=1000)

    err_msg = r"keepdim=True is invalid for torch.max\(a, b\) overload."
    yield (SampleInput(make(3, 3), make(3, 3), keepdim=True), RuntimeError, err_msg)

    err_msg = r"keepdim=True is invalid for torch.max\(a\) overload."
    yield (SampleInput(make(3, 3), keepdim=True), RuntimeError, err_msg)


max_opinfo = OpInfo(
    ltorch.torch_max,
    supports_grad=True,
    sample_input_generator=max_sample_generator,
    error_input_generator=max_error_generator,
    torch_reference=torch.max,
    # Complex numbers are unordered
    dtypes=(datatypes.exact, datatypes.floating),
)

reduction_ops.append(max_opinfo)


def reduction_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape, dim, keepdim, dtype
    cases = (
        ((4, 4), None, False, None),
        ((5,), None, True, None),
        ((5,), 0, False, None),
        ((8, 1, 6), 1, True, None),
        ((8, 7, 5, 1), (0, 1), True, None),
        ((8, 7, 5, 1), (1, 3), False, None),
        # torch.prod() behaves differently when passing `dim=None` compared to
        # simply omitting the argument, due to pybind11's overload resolution
        # mechanism. Passing `None` in those cases instead leads to an error.
        # We test this behavior explicitly in order to try and catch such edge
        # cases.
        ((4, 4), None),
    )

    for c in cases:
        if len(c) == 2:
            shape, dim = c
            yield (SampleInput(make(shape), dtype=dtype))
        else:
            shape, dim, keepdim, dtype = c
            yield (SampleInput(make(shape), dim, keepdim=keepdim, dtype=dtype))


def logsumexp_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        # We set low (inclusive) and high (exclusive) here to avoid values
        # whose products can otherwise become extremely large
        low=-2,
        high=3,
    )

    # shape, dim, keepdim
    cases = (
        ((4, 4), (0, 1), False),
        ((5,), 0, True),
        ((5,), 0, False),
        ((8, 1, 6), 1, True),
        ((8, 0, 5, 1), (0, 2), True),
        ((8, 0, 5, 1), (0, 2), False),
        ((8, 7, 5, 1), (0, 1), True),
        ((8, 7, 5, 1), (1, 3), False),
    )

    for shape, dim, keepdim in cases:
        yield (SampleInput(make(shape), dim, keepdim))

        # Test that positive and negative infinity are set to zero
        if dtype.is_floating_point:
            inf_input_tensor = make(shape)
            # Set a quarter of elements to positive infinity
            inf_input_tensor = replace_random_percentage(inf_input_tensor, float("inf"), 0.25)
            # Set a quarter of elements to negative infinity
            inf_input_tensor = replace_random_percentage(inf_input_tensor, float("-inf"), 0.25)
            yield (SampleInput(inf_input_tensor, dim, keepdim))


logsumexp_opinfo = OpInfo(
    ltorch.logsumexp,
    # NOTE Pytorch logsumexp does not support complex dtypes.
    # RuntimeError: logsumexp(): Expected floating point type for result tensor, but got: ComplexFloat
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=logsumexp_sample_generator,
    torch_reference=torch.logsumexp,
    test_directives=(
        # NOTE Nvfuser fails with AssertionError: Tensor-likes are not close!
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.bfloat16, datatypes.float16),
            executors=("nvfuser",),
        ),
        # RuntimeError: "exp_vml_cpu" not implemented for 'Half'
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.bfloat16, datatypes.float16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
reduction_ops.append(logsumexp_opinfo)


prod_opinfo = OpInfo(
    ltorch.prod,
    sample_input_generator=reduction_sample_generator,
    torch_reference=torch._refs.prod,
    test_directives=(
        # NOTE Test fails due to precision
        # TODO Investigate or reduce test precision
        DecorateInfo(
            pytest.mark.skip, "test_core_vs_torch_consistency", dtypes=(datatypes.float32,), executors=("nvfuser",)
        ),
        # Torch doesn't support cpu real (float16) or complex half prod
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32, datatypes.float16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO Review this
        # Greatest absolute difference: 11723436.0
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvfuser",),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
            active_if=nvfuser_version < "0.0.4",
        ),
    ),
)
reduction_ops.append(prod_opinfo)


sum_opinfo = OpInfo(
    ltorch.sum,
    supports_grad=True,
    sample_input_generator=reduction_sample_generator,
    torch_reference=torch.sum,
    test_directives=(
        # Torch doesn't support cpu complex half sum
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # nvFuser had issues with complex reductions, now fixed; TODO re-enable
        # this test.
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvfuser",),
        ),
        # Some PyTorch versions before PyTorch 1.13 throw a runtime error
        #   insisting, incorrectly, that dimensions be specified by name
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            active_if=LooseVersion(torch.__version__) < "1.13",
        ),
    ),
)
reduction_ops.append(sum_opinfo)


# TODO Update this so we can access sample input args/kwargs by name
#   instead of by offset, as is done here
def var_sample_generator(op, device, dtype, requires_grad):
    correction = (None, 0, 1)
    samples = reduction_sample_generator(op, device, dtype, requires_grad)
    for c, sample in itertools.product(correction, samples):
        a = sample.args[0]
        dim = sample.args[1] if len(sample.args) > 1 else None
        keepdim = sample.args[2] if len(sample.args) > 2 else False

        if c is not None:
            yield SampleInput(a, dim, keepdim=keepdim, correction=c)
        else:
            yield SampleInput(a, dim, keepdim=keepdim)

    # Tests zero-dim tensor
    yield SampleInput(make_tensor((), device=device, dtype=dtype, requires_grad=requires_grad))


# glu requires that value of the shape of the input at index dim be even
def glu_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    cases = (
        ((4,), None),
        ((3, 4), 1),
        ((3, 4), None),
        ((3, 4), -1),
        ((4, 3), 0),
        ((4, 5, 8), None),
        ((4, 5, 8), 0),
    )

    for shape, dim in cases:
        if dim is None:
            yield SampleInput(make(*shape))
        else:
            yield SampleInput(make(*shape), dim)


def glu_error_generator(op, device, **kwargs):
    make = partial(
        make_tensor,
        device=device,
        dtype=torch.float,
    )
    err_msg = r"Halving dimension must be even, but dimension .* is size .*"
    # The value of the shape of the input in the default (last) dim is odd, which is unsupported.
    yield (SampleInput(make((3,))), RuntimeError, err_msg)
    yield (SampleInput(make((2, 2, 3))), RuntimeError, err_msg)
    # The value of the shape of the input at index dim=1 is odd, which is unsupported.
    yield (SampleInput(make((4, 5, 8)), dim=1), RuntimeError, err_msg)


glu_opinfo = OpInfo(
    ltorch.glu,
    sample_input_generator=glu_sample_generator,
    error_input_generator=glu_error_generator,
    dtypes=(datatypes.inexact,),
    torch_reference=torch.nn.functional.glu,
    test_directives=(),
)
reduction_ops.append(glu_opinfo)


mean_opinfo = OpInfo(
    ltorch.mean,
    sample_input_generator=reduction_sample_generator,
    torch_reference=torch.mean,
    dtypes=(datatypes.inexact,),
    test_directives=(
        # PyTorch doesn't support CPU and CUDA complex half mean
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        # nvFuser had issues with complex reductions, now fixed; TODO re-enable
        # this test.
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvfuser",),
        ),
        # The low precision floating point types sometimes fail
        #   test tolerances on CPU in CI
        DecorateInfo(
            pytest.mark.skip,
            dtypes=(datatypes.bfloat16, datatypes.float16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
reduction_ops.append(mean_opinfo)

var_opinfo = OpInfo(
    ltorch.var,
    sample_input_generator=var_sample_generator,
    torch_reference=torch.var,
    # Complex var is not supported yet
    dtypes=(datatypes.floating,),
    test_directives=(
        # bfloat16 on CPU has accuracy things
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch doesn't support float16 and bfloat16 on CUDA
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
reduction_ops.append(var_opinfo)

var_mean_opinfo = OpInfo(
    ltorch.var_mean,
    supports_grad=True,
    sample_input_generator=var_sample_generator,
    torch_reference=torch.var_mean,
    # Complex var is not supported yet
    dtypes=(datatypes.floating,),
    test_directives=(
        # See issue "nvFuser fails to compile some var_mean tests"
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
            executors=("nvfuser",),
        ),
    ),
)
reduction_ops.append(var_mean_opinfo)


def std_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim, correction, keepdim
    cases = (
        ((), 0, 0, True),
        ((5), -1, 1, False),
        ((4, 4), 1, 0, True),
        ((5, 1, 5), -2, 2, False),
        ((2, 3, 4, 5), -3, 1, True),
    )

    for shape, dim, correction, keepdim in cases:
        yield (SampleInput(make(shape), dim=dim, correction=correction, keepdim=keepdim))


def std_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    err_msg = "only tensors with up to 64 dims are supported"
    yield (SampleInput(make([1] * 65), dim=64), RuntimeError, err_msg)
    yield (SampleInput(make([1] * 65), dim=-1), RuntimeError, err_msg)

    err_msg = "Duplicate value in list of dimensions"
    yield (SampleInput(make((5, 5, 5, 5)), dim=(0, 0)), RuntimeError, err_msg)
    yield (SampleInput(make((5, 5, 5, 5)), dim=(0, -4)), RuntimeError, err_msg)


std_opinfo = OpInfo(
    ltorch.std,
    supports_grad=True,
    sample_input_generator=std_sample_generator,
    error_input_generator=std_error_generator,
    torch_reference=torch.std,
    dtypes=(datatypes.floating,),
)
reduction_ops.append(std_opinfo)


def cumsum_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim
    cases = (
        ((), 0),
        ((4), -1),
        ((4, 4), 1),
        ((8, 1, 6), -2),
        ((8, 7, 5, 1), -3),
    )

    for shape, dim in cases:
        # torch.cumsum not implemented for dtype='Bool'
        for output_dtype in (None, torch.float if dtype is torch.bool else dtype):
            yield (SampleInput(make(shape), dim, dtype=output_dtype))


cumsum_opinfo = OpInfo(
    ltorch.cumsum,
    sample_input_generator=cumsum_sample_generator,
    torch_reference=torch.cumsum,
    test_directives=(
        # Torch doesn't support cpu/cuda complex half cumsum
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # "cumsum_out_cpu" not implemented for 'Half'
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
reduction_ops.append(cumsum_opinfo)


def argmin_argmax_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim
    cases = (
        ((), 0),
        ((), None),
        ((3, 0), 0),
        ((4, 4), 1),
        ((4, 1, 6), -1),
        ((4, 2, 3), None),
        ((4, 7, 5, 1), -3),
        ((4, 2, 5, 1), None),
    )

    for shape, dim in cases:
        for keepdim in (True, False):
            yield (SampleInput(make(shape), dim, keepdim=keepdim))


def argmin_argmax_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    err_msg = r"Expected reduction dim .* to have non-zero size."
    yield (SampleInput(make(3, 0), 1), RuntimeError, err_msg)

    err_msg = r"Expected reduction dim to be specified for a.numel\(\) == 0."
    yield (SampleInput(make(3, 0)), RuntimeError, err_msg)


argmax_opinfo = OpInfo(
    clang.argmax,
    sample_input_generator=argmin_argmax_sample_generator,
    error_input_generator=argmin_argmax_error_generator,
    torch_reference=torch.argmax,
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
)
reduction_ops.append(argmax_opinfo)

argmin_opinfo = OpInfo(
    clang.argmin,
    sample_input_generator=argmin_argmax_sample_generator,
    error_input_generator=argmin_argmax_error_generator,
    torch_reference=torch.argmin,
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
)
reduction_ops.append(argmin_opinfo)


def make_sort_stable_sample(shape, dim, dtype, device, requires_grad):
    """
    Creates stable samples at which sort is differentiable.

    The following holds true for any sample `x` from this generator:
    sort(x, ...).indices == sort(x + eps, ...).indices for eps in (0, 1).
    """

    make = partial(make_tensor, device=device, dtype=dtype)

    noise = make(shape, low=0, high=1)
    data = 2 * torch.arange(0, noise.numel(), device=noise.device).reshape(noise.shape).to(noise.dtype)
    if data.ndim > 0 and data.shape[dim] > 0:
        perm = torch.randperm(data.shape[dim], device=data.device)
        data = data.index_select(dim, perm)

    sample = data + noise
    sample.requires_grad_(requires_grad)
    return sample


def topk_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_sort_stable_sample, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, k, dim
    # NOTE: k = 0 is not consistent between the CPU and the CUDA PyTorch implementations,
    # unless shape[dim] == 0
    cases = (
        ((), 1),
        ((), 1, 0),
        ((3, 0), 0),
        ((4, 4), 2, 1),
        ((4, 1, 6), 3, -1),
        ((4, 1, 6), 3),
        ((4, 7, 5, 1), 2, -3),
        ((4, 2, 5, 1), 1),
    )

    for shape, k, *dim in cases:
        if not dim:
            implied_dim = -1
        else:
            implied_dim = dim[0]

        for largest, sorted in itertools.product((True, False), repeat=2):
            yield SampleInput(make(shape, implied_dim), k, *dim, largest=largest, sorted=sorted)


def topk_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    err_msg = r"selected index .* is out of range"
    yield (SampleInput(make(3, 2), 3), RuntimeError, err_msg)
    yield (SampleInput(make(3, 0), 1), RuntimeError, err_msg)

    err_msg = "Dimension out of range"
    yield (SampleInput(make(3, 3), 1, 3), IndexError, err_msg)
    yield (SampleInput(make(3, 3), 1, -3), IndexError, err_msg)


topk_opinfo = OpInfo(
    clang.topk,
    name="topk",
    supports_grad=True,
    sample_input_generator=topk_sample_generator,
    error_input_generator=topk_error_generator,
    torch_reference=torch.topk,
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
)
reduction_ops.append(topk_opinfo)


def atleast_1d2d3d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (
        (),
        (4,),
        (5, 5),
        (6, 7, 8),
        (3, 3, 3, 3),
    )

    for c in cases:
        yield SampleInput(make(c))

    yield SampleInput(make(()), make((2,)))
    yield SampleInput(make((2,)), make((5, 5)))
    yield SampleInput(make(()), make((2,)), make((4, 4)))
    yield SampleInput(make(2, 3), make(4, 5), make(6, 6, 6), make(5, 5, 5, 5))


atleast_1d_opinfo = OpInfo(
    ltorch.atleast_1d,
    supports_grad=True,
    sample_input_generator=atleast_1d2d3d_sample_generator,
    torch_reference=torch.atleast_1d,
)
reduction_ops.append(atleast_1d_opinfo)


atleast_2d_opinfo = OpInfo(
    ltorch.atleast_2d,
    supports_grad=True,
    sample_input_generator=atleast_1d2d3d_sample_generator,
    torch_reference=torch.atleast_2d,
)
reduction_ops.append(atleast_2d_opinfo)


atleast_3d_opinfo = OpInfo(
    ltorch.atleast_3d,
    supports_grad=True,
    sample_input_generator=atleast_1d2d3d_sample_generator,
    torch_reference=torch.atleast_3d,
)
reduction_ops.append(atleast_3d_opinfo)


opinfos.extend(reduction_ops)


#
# Sort and dim permutations operations
#
dim_perm_ops = []


def sort_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_sort_stable_sample, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, dim
    cases = (
        ((),),
        ((), 0),
        ((3, 0),),
        ((4, 4), 1),
        ((4, 1, 6), -1),
        ((4, 1, 6), 0),
        ((4, 7, 5, 1), -3),
        ((4, 2, 5, 1),),
    )

    for shape, *dim in cases:
        if dim:
            dim = dim[0]
        else:
            dim = -1

        for descending, stable in itertools.product((True, False), repeat=2):
            yield SampleInput(make(shape, dim), dim=dim, descending=descending, stable=stable)


sort_opinfo = OpInfo(
    clang.sort,
    name="sort",
    supports_grad=True,
    sample_input_generator=sort_sample_generator,
    torch_reference=torch.sort,
    dtypes=(datatypes.bool8, datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-6, rtol=1e-6)),
            "test_vjp_correctness",
        ),
        DecorateInfo(
            pytest.mark.skip(reason="PyTorch does not yet support boolean types in sort for CUDA"),
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
dim_perm_ops.append(sort_opinfo)


argsort_opinfo = OpInfo(
    ltorch.argsort,
    name="argsort",
    supports_grad=False,
    sample_input_generator=sort_sample_generator,
    torch_reference=torch.argsort,
    dtypes=(datatypes.bool8, datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
    test_directives=(
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-6, rtol=1e-6)),
            "test_vjp_correctness",
        ),
        DecorateInfo(
            pytest.mark.skip(reason="PyTorch does not yet support boolean types in argsort for CUDA"),
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
    ),
)
dim_perm_ops.append(argsort_opinfo)


opinfos.extend(dim_perm_ops)


#
# Tensor Creation OpInfos
#
tensor_creation_ops = []


def arange_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # start, end, step
    common_cases = (
        (0, 1, 2),
        (-5, -8, -1),
        (-3, 11, 3),
    )
    extra_cases = ()

    if datatypes.is_inexact_dtype(dtype):
        # start, end, step
        extra_cases = (
            (5, 11, 0.3),
            (3, -4.2, -1),
        )

    for start, end, step in itertools.chain(common_cases, extra_cases):
        yield SampleInput(start=start, end=end, step=step, dtype=dtype, device=device)

    # arange only requires end be specified
    partial_cases = (
        (5,),
        (3, 7),
    )

    for case in partial_cases:
        yield SampleInput(*case)


arange_opinfo = OpInfo(
    ltorch.arange,
    sample_input_generator=arange_sample_generator,
    torch_reference=torch.arange,
    dtypes=(datatypes.signedinteger, datatypes.unsignedinteger, datatypes.floating),
)
tensor_creation_ops.append(arange_opinfo)


def vargs_shape_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # these shapes are valid for zeros, ones, empty, and randn
    cases = (
        (1,),
        (4, 4),
        (2, 0, 3),
        (8, 1, 6),
        (8, 7, 5, 1),
    )

    yield SampleInput((), device=device, dtype=dtype)
    for shape in cases:
        yield SampleInput(shape, device=device, dtype=dtype)
        yield SampleInput(*shape, device=device, dtype=dtype)


# TODO Think of how to test empty (the tensor values cannot be compared as close -- maybe pass custom comparator?)
# empty_opinfo = OpInfo(
#     ltorch.empty,
#     sample_input_generator=vargs_shape_sample_generator,
#     torch_reference=torch.zeros,
# )
# tensor_creation_ops.append(empty_opinfo)


def full_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make_fv = partial(make_number, dtype=dtype)

    # shape, fill_value
    cases = (
        ((), make_fv()),
        ((4, 4), make_fv()),
        ((8, 1, 6), make_fv()),
        ((8, 7, 5, 1), make_fv()),
        ((4, 4), make_tensor((), dtype=dtype, device=device)),
    )

    for shape, fill_value in cases:
        yield SampleInput(shape, fill_value, device=device, dtype=dtype)
    # Tests dtype is inferred correctly
    yield SampleInput(shape, fill_value, device=device)


def full_error_generator(op, device, **kwargs):
    err_msg = "Can't safely cast fill_value of numbertype <class 'complex'> to dtype thunder.dtypes.float32"
    yield (SampleInput((1, 2), 1j, device=device, dtype=torch.float), RuntimeError, err_msg)


full_opinfo = OpInfo(
    ltorch.full,
    sample_input_generator=full_sample_generator,
    error_input_generator=full_error_generator,
    torch_reference=torch.full,
)
tensor_creation_ops.append(full_opinfo)


def full_like_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    make_fv = partial(make_number, dtype=dtype)

    # shape, fill_value
    cases = (
        ((), make_fv()),
        ((4, 4), make_fv()),
        ((8, 1, 6), make_fv()),
        ((8, 7, 5, 1), make_fv()),
    )

    for shape, fill_value in cases:
        yield SampleInput(make(shape), fill_value)


full_like_opinfo = OpInfo(
    ltorch.full_like,
    sample_input_generator=full_like_sample_generator,
    torch_reference=torch.full_like,
)
tensor_creation_ops.append(full_like_opinfo)


def empty_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # shape, fill_value
    cases = (
        (()),
        ((4, 4)),
        ((8, 1, 6)),
        ((8, 7, 5, 1)),
    )

    for shape in cases:
        yield SampleInput(shape, device=device, dtype=dtype)
        if len(shape) > 0:  # *() will lead to no shape being passed to `empty`.
            yield SampleInput(*shape, device=device, dtype=dtype)


def empty_error_generator(op, device, **kwargs):
    err_msg = "Can't safely cast fill_value of numbertype <class 'complex'> to dtype float32"
    yield (SampleInput((1, 2), 1j, device=device, dtype=torch.float), RuntimeError, err_msg)


# Helper function for `empty` opinfo.
# It always returns zero tensors, so that the consistency tests pass.
def torch_empty_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.empty(*args, **kwargs), 0)


empty_opinfo = OpInfo(
    name="empty", op=torch_empty_and_zero, sample_input_generator=empty_sample_generator, torch_reference=torch.zeros
)
tensor_creation_ops.append(empty_opinfo)


def fixed_value_tensor_creation_op_sample_generator_with_bounds(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        (4, 4),
        (8, 1, 6),
        (8, 7, 5, 1),
        [
            4,
        ],  # Using `list[int]` should also work.
    )

    bounds = (
        (0, 2),
        (2,),  # we want to support the case when low is not given, like PyTorch
    )

    for shape in cases:
        for bound in bounds:
            yield SampleInput(*bound, shape, device=device, dtype=dtype)


def fixed_value_tensor_creation_op_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        (),
        (4, 4),
        (8, 1, 6),
        (8, 7, 5, 1),
        [
            4,
        ],  # Using `list[int]` should also work.
    )

    for shape in cases:
        yield SampleInput(shape, device=device, dtype=dtype)


# TODO Test overriding the "like" values, like by setting a different requires grad
#   This would probably require the "no dtype" and "no device" opinfo sample generator cases
def fixed_value_like_tensor_creation_op_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        5,
        (),
        (4, 4),
        (8, 1, 6),
        (8, 7, 5, 1),
    )

    for shape in cases:
        a = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(a)


def varargs_tensor_creation_op_sample_generator(*args, **kwargs):
    yield from fixed_value_tensor_creation_op_sample_generator(*args, **kwargs)
    yield from vargs_shape_sample_generator(*args, **kwargs)


def varargs_tensor_creation_op_sample_generator_with_bounds(*args, **kwargs):
    yield from fixed_value_tensor_creation_op_sample_generator_with_bounds(*args, **kwargs)


ones_opinfo = OpInfo(
    ltorch.ones,
    sample_input_generator=varargs_tensor_creation_op_sample_generator,
    torch_reference=torch.ones,
)
tensor_creation_ops.append(ones_opinfo)

zeros_opinfo = OpInfo(
    ltorch.zeros,
    sample_input_generator=varargs_tensor_creation_op_sample_generator,
    torch_reference=torch.zeros,
)
tensor_creation_ops.append(zeros_opinfo)

zeros_like_opinfo = OpInfo(
    ltorch.zeros_like,
    sample_input_generator=fixed_value_like_tensor_creation_op_sample_generator,
    torch_reference=torch.zeros_like,
)
tensor_creation_ops.append(zeros_like_opinfo)


# Helper function for `randn` opinfo.
# It always returns zero tensors, so that the consistency tests pass.
def torch_randn_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.randn(*args, **kwargs), 0)


def torch_rand_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.rand(*args, **kwargs), 0)


def torch_randint_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.randint(*args, **kwargs), 0)


def randn_error_generator(op, device, **kwargs):
    err_msg = "requires_grad=True is not yet supported"
    yield (SampleInput(1, 2, requires_grad=True), NotImplementedError, err_msg)
    err_msg = "generator is not None which"
    yield (SampleInput(1, 2, generator=torch.Generator()), NotImplementedError, err_msg)


# NOTE: This OpInfo ends up checking only `shape`, `device` and `dtype` consistency
# To test with OpInfo, we need operation to have deterministic output (for consistency tests).
# Since, randn returns random values, we call `full_like` on it to create output with fixed value.
# It is ok, as we just want to test `dtype`, `device` and `shape` for the output of `randn`
randn_opinfo = OpInfo(
    name="randn",
    op=torch_randn_and_zero,
    sample_input_generator=varargs_tensor_creation_op_sample_generator,
    error_input_generator=randn_error_generator,
    torch_reference=lambda *args, **kwargs: torch.randn(*args, **kwargs).fill_(0),
    dtypes=(datatypes.floating, datatypes.complexfloating),
)
tensor_creation_ops.append(randn_opinfo)

rand_opinfo = OpInfo(
    name="rand",
    op=torch_rand_and_zero,
    sample_input_generator=varargs_tensor_creation_op_sample_generator,
    error_input_generator=randn_error_generator,  # Does not depend on the distribution
    torch_reference=lambda *args, **kwargs: torch.rand(*args, **kwargs).fill_(0),
    dtypes=(datatypes.floating, datatypes.complexfloating),
)
tensor_creation_ops.append(rand_opinfo)

randint_opinfo = OpInfo(
    name="randint",
    op=torch_randint_and_zero,
    sample_input_generator=varargs_tensor_creation_op_sample_generator_with_bounds,
    error_input_generator=randn_error_generator,  # Does not depend on the distribution
    torch_reference=lambda *args, **kwargs: torch.randint(*args, **kwargs).fill_(0),
    dtypes=(datatypes.int64, datatypes.floating),
)
tensor_creation_ops.append(randint_opinfo)


# Helper function for `randn_like` opinfo.
# It always returns zero tensors, so that the consistency tests pass.
def torch_randn_like_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.randn_like(*args, **kwargs), 0)


# NOTE: This OpInfo ends up checking only `shape`, `device` and `dtype` consistency
# See the note on `randn` OpInfo for more details.
randn_like_opinfo = OpInfo(
    torch_randn_like_and_zero,
    sample_input_generator=fixed_value_like_tensor_creation_op_sample_generator,
    torch_reference=lambda *args, **kwargs: torch.randn_like(*args, **kwargs).fill_(0),
    dtypes=(datatypes.floating, datatypes.complexfloating),
)
tensor_creation_ops.append(randn_like_opinfo)


def torch_rand_like_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.rand_like(*args, **kwargs), 0)


rand_like_opinfo = OpInfo(
    torch_rand_like_and_zero,
    sample_input_generator=fixed_value_like_tensor_creation_op_sample_generator,
    torch_reference=lambda *args, **kwargs: torch.rand_like(*args, **kwargs).fill_(0),
    dtypes=(datatypes.floating, datatypes.complexfloating),
)
tensor_creation_ops.append(rand_like_opinfo)


def torch_empty_like_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.empty_like(*args, **kwargs), 0)


empty_like_opinfo = OpInfo(
    op=torch_empty_like_and_zero,
    sample_input_generator=fixed_value_like_tensor_creation_op_sample_generator,
    torch_reference=lambda *args, **kwargs: torch.empty_like(*args, **kwargs).fill_(0),
)
tensor_creation_ops.append(empty_like_opinfo)


def bernoulli_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make_t = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=0, high=1)

    shapes = ((), (2, 2), (2, 0, 1), (1, 2, 3))

    for shape in shapes:
        yield SampleInput(make_t(shape))


def bernoulli_error_generator(op, device, **kwargs):
    err_msg = "bernoulli only supports floating point dtypes, got thunder.dtypes.int64"
    yield (SampleInput(torch.ones(3, 3, device=device, dtype=torch.long)), RuntimeError, err_msg)

    err_msg = "generator is not None which is currently unsupported"
    yield (
        SampleInput(torch.ones(3, 3, device=device), generator=torch.Generator(device=device)),
        RuntimeError,
        err_msg,
    )


# Helper function for `bernoulli` opinfo.
# It always returns zero tensors, so that the consistency tests and grad tests pass.
def torch_bernoulli_and_zero(*args, **kwargs):
    return ltorch.full_like(ltorch.bernoulli(*args, **kwargs), 0)


# NOTE: This OpInfo ends up checking only `shape`, `device` and `dtype` consistency
# similar to `randn`
# See the note on `randn` OpInfo for more details.
bernoulli_opinfo = OpInfo(
    name="bernoulli",
    op=torch_bernoulli_and_zero,
    sample_input_generator=bernoulli_sample_generator,
    error_input_generator=bernoulli_error_generator,
    torch_reference=lambda *args, **kwargs: torch.bernoulli(*args, **kwargs).fill_(0),
    supports_grad=False,
    dtypes=(datatypes.floating,),
)
opinfos.append(bernoulli_opinfo)


def tensor_constructor_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # Used to generate sequence.
    make_t = partial(make_tensor, device="cpu", dtype=dtype, requires_grad=False)
    bool_list = [True, False] * 8
    unsigned_int_list = [0, 1, 2, 12, 24, 65, 127, 255] * 2
    small_int_list = [-127, -64, 0, 32, 64, 65, 123] * 2
    int_list = [-50000, -25000, 0, 1, 2, 5000, 10000, 150000] * 2
    INF = float("-inf")
    NEG_INF = -INF
    NAN = float("nan")
    float_list = [
        INF,
        NEG_INF,
        NAN,
        -0.0,
        0.0,
        2e30,
        -2e30,
        0.000012,
        -0.000012,
        -5e10,
        5e10,
        24e32,
        -24e32,
        1e-30,
        -1e-30,
        2.0,
    ]

    complex_list = [
        complex(INF, 0),
        complex(-0.0, NEG_INF),
        complex(NAN, NAN),
        complex(NAN, INF),
        10e5j,
        -10e5j,
        0.000012j,
        -0.0000012j,
        1 + 2j,
        -1 - 2j,
        1 - 2j,
        -1 + 2j,
        0.000012 + 0.0000012j,
        1e30 + 2e24j,
        -0.0 - 0j,
        0.0 + 0j,
    ]

    # Python Scalars
    yield SampleInput(False, device=device)
    yield SampleInput(2, device=device)
    yield SampleInput(3.14, device=device)
    yield SampleInput(2j, device=device)

    # Infer dtype
    yield SampleInput(
        bool_list,
        device=device,
    )
    yield SampleInput(
        [bool_list, int_list],
        device=device,
    )
    yield SampleInput(
        [[bool_list, int_list], [float_list, float_list]],
        device=device,
    )
    yield SampleInput(
        [[bool_list, int_list], [float_list, complex_list]],
        device=device,
    )

    # Interesting shapes
    yield SampleInput(make_t(()).tolist(), device=device)
    yield SampleInput(make_t(0).tolist(), device=device)
    yield SampleInput(make_t(0, 1, 2).tolist(), device=device)
    yield SampleInput(make_t(1, 2, 0).tolist(), device=device)
    yield SampleInput(make_t(1, 2, 0, 2, 3).tolist(), device=device)

    # Normal shapes
    yield SampleInput(make_t(1, 2, 3).tolist(), device=device)
    yield SampleInput(make_t(2, 2, 3).tolist(), device=device)
    yield SampleInput(make_t(1, 2, 1, 4, 3).tolist(), device=device)

    def get_seq_list(dtype):
        if dtype.is_complex:
            return complex_list
        elif dtype.is_floating_point:
            return float_list
        elif dtype == torch.bool:
            return bool_list
        elif dtype == torch.uint8:
            return unsigned_int_list
        elif dtype in (torch.int8, torch.int16):
            return small_int_list
        else:
            return int_list

    # dtype specified
    yield SampleInput(
        get_seq_list(dtype),
        dtype=dtype,
        device=device,
    )
    yield SampleInput(
        [[]],
        dtype=dtype,
        device=device,
    )


def tensor_constructor_error_generator(op, device, **kwargs):
    err_msg = "Expected seq of len=2 at dim 1"
    yield (SampleInput([[1, 0], [1]]), RuntimeError, err_msg)

    err_msg = "Expected sequences of numbers, but found type <class 'str'>"
    yield (SampleInput(["hi"]), ValueError, err_msg)

    err_msg = "Expected sequences of numbers, but found type <class 'str'>"
    yield (SampleInput([[1], ["hi"]]), ValueError, err_msg)

    err_msg = "Expected sequences of numbers, but found type <class 'list'>"
    yield (SampleInput([[1], [[6, 2]]]), ValueError, err_msg)

    err_msg = "Can't safely cast sequence with numbertype <class 'float'> to dtype thunder.dtypes.int32"
    yield (SampleInput([[1, 2.0], [6, 2]], dtype=torch.int32), RuntimeError, err_msg)

    err_msg = "Can't safely cast sequence with numbertype <class 'complex'> to dtype thunder.dtypes.int32"
    yield (SampleInput([[1, 2j], [6, 2]], dtype=torch.int32), RuntimeError, err_msg)

    err_msg = "Can't safely cast sequence with numbertype <class 'complex'> to dtype thunder.dtypes.float64"
    yield (SampleInput([[1, 2j], [6, 2]], dtype=torch.float64), RuntimeError, err_msg)


tensor_constructor_opinfo = OpInfo(
    ltorch.tensor,
    sample_input_generator=tensor_constructor_sample_generator,
    error_input_generator=tensor_constructor_error_generator,
    torch_reference=torch.tensor,
)

tensor_creation_ops.append(tensor_constructor_opinfo)

opinfos.extend(tensor_creation_ops)

#
# Linear algebra OpInfos
#
linear_algebra_ops = []


def normalize_sample_generator(op, device, dtype, requires_grad, **kwargs):
    def make(shape, p=2.0, dim=1, min_scalar_norm=0.3):
        t = make_tensor(shape, device=device, dtype=torch.double)
        t = t / t.numel()

        # avoid values close to zero
        t = torch.where(t.abs() >= min_scalar_norm, t, min_scalar_norm)

        # pre-normalize for general stability
        t = torch.nn.functional.normalize(t, p=p, dim=dim)

        # stabilize for inf norms
        # idea: make max/min element larger/smaller
        # to guarantee stable argmax/argmin for finite-diff
        # grad checks
        if p == math.inf or p == -math.inf:
            if p == math.inf:
                op = torch.argmax
                delta = +min_scalar_norm / 2
            else:
                op = torch.argmin
                delta = -min_scalar_norm / 2

            idx = op(t, dim=dim, keepdim=True).expand_as(t)
            delta = torch.tensor(delta, device=device, dtype=torch.double).expand_as(t)
            t.scatter_add_(dim=dim, index=idx, src=delta)

        return t.to(dtype).requires_grad_(requires_grad)

    # input shape
    shapes = (
        (4, 4),
        (32, 8),
        (16, 10, 16),
        (4, 2, 4, 5),
    )

    # TODO: add samples with negative `p`
    for shape in shapes:
        yield SampleInput(make(shape), eps=1e-6)
        yield SampleInput(make(shape, p=0), p=0, eps=1e-6)
        yield SampleInput(make(shape, p=1), p=1, eps=1e-6)
        yield SampleInput(make(shape, p=4), p=4, eps=1e-6)
        yield SampleInput(make(shape, p=math.inf), p=math.inf, eps=1e-6)


normalize_opinfo = OpInfo(
    ltorch.normalize,
    sample_input_generator=normalize_sample_generator,
    torch_reference=torch.nn.functional.normalize,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    supports_grad=True,
    test_directives=(
        # TODO Investigate the low precision difference
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-1, rtol=1e-1)),
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        # Ref -https://github.com/Lightning-AI/lightning-thunder/issues/2363
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=5e-4, rtol=5e-4)),
            "test_vjp_correctness",
            dtypes=(datatypes.float64,),
            executors=("nvfuser",),
        ),
    ),
)
linear_algebra_ops.append(normalize_opinfo)


def matmul_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    M = 4
    N = 3
    B = 2
    # shape_a, shape_b
    cases = (
        ((M,), (M,)),
        ((M,), (M, N)),
        ((M, N), (N,)),
        ((M,), (B, M, N)),
        ((B, M, N), (N,)),
        ((M, N), (N, M)),
        ((B, M, N), (B, N, M)),
        ((B, B, M, N), (B, B, N, M)),
        # cases nd @ 2d --> these should lower to 2d-gemms for efficiency
        ((1, M, N), (N, N)),
        ((B, M, N), (N, N)),
        ((B, N, M), (M, N)),
    )

    for shape_a, shape_b in cases:
        yield SampleInput(make(shape_a), make(shape_b))


matmul_opinfo = OpInfo(
    ltorch.matmul,
    supports_grad=True,
    sample_input_generator=matmul_sample_generator,
    torch_reference=torch.matmul,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch CPU doesn't support float16 matmul
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch doesn't support complex32 matmul
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        # TODO Investigate the low precision difference -- PyTorch is slightly more accurate at this
        #   computation
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-1, rtol=1e-1)),
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
    ),
)
linear_algebra_ops.append(matmul_opinfo)


def multi_dot_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shapes
    cases = [
        [(2), (2, 3)],
        [(1, 2), (2, 3)],
        [(2, 3), (3, 2), (2, 2)],
        [(2, 3), (3, 10), (10, 4), (4, 2)],
    ]

    for shapes in cases:
        yield SampleInput([make(s) for s in shapes])


multi_dot_opinfo = OpInfo(
    ltorch.multi_dot,
    sample_input_generator=multi_dot_sample_generator,
    torch_reference=torch.linalg.multi_dot,
    dtypes=(datatypes.floating,),
)
linear_algebra_ops.append(multi_dot_opinfo)


def _grouped_mm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    M = 16
    N = 64
    K = 32
    G = 2

    a = make_tensor((M, K), device=device, dtype=dtype, low=0, high=1.0, requires_grad=False)
    b = make_tensor((G, K, N), device=device, dtype=dtype, low=0, high=1.0, requires_grad=False)

    # torch._grouped_mm, the torchex implementation, requires offsets to be
    # multiples of 16. nvFuser doesn't have that restriction.
    for offsets in [[0, 16], [16, 16]]:
        c = torch.tensor(offsets, device=device, dtype=torch.int32)
        bfloat16_comp = TorchTensorComp(atol=1e-1, rtol=1e-1)
        si = SampleInput(a, b, c)
        si.set_comparator(bfloat16_comp)
        yield si


def _group_sizes_from_offsets(offsets: torch.Tensor) -> list[int]:
    group_sizes = []
    prev = 0
    for offset in offsets:
        group_sizes.append(offset - prev)
        prev = offset
    return group_sizes


# torch._grouped_mm has too many constraints to be used as a reference
# implementation. For example, it supports only Hopper and only bfloat16.
def _grouped_mm_reference(a, b, offsets):
    num_groups = offsets.numel()
    group_sizes = _group_sizes_from_offsets(offsets)

    if a.dim() == 2 and b.dim() == 2:
        # [m, k] @ [k, n] => [g, m, n]
        group_as = a.split(group_sizes, -1)
        group_bs = b.split(group_sizes, 0)
        out = torch.empty(num_groups, a.size(0), b.size(-1), dtype=a.dtype, device=a.device)
        group_outs = out.unbind()
    elif a.dim() == 3 and b.dim() == 2:
        # [g, m, k] @ [k, n] => [m, n]
        group_as = a.unbind()
        group_bs = b.split(group_sizes, -1)
        out = torch.empty(a.size(1), b.size(-1), dtype=a.dtype, device=a.device)
        group_outs = out.split(group_sizes, -1)
    elif a.dim() == 2 and b.dim() == 3:
        # [m, k] @ [g, k, n] => [m, n]
        group_as = a.split(group_sizes, 0)
        group_bs = b.unbind()
        out = torch.empty(a.size(0), b.size(-1), dtype=a.dtype, device=a.device)
        group_outs = out.split(group_sizes, 0)
    else:
        assert False, f"Unexpected ranks: {a.size()} and {b.size()}"

    for group_a, group_b, group_out in zip(group_as, group_bs, group_outs):
        torch.matmul(group_a, group_b, out=group_out)

    return out


if LooseVersion(torch.__version__) >= "2.8":
    _grouped_mm_opinfo = OpInfo(
        prims._grouped_mm,
        supports_grad=False,
        sample_input_generator=_grouped_mm_sample_generator,
        torch_reference=_grouped_mm_reference,
        dtypes=(datatypes.bfloat16,),
        devicetypes=(devices.DeviceType.CUDA,),
        test_directives=(
            DecorateInfo(
                pytest.mark.skip,
                "test_core_vs_torch_consistency",
                executors=("torch",),
                # torch._grouped_mm, the torchex implementation, doesn't support pre-Hopper.
                active_if=(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0)),
            ),
        ),
    )
    linear_algebra_ops.append(_grouped_mm_opinfo)


def einsum_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shapes, equation
    cases = (
        # Basic views/diagonal-like results
        ([(3,)], "i"),
        ([(3, 3)], "ij"),
        ([(3, 3)], "ji"),
        ([(3, 3)], "ii->i"),
        ([(3, 3, 3)], "iji->j"),
        ([(2, 2, 3, 3)], "iijj->ij"),
        ([(2, 2, 3, 3)], "iijj->ji"),
        ([(3, 3, 3, 3)], "iiij->ij"),
        ([(3, 3, 3, 3)], "iiii->i"),
        # Basic GEMM/TDOT pairs
        ([(1, 3, 4), (4, 3)], "bij,jk"),
        ([(1, 3, 4), (4, 3)], "bij,jk->bik"),
        ([(1, 3, 4), (4, 3)], "bij,jk->bki"),
        ([(2, 3, 3), (1, 3, 3)], "bij,bjk"),
        ([(2, 3, 3), (1, 3, 3)], "bij,bkj"),
        ([(2, 3, 3), (1, 3, 3)], "bij,bjk->bik"),
        ([(2, 3, 3), (2, 3, 3)], "bij,bkj->bik"),
        # Tensor-like products, aka 'OUTER'
        ([(3,), (4,)], "i,j->ij"),
        ([(3,), (2, 2)], "i,jk->jik"),
        ([(3,), (4,), (5,)], "i,j,k->ijk"),
        ([(3,), (4,), (5,), (2,)], "i,j,k,l->lkji"),
        # Multiple reductions
        ([(1, 2, 3, 4), (4, 3, 2)], "ijkl,lkj->i"),
        ([(2, 2, 2), (2, 2, 2)], "ijk,ijk->i"),
        ([(2, 2, 2), (2, 2, 2)], "ijk,kji->i"),
        ([(2, 2, 2, 2), (2, 2, 2, 2)], "aijk,ijkb->ba"),
        ([(2, 2, 2, 2), (2, 2, 2, 2)], "aijk,jikb->ba"),
        ([(2, 2, 2, 2), (2, 2, 2, 2)], "aijk,kjib->ba"),
        # From a Transformer model (T5 uses it?).
        ([(1, 3, 4, 5), (2, 3, 2, 5)], "bnqd,bnkd->bnqk"),
        # Cases from BERT.
        ([(1, 3, 4, 5), (1, 4, 5)], "bhld,lrd->bhlr"),
        ([(1, 3, 4, 5), (4, 6, 5)], "bhld,lrd->bhlr"),
        # Basic ellipsis
        ([(3, 3)], "i...->..."),
        ([(1, 2, 3), (3,)], "...ik, ...j->ij"),
        ([(4,), (4, 4, 4)], "...a, ...a->..."),
        ([(3, 3, 2, 2)], "...ii->...i"),
        ([(2, 3, 3, 2)], "i...i->...i"),
        ([(1, 2, 3, 4)], "...ijk->...kji"),
        ([(2, 3, 2, 2), (3, 2, 2, 2)], "ij...,jk...->ik..."),
        ([(3, 2), (4, 3)], "k...,jk"),
        # Let's go with >=3 operands!
        ([(2, 2, 2), (2, 2, 2), (2, 2, 2)], "ijk,kji,jki->ij"),
        ([(1, 2, 3, 4), (1, 1, 5), (2, 1, 2)], "i...j,...k,l...m->ijklm"),
        ([(2, 2, 2), (2, 2, 2), (2, 2, 2)], "...i,...j,...kl->ijk"),
        ([(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)], "...i,...j,...kl,mn...->ijkm"),
    )

    for shapes, eq in cases:
        operands = [make(shape) for shape in shapes]
        yield SampleInput(eq, *operands)


def einsum_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    cases = (
        ([(3,)], "i->->j", "multiple arrows"),
        ([(3,), (3,)], "i->i", r"Found 1 operand\(s\) in the equation, but 2 tensor\(s\) were provided"),
        ([(3,)], "....i->i", "two or more ellipses"),
        ([(3,)], "...i...->i", "two or more ellipses"),
        ([(3, 3)], "i->i", "the number of subscripted dims 1 has to match the operand's dimensionality 2"),
        ([()], "i->i", r"subscripts more dimenions \(1\) then there are in the operand \(0\)"),
        ([(3, 3)], "ijk->i", r"subscripts more dimenions \(3\) then there are in the operand \(2\)"),
        ([(3,)], "i->j", "includes a 'j' label which does not apper in neither of the operand's subsripts"),
        ([(3,)], "i->ii", "Output subscript string 'ii' includes multiple 'i' labels"),
        ([(3, 1)], "ii->i", "label 'i' requires dimensions 1 and 0 to have the same lenght, but got 1 != 3"),
        ([(1, 3)], "ii->i", "label 'i' requires dimensions 1 and 0 to have the same lenght, but got 3 != 1"),
        (
            [(1, 2, 3), (2, 3, 1)],
            "i...j,k...l->ijkl",
            "Implied by ellipsis operands' dimensions do not jointy broadcast",
        ),
        (
            [(2, 2, 2), (3, 3, 3), (4, 4, 4)],
            "i...,...j,k...->ijk",
            "Implied by ellipsis operands' dimensions do not jointy broadcast",
        ),
        # opt_einsum throws.
        (
            [(1, 2, 3), (2, 3, 1)],
            "ijk,ljm->im",
            r"Size of label 'j' for operand 1 \(2\) does not match previous terms \(3\)",
        ),
    )

    for op_shapes, einsum_eq, err_msg in cases:
        yield (
            SampleInput(
                einsum_eq,
                *tuple(make(s) for s in op_shapes),
            ),
            ValueError,
            err_msg,
        )


einsum_opinfo = OpInfo(
    ltorch.einsum,
    sample_input_generator=einsum_sample_generator,
    error_input_generator=einsum_error_generator,
    torch_reference=torch.einsum,
    supports_grad=True,
    # TODO: test all integer types and figure out their dtype.
    dtypes=(datatypes.float32, datatypes.float64),
    # See issue "Disabled einsum tests might hide potential issues in our
    # testing/op implementations"
    # Testing only float32, float64 now.
    #  types=(datatypes.int64, datatypes.floating),
    #  domain=(-1, +1),
    test_directives=(
        DecorateInfo(
            pytest.mark.skip(reason="vjp is tested with manual tests"),
            "test_vjp_correctness",
        ),
        DecorateInfo(
            # Some flakiness in phantom grad tests.
            # TODO: investigate and restore lower values.
            custom_comparator(partial(assert_close, atol=1e-3, rtol=1e-3)),
            dtypes=(datatypes.float32,),
        ),
    ),
    #      DecorateInfo(
    #          pytest.mark.skip(reason="vjp is tested with manual tests"),
    #          "test_vjp_correctness",
    #      ),
    #      # RuntimeError: "addmm_impl_cpu" is not implemented for 'Half'
    #      DecorateInfo(
    #          pytest.mark.xfail,
    #          dtypes=(datatypes.float16,),
    #          devicetypes=(devices.DeviceType.CPU,),
    #      ),
    #      # PyTorch bug: tries to dispatch to bmm
    #      # which is not implemented for int64.
    #      DecorateInfo(
    #          pytest.mark.xfail,
    #          "test_core_vs_torch_consistency",
    #          dtypes=(datatypes.int64,),
    #          devicetypes=(devices.DeviceType.CUDA,),
    #      ),
    #      # Precision is very low on the CPU.
    #      # TODO: investigate.
    #      DecorateInfo(
    #          pytest.mark.xfail,
    #          "test_core_vs_torch_consistency",
    #          dtypes=(datatypes.bfloat16,),
    #          devicetypes=(devices.DeviceType.CPU,),
    #      ),
    #      DecorateInfo(
    #          custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
    #          dtypes=(datatypes.float16,),
    #      ),
    #      # Spurious single values.
    #      # TODO: investigate.
    #      DecorateInfo(
    #          custom_comparator(partial(assert_close, atol=1e-1, rtol=1e-1)),
    #          dtypes=(datatypes.bfloat16,),
    #      ),
)
linear_algebra_ops.append(einsum_opinfo)


def linear_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    in_features = 3
    out_features = 5
    batch_size = 2
    # shape_input, shape_weight
    cases_no_bias = (
        ((in_features,), (out_features, in_features)),
        ((batch_size, in_features), (out_features, in_features)),
    )

    for shape_input, shape_weight in cases_no_bias:
        yield SampleInput(make(shape_input), make(shape_weight))

    # shape_input, shape_weight, shape_bias
    cases_with_bias = (
        ((in_features,), (out_features, in_features), (out_features,)),
        ((batch_size, in_features), (out_features, in_features), (out_features,)),
    )

    for shape_input, shape_weight, shape_bias in cases_with_bias:
        yield SampleInput(make(shape_input), make(shape_weight), make(shape_bias))


linear_opinfo = OpInfo(
    ltorch.linear,
    supports_grad=True,
    sample_input_generator=linear_sample_generator,
    torch_reference=torch.nn.functional.linear,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch CPU doesn't support float16 linear
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch doesn't support complex32 linear
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
    ),
)
linear_algebra_ops.append(linear_opinfo)


def tensor_1d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (4, 3),
        (2, 1),
        (5, 0),
    )

    for shape_a, shape_b in cases:
        yield SampleInput(make(shape_a), make(shape_b))


outer_opinfo = OpInfo(
    ltorch.outer, supports_grad=True, sample_input_generator=tensor_1d_sample_generator, torch_reference=torch.outer
)
linear_algebra_ops.append(outer_opinfo)


opinfos.extend(linear_algebra_ops)

#
# NN Ops
#
nn_ops = []


def baddbmm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), 0.0, None),
        ((3, 0, 5), (3, 0, 0), (3, 0, 5), 0.0, None),
        ((0, 5, 6), (0, 5, 0), (0, 0, 6), 0.0, None),
        ((3, 5, 6), (3, 5, 0), (3, 0, 6), 0.0, None),
        ((3, 5, 6), (3, 5, 8), (3, 8, 6), 0.0, None),
        ((3, 5, 6), (3, 5, 8), (3, 8, 6), 0.25, float("inf")),
        ((3, 5, 6), (3, 5, 8), (3, 8, 6), 0.25, float("-inf")),
        ((3, 5, 6), (3, 5, 8), (3, 8, 6), 0.25, float("nan")),
    )

    int_constants_cases = ((2, 2), (1, 0), (0, 1))

    float_constants_cases = (
        (2.0, 2.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (float("inf"), 2.0),
        (2.0, float("inf")),
        (float("inf"), float("inf")),
        (float("-inf"), 2.0),
        (2.0, float("-inf")),
        (float("-inf"), float("-inf")),
        (float("nan"), 2.0),
        (2.0, float("nan")),
        (float("nan"), float("nan")),
    )

    for shape_in, shape_batch1, shape_batch2, singularity_amount, singularity_value in cases:
        a = make(shape_in)
        b1 = make(shape_batch1)
        b2 = make(shape_batch2)

        if isinstance(to_dtype(dtype), datatypes.exact):
            for alpha, beta in int_constants_cases:
                yield SampleInput(make(shape_in), make(shape_batch1), make(shape_batch2), alpha=alpha, beta=beta)
        else:
            if singularity_value is not None:
                a = replace_random_percentage(a, singularity_value, singularity_amount)
                b1 = replace_random_percentage(b1, singularity_value, singularity_amount)
                b2 = replace_random_percentage(b2, singularity_value, singularity_amount)

            for alpha, beta in float_constants_cases:
                yield SampleInput(make(shape_in), make(shape_batch1), make(shape_batch2), alpha=alpha, beta=beta)

    if isinstance(to_dtype(dtype), datatypes.exact):
        yield SampleInput(make(3, 5, 6), batch1=make(3, 5, 0), batch2=make(3, 0, 6), alpha=2, beta=2)


def baddbmm_error_generator(op, device, dtype=torch.int32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)
    cases = (
        ((3, 5, 6), (3, 5), (3, 0, 6), RuntimeError, "batch1 must be a 3D tensor, found 2 instead."),
        ((3, 5, 6), (3, 5, 0), (3, 0), RuntimeError, "batch2 must be a 3D tensor, found 2 instead."),
        (
            (3, 5, 6),
            (3, 5, 0),
            (3, 0, 6),
            ValueError,
            "2.0 had an unexpected type <class 'float'>. Supported types are <class 'int'>",
        ),
    )

    for shape_in, shape_batch1, shape_batch2, err_type, err_msg in cases:
        yield (
            SampleInput(make(shape_in), make(shape_batch1), make(shape_batch2), alpha=2.0, beta=2.0),
            err_type,
            err_msg,
        )


baddbmm_opinfo = OpInfo(
    ltorch.baddbmm,
    supports_grad=True,
    dtypes=(datatypes.floating, datatypes.signedinteger, datatypes.unsignedinteger),
    sample_input_generator=baddbmm_sample_generator,
    error_input_generator=baddbmm_error_generator,
    torch_reference=torch.baddbmm,
    test_directives=(
        # baddbmm not implemented on CUDA for int
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.exact,),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # test_phantom_grad_vs_torch_consistency does not support nan singularity
        DecorateInfo(
            pytest.mark.xfail,
            "test_phantom_grad_vs_torch_consistency",
        ),
        # InterpreterError: Encountered exception Failed: Timeout >240.0s while tracing
        # Appearing only in CI, passes locally.
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("torch", "nvfuser"),
        ),
    ),
)

nn_ops.append(baddbmm_opinfo)


def _convolution_get_default_args():
    defaults = {
        "stride": (1,),
        "padding": (0,),
        "dilation": (1,),
        "transposed": False,
        "output_padding": (0,),
        "groups": 1,
    }

    return defaults


def convolution_1d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # Mostly taken from PyTorch

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple = (
        ((1, 3, 4), (3, 3, 3), (3,), {"stride": (2,), "padding": 2, "groups": 1}),
        ((2, 4, 8), (2, 2, 3), (2,), {"stride": 3, "padding": 1, "groups": 2, "dilation": 2}),
        ((1, 4, 5), (1, 4, 3), None, {"stride": (2,), "padding": "valid"}),
        ((2, 2, 4), (2, 1, 4), (2,), {"stride": (1,), "padding": "same", "groups": 2, "dilation": (2,)}),
        # With defaults
        ((1, 4, 5), (3, 4, 3), None, {}),
        # Empty inputs
        ((0, 3, 4), (3, 3, 3), None, {}),  # Empty batch
        # We cannot test empty out_channels, because:
        # - we do not allow groups == 0.
        # - PyTorch will error unless out_channels == 0 >= groups == 1,
        #   otherwise it will error if groups == 0.
        # ((1, 3, 4), (0, 3, 3), None, {}),  # Empty out_channels,
        # Empty in_channels (i.e. a.shape[1]) implies empty weight.shape[1]
        ((1, 0, 4), (3, 0, 3), None, {}),  # Empty in_channels (i.e. a.shape[1])
        ((0, 0, 4), (3, 0, 3), None, {}),  # Empty batch and in_channels (i.e. a.shape[1])
    )

    for a_shape, weight_shape, bias_shape, kwargs in cases:
        yield SampleInput(
            make(a_shape), make(weight_shape), make(bias_shape) if bias_shape is not None else None, **kwargs
        )


def convolution_1d_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple = (
        # groups should be > 0
        ((1, 1, 1), (1, 1, 1), None, {"groups": 0}, "groups(.*?) should be greater than 0"),
        # Wrong weight dim.
        # The string match is partial because
        # this generator is used for both conv1d and convolution,
        # and they produce different error messages.
        ((1, 1, 1), (1, 1), None, {}, ""),  # weight.ndim is too low
        # Wrong a dim,
        ((1, 1, 1, 1, 1), (1, 1, 1), None, {}, ""),  # a.ndim > weight.ndim
        # Zero-dim features
        ((1, 1, 0), (1, 1, 1), None, {}, "Input's shape(.*?) can be zero only in the(.*?)batch(.*?)channel"),
        # Zero-dim kernel
        ((1, 1, 1), (1, 1, 0), None, {}, "kernel_size(.*?) should not contain zero dimensions"),
        # weight.shape[1] ==  a.shape[1] / groups, i.e. in_channels / groups
        ((1, 4, 1), (1, 1, 1), None, {"groups": 2}, r"weight.shape(.*?)equal to \(in_channels / groups\)"),
        # groups should divide out_channels, i.e. weight.shape[0]
        ((1, 4, 1), (3, 2, 1), None, {"groups": 2}, "out_channels(.*?) should be divisible by groups"),
        # Wrong bias ndim/numel
        ((1, 2, 2), (3, 2, 2), (1, 1), {}, "bias.ndim(.*?) should be 1"),
        ((1, 2, 2), (3, 2, 2), (1,), {}, r"bias.numel(.*?) should match out_channels, \(i.e. weight.shape\[0\]=3"),
    )

    for a_shape, weight_shape, bias_shape, kwargs, err_msg in cases:
        yield (
            SampleInput(
                make(a_shape), make(weight_shape), make(bias_shape) if bias_shape is not None else None, **kwargs
            ),
            RuntimeError,
            err_msg,
        )

    # Produce all sorts of wrong values for stride, padding, dilation
    def incorrect_seq_gen():
        min_val_map = {"stride": 1, "dilation": 1, "padding": 0}

        for param in ("stride", "padding", "dilation"):
            yield param, (1, 1), rf"len\({param}\) should be(.*?) 1 or"
            # convolution does not support scalars for sequence params,
            # only conv does. However, scalars are wrapped to a sequence
            # before convolution fallback only if they are integers.
            # To trigger the right exeption, this wrontly typed scalar
            # is passed as a sequence.
            yield param, (1.0,), "should be integers"
            yield param, (-1), f"should be (.*?) at least {min_val_map[param]}"

    for param, param_val, err_msg in incorrect_seq_gen():
        yield (SampleInput(make(1, 1, 1), make(1, 1, 1), None, **{param: param_val}), RuntimeError, err_msg)

    # padding == 'same' only works with all-1 strides
    yield (
        SampleInput(make(1, 1, 1), make(1, 1, 1), None, **{"padding": "same", "stride": 2}),
        RuntimeError,
        "padding='same' requires all `strides` to be 1",
    )

    # padded_a_dim = a_dim + 2 * padding
    # dilated_kernel = dilation * (kernel_size - 1) + 1
    # padded_a_dim < dilation_kernel signals shape inconsistency
    yield (
        SampleInput(make(2, 2, 2), make(2, 2, 2), None, **{"padding": 0, "dilation": 2}),
        RuntimeError,
        "Inconsistent shape",
    )


def _convolution_sample_dim_lifter(sample, **kwargs):
    make = partial(make_tensor, **kwargs)

    # Take an nD sample and lift it to (n+1)D by duplicating
    # the last dimension in the tensor inputs like a and weight.
    a, weight, bias = sample.args
    a = make(a.shape + (a.shape[-1],))
    weight = make(weight.shape + (weight.shape[-1],))
    bias = make(bias.shape) if bias is not None else None

    # Update the len of input sequences which are not strings
    for param_name, param_val in sample.kwargs.items():
        if not isinstance(param_val, str) and (isinstance(param_val, Sequence) and len(param_val) != 1):
            sample.kwargs[param_name] = tuple(param_val) + (param_val[-1],)

    return SampleInput(a, weight, bias, **sample.kwargs)


def _convolution_dim_lifter_sample_generator(sample_gen, op, **kwargs):
    # Iterate over nD samples from sample_gen and lift the dimension to (n+1)D
    # by duplicating the last dimension in the tensor inputs like a and weight.
    for sample in sample_gen(op, **kwargs):
        yield _convolution_sample_dim_lifter(sample, **kwargs)


def convolution_2d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # Mostly taken from PyTorch

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, groups, dilation)
    cases: tuple = (
        ((1, 3, 4, 4), (3, 3, 3, 3), (3,), {"stride": (2, 2), "padding": 2, "groups": 1}),
        ((2, 4, 8, 8), (2, 2, 3, 3), (2,), {"stride": (3, 2), "padding": (2, 1), "groups": 2, "dilation": (4, 4)}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {"stride": 2, "padding": 1, "groups": 1, "dilation": (2, 3)}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {"stride": 2, "padding": 1, "groups": 1, "dilation": (2, 3)}),
        ((1, 2, 4, 3), (4, 2, 3, 4), None, {"stride": 2, "padding": 1, "groups": 1}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {"stride": 2, "padding": "valid"}),
        ((1, 4, 5, 6), (1, 4, 2, 3), (1,), {"stride": 1, "padding": "same", "dilation": 3}),
        # Below are the group related samples from common_nn.py
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {"groups": 4}),
        ((2, 4, 6, 6), (8, 1, 3, 3), (8,), {"groups": 4}),
        ((2, 4, 6, 6), (8, 1, 3, 3), None, {"groups": 4}),
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {"groups": 4, "stride": (3, 2)}),
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {"groups": 4, "padding": (1, 1)}),
        ((2, 4, 5, 5), (4, 1, 2, 2), (4,), {"groups": 4, "dilation": (2, 2)}),
        ((2, 4, 6, 5), (6, 2, 3, 2), (6,), {"groups": 2}),
        # With defaults
        ((1, 4, 5, 5), (3, 4, 3, 3), None, {}),
        # Empty inputs
        ((0, 3, 4, 4), (3, 3, 3, 3), None, {}),  # Empty batch
        # We cannot test empty out_channels, because:
        # - we do not allow groups == 0.
        # - PyTorch will error unless out_channels == 0 >= groups == 1,
        #   otherwise it will error if groups == 0.
        # ((1, 3, 4, 4), (0, 3, 3, 3), None, {}),  # Empty out_channels,
        # Empty in_channels (i.e. a.shape[1]) implies empty weight.shape[1]
        ((1, 0, 4, 4), (3, 0, 3, 3), None, {}),  # Empty in_channels (i.e. a.shape[1])
        ((0, 0, 4, 4), (3, 0, 3, 3), None, {}),  # Empty batch and in_channels (i.e. a.shape[1])
    )

    for a_shape, weight_shape, bias_shape, kwargs in cases:
        yield SampleInput(
            make(a_shape), make(weight_shape), make(bias_shape) if bias_shape is not None else None, **kwargs
        )


def convolution_2d_error_generator(op, device, dtype=torch.float32, **kwargs):
    # We re-use 1D samples and lift them to 2D
    for sample, ex_type, err_msg in convolution_1d_error_generator(op, device, dtype, **kwargs):
        yield (_convolution_sample_dim_lifter(sample, device=device, dtype=dtype, **kwargs), ex_type, err_msg)


def convolution_3d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # PyTorch does not support 3D convolutions for bfloat16 CUDA.
    if torch.device(device).type == "cuda" and dtype is torch.bfloat16:
        return

    # We re-use 2D samples and lift them to 3D
    yield from _convolution_dim_lifter_sample_generator(
        convolution_2d_sample_generator, op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )


def convolution_3d_error_generator(op, device, dtype=torch.float32, **kwargs):
    # We re-use 2D samples and lift them to 3D
    for sample, ex_type, err_msg in convolution_2d_error_generator(op, device, dtype, **kwargs):
        yield (_convolution_sample_dim_lifter(sample, device=device, dtype=dtype, **kwargs), ex_type, err_msg)


def _convolution_sample_materialize_defaults(sample, **kwargs):
    make = partial(make_tensor, **kwargs)

    # Materialize defaults
    defaults = _convolution_get_default_args()
    sample.kwargs = defaults | sample.kwargs

    for key in ("stride", "padding", "dilation"):
        param = sample.kwargs[key]
        if isinstance(param, int):
            sample.kwargs[key] = (param,)

    # Handle string padding
    padding = sample.kwargs["padding"]
    if isinstance(padding, str):
        if padding == "valid":
            sample.kwargs["padding"] = defaults["padding"]
        elif padding == "same":
            # Pad a, replace padding == 'same' with zero
            sample.kwargs["padding"] = (0,)

            a, weight, bias = sample.args

            def produce_pad_args():
                dim = weight.ndim - 2
                dilation = sample.kwargs.get("dilation", defaults["dilation"])
                if isinstance(dilation, int):
                    dilation = (dilation,) * dim
                elif len(dilation) == 1:
                    dilation = (dilation[0],) * dim

                pad = []
                _, _, *kernel_size = weight.shape
                for d, k in zip(reversed(dilation), reversed(kernel_size)):
                    total_pad = d * (k - 1)
                    lo = total_pad // 2
                    hi = total_pad - lo
                    pad.append(lo)
                    pad.append(hi)
                # No need to pad batch and channel dims
                pad.extend([0, 0, 0, 0])
                return pad

            return SampleInput(
                torch.nn.functional.pad(make(a.shape), produce_pad_args(), mode="constant", value=0),
                make(weight.shape),
                make(bias.shape) if bias is not None else None,
                **sample.kwargs,
            )

    return sample


def convolution_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample in itertools.chain(
        convolution_1d_sample_generator(op, device, dtype, requires_grad, **kwargs),
        convolution_2d_sample_generator(op, device, dtype, requires_grad, **kwargs),
        convolution_3d_sample_generator(op, device, dtype, requires_grad, **kwargs),
    ):
        yield _convolution_sample_materialize_defaults(
            sample, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )


def convolution_error_generator(op, device, dtype=torch.float32, **kwargs):
    for sample, ex_type, err_msg in itertools.chain(
        convolution_1d_error_generator(op, device, dtype, **kwargs),
        convolution_2d_error_generator(op, device, dtype, **kwargs),
        convolution_3d_error_generator(op, device, dtype, **kwargs),
    ):
        # Leave padding == "same" to conv{1, 2, 3}d
        padding = sample.kwargs.get("padding", None)
        if padding == "same":
            continue

        yield (_convolution_sample_materialize_defaults(sample, device=device, dtype=dtype, **kwargs), ex_type, err_msg)


def _conv_remove_batch(sample, **kwargs):
    make = partial(make_tensor, **kwargs)

    a, weight, bias = sample.args
    # Batch dim is present in a
    if a.ndim == weight.ndim:
        a = make(a.shape[1:])
        weight = make(weight.shape)
        bias = make(bias.shape) if bias is not None else None
        return (SampleInput(a, weight, bias, **sample.kwargs),)
    else:
        return ()


def conv1d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample in itertools.chain(
        convolution_1d_sample_generator(op, device, dtype, requires_grad, **kwargs),
    ):
        yield sample
        yield from _conv_remove_batch(sample, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)


def conv1d_error_generator(op, device, dtype=torch.float32, **kwargs):
    for sample, ex_type, err_msg in itertools.chain(
        convolution_1d_error_generator(op, device, dtype, **kwargs),
    ):
        yield (sample, ex_type, err_msg)
        for s in _conv_remove_batch(sample, device=device, dtype=dtype, **kwargs):
            yield (s, ex_type, err_msg)


def conv2d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample in itertools.chain(
        convolution_2d_sample_generator(op, device, dtype, requires_grad, **kwargs),
    ):
        yield sample
        yield from _conv_remove_batch(sample, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)


def conv2d_error_generator(op, device, dtype=torch.float32, **kwargs):
    for sample, ex_type, err_msg in itertools.chain(
        convolution_2d_error_generator(op, device, dtype, **kwargs),
    ):
        yield (sample, ex_type, err_msg)
        for s in _conv_remove_batch(sample, device=device, dtype=dtype, **kwargs):
            yield (s, ex_type, err_msg)


def conv3d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample in itertools.chain(
        convolution_3d_sample_generator(op, device, dtype, requires_grad, **kwargs),
    ):
        yield sample
        yield from _conv_remove_batch(sample, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)


def conv3d_error_generator(op, device, dtype=torch.float32, **kwargs):
    for sample, ex_type, err_msg in itertools.chain(
        convolution_3d_error_generator(op, device, dtype, **kwargs),
    ):
        yield (sample, ex_type, err_msg)
        for s in _conv_remove_batch(sample, device=device, dtype=dtype, **kwargs):
            yield (s, ex_type, err_msg)


def generic_max_pool_sample_generator(conv_sample_generator):
    def sample_generator(op, device, dtype, requires_grad, **kwargs):
        make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

        for sample in conv_sample_generator(op, device, dtype, requires_grad, **kwargs):
            # Filter out "same" padding for simplicity
            padding = sample.kwargs.get("padding", 0)
            if padding == "valid":
                padding = 0
            else:
                continue

            input, weights, *_ = sample.args

            # Empty channels is not supported!
            in_channels = input.shape[0] if input.ndim != weights.ndim else input.shape[1]
            if in_channels == 0:
                continue

            kernel_size = weights.shape[2:]
            stride = sample.kwargs.get("stride", None)
            dilation = sample.kwargs.get("dilation", 1)

            yield SampleInput(make(input.shape), kernel_size, padding=padding, stride=stride, dilation=dilation)
            # stride == None implies stride = kernel_size
            yield SampleInput(make(input.shape), kernel_size, padding=padding, stride=None, dilation=dilation)

    return sample_generator


def generic_avg_pool_sample_generator(max_pool_sample_generator):
    def sample_generator(op, device, dtype, requires_grad, **kwargs):
        make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

        for sample in max_pool_sample_generator(op, device, dtype, requires_grad, **kwargs):
            input, *rest = sample.args

            # No dilation for avg_pool for some reason in PyTorch.
            # Good news: we can easily support that if needed!
            kwargs = sample.kwargs.copy()
            del kwargs["dilation"]

            yield SampleInput(make(input.shape), *rest, **kwargs)

    return sample_generator


convolution_opinfo = OpInfo(
    clang.convolution,
    sample_input_generator=convolution_sample_generator,
    error_input_generator=convolution_error_generator,
    torch_reference=torch.convolution,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness", active_if=IS_WINDOWS),
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            # PyTorch supports complex dtypes only
            # in composite operations like
            # torch.nn.functional.conv{1, 2, 3}
            dtypes=(datatypes.complexfloating,),
            executors=("torch", "nvfuser"),
        ),
        # InterpreterError: Encountered exception Failed: Timeout >240.0s while tracing
        # Appearing only in CI, passes locally.
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("torch", "nvfuser"),
        ),
    ),
)
nn_ops.append(convolution_opinfo)


conv1d_opinfo = OpInfo(
    ltorch.conv1d,
    sample_input_generator=conv1d_sample_generator,
    error_input_generator=conv1d_error_generator,
    torch_reference=torch.nn.functional.conv1d,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            # We do not support complex convolutions
            # because there is no access to real/imag yet.
            dtypes=(datatypes.complexfloating,),
            executors=("torch", "nvfuser"),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(conv1d_opinfo)


conv2d_opinfo = OpInfo(
    ltorch.conv2d,
    sample_input_generator=conv2d_sample_generator,
    error_input_generator=conv2d_error_generator,
    torch_reference=torch.nn.functional.conv2d,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            # We do not support complex convolutions
            # because there is no access to real/imag yet.
            dtypes=(datatypes.complexfloating,),
            executors=("torch", "nvfuser"),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(conv2d_opinfo)


conv3d_opinfo = OpInfo(
    ltorch.conv3d,
    sample_input_generator=conv3d_sample_generator,
    error_input_generator=conv3d_error_generator,
    torch_reference=torch.nn.functional.conv3d,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            # We do not support complex convolutions
            # because there is no access to real/imag yet.
            dtypes=(datatypes.complexfloating,),
            executors=("torch", "nvfuser"),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(conv3d_opinfo)


avg_pool1d_opinfo = OpInfo(
    ltorch.avg_pool1d,
    sample_input_generator=generic_avg_pool_sample_generator(
        generic_max_pool_sample_generator(conv1d_sample_generator)
    ),
    torch_reference=torch.nn.functional.avg_pool1d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(avg_pool1d_opinfo)


avg_pool2d_opinfo = OpInfo(
    ltorch.avg_pool2d,
    sample_input_generator=generic_avg_pool_sample_generator(
        generic_max_pool_sample_generator(conv2d_sample_generator)
    ),
    torch_reference=torch.nn.functional.avg_pool2d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(avg_pool2d_opinfo)


avg_pool3d_opinfo = OpInfo(
    ltorch.avg_pool3d,
    sample_input_generator=generic_avg_pool_sample_generator(
        generic_max_pool_sample_generator(conv3d_sample_generator)
    ),
    torch_reference=torch.nn.functional.avg_pool3d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support {b}float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(avg_pool3d_opinfo)


def adaptive_avg_pool2d_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    cases_runtime_error = (
        ((3,), (1, 1), "adaptive_avg_pool2d: Expected 3D or 4D tensor, but got"),
        ((3, 4, 5), (3,), "adaptive_avg_pool2d: output_size must be 2"),
        (
            (3, 4, 5),
            (3, -2),
            "Found invalid length",
        ),
        (
            (3, 4, 0),
            (3, 2),
            "adaptive_avg_pool2d: Expected input to have non-zero size for non-batch dimensions, but input has sizes ",
        ),
        (
            (1, 3, 0, 4),
            (3, 2),
            "adaptive_avg_pool2d: Expected input to have non-zero size for non-batch dimensions, but input has sizes ",
        ),
    )
    cases_value_error = (
        ((3, 4, 5), (3.0, 3.0), r"Element (.*?) \((.*?)\) had an unexpected type"),
        ((3, 4, 5), 4.0, r"(.*?) had an unexpected type"),
    )
    for input_args, err_type in zip((cases_value_error, cases_runtime_error), (ValueError, RuntimeError)):
        for op_shapes, output_sizes, err_msg in input_args:
            yield (
                SampleInput(
                    make(op_shapes),
                    output_sizes,
                ),
                err_type,
                err_msg,
            )


def adaptive_avg_pool2d_sample_generator(op, device, dtype, requires_grad, **kwargs):
    cases = (
        ((3, 3, 3), 5),
        ((3, 4, 4), (3, 3)),
        ((1, 3, 5, 2), 3),
        ((3, 2, 3, 4), (2, 4)),
        ((3, 0, 3, 4), (0, 0)),
        ((0, 0, 3, 4), (0, 2)),
    )

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for input_shape, output_shape in cases:
        a = make(input_shape)
        yield SampleInput(a, output_shape)


adaptive_avg_pool2d_opinfo = OpInfo(
    ltorch.adaptive_avg_pool2d,
    sample_input_generator=adaptive_avg_pool2d_sample_generator,
    error_input_generator=adaptive_avg_pool2d_error_generator,
    torch_reference=torch.nn.functional.adaptive_avg_pool2d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # nvfuser does not support adaptive_avg_pool2d for now
        DecorateInfo(
            pytest.mark.skip,
            executors=("nvfuser",),
        ),
        # NOTE: Pytorch handles zero size non-batch dimension differently for adaptive_avg_pool2d_backward between CUDA and CPU
        # RuntimeError: adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, but grad_output has sizes [3, 0, 0, 0] with dimension 1 being empty
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(adaptive_avg_pool2d_opinfo)


max_pool1d_opinfo = OpInfo(
    ltorch.max_pool1d,
    sample_input_generator=generic_max_pool_sample_generator(conv1d_sample_generator),
    torch_reference=torch.nn.functional.max_pool1d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(max_pool1d_opinfo)


max_pool2d_opinfo = OpInfo(
    ltorch.max_pool2d,
    sample_input_generator=generic_max_pool_sample_generator(conv2d_sample_generator),
    torch_reference=torch.nn.functional.max_pool2d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(max_pool2d_opinfo)


max_pool3d_opinfo = OpInfo(
    ltorch.max_pool3d,
    sample_input_generator=generic_max_pool_sample_generator(conv3d_sample_generator),
    torch_reference=torch.nn.functional.max_pool3d,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Skipped because it is slow.
        # TODO: remove once the grad tests are fast.
        DecorateInfo(
            pytest.mark.skip(reason="Slow test. Skipping for now."),
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(max_pool3d_opinfo)


def one_hot_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    test_shapes = [
        (0, 512),
        (10,),
        (5, 10),
        (3, 5, 10),
    ]

    max_value = 9
    for shape in test_shapes:
        for num_classes in range(1, max_value + 1):
            a = make(shape, low=0, high=num_classes - 1)  # use non-negative integers

            yield SampleInput(a, num_classes=num_classes)


one_hot_opinfo = OpInfo(
    ltorch.one_hot,
    sample_input_generator=one_hot_sample_generator,
    torch_reference=torch.nn.functional.one_hot,
    dtypes=(datatypes.int64,),  # akin to torch.long. F.one_hot expects input LongTensor
)
nn_ops.append(one_hot_opinfo)


def group_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # NOTE: we set low/high to -+ 1 to avoid numerical issues with reduced float types.
    make = partial(make_tensor, low=-1, high=+1, device=device, dtype=dtype, requires_grad=requires_grad)

    dim_len = (0, 2)
    groups = (1,)
    ndims = (2, 3, 4)
    for ndim in ndims:
        for shape in itertools.product(dim_len, repeat=ndim):
            _, num_channels, *inner_dims = shape

            # PyTorch has a bug, it causes:
            # RuntimeError: CUDA error: invalid configuration argument
            # for inputs with ndim >= 3 and num_channels == 0 with empty
            # weight and/or bias.
            if torch.device(device).type == "cuda" and ndim >= 3 and num_channels == 0:
                continue

            a = make(shape)

            for weight, bias in itertools.product((False, True), repeat=2):
                for group_len in groups:
                    yield SampleInput(
                        a, group_len, make((num_channels,)) if weight else None, make((num_channels,)) if bias else None
                    )


def group_norm_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    yield (
        SampleInput(make((1,)), 1),
        RuntimeError,
        "a.ndim=1 should be at least 2",
    )
    yield (SampleInput(make((1, 1)), 0), RuntimeError, "num_groups=(.*?) should be greater than 0")
    yield (SampleInput(make((1, 1)), 2), RuntimeError, "num_channels=(.*?) should be divisible by num_groups")
    for param in ("weight", "bias"):
        yield (
            SampleInput(make((1, 1)), 1, **{param: make((1, 1))}),
            RuntimeError,
            f"{param}.ndim=(.*?) should be equal to 1",
        )
        yield (
            SampleInput(make((2, 3)), 1, **{param: make((4,))}),
            RuntimeError,
            f"{param}.numel=(.*?) to num_channels=3",
        )


group_norm_opinfo = OpInfo(
    ltorch.group_norm,
    sample_input_generator=group_norm_sample_generator,
    error_input_generator=group_norm_error_generator,
    torch_reference=torch.nn.functional.group_norm,
    # Complex var is not supported yet
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # PyTorch doesn't support float16 and bfloat16 on CUDA
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
            devicetypes=(devices.DeviceType.CUDA,),
        ),
        # This should be fixed now; TODO re-enable, test
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvfuser",),
        ),
    ),
)
nn_ops.append(group_norm_opinfo)


def layer_norm_reference_generator(op, device, dtype, requires_grad, **kwargs):
    yield from layer_norm_sample_generator(op, device, dtype, requires_grad, **kwargs)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # batch, seq, embedding
    cases = (
        (16, 128, 768),
        (16, 128, 1024),
        (16, 128, 1280),
        (16, 128, 1600),
        (16, 512, 768),
        (16, 512, 1024),
        (16, 512, 1280),
        (16, 512, 1600),
    )

    for batch, seq_len, embedding in cases:
        a = make_arg(batch, seq_len, embedding)
        normalized_shape = (embedding,)

        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)

        yield SampleInput(a, normalized_shape, weight, bias, 1e-03)


def layer_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # input_shape, normalized_shape, kwargs
    cases = (
        ((1, 2, 3), (1, 2, 3), {"eps": 0.5}),
        ((2, 2, 3), (2, 3), {"eps": 0.5}),
        ((1,), (1,), {}),
        ((1, 2), (2,), {}),
        # ((0, 1), (1,), {}),  # nvfuser doesn't handle tensors with zeros in shape.
    )

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    for input_shape, normalized_shape, kwargs in cases:
        # Shape of weight and bias should be the same as normalized_shape
        a = make_arg(input_shape)
        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)
        yield SampleInput(a, normalized_shape, weight, bias, **kwargs)


def layer_norm_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    yield (SampleInput(make(1, 1), ()), RuntimeError, "Expected normalized_shape=(.*?) to have length >= 1")
    yield (
        SampleInput(make(1), (1, 1)),
        RuntimeError,
        "Expected a.ndim=1 to be greater than or equal to len\\(normalized_shape\\)=2",
    )
    yield (
        SampleInput(make(1, 2, 3), (2, 1)),
        RuntimeError,
        "Expected the last 2 dimensions",  # a.shape[-len(normalized_shape):] == normalized_shape
    )
    for param in ("weight", "bias"):
        yield (
            SampleInput(make((1, 2, 3)), (2, 3), **{param: make((1, 1))}),
            RuntimeError,
            f"Expected {param}.shape(.*?) to be the same as normalized_shape",
        )


layer_norm_opinfo = OpInfo(
    ltorch.layer_norm,
    sample_input_generator=layer_norm_sample_generator,
    error_input_generator=layer_norm_error_generator,
    reference_input_generator=layer_norm_reference_generator,
    torch_reference=torch.nn.functional.layer_norm,
    # Complex var is not supported yet
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(layer_norm_opinfo)


def rms_norm_reference_generator(op, device, dtype, requires_grad, **kwargs):
    for sample_inputs in layer_norm_reference_generator(op, device, dtype, requires_grad, **kwargs):
        if len(sample_inputs.args) > 3:  # positional bias
            sample_inputs.args = sample_inputs.args[:3] + sample_inputs.args[4:]
        sample_inputs.kwargs.pop("bias", None)
        yield sample_inputs


def rms_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    for sample_inputs in layer_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
        if len(sample_inputs.args) > 3:  # positional bias
            sample_inputs.args = sample_inputs.args[:3] + sample_inputs.args[4:]
        sample_inputs.kwargs.pop("bias", None)
        yield sample_inputs


def rms_norm_error_generator(op, device, **kwargs):
    for sample_inputs, exc_type, msg in layer_norm_error_generator(op, device, **kwargs):
        if len(sample_inputs.args) > 3:  # positional bias
            sample_inputs.args = sample_inputs.args[:3] + sample_inputs.args[4:]
        sample_inputs.kwargs.pop("bias", None)
        if "bias" not in msg:
            yield sample_inputs, exc_type, msg


if LooseVersion(torch.__version__) >= "2.4":
    rms_norm_opinfo = OpInfo(
        ltorch.rms_norm,
        sample_input_generator=rms_norm_sample_generator,
        error_input_generator=rms_norm_error_generator,
        reference_input_generator=rms_norm_reference_generator,
        torch_reference=torch.nn.functional.rms_norm,
        # Complex var is not supported yet
        dtypes=(datatypes.floating,),
        test_directives=(
            # PyTorch does not support float16 on CPU
            DecorateInfo(
                pytest.mark.xfail,
                "test_core_vs_torch_consistency",
                dtypes=(datatypes.float16,),
                devicetypes=(devices.DeviceType.CPU,),
            ),
            DecorateInfo(
                pytest.mark.xfail,
                dtypes=(datatypes.float16, datatypes.bfloat16),
                devicetypes=(devices.DeviceType.CUDA, devices.DeviceType.CPU),
                active_if=LooseVersion(torch.__version__) < "2.7",
            ),
        ),
    )
    nn_ops.append(rms_norm_opinfo)


def batch_norm_reference_generator(op, device, dtype, requires_grad, **kwargs):
    yield from layer_norm_sample_generator(op, device, dtype, requires_grad, **kwargs)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # batch, seq, embedding
    cases = (
        (16, 128, 768),
        (16, 128, 1024),
        (16, 128, 1280),
        (16, 128, 1600),
        (16, 512, 768),
        (16, 512, 1024),
        (16, 512, 1280),
        (16, 512, 1600),
    )

    for batch, seq_len, embedding in cases:
        a = make_arg(batch, seq_len, embedding)
        normalized_shape = (batch,)

        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)
        running_mean = make_arg(normalized_shape)
        running_var = make_arg(normalized_shape)

        yield SampleInput(a, running_mean, running_var, weight, bias, 1e-03)


def batch_norm_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    yield (
        SampleInput(
            make(
                0,
            ),
            (),
            (),
            (),
            (),
        ),
        RuntimeError,
        "Expected input_shape=(.*?) to have length >= 2",
    )
    yield (
        SampleInput(
            make(
                2,
            ),
            (),
            (),
            make(
                2,
            ),
            make(
                2,
            ),
        ),
        RuntimeError,
        "Expected input_shape=(.*?) to have length >= 2",
    )
    yield (
        SampleInput(
            make(
                2,
                2,
            ),
            (),
            (),
            make(2, 2),
            make(
                2,
            ),
        ),
        RuntimeError,
        "Expected weight.shape=(.*?) to be (.*?)!",
    )
    yield (
        SampleInput(
            make(
                2,
                2,
            ),
            (),
            (),
            make(
                2,
            ),
            make(
                2,
                2,
            ),
        ),
        RuntimeError,
        "Expected bias.shape=(.*?) to be (.*?)!",
    )


def batch_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # input_shape, kwargs
    # TODO: implement running_mean and running_var
    cases = (
        ((3, 3, 3), {"momentum": 0.2}),
        ((3, 3, 3), {"momentum": -1.2}),
        ((3, 3, 5, 6), {"momentum": 0.0}),
        ((3, 2, 3, 4), {"momentum": -1.0, "eps": 0.5}),
        ((3, 2, 3, 4, 12), {"momentum": -1.0, "eps": 0.5}),
    )
    if op.name != "instance_norm":
        cases += (((3, 4), {"momentum": 0.2, "eps": 0.5}),)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    for input_shape, kwargs in cases:
        # Shape of weight and bias should be the same as normalized_shape
        normalized_shape = (input_shape[1],)
        for mean_var, w, b, training in itertools.product((True, False), (True, False), (True, False), (True, False)):
            if not training and not mean_var:
                continue
            a = make_arg(input_shape)
            weight = make_arg(normalized_shape) if w else None
            bias = make_arg(normalized_shape) if b else None
            # 'batch_norm' is not differentiable with respect to argument 'running_mean' and 'running_var'
            running_mean = make(normalized_shape) if mean_var else None
            running_var = make(normalized_shape) if mean_var else None
            yield SampleInput(a, running_mean, running_var, weight, bias, training, **kwargs)


batch_norm_opinfo = OpInfo(
    ltorch.batch_norm,
    sample_input_generator=batch_norm_sample_generator,
    error_input_generator=batch_norm_error_generator,
    reference_input_generator=batch_norm_reference_generator,
    torch_reference=torch.nn.functional.batch_norm,
    # Complex var is not supported yet
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(batch_norm_opinfo)


instance_norm_opinfo = OpInfo(
    ltorch.instance_norm,
    sample_input_generator=batch_norm_sample_generator,
    torch_reference=torch.nn.functional.instance_norm,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(instance_norm_opinfo)


def local_response_norm_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # shape, size, {alpha, beta, k}
    cases = (
        ((1, 1, 1), 1, {}),
        ((3, 3, 3), 3, {}),
        ((3, 3, 3, 3), 1, {}),
        ((3, 3, 3), 1, {"alpha": 0.01, "beta": 0.8, "k": 1.5}),
        ((2, 5, 3, 2), 2, {"k": 1.2}),
        ((3, 5, 512), 1, {"alpha": 0.001, "beta": 1.2, "k": 1.5}),
    )

    for shape, size, kwargs in cases:
        sample = SampleInput(make(shape), size, **kwargs)
        if dtype == torch.bfloat16 or dtype == torch.float16:
            sample.set_comparator(TorchTensorComp(atol=1e-1, rtol=1e-1))

        yield sample


local_response_norm_opinfo = OpInfo(
    ltorch.local_response_norm,
    sample_input_generator=local_response_norm_sample_generator,
    torch_reference=torch.nn.functional.local_response_norm,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support b/float16 on CPU
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)
nn_ops.append(local_response_norm_opinfo)


def softmax_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    S = 2
    M = 5
    # shape, dim
    cases = (
        ((S,), 0),
        ((S, S), 0),
        ((S, S), 1),
        ((S, S), -1),
        ((S, M, S), 2),
        ((), 0),
    )

    for shape, dim in cases:
        yield SampleInput(make(shape), dim=dim)

    # Adds dtype parameter testing when not doing grad testing
    # TODO Reconcile the grad and non-grad samples
    if not requires_grad:
        # NOTE: torch.log_softmax(x, dim, dtype=float) returns a float64 tensor while thunder returns a float32 tensor.
        supported_float_dtypes = {None, torch.float32, torch.float64}
        if thunder.tests.bf16.device_supports_bf16(device):
            supported_float_dtypes.update({torch.bfloat16})

        # Defines a custom comparator for when the output is bfloat16
        # TODO These are very loose tolerances, but observered differences can be up to 0.019 in absolute difference
        #   and .02 in relative difference
        bfloat16_comp = TorchTensorComp(atol=1e-1, rtol=1e-1)

        for (shape, dim), dtype_option in itertools.product(cases, supported_float_dtypes):
            si = SampleInput(make(shape), dim=dim, dtype=dtype_option)

            # Sets the bfloat16 comparator with custom tolerances when the output is requrested
            #   to be in bfloat16
            if dtype_option is torch.bfloat16:
                si.set_comparator(bfloat16_comp)

            yield si


softmax_opinfo = OpInfo(
    ltorch.softmax,
    supports_grad=True,
    sample_input_generator=softmax_sample_generator,
    torch_reference=torch.softmax,
    dtypes=(datatypes.floating,),
)
nn_ops.append(softmax_opinfo)


softmin_opinfo = OpInfo(
    ltorch.softmin,
    supports_grad=True,
    sample_input_generator=softmax_sample_generator,
    torch_reference=torch.nn.functional.softmin,
    dtypes=(datatypes.floating,),
)
nn_ops.append(softmin_opinfo)


log_softmax_opinfo = OpInfo(
    ltorch.log_softmax,
    sample_input_generator=softmax_sample_generator,
    torch_reference=None if LooseVersion(torch.__version__) < "1.13" else torch._refs.log_softmax,
    dtypes=(datatypes.floating,),
    test_directives=(
        # Sets more permissive atol and rtol precisions for bfloat16 than assert_close's defaults
        #   (which are 1.6e-2 and 1e-5)
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
    ),
)
nn_ops.append(log_softmax_opinfo)


def embedding_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    N = 5
    S = 2
    # indices_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    cases = (
        ((S,), (N, S), None, None, 2.0, False, False),
        ((S,), (N, S), 0, None, 2.0, False, False),
        ((S,), (N, S), None, None, 2.0, True, False),
        # nvfuser executor would raise an error when running this test
        # PyTorch works fine
        # RuntimeError: unsupported memory format option Contiguous
        # Because sparse=True, the output tensor is always in sparse format
        # ((S,), (N, S), None, None, 2.0, False, True),
    )

    for indices_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse in cases:
        indices = make(indices_shape, low=0, high=N, dtype=torch.long, requires_grad=False)
        if padding_idx is not None:
            # ensure that padding_idx is present to ensure grad computation is correct
            indices[0] = padding_idx
        yield SampleInput(
            indices,
            make(weight_shape),
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        )


embedding_opinfo = OpInfo(
    ltorch.embedding,
    supports_grad=True,
    sample_input_generator=embedding_sample_generator,
    torch_reference=torch.nn.functional.embedding,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # TODO Investigate these discrepancies -- some dtype x executor configurations seem to be fine
        # See issue "phantom grad's embedding computation is divergent from
        # PyTorch's"
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1, rtol=2)),
            "test_phantom_grad_vs_torch_consistency",
        ),
    ),
)
nn_ops.append(embedding_opinfo)


def gelu_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for sample in elementwise_unary_generator(op, device, dtype, requires_grad, **kwargs):
        a, *_ = sample.args
        yield sample
        yield SampleInput(make(a.shape), approximate="none")
        yield SampleInput(make(a.shape), approximate="tanh")


gelu_opinfo = OpInfo(
    ltorch.gelu,
    # Note that Pytorch does not support complex inputs in gelu.
    dtypes=(datatypes.floating, datatypes.complexfloating),
    supports_grad=True,
    sample_input_generator=gelu_sample_generator,
    torch_reference=torch.nn.functional.gelu,
    test_directives=(
        # PyTorch does not support complex types
        # for both the CPU and CUDA gelu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
        # PyTorch does not support CPU Half gelu
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # CPU Gelu phantom grad tests are very slow in CI
        DecorateInfo(
            pytest.mark.skip,
            "test_phantom_grad_vs_torch_consistency",
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # TODO Investigate the low precision difference -- PyTorch is slightly more accurate at this
        #   computation
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1, rtol=2)),
            dtypes=(datatypes.bfloat16, datatypes.float16),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
    ),
)
nn_ops.append(gelu_opinfo)


def scaled_dot_product_attention_reference_generator(op, device, dtype, requires_grad, **kwargs):
    yield from scaled_dot_product_attention_sample_generator(op, device, dtype, requires_grad, **kwargs)

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 4-dim (multiheaded) causal cases
    n_head = 8
    N, L, S, E, Ev = 2, 2, 64, 64, 64
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    yield SampleInput(q, k, v, None, 0.0, True)

    # non-contiguous with stride 0 cases
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    q_broadcast = torch.as_strided(q, size=q.shape, stride=(0, 0, E, 1))
    k_broadcast = torch.as_strided(k, size=k.shape, stride=(0, 0, E, 1))
    v_broadcast = torch.as_strided(v, size=v.shape, stride=(0, 0, Ev, 1))
    yield SampleInput(q_broadcast, k_broadcast, v_broadcast, None, 0.0, True)


def scaled_dot_product_attention_sample_generator(op, device, dtype, requires_grad, **kwargs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 3-dim causal cases. dropout_p is not parametrized because we don't have a way to control for rng
    for N, (L, S), (E, Ev) in itertools.product((1, 2), ((2, 2), (2, 3)), ((2, 2), (2, 3))):
        q, k, v = make(N, L, E), make(N, S, E), make(N, S, Ev)
        yield SampleInput(q, k, v, dropout_p=0.0, is_causal=True)

    # 4-dim (multiheaded) causal cases
    n_head = 3
    N, L, S, E, Ev = 2, 2, 3, 2, 3
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    yield SampleInput(q, k, v, None, 0.0, True)

    # test the scale factor which was added in torch 2.1
    if LooseVersion(torch.__version__) >= LooseVersion("2.1.0"):
        q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
        yield SampleInput(q, k, v, is_causal=True, scale=0.123)

    # mask cases
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    bool_attn_mask = make((L, S), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    yield SampleInput(q, k, v, attn_mask=bool_attn_mask, is_causal=False)

    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    additive_attn_mask = make((L, S), dtype=q.dtype).tril()
    yield SampleInput(q, k, v, attn_mask=additive_attn_mask, is_causal=False)

    # mask with extra padding: this case will raise if https://github.com/pytorch/pytorch/issues/103749 is fixed
    # when that happens, update the SDPA impl and remove this comment
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    bool_attn_mask = make((L, S), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    bool_attn_mask[-1, :] = False
    yield SampleInput(q, k, v, attn_mask=bool_attn_mask, dropout_p=0.0, is_causal=False)


def scaled_dot_product_attention_error_generator(op, device, **kwargs):
    make = partial(make_tensor, device=device, dtype=torch.float32)

    q, k, v = make(1, 1, 1), make(1, 1, 1), make(1, 1, 1, 1)
    bool_attn_mask = make((1, 1), dtype=torch.bool, low=1, high=1)
    yield (
        SampleInput(q, k, v, attn_mask=bool_attn_mask, is_causal=True),
        ValueError,
        "Explicit attn_mask should not be set when is_causal=True",
    )

    q, k, v = make(1, 1, 1), make(1, 1, 1), make(1, 1, 1, 1)
    attn_mask = make((1, 1), dtype=torch.cfloat)
    yield (
        SampleInput(q, k, v, attn_mask=attn_mask, is_causal=False),
        ValueError,
        "attn_mask.dtype=(.*?) is expected to be of the boolean or a floating type",
    )

    # make q, k, v a non-floating tensor
    var_names = ("query", "key", "value")
    for pos, var_name in enumerate(var_names):
        args = [make(1, 1, 1, dtype=torch.bool) if i == pos else make(1, 1, 1) for i in range(3)]
        yield (SampleInput(*args), ValueError, f"{var_name}.dtype(.*?) is expected to be a floating type")


sdpa_opinfo = OpInfo(
    ltorch.scaled_dot_product_attention,
    sample_input_generator=scaled_dot_product_attention_sample_generator,
    reference_input_generator=scaled_dot_product_attention_reference_generator,
    error_input_generator=scaled_dot_product_attention_error_generator,
    torch_reference=torch.nn.functional.scaled_dot_product_attention,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch CPU doesn't support float16 matmul
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(
                # Numerical issue with bfloat16: skipped since CPU is not a priority
                devices.DeviceType.CPU,
                # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
                devices.DeviceType.CUDA,
            ),
        ),
        # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
        # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention_backward' with arguments from the 'CPU' backend
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("torch",),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # RuntimeError: Only fp32, half & bf16 supported at the moment
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            dtypes=(datatypes.float64,),
        ),
    ),
)
nn_ops.append(sdpa_opinfo)


def grad_scaled_dot_product_attention_sample_generator(op, device, dtype, requires_grad, **kwargs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    from thunder.executors.sdpaex import SpdaBackend

    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-0.5, high=0.5)
    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L4863-L4899
    # * query (batch_size, num_heads, query_seq_len, E)
    # * key (batch_size, num_heads, key_seq_len, E)
    # * value (batch_size, num_heads, key_seq_len, Ev)
    # * attn_mask (batch_size, num_heads, query_seq_len, key_seq_len)

    # NOTE: aten::scaled_dot_product_efficient_attention does not support broadcastable batch sizes.
    n_head = 2
    N = 8
    alignment_factor = 8

    # NOTE If (6 * flash_threads) > L where flash_threads = N * n_head, then the cutlass memory efficient sdpa
    # is prioritized over flash attention sdpa. Reference: See "priority_order" function in
    # aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
    flash_attn_threshold = 6 * N * n_head
    query_seq_length = (flash_attn_threshold - 32, flash_attn_threshold + 32)

    # We pad and slice the attention bias to ensure contiguity for the memory efficient sdpa.
    # Test Case: kv_seq_len > head_dim
    q, k, v = make(1, n_head, 12, 8), make(1, n_head, 14, 8), make(1, n_head, 14, 8)
    bool_attn_mask = make((1, n_head, 12, 14), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    yield SampleInput(q, k, v, bool_attn_mask, dropout_p := 0.0, is_causal := False, scale=0.5)

    # Test Case: kv_seq_len < head_dim
    q, k, v = make(1, n_head, 12, 16), make(1, n_head, 14, 16), make(1, n_head, 14, 16)
    bool_attn_mask = make((1, n_head, 12, 14), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
    yield SampleInput(q, k, v, bool_attn_mask, dropout_p := 0.0, is_causal := False, scale=0.5)

    for L in query_seq_length:
        is_flash_attention = L <= flash_attn_threshold
        S = random.randint(1, 10) * alignment_factor

        # NOTE Flash attention requires the head dim be divisible by 8.
        # If input tensors requires_grad=True and gpu is sm86 or sm89, then head dim must be less than 64.
        if is_flash_attention:
            E = random.randint(1, 8) * alignment_factor
        else:
            E = random.randint(8, 20) * alignment_factor

        # NOTE Flash attention requires Ev == E.
        if is_flash_attention:
            Ev = E
        else:
            Ev = random.randint(1, 10) * alignment_factor

        # 4-dim (multiheaded) causal cases
        q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
        yield SampleInput(q, k, v, attn_mask := None, dropout_p := 0.0, is_causal := False)

        # Test padding case when head size is not a multiple of 8
        if is_flash_attention:
            offset = random.randint(1, 7)
            q, k, v = make(N, n_head, L, E + offset), make(N, n_head, S, E + offset), make(N, n_head, S, Ev + offset)

            # Skip these test cases if the flash attention kernel is not selected.
            # If the flash attention kernel is unavailable, _fused_sdp_choice uses the math reference.
            # When the dtype is not fp64, there is inconsistent results with torch autograd.
            backend = torch._fused_sdp_choice(q, k, v, attn_mask := None, dropout_p := 0.0, is_causal := False)
            if SpdaBackend(backend) == SpdaBackend.FLASH_ATTENTION:
                # fixed scale
                yield SampleInput(q, k, v, attn_mask := None, dropout_p := 0.0, is_causal := False, scale=0.5)

                # default scale
                yield SampleInput(q, k, v, attn_mask := None, dropout_p := 0.0, is_causal := False)

        # Non-contiguous input tensor case
        nq = make(N, n_head, L, E).permute(0, 1, 3, 2)
        nk = make(N, n_head, L, E).permute(0, 1, 3, 2)
        nv = make(N, n_head, L, E).permute(0, 1, 3, 2)
        yield SampleInput(nq, nk, nv, attn_mask := None, dropout_p := 0.0, is_causal := False)

        # Test the scale factor which was added in torch 2.1
        if LooseVersion(torch.__version__) >= LooseVersion("2.1.0"):
            q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
            yield SampleInput(q, k, v, attn_mask := None, dropout_p := 0.0, is_causal := False, scale=0.125)

        # NOTE Flash attention sdpa does not support attn_mask argument; These cases always use memory efficient sdpa.
        q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
        bool_attn_mask = make((N, n_head, L, S), dtype=torch.bool, low=1, high=1, requires_grad=False).tril()
        yield SampleInput(q, k, v, attn_mask := bool_attn_mask, is_causal=False)

        q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
        additive_attn_mask = make((N, n_head, L, S), dtype=q.dtype).tril()
        yield SampleInput(q, k, v, attn_mask := additive_attn_mask, is_causal=False)


# NOTE When calculating the gradient in the backwards pass, the torch executor calls fused sdpa functions.
# This opinfo test creates inputs that are valid for those functions.
grad_sdpa_opinfo = OpInfo(
    ltorch.scaled_dot_product_attention,
    name="grad_forward_scaled_dot_product_attention",
    sample_input_generator=grad_scaled_dot_product_attention_sample_generator,
    torch_reference=torch.nn.functional.scaled_dot_product_attention,
    # RuntimeError: Only fp32, half & bf16 supported at the moment
    dtypes=(
        datatypes.float32,
        datatypes.float16,
        datatypes.bfloat16,
    ),
    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention_backward' with arguments from the 'CPU' backend
    devicetypes=(devices.DeviceType.CUDA,),
)
nn_ops.append(grad_sdpa_opinfo)


# TODO When more bwd support is added merge the logic (but not all the cases) for sample generation
def cross_entropy_reference_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, target_shape
    shapes = (
        ((2, 16), (2,)),
        ((7, 18), (7,)),
        ((7, 18), (7, 18)),
        ((3, 4, 2, 3), (3, 4, 2, 3)),
        ((3, 4, 2, 3), (3, 2, 3)),
        ((5,), ()),
        ((3, 4, 0), (3, 0)),
        ((3, 4, 0), (3, 4, 0)),
        ((256, 1024), (256,)),
        ((256, 32768), (256,)),
        ((256, 1024), (256, 1024)),
        ((256, 32768), (256, 32768)),
    )

    weight_options = (True, False)
    reduction_options = ("none", "mean", "sum")
    label_smoothing_options = (0.0, 0.5)
    ignore_index_options = (-1, 3)

    for shape, weight_flag, reduction_str, label_smoothing, ignore_index in itertools.product(
        shapes, weight_options, reduction_options, label_smoothing_options, ignore_index_options
    ):
        input_shape, target_shape = shape
        probability_target = input_shape == target_shape
        # ignore_index can't be supplied with probablity target
        if probability_target and ignore_index >= 0:
            continue
        C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
        yield SampleInput(
            make(shape[0]),
            (
                make(shape[1], low=0, high=C, dtype=torch.long, requires_grad=False)
                if not probability_target
                else make(shape[1], low=0.0, high=1.0, requires_grad=True)
            ),
            weight=make(C, requires_grad=False) if weight_flag else None,
            ignore_index=ignore_index,
            reduction=reduction_str,
            label_smoothing=label_smoothing,
        )


def cross_entropy_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, target_shape
    shapes = (
        ((2, 16), (2,)),
        ((7, 18), (7,)),
        ((7, 18), (7, 18)),
        ((3, 4, 2, 3), (3, 4, 2, 3)),
        ((3, 4, 2, 3), (3, 2, 3)),
        ((5,), ()),
        ((3, 4, 0), (3, 0)),
        ((3, 4, 0), (3, 4, 0)),
    )

    weight_options = (False, True)
    reduction_options = ("none", "mean", "sum")
    label_smoothing_options = (0.0, 0.5)
    ignore_index_options = (-1, 3)

    for shape, weight_flag, reduction_str, label_smoothing, ignore_index in itertools.product(
        shapes, weight_options, reduction_options, label_smoothing_options, ignore_index_options
    ):
        # NOTE According to pytorch/pytorch#64572, nll_loss should return NaN when reduction = "mean"
        # and the whole target is equal to ignore_index. However, if the inputs are cuda tensors, PyTorch returns 0.
        # Skip this case because we are consistent across devices.
        if torch.device(device).type == "cuda" and reduction_str == "mean" and ignore_index > 0:
            continue

        input_shape, target_shape = shape
        probability_target = input_shape == target_shape
        # ignore_index can't be supplied with probablity target
        if probability_target and ignore_index >= 0:
            continue
        C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
        yield SampleInput(
            make(shape[0]),
            (
                make(shape[1], low=0, high=C, dtype=torch.long, requires_grad=False)
                if not probability_target
                else make(shape[1], low=0.0, high=1.0, requires_grad=True)
            ),
            weight=make(C, low=1.0, high=2.0, requires_grad=False) if weight_flag else None,
            ignore_index=ignore_index,
            reduction=reduction_str,
            label_smoothing=label_smoothing,
        )


def cross_entropy_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    input_shape = (7, 18)
    target_shape = (7,)
    C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
    valid_input = make(input_shape)
    valid_target = make(target_shape, low=0, high=C, dtype=torch.long, requires_grad=False)

    # unexpected reduction string argument
    yield (
        SampleInput(valid_input, valid_target, reduction="foo"),
        ValueError,
        'Expected reduction string to be "none", "sum", or "mean", but it is (.*?).',
    )

    # target tensor is not integer dtype
    float_target = make(target_shape, low=0, high=C, dtype=torch.float, requires_grad=False)
    yield (
        SampleInput(valid_input, float_target),
        RuntimeError,
        "Expected target to be a tensor with an integer dtype, but it has dtype (.*?).",
    )

    # input tensor has 0 dimensions
    scalar_input = make(scalar_shape := ())
    yield (
        SampleInput(scalar_input, valid_target),
        RuntimeError,
        f"Expected the input tensor to have more than 1 dimension, but it has {scalar_input.ndim} dimensions.",
    )

    # weight tensor has more than 1 dimension
    multiple_dim_weight = make((C, 10), requires_grad=False)
    yield (
        SampleInput(valid_input, valid_target, weight=multiple_dim_weight),
        RuntimeError,
        f"Expected a 1D tensor with {C} elements for weight argument, \
            but found a tensor with {multiple_dim_weight.ndim} dimensions and {multiple_dim_weight.shape[0]} elements.",
    )

    # weight tensor numel != C
    incorrect_numel_weight = make((C + 10,), requires_grad=False)
    yield (
        SampleInput(valid_input, valid_target, weight=incorrect_numel_weight),
        RuntimeError,
        f"Expected a 1D tensor with {C} elements for weight argument, \
            but found a tensor with {incorrect_numel_weight.ndim} dimensions and {incorrect_numel_weight.shape[0]} elements.",
    )

    # label_smoothing out-of-bounds
    out_of_bounds_label_smoothing = 1.5
    yield (
        SampleInput(valid_input, valid_target, label_smoothing=out_of_bounds_label_smoothing),
        RuntimeError,
        r"Expected label_smoothing to be in \[0, 1\] range but got 1.5.",
    )

    # target tensor is not integer dtype
    float_target = make(target_shape, low=0, high=C, dtype=torch.float, requires_grad=False)
    yield (
        SampleInput(valid_input, float_target),
        RuntimeError,
        "Expected target to be a tensor with an integer dtype, but it has dtype (.*?).",
    )

    # input ndims != (target ndims + 1)
    extra_dim_input = make(input_shape + (10,))
    yield (
        SampleInput(extra_dim_input, valid_target),
        RuntimeError,
        "Expected the input tensor to have (.*?) dimensions, but it has (.*?) dimensions.",
    )

    # target shape is input shape except class dimension
    incorrect_batch_target = make((10,), low=0, high=C, dtype=torch.long, requires_grad=False)
    yield (
        SampleInput(valid_input, incorrect_batch_target),
        RuntimeError,
        "Expected the target tensor to have the same shape as the input tensor except for the class dimension \
            (.*?), but it has shape (.*?).",
    )

    integer_prob_target = make(input_shape, low=0, high=C, dtype=torch.long, requires_grad=False)
    yield (
        SampleInput(valid_input, integer_prob_target),
        RuntimeError,
        "Expected the target to have float dtype when target contains class probabilities \
                but it is (.*?).",
    )

    valid_prob_target = make(input_shape, low=0.0, high=1.0, dtype=torch.float, requires_grad=False)
    yield (
        SampleInput(valid_input, valid_prob_target, ignore_index=5),
        RuntimeError,
        "ignore_index argument is not supported when target contains class probabilities.",
    )


cross_entropy_opinfo = OpInfo(
    ltorch.cross_entropy,
    sample_input_generator=cross_entropy_sample_generator,
    reference_input_generator=cross_entropy_reference_generator,
    error_input_generator=cross_entropy_error_generator,
    torch_reference=torch.nn.functional.cross_entropy,
    dtypes=(datatypes.floating,),
    test_directives=(
        # TODO Investigate why CPU torch executor tests fail in CI (but not locally)
        DecorateInfo(
            pytest.mark.skip,
            devicetypes=(devices.DeviceType.CPU,),
            executors=("torch",),
        ),
        # Grad tests are slightly inaccurate in lower precision floating-point types
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-2, rtol=1e-2)),
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
        # TODO FIXME -- These tests are hitting an odd issue where real torch tensors are being passed to nll_loss
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
nn_ops.append(cross_entropy_opinfo)


def nll_loss_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, target_shape
    shapes = (
        ((7, 18), (7,)),
        ((3, 4, 2, 3), (3, 2, 3)),
        ((5,), ()),
        ((3, 4, 0), (3, 0)),
    )

    weight_options = (True, False)
    reduction_options = ("none", "mean", "sum")
    ignore_index_options = (-100, 3)

    # NOTE: The size_average and reduce parameters are not tested because they are deprecated.
    for shape, weight_flag, reduction_str, ignore_index in itertools.product(
        shapes, weight_options, reduction_options, ignore_index_options
    ):
        # NOTE According to pytorch/pytorch#64572, nll_loss should return NaN when reduction = "mean"
        # and the whole target is equal to ignore_index. However, if the inputs are cuda tensors, PyTorch returns 0.
        # Skip this case because we are consistent across devices.
        if torch.device(device).type == "cuda" and reduction_str == "mean" and ignore_index > 0:
            continue

        input_shape, target_shape = shape
        C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]

        # Input is expected to be log-probs.
        # We provide a which is log-stochastic in channel dims.
        a = make(input_shape, requires_grad=False)
        class_dim = 1 if len(input_shape) >= 2 else 0
        for dim in range(class_dim + 1, a.ndim):
            a = a.log_softmax(dim=dim)
        a.requires_grad_(requires_grad)

        target = make(target_shape, low=0, high=C, dtype=torch.long, requires_grad=False)
        # NOTE pytorch `cuda` behavior for scalar target == ignore_index diverges from the cpu behavior.
        # def foo(device):
        #     ignore_index = -100
        #     logits = torch.randn(5, device=device, dtype=torch.float32, requires_grad=False)
        #     weight = torch.randn(5, device=device, dtype=torch.float32, requires_grad=False)
        #     labels = torch.tensor(ignore_index, device=device)
        #     print(torch.nn.functional.cross_entropy(logits, labels, weight, ignore_index=ignore_index))
        # foo("cpu")  # tensor(nan)
        # foo("cuda") # tensor(0., device='cuda:0')
        if target.ndim != 0:
            # sprinkle ignore_index in the target, verify correctness, see issue 1744.
            target = torch.where(make(target_shape, low=0.0, high=1.0, requires_grad=False) > 0.3, target, ignore_index)

        yield SampleInput(
            a,
            target=target,
            weight=make(C, low=1.0, high=2.0, requires_grad=False) if weight_flag else None,
            ignore_index=ignore_index,
            reduction=reduction_str,
        )

    # Test empty input and target tensor short-circuit
    for reduction_str, ignore_index in itertools.product(reduction_options, ignore_index_options):
        yield SampleInput(
            empty_input_tensor := torch.tensor([], device=device, dtype=dtype),
            empty_target_tensor := torch.tensor([], device=device, dtype=torch.long),
            ignore_index=ignore_index,
            reduction=reduction_str,
        )


def nll_loss_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    input_shape = (7, 18)
    target_shape = (7,)
    C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
    valid_input = make(input_shape)
    valid_target = make(target_shape, low=0, high=C, dtype=torch.long, requires_grad=False)

    # unexpected reduction string argument
    yield (
        SampleInput(valid_input, valid_target, reduction="foo"),
        ValueError,
        'Expected reduction string to be "none", "sum", or "mean", but it is (.*?).',
    )

    # target tensor is not integer dtype
    float_target = make(target_shape, low=0, high=C, dtype=torch.float, requires_grad=False)
    yield (
        SampleInput(valid_input, float_target),
        RuntimeError,
        "Expected target to be a tensor with an integer dtype, but it has dtype (.*?).",
    )

    # input tensor has 0 dimensions
    scalar_input = make(scalar_shape := ())
    yield (
        SampleInput(scalar_input, valid_target),
        RuntimeError,
        f"Expected the input tensor to have more than 1 dimension, but it has {scalar_input.ndim} dimensions.",
    )

    # input ndims != (target ndims + 1)
    extra_dim_input = make(input_shape + (10,))
    yield (
        SampleInput(extra_dim_input, valid_target),
        RuntimeError,
        "Expected the input tensor to have (.*?) dimensions, but it has (.*?) dimensions.",
    )

    # target shape is input shape except class dimension
    incorrect_batch_target = make((10,), low=0, high=C, dtype=torch.long, requires_grad=False)
    yield (
        SampleInput(valid_input, incorrect_batch_target),
        RuntimeError,
        "Expected the target tensor to have the same shape as the input tensor except for the class dimension \
            (.*?), but it has shape (.*?).",
    )

    # weight tensor has more than 1 dimension
    multiple_dim_weight = make((C, 10), requires_grad=False)
    yield (
        SampleInput(valid_input, valid_target, weight=multiple_dim_weight),
        RuntimeError,
        f"Expected a 1D tensor with {C} elements for weight argument, \
            but found a tensor with {multiple_dim_weight.ndim} dimensions and {multiple_dim_weight.shape[0]} elements.",
    )

    # weight tensor numel != C
    incorrect_numel_weight = make((C + 10,), requires_grad=False)
    yield (
        SampleInput(valid_input, valid_target, weight=incorrect_numel_weight),
        RuntimeError,
        f"Expected a 1D tensor with {C} elements for weight argument, \
            but found a tensor with {incorrect_numel_weight.ndim} dimensions and {incorrect_numel_weight.shape[0]} elements.",
    )


nll_loss_opinfo = OpInfo(
    ltorch.nll_loss,
    sample_input_generator=nll_loss_sample_generator,
    error_input_generator=nll_loss_error_generator,
    torch_reference=torch.nn.functional.nll_loss,
    dtypes=(datatypes.floating,),
    test_directives=(
        # take_along_axis is disabled with nvfuser, which this operator relies on.
        DecorateInfo(
            pytest.mark.skip,
            executors=("nvfuser",),
        ),
        # FP16: RuntimeError: "nll_loss_out_frame" not implemented for 'Half'
        # BF16: AssertionError: Scalars are not close!
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(
                datatypes.float16,
                datatypes.bfloat16,
            ),
        ),
        # TODO FIXME -- These tests are hitting an odd issue where real torch tensors are being passed to nll_loss
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
        # NOTE PyTorch returns NaN if ignore_index == target_index and reduction='mean'
        DecorateInfo(
            custom_comparator(partial(assert_close, equal_nan=True)),
        ),
    ),
)
nn_ops.append(nll_loss_opinfo)


def mse_loss_sample_generator(op, device, dtype, requires_grad, **kwards):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, target_shape
    shapes = (
        ((2, 16), (2, 16)),
        ((7, 18), (7, 18)),
        ((3, 4, 2, 3), (3, 4, 2, 3)),
        ((3, 4, 2, 3), (4, 1, 3)),
        ((2, 3, 1), (3, 1)),
    )

    reduction_options = ("none", "mean", "sum")

    for shape, reduction_str in itertools.product(shapes, reduction_options):
        input_shape, target_shape = shape

        C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
        yield SampleInput(
            make(input_shape, low=0.0, high=1.0, dtype=dtype, requires_grad=True),
            make(target_shape, low=0.0, high=1.0, dtype=dtype, requires_grad=True),
            reduction=reduction_str,
        )


mse_loss_opinfo = OpInfo(
    ltorch.mse_loss,
    sample_input_generator=mse_loss_sample_generator,
    torch_reference=torch.nn.functional.mse_loss,
    dtypes=(datatypes.floating,),
    test_directives=(
        # NOTE: PyTorch does not support bf16 mse_loss
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # NOTE: currently, mse_loss is encountering the following errors
        # RuntimeError: "mse_cpu" not implemented for 'BFloat16'
        # RuntimeError: "mse_backward_cpu_out" not implemented for 'Half'
        DecorateInfo(
            pytest.mark.skip,
            "test_phantom_grad_vs_torch_consistency",
            dtypes=(
                datatypes.bfloat16,
                datatypes.float16,
            ),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Sets more permissive atol and rtol precisions for float16 than assert_close's defaults
        #   (which are 1e-3 and 1e-5)
        DecorateInfo(
            custom_comparator(partial(assert_close, atol=1e-3, rtol=1e-2)),
            executors=("nvfuser",),
            dtypes=(datatypes.float16,),
        ),
    ),
)

nn_ops.append(mse_loss_opinfo)


def interpolate_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Not that much variability in the batch and channels dims
    # since they are never modified.
    # NOTE: we had to cut it down.
    # Before:
    # batch = ((1,), (2,))
    # After:
    batch = ((1,),)
    channels = ((2,),)

    # For shape in dim_options we try size=dim_options[shape]
    dim_options = {
        # Nice prime number. We test co-prime with 5 sizes.
        # NOTE: if the tests take too much time, reducing
        # the tuple length is the best way to cut it down.
        # NOTE: we had to cut it down.
        # Before:
        # 5: (2, 3, 4, 7, 8, 9),
        # After:
        5: (3, 7),
        # Nice even number. We test *0.5 *1, and *2.
        # NOTE: we had to cut it down.
        # Before:
        # 6: (3, 6, 12),
        # After:
    }
    # Testing 3D-5D inputs only since they are the ones that PyTorch supports.
    # NOTE: we had to cut it down.
    # Before:
    # n_spatial_dims = (1, 2, 3)
    # After:
    n_spatial_dims = (1, 3)

    # All possible combinations to test that dependencies between dimensions are captured correctly.
    # Since specifying size will call the scale_factor path, we do not explicitly test scale_factor
    # in the loop below.
    for b, c, l, dim in itertools.product(batch, channels, dim_options, n_spatial_dims):
        for size in itertools.product(dim_options[l], repeat=dim):
            spatial_dims = (l,) * dim
            a_shape = b + c + spatial_dims

            yield SampleInput(make(a_shape), size=size)
            yield SampleInput(make(a_shape), size=size, mode="nearest-exact")

    # mode = "bilinear" supports only 4D inputs in PyTorch, so 2 spatial dimensions
    n_spatial_dims_bilinear = (2,)
    for b, c, l, dim in itertools.product(batch, channels, dim_options, n_spatial_dims_bilinear):
        for size in itertools.product(dim_options[l], repeat=dim):
            spatial_dims = (l,) * dim
            a_shape = b + c + spatial_dims

            yield SampleInput(make(a_shape), size=size, mode="bilinear")

    # Test scale/scale_factor passed as a scalar
    yield SampleInput(make(1, 1, 5, 5), scale_factor=0.5)
    yield SampleInput(make(1, 1, 5, 5), size=10)

    # Let's try some crazy scale_factor
    yield SampleInput(make(1, 1, 5, 5), scale_factor=(1.37, 0.26))
    yield SampleInput(make(1, 1, 5, 5), scale_factor=(0.26, 1.37))


def interpolate_error_generator(op, device, dtype=torch.float32, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype)

    yield (SampleInput(make(1, 1), scale_factor=2.0), RuntimeError, "Expected (.*?)ndim(.*?) >= 3")
    yield (SampleInput(make(1, 1, 0), scale_factor=2.0), RuntimeError, "Expected (.*?)numel(.*?) to be greater than 0")

    yield (SampleInput(make(1, 1, 1)), RuntimeError, "Only one of `size` or `scale_factor` has to be specified")
    yield (
        SampleInput(make(1, 1, 1), size=(2,), scale_factor=2.0),
        RuntimeError,
        "Only one of `size` or `scale_factor` has to be specified",
    )

    yield (SampleInput(make(1, 1, 1), size=0), RuntimeError, "size(.*?) is expected to be greater than zero")
    yield (
        SampleInput(make(1, 1, 1), size=2.0),
        RuntimeError,
        "size(.*?) is expected to be a greater than zero integer",
    )
    yield (
        SampleInput(make(1, 1, 1), size=(2, 2)),
        RuntimeError,
        "size(.*?) is expected to be (.*?) a sequence (.*?) of length 1",
    )
    yield (
        SampleInput(make(1, 1, 1, 1), size=(2.0, 2)),
        RuntimeError,
        "size(.*?) is expected to be (.*?) a sequence of strictly positive integers",
    )

    yield (
        SampleInput(make(1, 1, 1), scale_factor=0.0),
        RuntimeError,
        "scale_factor(.*?) is expected to be strictly positive",
    )
    yield (
        SampleInput(make(1, 1, 1), scale_factor=2),
        RuntimeError,
        "scale_factor(.*?) is expected to be a strictly positive floating point number",
    )
    yield (
        SampleInput(make(1, 1, 1), scale_factor=(2.0, 2.0)),
        RuntimeError,
        "scale_factor(.*?) is expected to be (.*?) a sequence (.*?) of length 1",
    )
    yield (
        SampleInput(make(1, 1, 1, 1), scale_factor=(2.0, 2)),
        RuntimeError,
        "scale_factor(.*?) is expected to be (.*?) a sequence of strictly positive floating point numbers",
    )
    yield (
        SampleInput(make(1, 1, 1, 1), mode="trilinear"),
        RuntimeError,
        "only modes 'nearest', 'nearest-exact' and 'bilinear' are supported at the moment, but got mode=(.*?)",
    )
    yield (
        SampleInput(make(1, 1, 5), scale_factor=2.0, mode="bilinear"),
        RuntimeError,
        "bilinear interpolation supports exactly two spatial dims, got 1",
    )
    yield (
        SampleInput(make(1, 1, 5, 5, 5), scale_factor=2.0, mode="bilinear"),
        RuntimeError,
        "bilinear interpolation supports exactly two spatial dims, got 3",
    )


interpolate_opinfo = OpInfo(
    ltorch.interpolate,
    sample_input_generator=interpolate_sample_generator,
    error_input_generator=interpolate_error_generator,
    torch_reference=torch.nn.functional.interpolate,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch does not support CPU Half upsample used in interpolate
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # This should be fixed now; TODO re-enable and test
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
        ),
        DecorateInfo(
            pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2452"),
            "test_core_vs_torch_consistency",
            executors=("nvfuser",),
        ),
    ),
)
nn_ops.append(interpolate_opinfo)


opinfos.extend(nn_ops)


# Ops related to Probability Distributions.
prob_distr_ops = []


# multinomial testing is currently disabled due to issue "randomness: enable
# PyTorch generators for operations like multinomial"
# def multinomial_sample_generator(op, device, dtype, requires_grad, **kwargs):
#     make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

#     shapes = [
#         (10),
#         (10, 10),
#     ]
#     num_samples = (1, 3, 5)
#     replacement = (True, False)
#     seed = 13

#     for shape, ns, r in itertools.product(shapes, num_samples, replacement):
#         weights = make(shape).abs()
#         gen = torch.Generator(device).manual_seed(seed)
#         yield SampleInput(weights, ns, r, generator=gen)


# def torch_multinomial_like(
#     a: torch.Tensor,
#     num_samples: int,
#     replacement: bool,
#     *,
#     generator: torch.Generator,
# ):
#     return prims.multinomial(a, num_samples, replacement, generator.initial_seed())


# multinomial_prim_opinfo = OpInfo(
#     torch_multinomial_like,
#     name="multinomial_prim",
#     supports_grad=False,
#     sample_input_generator=multinomial_sample_generator,
#     torch_reference=torch.multinomial,
#     dtypes=(datatypes.floating,),
#     test_directives=(
#         # PyTorch does not support CUDA BFloat16 multinomial
#         DecorateInfo(
#             pytest.mark.xfail,
#             "test_core_vs_torch_consistency",
#             dtypes=(datatypes.bfloat16,),
#             devicetypes=(devices.DeviceType.CUDA,),
#         ),
#         # PyTorch does not support CPU Half multinomial
#         DecorateInfo(
#             pytest.mark.xfail,
#             "test_core_vs_torch_consistency",
#             dtypes=(datatypes.float16,),
#             devicetypes=(devices.DeviceType.CPU,),
#         ),
#     ),
# )
# prob_distr_ops.append(multinomial_prim_opinfo)


opinfos.extend(prob_distr_ops)


# Memory access ops
memory_access_ops = []


def item_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Different flavors of 1-numel arrays with dims 0-4
    for d in range(5):
        yield SampleInput(make((1,) * d))


item_opinfo = OpInfo(
    prims.item,
    sample_input_generator=item_sample_generator,
    torch_reference=torch.Tensor.item,
)
memory_access_ops.append(item_opinfo)


opinfos.extend(memory_access_ops)
