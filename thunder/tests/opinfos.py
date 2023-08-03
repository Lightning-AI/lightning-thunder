import itertools
import math
import operator
from collections import namedtuple
from functools import partial, wraps
from numbers import Number
from typing import Union, Callable, Optional
from collections.abc import Sequence

import numpy as np
import pytest

# TODO: make this import conditional on Torch being available and querying if should test with torch
import torch
from looseversion import LooseVersion

import thunder.core.devices as devices
import thunder.core.dtypes as datatypes
import thunder.clang as clang
import thunder.core.prims as prims
import thunder.torch as ltorch
from thunder.core.pytree import tree_map
from thunder.tests.framework import _all_devicetypes, JAX_AVAILABLE
from thunder.tests.make_tensor import make_tensor
from thunder.core.symbol import Symbol

import thunder.executors as executors

#
# Helpful constants and utility functions
#

# TODO This is a hack to support comparisons like nvfuser_version > LooseVersion("0.0.3") even when
#   nvfuser_version is None. A better approach would probably be to create a helper function
#   nvfuser_atleast(X) which handles nvfuser_version being None properly
nvfuser_version = executors.nvfuser_version()
nvfuser_version = nvfuser_version if nvfuser_version is not None else LooseVersion("0.0.0")


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


def push_away_from_singularities(x, singularity_fn, eps):
    """This function takes a tensor and moves individual values away
    from singularities in `eps` increments, until they are further than
    `eps` away from them. The `singularity_fn`  returns the (signed)
    distance from `x` to the nearest singularity."""
    x_dist = singularity_fn(x)
    x_ = torch.where((x_dist > 0) & (x_dist < eps), x + eps, x)
    return torch.where((x_dist < 0) & (x_dist > -eps), x - eps, x_)


def make_number(**kwargs):
    v = make_tensor((), device="cpu", **kwargs).item()
    return v


# Returns a noncontiguous (tensor with the same shape and values as t
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


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"[SampleInput args={self.args} kwargs={self.kwargs}]"

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        args, kwargs = tree_map(to_noncontiguous, self.args), tree_map(to_noncontiguous, self.kwargs)
        return SampleInput(*args, **kwargs)

    def jax(self):
        def to_jax(t):
            if isinstance(t, torch.Tensor):
                return jnp.array(t.cpu().numpy())
            if isinstance(t, torch.dtype):
                return _torch_to_jax_dtype_map[t]

            return t

        args, kwargs = tree_map(to_jax, self.args), tree_map(to_jax, self.kwargs)
        return SampleInput(*args, **kwargs)

    def thunder(self):
        def to_thunder(t):
            if isinstance(t, torch.dtype):
                return ltorch.to_thunder_dtype(t)
            return t

        args, kwargs = tree_map(to_thunder, self.args), tree_map(to_thunder, self.kwargs)
        return SampleInput(*args, **kwargs)


# TODO: add executor
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
        decorator,
        test_template_name=None,
        *,
        executors=None,
        devicetypes: Optional[Sequence[devices.DeviceType]] = None,
        dtypes=None,
        active_if=True,
    ):
        self.decorator = decorator
        self.test_template_name = test_template_name
        self.executors = executors
        self.devicetypes = devicetypes

        if devicetypes is not None:
            for x in devicetypes:
                assert isinstance(
                    x, devices.DeviceType
                ), f"Found non-devicetype {x} when initializing a DecorateInfo's devicetypes"

        self.dtypes = None if dtypes is None else datatypes.resolve_dtypes(dtypes)
        self.active_if = active_if

    def is_active(
        self, test_template_name, executor, device_or_devicetype: Union[str, devices.Device, devices.DeviceType], dtype
    ):
        # Acquires devicetype
        devicetype_: devices.DeviceType
        if isinstance(device_or_devicetype, str):
            devicetype_ = devices.device_from_string(device_or_devicetype).devicetype
        elif isinstance(device_or_devicetype, devices.Device):
            devicetype_ = device_or_devicetype.devicetype
        else:
            assert False, f"Unknown device or devicetype {device_or_devicetype}, expect a string, device, or devicetype"

        executor_match = self.executors is None or executor.name in self.executors
        test_name_match = self.test_template_name is None or self.test_template_name == test_template_name
        devicetype_match = self.devicetypes is None or devicetype_ in self.devicetypes
        dtype_match = self.dtypes is None or dtype in self.dtypes

        return self.active_if and executor_match and test_name_match and devicetype_match and dtype_match


Domain = namedtuple("Domain", "low high")
opinfos = []


# TODO: require use of generic Thunder dtypes (once they exist)
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op: Union[Symbol, Callable],
        *,
        name: Optional[str] = None,
        devicetypes: Optional[Sequence[devices.DeviceType]] = None,
        dtypes=None,
        sample_input_generator,
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
        self.sample_input_generator = sample_input_generator
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

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    # TODO Maybe allow sample input generation not using torch?
    # NOTE Today all sample inputs are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def sample_inputs(self, device: devices.Device, dtype: datatypes.dtype, *, requires_grad: bool = False, **kwargs):
        torch_dtype = ltorch.to_torch_dtype(dtype)
        torch_device = str(device)
        return self.sample_input_generator(self, torch_device, torch_dtype, requires_grad, **kwargs)

    def error_inputs(self, device: devices.Device, **kwargs):
        torch_device = str(device)
        return self.error_input_generator(self, torch_device, **kwargs)

    # NOTE Today all benchmarks are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def benchmarks(self, device: devices.Device, dtype: datatypes.dtype, *, requires_grad: bool = False, **kwargs):
        torch_dtype = ltorch.to_torch_dtype(dtype)
        torch_device = str(device)
        return self.benchmark_generator(self, torch_device, dtype, requires_grad, **kwargs)

    def devicetypes(self):
        return set(self._devicetypes)

    # TODO Add per-device dtype support
    def dtypes(self, devicetype: devices.DeviceType = None):
        if devicetype is not None:
            raise NotImplementedError

        return datatypes.resolve_dtypes(self._dtypes)

    def test_decorators(self, test_name, executor, devicetype: devices.DeviceType, dtype: datatypes.dtype):
        return [d.decorator for d in self.test_directives if d.is_active(test_name, executor, devicetype, dtype)]


#
# Elementwise Unary OpInfos
#

# TODOA Create elementwise unary OpInfo subclass and maybe auto add to list
elementwise_unary_ops = []


# TODO Add small value, large value, and extremal-valued samples
def elementwise_unary_generator(
    op, device: torch.device, dtype: torch.dtype, requires_grad: bool, *, supports_numbers: bool = True, **kwargs
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
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)

        return op(torch.tensor(x)).item()

    return _fn


# NOTE: slightly different from generic _elementwise_unary_torch helper
#   because this returns the input when given an unsigned type
@wraps(torch.abs)
def _abs_torch(x: Union[torch.Tensor, Number]):
    if isinstance(x, torch.Tensor):
        if datatypes.is_unsigned_dtype(ltorch.to_thunder_dtype(x.dtype)):
            return x
        return torch.abs(x)

    # Handles numbers
    assert isinstance(x, Number)
    if datatypes.is_unsigned_dtype(type(x)):
        return x
    return torch.abs(torch.tensor(x)).item()


abs_opinfo = ElementwiseUnaryOpInfo(
    ltorch.abs,
    torch_reference=_abs_torch,
    test_directives=(
        # complex32 cpu abs is sometimes flaky in CI
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
    ),
)

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
            executors=("nvFuser",),
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
            executors=("nvFuser",),
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
            executors=("nvFuser",),
            active_if=nvfuser_version < LooseVersion("0.0.3"),
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

erf_opinfo = OpInfo(
    clang.erf,
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
            executors=("TorchEx"),
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
            executors=("nvFuser",),
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
            executors=("nvFuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
    ),
)
elementwise_unary_ops.append(erfinv_opinfo)

exp_opinfo = OpInfo(
    clang.exp,
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
            executors=("nvFuser",),
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

rsqrt_opinfo = OpInfo(
    clang.rsqrt,
    domain=(0, math.inf),
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
        # see https://github.com/csarofeen/pytorch/issues/2367
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

silu_opinfo = OpInfo(
    clang.silu,
    dtypes=(datatypes.floating,),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.silu),
    test_directives=(
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
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

sigmoid_opinfo = OpInfo(
    clang.sigmoid,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sigmoid),
    test_directives=(
        # torch.sigmoid is not implemented for CPU float16 or complex32
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("TorchEx",),
            devicetypes=(devices.DeviceType.CPU,),
            dtypes=(datatypes.float16, datatypes.complex32),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("TorchEx",),
            devicetypes=(devices.DeviceType.CUDA,),
            dtypes=(
                # reciprocal_cuda for ComplexHalf is not implemented in torch
                datatypes.complex32,
            ),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            executors=("TorchEx",),
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
            executors=("nvFuser",),
            dtypes=(datatypes.complex64,),
        ),
    ),
)
elementwise_unary_ops.append(sigmoid_opinfo)

sign_opinfo = OpInfo(
    clang.sign,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.sgn),
    test_directives=(
        # TODO Need to add nvFuser specific support for complex sign
        # https://github.com/csarofeen/pytorch/issues/2492
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvFuser",),
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
    clang.silu,
    dtypes=(datatypes.floating,),
    sample_input_generator=partial(elementwise_unary_generator, supports_numbers=False),
    torch_reference=_elementwise_unary_torch(torch.nn.functional.silu),
    test_directives=(
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
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
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
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
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.tanh),
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
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
    domain=(-1.0 + eps, math.inf),
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
    sample_input_generator=partial(elementwise_unary_generator, exclude_zero=True),
    torch_reference=_elementwise_unary_torch(torch.log),
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
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
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
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
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvFuser",),
            dtypes=(datatypes.complexfloating,),
        ),
        # NOTE: Torch gives wrong result: https://github.com/pytorch/pytorch/issues/94333
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
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
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
    dtypes=set(datatypes.all_dtypes) - set(datatypes.boolean_dtypes),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=_elementwise_unary_torch(torch.neg),
)
elementwise_unary_ops.append(neg_opinfo)

ndtri_opinfo = OpInfo(
    clang.ndtri,
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
        # TODO: nvFuser needs to return copy for integer dtypes.
        # https://github.com/csarofeen/pytorch/issues/2499
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvFuser",),
            dtypes=(datatypes.int32, datatypes.int64),
        ),
    ),
)
elementwise_unary_ops.append(trunc_opinfo)


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


# TODO Extend this generator
def elementwise_binary_generator(op, device, dtype, requires_grad, *, no_rhs_numbers: bool = False, **kwargs):
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
        yield SampleInput(c, d)


# TODO: update dtypes with Thunder dtypes (when they exist)
add_opinfo = OpInfo(
    clang.add,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.add,
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2549
        DecorateInfo(
            pytest.mark.skip,
            "test_jvp_correctness",
            executors=("nvFuser",),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
elementwise_binary_ops.append(add_opinfo)

# NOTE: nvFuser does not currently support uint8, int8, or int16
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
)
elementwise_binary_ops.append(copysign_opinfo)

# For grad test stability it's better to use wider range of values
elementwise_comparison_generator = partial(elementwise_binary_generator, low=-1000, high=1000)

eq_opinfo = OpInfo(
    clang.eq,
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.eq,
    test_directives=(
        # There's a problem of reducing a tensor produced by full op
        # See https://github.com/NVIDIA/Fuser/issues/132
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
elementwise_binary_ops.append(eq_opinfo)

# NOTE floor division is not defined for complex numbers
floor_divide_opinfo = OpInfo(
    clang.floor_divide,
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=partial(elementwise_binary_generator, exclude_zero=True),
    torch_reference=torch.floor_divide,
    test_directives=(
        # TODO FIXME
        # nvFuser's division operation is true division, so the dtypes are wrong
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.exact,),
            executors=("nvFuser,"),
        ),
        # TODO FIXME Connect to nvFuser's trunc division correctly
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float32,),
            executors=("nvFuser,"),
        ),
        # TODO FIXME AssertionError: Tensor-likes are not close!
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
            executors=("TorchEx,"),
        ),
        # PyTorch doesn't support boolean floor division
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
    ),
)
elementwise_binary_ops.append(floor_divide_opinfo)

fmod_opinfo = OpInfo(
    clang.fmod,
    sample_input_generator=partial(elementwise_binary_generator, exclude_zero=True),
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
    test_directives=(
        # There's a problem of reducing a tensor produced by full op
        # See https://github.com/NVIDIA/Fuser/issues/132
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
        # This test is flaky in CI (which seems odd)
        # AssertionError: Scalars are not close!
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
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

le_opinfo = OpInfo(
    clang.le,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.le,
    test_directives=(
        # There's a problem of reducing a tensor produced by full op
        # See https://github.com/NVIDIA/Fuser/issues/132
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
elementwise_binary_ops.append(le_opinfo)

lt_opinfo = OpInfo(
    clang.lt,
    # NOTE Comparison operations are only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_comparison_generator,
    torch_reference=torch.lt,
    test_directives=(
        # There's a problem of reducing a tensor produced by full op
        # See https://github.com/NVIDIA/Fuser/issues/132
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
elementwise_binary_ops.append(lt_opinfo)

mul_opinfo = OpInfo(
    clang.mul,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.mul,
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2549
        DecorateInfo(
            pytest.mark.skip,
            "test_jvp_correctness",
            executors=("nvFuser",),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvFuser",),
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
    test_directives=(
        # There's a problem of reducing a tensor produced by full op
        # See https://github.com/NVIDIA/Fuser/issues/132
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
elementwise_binary_ops.append(ne_opinfo)

# NOTE torch.nextafter doens't support RHS numbers
nextafter_opinfo = OpInfo(
    clang.nextafter,
    sample_input_generator=partial(elementwise_binary_generator, no_rhs_numbers=True),
    torch_reference=torch.nextafter,
    # NOTE: nextafter is supported by PyTorch only for bfloat16, float32,
    # and float64 arguments (after normal promotion rules) and by NVFuser
    # only for float32, and float64 arguments (after normal promotion rules).
    dtypes=(datatypes.floating,),
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.bfloat16),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
            active_if=nvfuser_version < "0.0.7",
        ),
    ),
)
elementwise_binary_ops.append(nextafter_opinfo)


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
        # See https://github.com/csarofeen/pytorch/issues/2361
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            executors=("nvFuser,"),
            dtypes=(datatypes.complex64, datatypes.complex128),
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
        DecorateInfo(pytest.mark.xfail, dtypes=(datatypes.bool8, datatypes.complexfloating), executors=("TorchEx",)),
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
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.true_divide,
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
        # See https://github.com/csarofeen/pytorch/issues/2549
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
        # This test sometimes fails in CI
        #   Absolute difference: 12.348295743693598 (up to 1e-05 allowed)
        #   Relative difference: 6.48032003869975e-05 (up to 1.3e-06 allowed)
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("TorchEx",),
            dtypes=(datatypes.float64,),
        ),
    ),
)
elementwise_binary_ops.append(true_divide_opinfo)

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_binary_ops)

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


where_opinfo = OpInfo(
    clang.where,
    sample_input_generator=where_sample_generator,
    torch_reference=torch.where,
)
conditional_and_mask_ops.append(where_opinfo)


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
    ),
)
conditional_and_mask_ops.append(tril_opinfo)

# Puts all elementwise ternary opinfos into the "opinfos" list
opinfos.extend(conditional_and_mask_ops)

#
# Data movement ops
#
data_movement_ops = []


def convert_element_type_sample_generator(op, device, dtype, requires_grad, **kwargs):
    a = make_tensor((2, 3, 4), device=device, dtype=dtype, requires_grad=requires_grad)

    # TODO: add more source and target dtype pairs
    yield SampleInput(a, torch.float32)


convert_element_type_opinfo = OpInfo(
    prims.convert_element_type,
    sample_input_generator=convert_element_type_sample_generator,
    torch_reference=torch.Tensor.to,
    jax_reference=jax.lax.convert_element_type if JAX_AVAILABLE else None,
    test_directives=(
        # These usually pass but tols are still too tight to perform these tests
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
        ),
    ),
)
data_movement_ops.append(convert_element_type_opinfo)


def to_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # None
    yield SampleInput(make(4, 4))

    # device
    yield SampleInput(make(4, 4), device)
    yield SampleInput(make(4, 4), "cpu")
    yield SampleInput(make(4, 4), "cpu", dtype=torch.complex128)

    # dtype
    yield SampleInput(make(4, 4), dtype)

    # device and dtype
    yield SampleInput(make(4, 4), device, dtype)
    yield SampleInput(make(4, 4), "cpu", torch.complex128)

    # tensor
    yield SampleInput(make(4, 4), make(2, 2))
    yield SampleInput(make(4, 4), make(2, 2, device="cpu", dtype=torch.complex128))


to_opinfo = OpInfo(
    ltorch.to,
    sample_input_generator=to_sample_generator,
    torch_reference=torch.Tensor.to,
)
data_movement_ops.append(to_opinfo)

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
        # See https://github.com/csarofeen/pytorch/issues/2549
        DecorateInfo(
            pytest.mark.skip,
            "test_jvp_correctness",
            executors=("nvFuser",),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvFuser",),
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
        ([(2,), (4,)], 0),
        ([(1,), (2,), (3,)], 0),
        ([(0,), (2,)], 0),
        ([(0,), (2,)], -1),
        ([(2, 3), (2, 4)], 1),
        ([(2, 3), (2, 4), (2, 5)], 1),
    ]

    for shapes, dim in cases:
        yield SampleInput([make(s) for s in shapes], dim)


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
        yield SampleInput([make(s) for s in shapes], dim), exc_type, err_msg_match


cat_opinfo = OpInfo(
    clang.cat,
    sample_input_generator=cat_sample_generator,
    error_input_generator=cat_error_generator,
    torch_reference=torch.cat,
    test_directives=(
        # cat op was introduced in nvFuser 0.0.5
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
            active_if=nvfuser_version < "0.0.5",
        ),
        # vjp and jvp not yet implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
        DecorateInfo(pytest.mark.xfail, "test_jvp_correctness"),
    ),
)
shape_ops.append(cat_opinfo)


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
        yield SampleInput([make(s) for s in shapes], dim)


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
        yield SampleInput([make(s) for s in shapes], dim), exc_type, err_msg_match


stack_opinfo = OpInfo(
    clang.stack,
    sample_input_generator=stack_sample_generator,
    error_input_generator=stack_error_generator,
    torch_reference=torch.stack,
    test_directives=(
        # vjp and jvp not yet implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
        DecorateInfo(pytest.mark.xfail, "test_jvp_correctness"),
    ),
)
shape_ops.append(stack_opinfo)


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
        # vjp and jvp not yet implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
        DecorateInfo(pytest.mark.xfail, "test_jvp_correctness"),
    ),
)
shape_ops.append(expand_opinfo)


def getitem_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # TODO Test advanced indexing, all cases below are basic indexing
    # NOTE PyTorch does not allow negative steps
    # a.shape, key
    cases = (
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
        # Slicing and numbers
        ((1, 5, 3), (0, slice(2, 3), 2)),
        ((1, 5, 3), (-1, slice(2, 3), -2)),
        # All numbers
        ((1, 5, 3), (-1, 3, -2)),
        # Ellipses
        ((1, 5, 3), (..., slice(1, 2))),
        ((1, 5, 3), (0, ..., slice(1, 2))),
        # Newaxis/None
        ((1, 5, 3), (None, None, 0, None, 2, ..., None, None, None)),
        ((1, 5, 3), (None, None, 0, None, 2, ..., None, None)),
        # Addtl. cases
        ((7, 9, 5), (slice(2, 6, 2), None, ..., slice(3, 7), None, 2, None)),
        ((11, 7, 9, 5), (None, slice(2, 6, 2), None, ..., slice(3, 7), None, 2, None, None)),
    )

    for shape, key in cases:
        a = make(shape)
        yield SampleInput(a, key)


getitem_opinfo = OpInfo(
    operator.getitem,
    sample_input_generator=getitem_sample_generator,
    torch_reference=operator.getitem,
    jax_reference=operator.getitem,
    test_directives=(
        # TODO https://github.com/Lightning-AI/lightning-thunder/issues/422
        DecorateInfo(pytest.mark.xfail, executors=("nvFuser",)),
        # NotImplementedError: VJP for Ops.SQUEEZE is not implemented
        DecorateInfo(pytest.mark.xfail, "test_vjp_correctness"),
    ),
)
shape_ops.append(getitem_opinfo)


# TODO: only remove these cases when the executor is nvFuser
# FIXME: Zero-dim cases are skipped due to https://github.com/csarofeen/pytorch/issues/2383
# FIXME: tensors with no elements are skipped because of no nvFuser support
def reshape_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # tensor shape, shape
    cases = (
        ((4, 2), (2, -1, 2)),
        # ((), (-1,)),  # neg index, empty
        ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1)),  # neg index
    )

    reversible_cases = (
        ((4,), (4,)),
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
    sample_input_generator=reshape_sample_generator,
    torch_reference=torch.reshape,
)
shape_ops.append(reshape_opinfo)


def pad_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, padding_config
    cases = (
        ((1, 3), ((0, 0, 0), (0, 0, 0))),
        ((3, 7, 5), ((-2, 1, 0), (1, 3, 0), (-1, 2, 0))),
        ((2, 2), ((1, 1, 1), (-1, 2, 0))),
        ((2, 0, 3), ((1, 0, 0), (1, 1, 2), (0, 0, 0))),
        ((7, 5), ((0, 0, 3), (-6, 2, 1))),
        ((3, 2, 5), ((-2, 1, 0), (1, -1, 0), (-1, 3, 1))),
        # Versions of above examples but with padding between elements set to 0
        ((2, 2), ((1, 1, 0), (-1, 2, 0))),
        ((2, 0, 3), ((1, 0, 0), (1, 1, 0), (0, 0, 0))),
        # See https://github.com/Lightning-AI/lightning-thunder/issues/415
        #   The PyTorch lowering does not handle this case properly
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
        # TODO FIXME nvFuser's pad translation likely just needs an update
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
            dtypes=(datatypes.complexfloating,),
        ),
        # PyTorch's pad doesn't support complex padding values
        DecorateInfo(
            pytest.mark.xfail,
            executors=("TorchEx",),
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
shape_ops.append(pad_opinfo)


def slice_in_dim_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, start_index, limit_index, stride, dim
    cases = (
        ((4, 6, 7), 1, 3, 2, 1),
        ((4, 6, 7), 0, -1, 3, 2),
    )

    for shape, start_idx, limit_idx, stride, dim in cases:
        a = make(shape)
        yield SampleInput(a, start_idx, limit_idx, stride, dim)


slice_in_dim = OpInfo(
    clang.slice_in_dim,
    sample_input_generator=slice_in_dim_sample_generator,
    jax_reference=jax.lax.slice_in_dim if JAX_AVAILABLE else None,
    test_directives=(
        # nvFuser executor doesn't support pad correctly
        # See https://github.com/Lightning-AI/lightning-thunder/issues/285
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
shape_ops.append(slice_in_dim)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/416
#   Add strides and slicing outside tensor boundaries
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
    sample_input_generator=slice_prim_sample_generator,
    jax_reference=jax.lax.slice if JAX_AVAILABLE else None,
    test_directives=(
        # nvFuser executor doesn't support pad correctly
        # See https://github.com/Lightning-AI/lightning-thunder/issues/285
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
shape_ops.append(slice_prim_opinfo)


def split_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, size_or_sections, dim
    cases = (
        ((4, 6, 7), 2, 0),
        ((4, 6, 7), 3, 0),
        ((4, 6, 7), 3, -1),
        ((4, 6, 7), 9, 1),
        ((4, 6, 7), (1, 2, 1, 2), 1),
        # TODO https://github.com/Lightning-AI/lightning-thunder/issues/420
        # ((4, 6, 7), (3, 1, 2, 0, 0, 1), -1),
        ((4, 4, 12), 4, 2),
    )

    for shape, size_or_sections, dim in cases:
        yield SampleInput(make(shape), size_or_sections, dim)


split_opinfo = OpInfo(
    ltorch.split,
    sample_input_generator=split_sample_generator,
    torch_reference=torch.split,
    test_directives=(
        # nvFuser executor doesn't support pad correctly
        # See https://github.com/Lightning-AI/lightning-thunder/issues/285
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
shape_ops.append(split_opinfo)


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
    sample_input_generator=squeeze_sample_generator,
    jax_reference=jax.lax.squeeze if JAX_AVAILABLE else None,
)
shape_ops.append(squeeze_opinfo)


def tensor_split_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, indices_or_sections, dim
    cases = (
        ((4, 6, 7), 2, 1),
        ((4, 6, 7), 2, 2),
        ((4, 6, 7), 3, 0),
        ((4, 6, 7), 5, -1),
        # TODO https://github.com/Lightning-AI/lightning-thunder/issues/421
        # ((4, 6, 7), (0, 1), 1),
        ((4, 6, 7), (1, 5, 6), 2),
        ((4, 6, 7), (1, 5, 9, 9), 2),
        ((4, 6, 7), (1, 5, 6, 7), 2),
        # TODO https://github.com/Lightning-AI/lightning-thunder/issues/421
        # ((4, 6, 7), (0, 0, 1, 1, 2), -2),
    )

    for shape, indices_or_sections, dim in cases:
        yield SampleInput(make(shape), indices_or_sections, dim)


tensor_split_opinfo = OpInfo(
    ltorch.tensor_split,
    sample_input_generator=tensor_split_sample_generator,
    torch_reference=torch.tensor_split,
    test_directives=(
        # nvFuser executor doesn't support pad correctly
        # See https://github.com/Lightning-AI/lightning-thunder/issues/285
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("nvFuser",),
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
    )

    for shape, perm in cases:
        yield SampleInput(make(shape), perm)


transpose_opinfo = OpInfo(
    clang.transpose,
    sample_input_generator=transpose_sample_generator,
    torch_reference=torch.permute,
)
shape_ops.append(transpose_opinfo)


def permute_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape, perm
    cases = (
        ((2, 3, 4), (0, 1, 2)),
        ((2, 3, 4), (1, 2, 0)),
        ((2, 3, 4), (2, 1, 0)),
        ((2, 3, 4), (0, 2, 1)),
        ((2, 3, 4), (0, -1, 1)),
        ((4, 7), (1, 0)),
        ((3,), (0,)),
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
    sample_input_generator=permute_sample_generator,
    error_input_generator=permute_error_generator,
    torch_reference=torch_permute_reference,
)
shape_ops.append(permute_opinfo)


def matrix_transpose_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape
    cases = (
        (4, 7, 8),
        (4, 7),
    )

    for shape in cases:
        yield SampleInput(make(shape))


transpose_opinfo = OpInfo(
    clang.matrix_transpose,
    sample_input_generator=matrix_transpose_sample_generator,
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
    sample_input_generator=take_sample_generator,
    torch_reference=torch_index_select_wrapper,
    test_directives=(
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
            active_if=nvfuser_version < "0.0.3",
        ),
    ),
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
            yield SampleInput(a, b, c, dim)


# signature order mismatch, use a wrapper to resolve
def torch_index_add_wrapper(a, b, c, dim):
    return torch.index_add(a, dim, b, c)


# TODO: mapping jax.lax.gather for testing
index_add_opinfo = OpInfo(
    clang.index_add,
    sample_input_generator=index_add_sample_generator,
    torch_reference=torch_index_add_wrapper,
)
shape_ops.append(index_add_opinfo)


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
    sample_input_generator=take_along_axis_sample_generator,
    torch_reference=torch.take_along_dim,
    # Torch doesn't support complex half on take_along_dim
    test_directives=(
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # Support for take_along_axis was added in nvFuser v0.0.10
        DecorateInfo(
            pytest.mark.skip,
            executors=("nvFuser"),
            active_if=nvfuser_version < LooseVersion("0.0.10"),
        ),
    ),
)
shape_ops.append(take_along_axis_opinfo)


def scatter_add_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # torch.scatter_add expects index to be long but not int
    # Index is not differentiable! Marking requires_grad as False
    make_index = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    # Not sure if we need to consider higher order gradient, marking requires_grad as False
    make_source = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    for shape_a, dim, shape_b in take_along_axis_cases:
        canonicalized_dim = dim if dim >= 0 else dim + len(shape_a)
        shape_source = list(shape_a)
        shape_source[canonicalized_dim] = shape_b[canonicalized_dim]
        a = make(shape_a)
        b = make_index(shape_b, low=0, high=shape_a[dim])
        c = make_source(shape_source)
        yield SampleInput(a, b, c, dim)

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
        c = make_source(shape_source)
        yield SampleInput(a, b, c, dim)


# signature order mismatch, use a wrapper to resolve
def torch_scatter_add_wrapper(a, b, c, dim):
    return torch.scatter_add(a, dim, b, c)


scatter_add_opinfo = OpInfo(
    clang.scatter_add,
    sample_input_generator=scatter_add_sample_generator,
    torch_reference=torch_scatter_add_wrapper,
    test_directives=(
        # Torch doesn't support complex half on scatter_add
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
        ),
        # float16 and bfloat16 has flaky accuracy fails
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16, datatypes.float16),
        ),
    ),
)
shape_ops.append(scatter_add_opinfo)


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
    sample_input_generator=unsqueeze_sample_generator,
    jax_reference=jax.lax.expand_dims if JAX_AVAILABLE else None,
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2549
        DecorateInfo(
            pytest.mark.skip,
            "test_jvp_correctness",
            executors=("nvFuser",),
        ),
        DecorateInfo(
            pytest.mark.skip,
            "test_vjp_correctness",
            executors=("nvFuser",),
        ),
    ),
)
shape_ops.append(unsqueeze_opinfo)

opinfos.extend(shape_ops)

#
# Reduction OpInfos
#
reduction_ops = []


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
            yield (SampleInput(make(shape), dim, keepdim, dtype=dtype))


prod_opinfo = OpInfo(
    ltorch.prod,
    sample_input_generator=reduction_sample_generator,
    torch_reference=torch._refs.prod,
    test_directives=(
        # NOTE Test fails due to precision
        # TODO Investigate or reduce test precision
        DecorateInfo(
            pytest.mark.skip, "test_core_vs_torch_consistency", dtypes=(datatypes.float32,), executors=("nvFuser",)
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
            executors=("nvFuser",),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
            active_if=nvfuser_version < "0.0.4",
        ),
    ),
)
reduction_ops.append(prod_opinfo)


sum_opinfo = OpInfo(
    ltorch.sum,
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
        # See https://github.com/csarofeen/pytorch/issues/2369
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvFuser",),
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
    unbiased = (None, True, False)
    correction = (None, 0, 1)
    samples = reduction_sample_generator(op, device, dtype, requires_grad)
    for u, c, sample in itertools.product(unbiased, correction, samples):
        a = sample.args[0]
        dim = sample.args[1] if len(sample.args) > 1 else None
        keepdim = sample.args[2] if len(sample.args) > 2 else False
        # cannot specify both correction and unbiased arguments
        if u is not None and c is not None:
            continue
        elif u is not None:
            yield SampleInput(a, dim, u, keepdim)
        elif c is not None:
            yield SampleInput(a, dim, keepdim=keepdim, correction=c)
        else:
            yield SampleInput(a, dim, keepdim)

    # Tests zero-dim tensor
    yield SampleInput(make_tensor((), device=device, dtype=dtype, requires_grad=requires_grad))


mean_opinfo = OpInfo(
    ltorch.mean,
    sample_input_generator=reduction_sample_generator,
    torch_reference=torch.mean,
    dtypes=(datatypes.floating, datatypes.complexfloating),
    test_directives=(
        # PyTorch doesn't support CPU and CUDA complex half mean
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complex32,),
            devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
        ),
        # See https://github.com/csarofeen/pytorch/issues/2369
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.complexfloating,),
            executors=("nvFuser",),
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
        # NotImplementedError: VJP for Ops.VAR is not implemented
        DecorateInfo(
            pytest.mark.xfail,
            "test_vjp_correctness",
            executors=("TorchEx",),
        ),
    ),
)
reduction_ops.append(var_opinfo)

var_mean_opinfo = OpInfo(
    ltorch.var_mean,
    sample_input_generator=var_sample_generator,
    torch_reference=torch.var_mean,
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
reduction_ops.append(var_mean_opinfo)

opinfos.extend(reduction_ops)

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


def full_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make_fv = partial(make_number, dtype=dtype)

    # shape, fill_value
    cases = (
        ((), make_fv()),
        ((4, 4), make_fv()),
        ((8, 1, 6), make_fv()),
        ((8, 7, 5, 1), make_fv()),
    )

    for shape, fill_value in cases:
        yield SampleInput(shape, fill_value, device=device, dtype=dtype)


full_opinfo = OpInfo(
    clang.full,
    sample_input_generator=full_sample_generator,
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


def ones_sample_generator(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        (()),
        ((4, 4)),
        ((8, 1, 6)),
        ((8, 7, 5, 1)),
    )

    for shape in cases:
        yield SampleInput(shape, device=device, dtype=dtype)


ones_opinfo = OpInfo(
    ltorch.ones,
    sample_input_generator=ones_sample_generator,
    torch_reference=torch.ones,
)
tensor_creation_ops.append(ones_opinfo)


def empty_sample_generator(op, device, dtype, requires_grad, **kwargs):
    cases = (
        # (),  # FIXME: https://github.com/csarofeen/pytorch/issues/2358
        (1,),
        (4, 4),
        # (2, 0, 3),  # FIXME: nvFuser does not yet support shapes with 0-sized dimensions
        (8, 1, 6),
        (8, 7, 5, 1),
    )

    for shape in cases:
        yield SampleInput(shape, device=device, dtype=dtype)


# empty_opinfo = OpInfo(
#     ltorch.empty,
#     sample_input_generator=empty_sample_generator,
#     torch_reference=torch.zeros,
# )
# tensor_creation_ops.append(empty_opinfo)


opinfos.extend(tensor_creation_ops)

#
# Matmul OpInfos
#
matmul_ops = []


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
    )

    for shape_a, shape_b in cases:
        yield SampleInput(make(shape_a), make(shape_b))


matmul_opinfo = OpInfo(
    ltorch.matmul,
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
    ),
)
matmul_ops.append(matmul_opinfo)


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
matmul_ops.append(linear_opinfo)

opinfos.extend(matmul_ops)

#
# NN Ops
#
nn_ops = []


# TODO: improve sample generation, test dtype argument
def softmax_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    S = 2
    M = 5
    # Shape, dim, dtype
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


softmax_opinfo = OpInfo(
    ltorch.softmax,
    sample_input_generator=softmax_sample_generator,
    torch_reference=None if LooseVersion(torch.__version__) < "1.13" else torch._refs.softmax,
    dtypes=(datatypes.floating,),
    test_directives=(
        # torch.softmax doesn't support float16 on CPU
        # RuntimeError: "softmax_lastdim_kernel_impl" not implemented for 'Half'
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        # Tolerances are currently too conservative for this test with half precision
        DecorateInfo(
            pytest.mark.skip,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bfloat16,),
        ),
    ),
)
nn_ops.append(softmax_opinfo)


def embedding_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    N = 5
    S = 2
    # indices_shape, weight_shape, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    cases = (
        ((S,), (N, S), None, None, 2.0, False, False),
        ((S,), (N, S), 0, None, 2.0, False, False),
        ((S,), (N, S), None, None, 2.0, True, False),
        # nvFuser executor would raise an error when running this test
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
    sample_input_generator=embedding_sample_generator,
    torch_reference=torch.nn.functional.embedding,
    dtypes=(datatypes.floating, datatypes.complexfloating),
)
nn_ops.append(embedding_opinfo)


def scaled_dot_product_attention_sample_generator(op, device, dtype, requires_grad, **kwargs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 3-dim causal cases. dropout_p is not parametrized because we don't have a way to control for rng
    for N, (L, S), (E, Ev) in itertools.product((1, 2), ((2, 2), (2, 3)), ((2, 2), (2, 3))):
        q, k, v = make(N, L, E), make(N, S, E), make(N, S, Ev)
        yield SampleInput(q, k, v, dropout_p=0.0, is_causal=True)

    # 4-dim (multiheaded) causal case
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
    bool_attn_mask = make((L, S), dtype=torch.bool, low=1, high=1).tril()
    yield SampleInput(q, k, v, attn_mask=bool_attn_mask, is_causal=False)

    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    additive_attn_mask = make((L, S), dtype=q.dtype).tril()
    yield SampleInput(q, k, v, attn_mask=additive_attn_mask, is_causal=False)

    # mask with extra padding: this case will raise if https://github.com/pytorch/pytorch/issues/103749 is fixed
    # when that happens, update the SDPA impl and remove this comment
    q, k, v = make(N, n_head, L, E), make(N, n_head, S, E), make(N, n_head, S, Ev)
    bool_attn_mask = make((L, S), dtype=torch.bool, low=1, high=1).tril()
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


sdpa_opinfo = OpInfo(
    ltorch.scaled_dot_product_attention,
    sample_input_generator=scaled_dot_product_attention_sample_generator,
    error_input_generator=scaled_dot_product_attention_error_generator,
    torch_reference=torch.nn.functional.scaled_dot_product_attention,
    dtypes=(datatypes.floating,),
    test_directives=(
        # PyTorch CPU doesn't support float16 matmul
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.float16,),
            devicetypes=(devices.DeviceType.CPU,),
        ),
        DecorateInfo(
            pytest.mark.xfail,
            dtypes=(datatypes.bfloat16,),
            devicetypes=(
                # Numerical issue with bfloat16: skipped since CPU is not a priority
                devices.DeviceType.CPU,
                # RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
                devices.DeviceType.CUDA,
            ),
        ),
    ),
)
nn_ops.append(sdpa_opinfo)


def cross_entropy_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # TODO Enable test cases after adding support nll_loss_nd, weight tensor, and label_smoothing options.
    # See https://github.com/Lightning-AI/lightning-thunder/issues/704
    # input_shape, target_shape
    shapes = (
        ((7, 18), (7,)),
        # ((7, 18), (7, 18)),
        # ((3, 4, 2, 3), (3, 4, 2, 3)),
        # ((3, 4, 2, 3), (3, 2, 3)),
        ((5,), ()),
        # ((3, 4, 0), (3, 0)),
        # ((3, 4, 0), (3, 4, 0)),
    )

    # weight_options = (True, False)
    reduction_options = ("none", "mean", "sum")
    # label_smoothing_options = (0.0, 0.5)
    ignore_index_options = (-1, 3)

    weight_options = (False,)
    label_smoothing_options = (0.0,)
    for shape, weight_flag, reduction_str, label_smoothing, ignore_index in itertools.product(
        shapes, weight_options, reduction_options, label_smoothing_options, ignore_index_options
    ):
        # TODO nvfuser segfaults here, let's skip it for now until we have the support
        if reduction_str == "mean" and weight_flag:
            continue

        input_shape, target_shape = shape
        probability_target = input_shape == target_shape
        # ignore_index can't be supplied with probablity target
        if probability_target and ignore_index >= 0:
            continue

        C = input_shape[1] if len(input_shape) >= 2 else input_shape[0]
        yield SampleInput(
            make(shape[0]),
            make(shape[1], low=0, high=C, dtype=torch.long, requires_grad=False)
            if not probability_target
            else make(shape[1], low=0.0, high=1.0, requires_grad=True),
            weight=make(C) if weight_flag else None,
            size_average=None,
            ignore_index=ignore_index,
            reduce=None,
            # NOTE: I have to use kwargs, otherwise tracing on string seems to return a tuple of char.
            reduction=reduction_str,
            label_smoothing=label_smoothing,
        )


cross_entropy_opinfo = OpInfo(
    ltorch.cross_entropy,
    sample_input_generator=cross_entropy_sample_generator,
    torch_reference=torch.nn.functional.cross_entropy,
    dtypes=(datatypes.float32, datatypes.float64),
    test_directives=(
        # nvFuser version 10 adds take_along_axis, which this
        #   operator relies on
        DecorateInfo(
            pytest.mark.skip,
            executors=("nvFuser",),
            active_if=nvfuser_version < LooseVersion("0.0.10"),
        ),
        # TODO These tests inexplicably fail in CI
        DecorateInfo(
            pytest.mark.skip,
            dtypes=(datatypes.float32, datatypes.float64),
            devicetypes=(devices.DeviceType.CPU,),
            executors=("TorchEx",),
        ),
    ),
)
nn_ops.append(cross_entropy_opinfo)


opinfos.extend(nn_ops)
