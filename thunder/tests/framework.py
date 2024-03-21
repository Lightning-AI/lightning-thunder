import inspect
import os
import sys
from functools import wraps, singledispatchmethod, partial
from itertools import product
from typing import List, Optional
from collections.abc import Callable, Sequence, Iterable

import pytest
import torch
from torch.testing import assert_close

from looseversion import LooseVersion
from lightning_utilities.core.imports import package_available

from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.dtypes as datatypes
import thunder.core.devices as devices
import thunder.executors as executors
import thunder.extend as extend
import thunder.executors.triton_utils as triton_utils
import thunder.core.utils as utils

from thunder.core.trace import TraceCtx, detached_trace

import thunder

__all__ = [
    "TestExecutor",
    "nvFuserExecutor",
    "TorchExecutor",
]


# A marker for actually wanting NOTHING instead of an unspecified value (marked with None)
class NOTHING:
    pass


JAX_AVAILABLE = package_available("jax")

# Require Triton version 2.1 or greater, since our current Triton executor won't run
#   properly due to an error in 2.0
TRITON_AVAILABLE: bool = triton_utils.is_triton_version_at_least("2.1")

NVFUSER_AVAILABLE = executors.nvfuser_available()

IN_CI: bool = os.getenv("CI", None) == "true"
CUDA_AVAILABLE: bool = torch.cuda.is_available()
env_var_FORCE_CPU_TEST_INSTANTIATION: str = os.getenv("FORCE_CPU_TEST_INSTANTIATION", None)
FORCE_CPU_TEST_INSTANTIATION: bool = (
    env_var_FORCE_CPU_TEST_INSTANTIATION == "true" or env_var_FORCE_CPU_TEST_INSTANTIATION == "1"
)
env_var_DISABLE_CUDA_TEST_INSTANTIATION: str = os.getenv("DISABLE_CUDA_TEST_INSTANTIATION", None)
DISABLE_CUDA_TEST_INSTANTIATION: bool = (
    env_var_DISABLE_CUDA_TEST_INSTANTIATION == "true" or env_var_DISABLE_CUDA_TEST_INSTANTIATION == "1"
)


# Filters the CPU devicetype when in CI, CUDA is available, and the environment variable
#   FORCE_CPU_TEST_INSTANTIATION isn't forcing CPU test instantiation
def filter_ci_devicetypes(devicetypes: Iterable[devices.DeviceType]) -> tuple[devices.DeviceType]:
    filtered: tuple[devices.DeviceType]
    if IN_CI and CUDA_AVAILABLE and not FORCE_CPU_TEST_INSTANTIATION:
        filtered = tuple(x for x in devicetypes if x is not devices.DeviceType.CPU)
    else:
        filtered = tuple(x for x in devicetypes)

    if DISABLE_CUDA_TEST_INSTANTIATION:
        filtered = tuple(x for x in devicetypes if x is not devices.DeviceType.CUDA)

    return filtered


# Asserts that a candidate is closer to a reference than a competitor
# This is useful when trying to compare low precision operators; the
#   reference is typically the result in a higher precision datatype, like double,
#   while the candidate and competitor are often in bfloat16 or float16
def assert_closer(*, reference, candidate, competitor, comparator):
    def _to_meta(x):
        if isinstance(x, torch.Tensor):
            return x.to(device="meta")

        return x

    # Validates metadata
    reference_meta = tree_map(_to_meta, reference)
    candidate_meta = tree_map(_to_meta, candidate)
    competitor_meta = tree_map(_to_meta, competitor)

    assert_close(reference_meta, candidate_meta, check_dtype=False)
    assert_close(reference_meta, competitor_meta, check_dtype=False)
    comparator(candidate_meta, competitor_meta)

    reference_flats, _ = tree_flatten(reference)
    candidate_flats, _ = tree_flatten(candidate)
    competitor_flats, _ = tree_flatten(competitor)

    for ref, cand, com in zip(reference_flats, candidate_flats, competitor_flats):
        if isinstance(ref, torch.Tensor):
            candidate_dist = torch.abs(ref - cand)
            competitor_dist = torch.abs(ref - com)
            minimum_dist = torch.minimum(candidate_dist, competitor_dist)

            signed_minimum_dist = torch.where(candidate_dist < 0, -minimum_dist, minimum_dist)
            target = ref + signed_minimum_dist

            comparator(cand, target, check_dtype=False)


# TODO: Add device type functionality to an object in this list
def _all_devicetypes() -> Sequence[devices.DeviceType]:
    return devices.all_devicetypes


# TODO Technically CUDA can be available without a CUDA device and that might be interesting to test
def available_devicetypes():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return devices.all_devicetypes
    return (devices.DeviceType.CPU,)


class TestExecutor:
    def supports_dtype(self, dtype: datatypes.dtype) -> bool:
        return dtype in datatypes.resolve_dtypes(self.supported_dtypes)

    def supports_devicetype(self, devicetype: devices.DeviceType) -> bool:
        return devicetype in self.supported_devicetypes

    # NOTE This method should be overridden by subclasses
    def executors_list(self) -> list[extend.Executor]:
        return []

    @singledispatchmethod
    def make_callable_legacy(self, fn, **kwargs):
        assert kwargs.pop("disable_preprocessing", True)
        return thunder.compile(fn, executors_list=self.executors_list(), disable_preprocessing=True, **kwargs)

    @singledispatchmethod
    def make_callable(self, fn, **kwargs):
        return thunder.jit(fn, executors=self.executors_list(), **kwargs)

    @make_callable.register
    def make_callable_from_trace(self, trace: TraceCtx, **kwargs):
        executors = thunder.executors
        # transform_for_execution doesn't work without a set trace
        # So we use detached_trace to get the tracectx and then use it
        with detached_trace():
            traces = thunder.common.transform_for_execution(trace, executors_list=self.executors_list(), **kwargs)
        return traces[-1].python_callable()

    # TODO Remove this
    def make_callable_with_info(self, fn, **kwargs):
        disable_preprocessing = kwargs.pop("disable_preprocessing", True)
        return thunder.compile(
            fn, executors_list=self.executors_list(), disable_preprocessing=disable_preprocessing, **kwargs
        )


# TODO Convert to singletons or just add to executor logic
class nvFuserTestExecutor(TestExecutor):
    name = "nvfuser"
    supported_devicetypes = (devices.DeviceType.CUDA,)
    supported_dtypes = (
        datatypes.floating,
        datatypes.bool8,
        datatypes.int32,
        datatypes.int64,
        datatypes.complex64,
        datatypes.complex128,
    )

    def executors_list(self) -> list[extend.Executor]:
        return [executors.get_nvfuser_executor()]

    def version(self):
        return executors.get_nvfuser_executor().version()


# TODO Convert to singletons or just add to executor logic
class TorchTestExecutor(TestExecutor):
    name = "torch"
    supported_devicetypes = (devices.DeviceType.CPU, devices.DeviceType.CUDA)
    supported_dtypes = (datatypes.dtype,)

    def executors_list(self) -> list[extend.Executor]:
        return [executors.get_torch_executor()]

    def version(self):
        return torch.__version__


class TorchCompileTestExecutor(TestExecutor):
    name = "torchcompile"
    supported_devicetypes = (devices.DeviceType.CPU, devices.DeviceType.CUDA)
    supported_dtypes = (datatypes.dtype,)

    def executors_list(self) -> list[extend.Executor]:
        from thunder.executors.torch_compile import torch_compile_executor

        return [torch_compile_executor]

    def version(self):
        return torch.__version__


# TODO Refactor these executors into the actual executor (sub)modules
TorchExecutor: TorchTestExecutor = TorchTestExecutor()
TorchCompileExecutor: TorchCompileTestExecutor = TorchCompileTestExecutor()
nvFuserExecutor: None | nvFuserTestExecutor = None

if NVFUSER_AVAILABLE:
    nvFuserExecutor = nvFuserTestExecutor()


def _all_test_executors():
    """Constructs a list of all Thunder executors to be used when generating tests."""
    executors = [TorchExecutor]

    if NVFUSER_AVAILABLE:
        executors.append(nvFuserExecutor)

    return executors


# Translates test templates with names like test_foo into instantiated tests with names like
#   test_foo_nvFuser_CUDA_float32
# TODO Fix test name when dtype is None
# TODO Refactor with _instantiate_opinfo_test_template
# TODO Should the device str include the actual device number, or just the device type like now?
# TODO Support multiple devices
def _instantiate_executor_test_template(
    template: Callable,
    scope,
    *,
    executor: TestExecutor,
    device_or_devices: devices.Device | Sequence[devices.Device],
    dtype: datatypes.dtype,
    as_name: str | None = None,
) -> Callable:
    devicetype: devices.DeviceType
    device_str: str | list[str]
    if isinstance(device_or_devices, devices.Device):
        devicetype = device_or_devices.devicetype
        device_str = str(device_or_devices)
    else:
        devicetype = device_or_devices[0].devicetype
        device_str = []
        for device in device_or_devices:
            device_str.append(str(device))

    devicetype_str = devices.devicetype_string(devicetype)
    template_name = as_name if as_name is not None else template.__name__
    test_name = "_".join((template_name, executor.name, devicetype_str, str(dtype)))

    test = partial(template, executor, device_str, dtype)

    # Mimics the instantiated test
    # TODO Review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO Support multiple devices
def _instantiate_opinfo_test_template(
    template: Callable, scope, *, opinfo, executor: TestExecutor, device: devices.Device, dtype: datatypes.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    device_str = devices.devicetype_string(device.devicetype)

    test_name = "_".join((template.__name__, opinfo.name, executor.name, device_str, str(dtype)))

    # Acquires the comparator
    # TODO If multiple decorators define custom comparators and they "overlap", then which
    #   custom comparator is applied by this is uncertain
    comp = assert_close
    for decorator in opinfo.test_decorators(template.__name__, executor, device, dtype):
        if isinstance(decorator, custom_comparator):
            comp = decorator.comparator

    def test():
        result = template(opinfo, device_str, dtype, executor, comp)
        return result

    # Applies decorators
    for decorator in opinfo.test_decorators(template.__name__, executor, device, dtype):
        test = decorator(test)

    # Mimics the instantiated test
    # TODO Review this mimicry -- are there other attributes to mimic?
    #   Probably want to refactor this to codeutils
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO Add documentation and example uses; not this must be the LAST decorator applied
# TODO Support more dtype specification flexibility
# TODO Add ability to run on other devices by default (like cuda:1 instead of cuda:0 being the default)
class ops:
    def __init__(
        self, opinfos, *, supported_executors=None, supported_devicetypes=None, supported_dtypes=None, scope=None
    ):
        self.opinfos = opinfos

        self.supported_executors = (
            set(supported_executors)
            if supported_executors is not None
            else set(_all_test_executors() + [TorchCompileExecutor])
        )
        for ex in self.supported_executors:
            assert isinstance(ex, TestExecutor)

        self.supported_devicetypes = (
            set(supported_devicetypes) if supported_devicetypes is not None else set(_all_devicetypes())
        )
        self.supported_devicetypes = set(filter_ci_devicetypes(self.supported_devicetypes))

        self.supported_dtypes = (
            datatypes.resolve_dtypes(supported_dtypes) if supported_dtypes is not None else datatypes.all_dtypes
        )

        if supported_dtypes == NOTHING:
            self.supported_dtypes = NOTHING

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    def __call__(self, test_template):
        # NOTE Unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)

        for opinfo in self.opinfos:
            devicetypes = (
                opinfo.devicetypes().intersection(self.supported_devicetypes).intersection(set(available_devicetypes()))
            )

            for executor, devicetype in product(
                sorted(self.supported_executors, key=lambda x: repr(x)), sorted(devicetypes, key=lambda x: repr(x))
            ):
                if not executor.supports_devicetype(devicetype):
                    continue

                if executor == TorchCompileExecutor and (
                    not opinfo.test_torch_compile_executor or sys.platform == "win32"
                ):
                    continue

                device = devices.Device(devicetype, None)

                # TODO Pass device_type to dtypes()
                dtypes = opinfo.dtypes()
                if self.supported_dtypes != (None,):
                    dtypes = dtypes.intersection(self.supported_dtypes)

                for dtype in sorted(dtypes, key=lambda t: repr(t)):
                    if not executor.supports_dtype(dtype):
                        continue

                    test = _instantiate_opinfo_test_template(
                        test_template,
                        self.scope,
                        opinfo=opinfo,
                        executor=executor,
                        device=device,
                        dtype=dtype,
                    )
                    # Adds the instantiated test to the requested scope
                    self.scope[test.__name__] = test


# TODO Allow executing the test suite on different devices (not just always cuda:0)
# TODO Example uses, note this must be the LAST decorator applied
class instantiate:
    # TODO: support other kinds of dtype specifications
    def __init__(
        self,
        *,
        executors=None,
        devicetypes=None,
        dtypes=None,
        num_devices: int = 1,
        decorators: None | Sequence = None,
        scope=None,
        as_name: str | None = None,
    ):
        self.executors = set(executors) if executors is not None else set(_all_test_executors())
        self.devicetypes = set(devicetypes) if devicetypes is not None else set(available_devicetypes())

        self.devicetypes = set(filter_ci_devicetypes(self.devicetypes))

        if dtypes == NOTHING:
            self.dtypes = (None,)
        else:
            self.dtypes = datatypes.resolve_dtypes(dtypes) if dtypes is not None else datatypes.all_dtypes

        self.num_devices = num_devices

        self.decorators = decorators

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

        self.as_name = as_name

    # TODO: refactor with the ops class above
    def __call__(self, test_template):
        # NOTE: unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)

        for executor, devicetype in product(
            sorted(self.executors, key=lambda x: repr(x)), sorted(self.devicetypes, key=lambda x: repr(x))
        ):
            if executor is None:
                continue

            if not executor.supports_devicetype(devicetype):
                continue

            # Identifies devices to run the test on
            available_devices = devices.available_devices()
            filtered_devices = list([x for x in available_devices if x.devicetype is devicetype])
            utils.check(self.num_devices > 0, lambda: f"Received an invalid request for {self.num_devices} devices")

            if devicetype is not devices.DeviceType.CPU and len(filtered_devices) < self.num_devices:
                continue

            device_or_devices = None
            if self.num_devices == 1:
                device_or_devices = devices.Device(devicetype, None)
            else:
                device_or_devices = []
                for idx in range(self.num_devices):
                    dev = devices.Device(devicetype, idx)
                    device_or_devices.append(dev)

            for dtype in sorted(self.dtypes, key=lambda t: repr(t)):
                if dtype is not None and not executor.supports_dtype(dtype):
                    continue

                test = _instantiate_executor_test_template(
                    test_template,
                    self.scope,
                    executor=executor,
                    device_or_devices=device_or_devices,
                    dtype=dtype,
                    as_name=self.as_name,
                )

                # Applies decorators
                if self.decorators is not None:
                    for dec in self.decorators:
                        test = dec(test)

                # Adds the instantiated test to the requested scope
                self.scope[test.__name__] = test


def run_snippet(snippet, opinfo, devicetype, dtype, *args, **kwargs):
    try:
        snippet(*args, **kwargs)
    except Exception as ex:
        exc_info = sys.exc_info()

        # Raises exceptions that occur with pytest, and returns debug information when
        # called otherwise
        # NOTE: PYTEST_CURRENT_TEST is set by pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise ex
        return ex, exc_info, snippet, opinfo, devicetype, dtype, args, kwargs

    return None


def requiresJAX(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        if not JAX_AVAILABLE:
            pytest.skip("Requires JAX")
        return fn(*args, **kwargs)

    return _fn


def requiresTriton(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        if not TRITON_AVAILABLE:
            pytest.skip("Requires Triton")
        return fn(*args, **kwargs)

    return _fn


def requiresCUDA(fn):
    import torch

    @wraps(fn)
    def _fn(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA")
        return fn(*args, **kwargs)

    return _fn


def requiresNVFuser(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        if not NVFUSER_AVAILABLE:
            pytest.skip("Requires nvFuser")
        return fn(*args, **kwargs)

    return _fn


# A dummy decorator that passes the comparator metadata
class custom_comparator:
    def __init__(self, comparator):
        self.comparator = comparator

    def __call__(self, test_template):
        return test_template
