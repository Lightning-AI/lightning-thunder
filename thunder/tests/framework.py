import inspect
import os
import sys
from functools import wraps, singledispatchmethod
from itertools import product
from typing import Callable, List, Optional
from collections.abc import Sequence

import pytest
import torch

from lightning_utilities.core.imports import package_available

import thunder.core.dtypes as datatypes
import thunder.core.devices as devices
import thunder.executors as executors
import thunder.core.utils as utils

from thunder.core.trace import TraceCtx, detached_trace

import thunder


# A marker for actually wanting NOTHING instead of an unspecified value (marked with None)
class NOTHING:
    pass


JAX_AVAILABLE = package_available("jax")
NVFUSER_AVAILABLE = executors.nvfuser_available()


# TODO: Add device type functionality to an object in this list
def _all_devicetypes() -> Sequence[devices.DeviceType]:
    return devices.all_devicetypes


# TODO Technically CUDA can be available without a CUDA device and that might be interesting to test
def available_devicetypes():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return devices.all_devicetypes
    return (devices.DeviceType.CPU,)


class Executor:
    def supports_dtype(self, dtype: datatypes.dtype) -> bool:
        return dtype in datatypes.resolve_dtypes(self.supported_dtypes)

    def supports_devicetype(self, devicetype: devices.DeviceType) -> bool:
        return devicetype in self.supported_devicetypes

    # NOTE This method should be overridden by subclasses
    def executors_list(self) -> list[executors.Executor]:
        return []

    @singledispatchmethod
    def make_callable(self, fn, **kwargs):
        # TODO: an error is thrown for many functions because __code__ and
        # inspect.signature for wrapped functions is not matching.
        # KeyError: 'args'
        # thunder/core/script/frontend.py:125: KeyError
        # with disable_preprocessing=False
        # See: https://github.com/Lightning-AI/lightning-thunder/issues/386
        disable_preprocessing = kwargs.pop("disable_preprocessing", True)
        return thunder.compile(
            fn, executors_list=self.executors_list(), disable_preprocessing=disable_preprocessing, **kwargs
        )

    @make_callable.register
    def make_callable_from_trace(self, trace: TraceCtx, **kwargs):
        executors = thunder.executors
        # transform_for_execution doesn't work without a set trace
        # So we use detached_trace to get the tracectx and then use it
        with detached_trace():
            executing_trace, history = executors.transform_for_execution(trace, executors_list=self.executors_list(), **kwargs)
        return executing_trace.python_callable()

    # TODO Remove this
    def make_callable_with_info(self, fn, **kwargs):
        disable_preprocessing = kwargs.pop("disable_preprocessing", True)
        return thunder.compile(
            fn, executors_list=self.executors_list(), disable_preprocessing=disable_preprocessing, **kwargs
        )


# TODO Convert to singletons or just add to executor logic
class nvFuser(Executor):
    name = "nvFuser"
    supported_devicetypes = (devices.DeviceType.CUDA,)
    supported_dtypes = (
        datatypes.floating,
        datatypes.bool8,
        datatypes.int32,
        datatypes.int64,
        datatypes.complex64,
        datatypes.complex128,
    )

    def executors_list(self) -> list[Executor]:
        return [executors.NVFUSER, executors.TORCH, executors.PYTHON]

    def version(self):
        return executors.nvfuser_version()


# TODO Convert to singletons or just add to executor logic
class TorchEx(Executor):
    name = "TorchEx"
    supported_devicetypes = (devices.DeviceType.CPU, devices.DeviceType.CUDA)
    supported_dtypes = (datatypes.dtype,)

    def executors_list(self) -> list[Executor]:
        return [executors.TORCH, executors.PYTHON]

    def version(self):
        return torch.__version__


# TODO Refactor these executors into the actual executor (sub)modules
TorchExecutor = TorchEx()
nvFuserExecutor = None

if NVFUSER_AVAILABLE:
    nvFuserExecutor = nvFuser()


def _all_executors():
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
    executor: Executor,
    device_or_devices: devices.Device | Sequence[devices.Device],
    dtype: datatypes.dtype,
    as_name: Optional[str] = None,
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

    def test():
        result = template(executor, device_str, dtype)
        return result

    # Mimics the instantiated test
    # TODO Review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO Support multiple devices
def _instantiate_opinfo_test_template(
    template: Callable, scope, *, opinfo, executor: Executor, device: devices.Device, dtype: datatypes.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    device_str = devices.devicetype_string(device.devicetype)

    test_name = "_".join((template.__name__, opinfo.name, executor.name, device_str, str(dtype)))

    def test():
        result = template(opinfo, device_str, dtype, executor)
        return result

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
            set(supported_executors) if supported_executors is not None else set(_all_executors())
        )

        self.supported_devicetypes = (
            set(supported_devicetypes) if supported_devicetypes is not None else set(_all_devicetypes())
        )
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

                device = devices.Device(devicetype, 0)

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
        scope=None,
        as_name: Optional[str] = None,
    ):
        self.executors = set(executors) if executors is not None else set(_all_executors())
        self.devicetypes = set(devicetypes) if devicetypes is not None else set(available_devicetypes())

        if dtypes == NOTHING:
            self.dtypes = (None,)
        else:
            self.dtypes = datatypes.resolve_dtypes(dtypes) if dtypes is not None else datatypes.all_dtypes

        self.num_devices = num_devices

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
                device_or_devices = devices.Device(devicetype, 0)
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
