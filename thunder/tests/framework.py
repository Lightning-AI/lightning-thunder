import inspect
import os
import sys
from functools import wraps
from itertools import product
from looseversion import LooseVersion

import pytest

import thunder.core.dtypes as datatypes
from thunder import make_traced
from thunder.core.trace import reset_executor_context, set_executor_context

__all__ = [
    "available_device_types",
    "executors",
    "ops",
    "NOTHING",
    "JAX_AVAILABLE",
]

# A marker for actually wanting NOTHING instead of specifying nothing
class NOTHING:
    pass


def _jax_available():
    try:
        import jax
    except Exception:
        return False

    return True


JAX_AVAILABLE = _jax_available()


# TODO: Add device type functionality to an object in this list
def _all_device_types():
    return ("cpu", "cuda")


def available_device_types():
    try:
        import torch

        # TODO: technically CUDA can be available without a CUDA device and that might
        #   be interesting to test
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return ("cpu", "cuda")
        return ("cpu",)
    except ModuleNotFoundError:
        ("cpu",)


class Executor:
    def supports_dtype(self, dtype):
        return dtype in datatypes.resolve_dtypes(self.supported_dtypes)

    def supports_devicetype(self, devicetype):
        return devicetype in self.supported_devicetypes


# TODO: extend with the ability to convert sample inputs to be appropriate for the executor
class nvFuser(Executor):
    name = "nvFuser"
    supported_devicetypes = ("cuda",)
    supported_dtypes = (
        datatypes.floating,
        datatypes.bool8,
        datatypes.int32,
        datatypes.int64,
        datatypes.complex64,
        datatypes.complex128,
    )

    ctx = None

    def get_executor_context(self):
        if self.ctx is None:
            from thunder.executors.nvfuser import nvFuserCtx

            self.ctx = nvFuserCtx()

        return self.ctx

    def make_callable(self, fn, **kwargs):
        return make_traced(fn, executor="nvfuser", **kwargs)

    # TODO: refactor this so can query the nvFuserCtx for the version
    def version(self):
        try:
            import nvfuser

            if hasattr(nvfuser, "version"):
                return nvfuser.version()
            return -1
        except:
            return -1


class TorchEx(Executor):
    name = "TorchEx"
    supported_devicetypes = ("cpu", "cuda")
    supported_dtypes = (datatypes.dtype,)

    ctx = None

    def get_executor_context(self):
        if self.ctx is None:
            from thunder.executors.torch import torchCtx

            self.ctx = torchCtx()

        return self.ctx

    def make_callable(self, fn, **kwargs):
        return make_traced(fn, executor="torch", **kwargs)

    def version(self):
        return torch.__version__


def _all_executors():
    """Constructs a list of all Thunder executors to be used when generating tests."""
    executors = []

    try:
        import torch

        executors.append(TorchEx())
    except ModuleNotFoundError:
        pass

    try:
        # TODO: refactor this so can query the nvFuserCTX for nvfuser
        #   (this requires making the ctx importable without nvFuser)
        import torch

        try:
            import nvfuser

            executors.append(nvFuser())
        except ImportError:
            try:
                import torch._C._nvfuser

                executors.append(nvFuser())
            except ImportError:
                pass
    except ModuleNotFoundError:
        pass

    return executors


def benchmark_executors():
    """Constructs a list of executors to use when benchmarking.

    These executors should define "get_callable", which returns a callable version of a given function.
    """
    executors = []

    try:
        import torch

        if LooseVersion(torch.__version__) >= "2.0":
            import nvfuser
        else:
            import torch._C._nvfuser

        executors.append(nvFuser())
    except ModuleNotFoundError:
        pass

    return executors


# TODO: refactor with _instantiate_opinfo_test_template
def _instantiate_executor_test_template(template, scope, *, executor, device, dtype):
    # Ex. test_foo_CUDA_float32
    # TODO: fix test name when dtype is None
    test_name = "_".join((template.__name__, executor.name, device.upper(), str(dtype)))

    def test():
        # TODO: currently this passes the device type as a string, but actually a device or multiple devices
        #   should be passed to the test
        result = template(executor, device, dtype)
        return result

    # Mimics the instantiated test
    # TODO: review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO: add decorator support, support for test directives -- how would this control assert_close behavior?
def _instantiate_opinfo_test_template(template, scope, *, opinfo, executor, device, dtype):
    """Instanties a test template for an operator."""

    # Ex. test_foo_CUDA_float32
    test_name = "_".join((template.__name__, opinfo.name, executor.name, device.upper(), str(dtype)))

    def test():
        # TODO: currently this passes the device type as a string, but actually a device or multiple devices
        #   should be passed to the test
        result = template(opinfo, device, dtype, executor)
        return result

    # TODO: pass device type explicitly
    for decorator in opinfo.test_decorators(template.__name__, executor, device, dtype):
        test = decorator(test)

    # Mimics the instantiated test
    # TODO: review this mimicry -- are there other attributes to mimic?
    test.__name__ = test_name
    test.__module__ = test.__module__

    return test


# TODO: don't pass the device type to the test, select an actual device
# TODO: example uses, note this must be the LAST decorator applied
class ops:

    # TODO: support other kinds of dtype specifications
    def __init__(
        self, opinfos, *, supported_executors=None, supported_device_types=None, supported_dtypes=None, scope=None
    ):
        self.opinfos = opinfos

        self.supported_executors = (
            set(supported_executors) if supported_executors is not None else set(_all_executors())
        )

        self.supported_device_types = (
            set(supported_device_types) if supported_device_types is not None else set(_all_device_types())
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
        # NOTE: unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)

        for opinfo in self.opinfos:
            device_types = (
                opinfo.device_types()
                .intersection(self.supported_device_types)
                .intersection(set(available_device_types()))
            )
            for executor, devicetype in product(self.supported_executors, device_types):
                if not executor.supports_devicetype(devicetype):
                    continue

                # TODO: pass device_type to dtypes()
                dtypes = opinfo.dtypes()
                if self.supported_dtypes != (None,):
                    dtypes = dtypes.intersection(self.supported_dtypes)

                for dtype in dtypes:
                    if not executor.supports_dtype(dtype):
                        continue

                    test = _instantiate_opinfo_test_template(
                        test_template,
                        self.scope,
                        opinfo=opinfo,
                        executor=executor,
                        device=devicetype,
                        dtype=dtype,
                    )
                    # Adds the instantiated test to the requested scope
                    self.scope[test.__name__] = test


# TODO: don't pass the device type to the test, select an actual device
# TODO: example uses, note this must be the LAST decorator applied
class executors:

    # TODO: support other kinds of dtype specifications
    def __init__(self, *, executors=None, devicetypes=None, dtypes=None, scope=None):
        self.executors = set(executors) if executors is not None else set(_all_executors())
        self.devicetypes = set(devicetypes) if devicetypes is not None else set(available_device_types())

        if dtypes == NOTHING:
            self.dtypes = (None,)
        else:
            self.dtypes = datatypes.resolve_dtypes(dtypes) if dtypes is not None else datatypes.all_dtypes

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    # TODO: refactor with the ops class above
    def __call__(self, test_template):
        # NOTE: unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)

        for executor, devicetype in product(self.executors, self.devicetypes):
            if not executor.supports_devicetype(devicetype):
                continue

            for dtype in self.dtypes:
                if dtype is not None and not executor.supports_dtype(dtype):
                    continue

                test = _instantiate_executor_test_template(
                    test_template,
                    self.scope,
                    executor=executor,
                    device=devicetype,
                    dtype=dtype,
                )
                # Adds the instantiated test to the requested scope
                self.scope[test.__name__] = test


def run_snippet(snippet, opinfo, device_type, dtype, *args, **kwargs):
    try:
        snippet(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()

        # Raises exceptions that occur with pytest, and returns debug information when
        # called otherwise
        # NOTE: PYTEST_CURRENT_TEST is set by pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise e
        return e, exc_info, snippet, opinfo, device_type, dtype, args, kwargs

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
