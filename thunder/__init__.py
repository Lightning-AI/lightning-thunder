from functools import wraps, partial
from typing import Dict, Set, Optional, Any, List, Tuple, Type
from types import EllipsisType
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import os
import dis
from enum import Enum, auto
import time
from numbers import Number
from itertools import chain
from types import NoneType
import optree
import warnings

from looseversion import LooseVersion

from thunder.core.options import (
    INTERPRETATION_OPTIONS,
    resolve_interpretation_option,
    CACHE_OPTIONS,
    resolve_cache_option,
    SHARP_EDGES_OPTIONS,
    resolve_sharp_edges_option,
)
from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
    tracectx,
    is_tracing,
)

from thunder import functional as functional
import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
from thunder.core.symbol import BoundSymbol
import thunder.core.devices as devices
from thunder.common import (
    CompileData,
    CompileStats,
    _create_callable,
    trace,
    transform_for_execution,
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor
from thunder.core.compile_data import compile_data_and_stats, get_cache_option, using_symbolic_values
from thunder.core.langctxs import LanguageContext, resolve_language, Languages
import thunder.core.langctxs as langctxs
from thunder.core.baseutils import is_base_printable, run_once
from thunder.core.codeutils import get_siginfo, SigInfo, is_simple_printable_collection
from thunder.core.proxies import (
    is_proxyable,
    proxy,
    Proxy,
    TensorProxy,
    pyval,
    pytype,
    NumberProxy,
    StringProxy,
    IntegerProxy,
    FloatProxy,
    ComplexProxy,
    TupleProxy,
    ListProxy,
    DictProxy,
    AnyProxy,
)
from thunder.core.jit_ext import minimal_thunder_jit, thunder_general_jit
from thunder.core.pytree import tree_flatten
from thunder.executors.torch_autograd import split_forward_backward, ThunderFunction
from thunder.cudagraphs import CUDAGraphExecutor

# NOTE This import is intentionally pytorch so that it thunder.torch doesn't import this
import torch as pytorch

import thunder.clang as clang

# Imports executors (to populate default executors and make them accessible)
import thunder.executors.pythonex
import thunder.executors.torchex
import thunder.executors.nvfuserex

pythonex = extend.get_executor("python")


_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# TODO RC1 Review exposed names
__all__ = [
    # dtype aliases
    "bool8",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
    # language aliases
    "torch",
    "numpy",
    "prims"
    # interface functions
    # TODO Extend this
    # TODO Add device aliases
    # TODO Add executor aliases
    "sdpa_executor",
    "nvfuser_executor",
    "pytorch_executor",
    # debugging functions
    "set_execution_callback_file",
]


def __version__():
    return LooseVersion("0.0.1")


# TODO maybe move these aliases to the core language?
#
# dtype aliases
#
bool8 = dtypes.bool8
uint8 = dtypes.uint8
int8 = dtypes.int8
int16 = dtypes.int16
int32 = dtypes.int32
int64 = dtypes.int64
bfloat16 = dtypes.bfloat16
float16 = dtypes.float16
float32 = dtypes.float32
float64 = dtypes.float64
complex32 = dtypes.complex32
complex64 = dtypes.complex64
complex128 = dtypes.complex128

#
# Module aliases
#

# NOTE this allows clang.foo() to be called directly as thunder.foo()
from thunder.clang import *

#
# Promoted executor-related functions and objects
#

# TODO Add more of these functions
get_all_executors = extend.get_all_executors
get_default_executors = extend.get_default_executors
get_always_executors = extend.get_always_executors

sdpa_executor: None | extend.Executor = extend.get_executor("sdpa")
nvfuser_executor: None | extend.Executor = extend.get_executor("nvfuser")
pytorch_executor: None | extend.Executor = extend.get_executor("torch")

# Default executor list is [sdpa -> nvfuser -> torch -> python]
if nvfuser_executor:
    add_default_executor(nvfuser_executor)

if sdpa_executor:
    add_default_executor(sdpa_executor)

#
# Promoted debugging functions
#

# If set, Python programs will be written to this file before being executed, and if the
#   the file is modified then the modified version of the program will be compiled and executed, instead.
from thunder.core.trace import _set_execution_file

set_execution_callback_file = _set_execution_file


def _eager_validate_tensor(p: TensorProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a number with symbolic values, but this is not supported yet")

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_tensor_shape_and_metadata(p)

    return ([p], [p])


def _eager_unpack_tensor(
    t: pytorch.Tensor, /, name: None | str, *, co: CACHE_OPTIONS
) -> tuple[TensorProxy, TensorProxy]:
    p = proxy(t, name=name)
    return p, p


def _eager_validate_literal_like(p: AnyProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(
            f"Trying to unpack a literal-like value of type {pytype(p)} with symbolic values, but this is not supported yet"
        )

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_literal_like(p, pyval(p))

    return ([], [pyval(p)])


def _eager_unpack_literal_like(x: Any, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[AnyProxy, None]:
    p = proxy(x, name=name)
    return p, x


def _eager_validate_none(p: AnyProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a None with symbolic values, but this is not supported yet")

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_none(p)

    return ([], [None])


def _eager_unpack_none(n: None, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[AnyProxy, None]:
    assert n is None
    p = proxy(None, name=name)
    return p, None


def _eager_validate_number(p: NumberProxy, /, *, co: CACHE_OPTIONS):
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a number with symbolic values, but this is not supported yet")

    # When not using symbolic values, numbers are compile-time constants, so an actual
    #   Python number is used when interpreting the function, and no number is passed
    #   from the prologue to the computation in the eventual thunder program
    val = pyval(p)

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_number_type_and_value(p, val)

    return ([], [val])


def _eager_unpack_number(num: Number, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[NumberProxy, Number]:
    p = proxy(num, name=name)
    return p, num


def _eager_validate_string(p: StringProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a string with symbolic values, but this is not supported yet")

    # When not using symbolic values, strings are compile-time constants, so an actual
    #   Python string is used when interpreting the function, and no string is passed
    #   from the prologue to the computation in the eventual thunder program
    val = pyval(p)

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_string_value(p, val)

    return ([], [val])


def _eager_unpack_string(
    s: str,
    /,
    name: None | str,
    *,
    co: CACHE_OPTIONS,
) -> tuple[StringProxy, str]:
    p = proxy(s, name=name)
    return p, s


# NOTE When unpacking a tuple...
#   - the values in the TupleProxy are the interpreter values
#   - the interpreter is given the tuple, the computation is given the tuple and a flat list of its proxied elements
#   - non-tuple values within the tuple are temporarily assigned proxies by clang.unpack_tuple so they can be
#       validated
def _eager_validate_tuple(p: TupleProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    unpacked = clang.unpack_tuple(p)

    computation_args = [p]

    clang.check_instance(p, (tuple, pytorch.Size))
    if len(p) == 0:
        clang.check_empty(p)

    for x in unpacked:
        cargs, iargs = _eager_validate(x, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_tuple(tup: tuple, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)

    values = []
    for x in tup:
        p, a = unpack(x)
        values.append(a)

    p = proxy(tuple(values), name=name)
    return p, p


def _eager_validate_list(p: ListProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    unpacked = clang.unpack_list(p)

    computation_args = [p]

    clang.check_type(p, list)
    if len(p) == 0:
        clang.check_empty(p)

    for x in unpacked:
        cargs, iargs = _eager_validate(x, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_list(lst: list, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)

    values = []
    for x in lst:
        p, a = unpack(x)
        values.append(a)

    p = proxy(list(values), name=name)
    return p, p


def _eager_validate_dict(p: DictProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    clang.check_type(p, dict)

    if len(p) == 0:
        clang.check_empty(p)

    computation_args = [p]
    for k, v in p.items():
        pv = clang.unpack_dict_key(p, k)
        cargs, iargs = _eager_validate(pv, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_dict(d: dict, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)
    proxied = {}
    for k, v in d.items():
        if not isinstance(k, (int, str)):
            raise ValueError(f"Unsupported input dict key type {type(k)}. Supported types are int and str.")

        vp, a = unpack(v)
        proxied[k] = a

    p = proxy(proxied, name=name)
    return p, p


def _eager_validate_any(p: Proxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    typ: type = pytype(p)
    if typ is NoneType:
        return _eager_validate_none(p, co=co)
    if typ in (pytorch.dtype, pytorch.device, slice, EllipsisType):
        return _eager_validate_literal_like(p, co=co)

    raise NotImplementedError("Trying to validate an object with type {typ}, but this is not implemented")


_type_to_unpack_map: dict[type, Callable] = {
    pytorch.Tensor: _eager_unpack_tensor,
    bool: _eager_unpack_number,
    int: _eager_unpack_number,
    float: _eager_unpack_number,
    complex: _eager_unpack_number,
    str: _eager_unpack_string,
    tuple: _eager_unpack_tuple,
    pytorch.Size: _eager_unpack_tuple,
    list: _eager_unpack_list,
    dict: _eager_unpack_dict,
    slice: _eager_unpack_literal_like,
    EllipsisType: _eager_unpack_literal_like,
    NoneType: _eager_unpack_none,
    pytorch.dtype: _eager_unpack_literal_like,
    pytorch.device: _eager_unpack_literal_like,
}

_type_to_validation_map: dict[type, Callable] = {
    TensorProxy: _eager_validate_tensor,
    IntegerProxy: _eager_validate_number,
    FloatProxy: _eager_validate_number,
    ComplexProxy: _eager_validate_number,
    StringProxy: _eager_validate_string,
    TupleProxy: _eager_validate_tuple,
    ListProxy: _eager_validate_list,
    DictProxy: _eager_validate_dict,
    AnyProxy: _eager_validate_any,
}


def _eager_validate(x: Any, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    typ: type = type(x)
    unpack_fn = _type_to_validation_map.get(typ, None)
    if unpack_fn is None:
        raise ValueError(f"Cannot validate object of type {typ}. Please file an issue requesting support.")

    return unpack_fn(x, co=co)


def _eager_unpack(x: Any, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[Proxy, Any]:
    typ: type = type(x)
    unpack_fn = _type_to_unpack_map.get(typ, None)
    if unpack_fn is None:
        raise ValueError(f"Cannot unpack object of type {typ}. Please file an issue requesting support.")

    return unpack_fn(x, name, co=co)


# A helper for "eager unpacking" interpreters that eagerly unpack their arguments as inputs
# An interpreter must do two things:
#   1) Create a prologue function with the same signature as the original function that
#       acquires all supported inputs and validates them according to the caching option
#   2) Creates a computation function that accepts the output of the prologue function and
#       returns what the original function did
def _eager_unpacking_interpreter(
    interpreter: Callable, fn: Callable, args, kwargs, /, *, interpreter_name: str
) -> tuple[TraceCtx, TraceCtx]:
    # Unpacks the inputs
    si: SigInfo = get_siginfo(fn, args, kwargs)

    prologue_trc: TraceCtx = TraceCtx(si.unwrapped_fn)
    computation_trc: TraceCtx = TraceCtx()

    # Constructs the prologue trace (which just trivially unpacks the tensor arguments for now)
    # TODO RC1 Remove the no_grad and no_autocast context managers from this trace
    # TODO RC1 Provide a mechanism to add context managers to the prologue and computation functions
    # TODO RC1 Don't always import torch in traces (particularly the prologue trace)
    csi = SigInfo("computation")
    csi.args = []
    prologue_args = []  # Arguments to the prologue
    prologue_kwargs = {}  # Kwargs to the prologue
    computation_args = []  # Arguments to the computation
    interpretation_args = []  # Arguments to interpret with
    interpretation_kwargs = {}  # Kwargs to interpret with
    co: CACHE_OPTIONS = get_cache_option()
    with tracectx(prologue_trc):
        # Unpacks args
        for name, x in si.args:
            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_args.append(p)
            interpretation_args.extend(iargs)

        # Unpacks varargs (if present)
        # NOTE varargs must follow other positional args
        if si.varargs is not None:
            name, x = si.varargs

            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_args.append(p)
            (iarg,) = iargs
            interpretation_args.extend(iarg)

        # Unpacks kwargs
        for name, x in si.kwargs.items():
            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_kwargs[name] = p
            (iarg,) = iargs
            interpretation_kwargs[name] = iarg

        if si.varkwargs is not None:
            name, x = si.varkwargs

            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_kwargs[name] = p
            (iarg,) = iargs

            for k, v in iarg.items():
                interpretation_kwargs[k] = v

    prologue_trc.args = prologue_args
    prologue_trc.kwargs = prologue_kwargs

    # Constructs the computation trace
    # TODO RC1 Only unpack what's used in the computation
    with tracectx(computation_trc):
        p: Proxy
        for p in computation_args:
            prims.unpack_trivial(p)
            csi.args.append((p.name, None))
            computation_trc.add_name(p.name)

        result = interpreter(si.unwrapped_fn)(*interpretation_args, **interpretation_kwargs)

        # Validates that the returned items are proxies or printable values
        def leaf_test(x: Any) -> bool:
            if isinstance(x, Proxy):
                return True
            if is_base_printable(x):
                return True
            if is_simple_printable_collection(x):
                return False

            raise RuntimeError(
                f"Trying to return object of type {type(x)}, but only proxies, strings, torch.device objects, numbers, tuples, lists, and dicts can be returned."
            )

        optree.tree_flatten(result, is_leaf=leaf_test)

        prims.python_return(result)

    # Creates hand-off from prologue to computation
    with tracectx(prologue_trc):
        prims.python_return(tuple(computation_args))

    # Constructs the computation trace's signature
    computation_trc._siginfo = csi
    computation_trc.args = computation_args

    return prologue_trc, computation_trc


# Translates the Python function a thunder program using the Python interpreter
def _python_interpreter(
    fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS
) -> tuple[TraceCtx, TraceCtx]:
    if sharp_edges is not SHARP_EDGES_OPTIONS.ALLOW:
        raise ValueError(
            f"Detecting sharp edges is not supported when using the Python interpreter. To detect sharp edges use another interpretation option."
        )

    def _interpreter(fn_):
        return fn_

    return _eager_unpacking_interpreter(_interpreter, fn, args, kwargs, interpreter_name="Python")


# Translates the Python function to a thunder program using the thunder interpreter
def _translate_functions_interpreter(
    fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS
) -> tuple[TraceCtx, TraceCtx]:
    pjit = partial(minimal_thunder_jit, sharp_edges=sharp_edges)
    return _eager_unpacking_interpreter(pjit, fn, args, kwargs, interpreter_name="translate functions")


# Translates the Python function to a thunder program using the thunder interpreter
def _general_frontend(fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS) -> tuple[TraceCtx, TraceCtx]:
    return thunder_general_jit(fn, args, kwargs, sharp_edges=sharp_edges)


class ThunderModule(pytorch.nn.Module):
    def __init__(self, model, compiled_model_call):
        super().__init__()
        self._model = model

        self._forward_fn = compiled_model_call

    def forward(self, *args, **kwargs):
        res = self._forward_fn(*args, **kwargs)
        return res

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization in data parallel mode.

        This context manager is intended to be used in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel` to disable gradient
        synchronization in the backward pass. It will not have any effect when
        used with other modules.

        .. note::

            This could lead to different accumulated gradients with ``torch.nn.parallel.distributed.DistributedDataParallel.no_sync``.
            PyTorch's gradient synchronization is implemented by applying all-reduce to gradient buckets of ``torch.nn.Parameter.grad``.
            Thus the ``no_sync`` context leads to :math:`\text{AllReduce} \\left( \\sum_{i = 0}^{\rm{num_grad_accum_steps}} g_i \right)`.
            In contrast, this synchronizes accumulated gradients when exiting, leading to
            :math:`\text{AllReduce} \\left( \\sum_{i = 0}^{\rm{num_grad_accum_steps - 1}} g_i \right) + \text{AllReduce}(g_{\rm{num_grad_accum_steps}})`.

        .. warning::

            You must reuse this context manager in each group of gradient accumulation iterations since gradients will get synchronized
            on context manager exit.

            .. code-block:: python

                with model.no_sync():
                    for _ in range(len(gradient_accumulation_iters)):
                        loss(model(x)).backward()  # uses no-sync-backward trace
                loss(model(x)).backward()  # uses the regular backward trace
                optimizer.step()

        """
        from thunder.distributed import (
            set_skip_data_parallel_grad_sync,
            reset_skip_data_parallel_grad_sync,
            _sync_grads,
        )

        token = set_skip_data_parallel_grad_sync(True)
        try:
            yield
        finally:
            reset_skip_data_parallel_grad_sync(token)
            _sync_grads(self)

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            return self._modules["_model"]
        return getattr(self._model, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.load_state_dict(*args, **kwargs)


# this captures the information needed to decide whether a cached function
# matches (e.g. ddp and autocast state)
_cache_info_ctx = ContextVar("cache_info_ctx")


def _with_cache_info_ctx(fn):
    def cache_info_wrapper(*args, **kwargs):
        tok = _cache_info_ctx.set({})
        try:
            res = fn(*args, **kwargs)
        finally:
            _cache_info_ctx.reset(tok)
        return res

    return cache_info_wrapper


def _get_cache_info():
    return _cache_info_ctx.get()


@run_once
def _recursive_jit_call_warning() -> None:
    warnings.warn(
        "Calling a jitted function from a jitted function currently uses all settings from the caller. In the future this behavior may change."
    )


# This function will replace compile() (below) before RC1
# TODO RC1 Consider adding a debug_log parameter to control debug printing
# TODO RC1 Consider renaming compile_options to additional_compile_options
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any | LanguageContext = None,
    executors: None | Sequence[Executor] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    interpretation: None | INTERPRETATION_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    disable_torch_autograd: bool = False,  # TODO Revisit this UX for RC1
    additional_transforms: list | None = None,
    **compile_options,  # TODO RC1 Make this explicit -- dict of options
) -> Callable:
    if "executors_list" in compile_options:
        warnings.warn("outdated argument executors_list= in call, please use executors=")
        if executors is None:
            executors = compile_options.pop("executors_list")

    # Resolves interpreter option
    interpretation = resolve_interpretation_option(interpretation)
    interpreter: Callable
    if interpretation is INTERPRETATION_OPTIONS.PYTHON_INTERPRETER:
        interpreter = _python_interpreter
    elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS:
        interpreter = _translate_functions_interpreter
    elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON:
        interpreter = _general_frontend

    if additional_transforms is None:
        additional_transforms = []

    # TODO RC1 Refine the compile data option to remove unused options
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors,
        cache_option=cache,
        sharp_edges=sharp_edges,
        using_jit=True,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=disable_torch_autograd,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=True,
        compile_options=compile_options,
    )
    cs = CompileStats()

    @_with_cache_info_ctx
    def get_computation_and_inputs(*args, **kwargs):
        # set up a record of things in the current environment that impact caching / prologues
        # this could be replaced by the respective querying in the prologues
        cache_info = _get_cache_info()

        # autocast related operations
        is_autocast_enabled = False
        autocast_key = None
        if pytorch.is_autocast_enabled() or pytorch.is_autocast_cpu_enabled():
            if pytorch.is_autocast_enabled() and pytorch.is_autocast_cpu_enabled():
                raise NotImplementedError(
                    "thunder.autocast does not support torch.is_autocast_enabled() and torch.is_autocast_cpu_enabled() simultaneously at this moment."
                )
            is_autocast_enabled = True
            autocast_gpu_dtype = dtypes.to_dtype(pytorch.get_autocast_gpu_dtype())
            autocast_cpu_dtype = dtypes.to_dtype(pytorch.get_autocast_cpu_dtype())
            cache_info.update(
                autocast_config_torch_enabled=pytorch.is_autocast_enabled(),
                autocast_config_torch_cpu_enabled=pytorch.is_autocast_cpu_enabled(),
                autocast_gpu_dtype=str(autocast_gpu_dtype),
                autocast_cpu_dtype=str(autocast_cpu_dtype),
            )
            autocast_thunder_dtype = autocast_cpu_dtype if pytorch.is_autocast_cpu_enabled() else autocast_gpu_dtype

        cache_info["is_autocast_enabled"] = is_autocast_enabled

        # TODO(crcrpar): support FSDP as well
        is_ddp_enabled = getattr(fn, "use_ddp", False)
        no_grad_sync = False
        if is_ddp_enabled:
            from thunder.distributed import get_skip_data_parallel_grad_sync

            no_grad_sync = get_skip_data_parallel_grad_sync()
        cache_info["no_grad_sync"] = no_grad_sync

        # TODO RC1 Add module and function checks to prologue (make it a compile option)

        # Checks cache
        cs.last_trace_cache_start = time.time_ns()
        if (cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES) or (cd.cache_option is CACHE_OPTIONS.SYMBOLIC_VALUES):
            for pro, pro_traces, comp, comp_traces, epilogue, epilogue_traces, backward_fn, backward_traces in reversed(
                cs.interpreter_cache
            ):
                try:
                    cs.last_prologue_execution_start = time.time_ns()
                    if epilogue:
                        inps, pro_to_epi = pro(*args, **kwargs)
                    else:
                        inps = pro(*args, **kwargs)
                        pro_to_epi = None
                    cs.last_prologue_execution_stop = time.time_ns()
                except Exception as ex:
                    continue

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro
                cs.last_prologue_transformation_start = 0
                cs.last_prologue_transformation_stop = 0
                cs.last_computation_transformation_start = 0
                cs.last_computation_transformation_stop = 0

                return inps, pro_to_epi, comp, epilogue, backward_fn

        if cd.cache_option is CACHE_OPTIONS.SAME_INPUT:
            if len(cs.interpreter_cache):
                (
                    pro,
                    pro_traces,
                    comp,
                    comp_traces,
                    epilogue,
                    epilogue_traces,
                    backward_fn,
                    backward_traces,
                ) = cs.interpreter_cache[0]

                cs.last_prologue_execution_start = time.time_ns()
                if epilogue:
                    inps, pro_to_epi = pro(*args, **kwargs)
                else:
                    inps = pro(*args, **kwargs)
                    pro_to_epi = None
                cs.last_prologue_execution_stop = time.time_ns()

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro

                return inps, pro_to_epi, comp, epilogue, backward_fn

        cs.cache_misses += 1
        cs.last_trace_cache_stop = time.time_ns()

        # Resets use of compile flags
        cs.last_compile_reasons = defaultdict(list)

        with compile_data_and_stats(cd, cs):
            # Acquires the trace OR inlines the trace into an existing trace and
            #   returns the (proxied) result of the operation
            cs.last_trace_tracing_start = time.time_ns()

            with langctxs.langctx(cd.langctx):
                prologue_trc: TraceCtx
                computation_trc: TraceCtx
                prologue_trc, computation_trc, *maybe_epilogue = interpreter(
                    fn, args, kwargs, sharp_edges=cd.sharp_edges
                )

            if maybe_epilogue:
                epilogue_traces = maybe_epilogue
                if epilogue_traces[-1] is not None:
                    epilogue = epilogue_traces[-1].python_callable()
                else:
                    epilogue_traces = None
                    epilogue = None
            else:
                epilogue_traces = None
                epilogue = None

            cs.last_trace_tracing_stop = time.time_ns()

            # Makes the prologue callable
            cs.last_prologue_transformation_start = time.time_ns()
            protraces = transform_for_execution(
                prologue_trc,
                executors_list=(pythonex,),
                use_del_last_used=False,
            )
            protrace = protraces[-1]
            pro = protrace.python_callable()

            cs.last_prologue_transformation_stop = time.time_ns()

            cs.last_prologue_execution_start = time.time_ns()
            if epilogue:
                inps, pro_to_epi = pro(*args, **kwargs)
            else:
                inps = pro(*args, **kwargs)
                pro_to_epi = None
            cs.last_prologue_execution_stop = time.time_ns()

            computation_traces = [computation_trc]

            if is_autocast_enabled:
                from thunder.core.transforms import autocast

                computation_trc = trace(compile_data=cd)(
                    autocast(computation_trc.python_callable(), dtype=autocast_thunder_dtype), *inps
                )
                computation_traces.append(computation_trc)

            backward_trc = None
            if not cd.disable_torch_autograd_support:
                tensor_cls = (pytorch.Tensor, TensorProxy)
                requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in inps)

                if requires_grad:
                    # thunder_backward may recursively call compile and wraps the result in a
                    # torch.autograd.Function to support embedding of Thunder-compiled
                    # functions in torch's Autograd
                    computation_trc, backward_trc = split_forward_backward(
                        computation_trc.python_callable(), cd, cs, *inps
                    )
                    computation_traces.append(computation_trc)

            cs.last_computation_transformation_start = time.time_ns()

            ## EPILOGUE and TRANSFORMS should not mix...
            # applies transforms
            for transform in additional_transforms:
                computation_trc = transform(computation_trc, executors_list=cd.executors_list)
                computation_traces.append(computation_trc)

            with langctxs.langctx(cd.langctx):
                extraces = transform_for_execution(
                    computation_trc,
                    executors_list=cd.executors_list,
                )
            extrace = extraces[-1]
            comp = extrace.python_callable()

            if backward_trc is not None:
                backward_fn = backward_trc.python_callable()
                backward_traces = [backward_trc]
            else:
                backward_fn = None
                backward_traces = []

            # TODO RC1 Update the cache
            if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
                cs.interpreter_cache.append(
                    (pro, protraces, comp, extraces, epilogue, epilogue_traces, backward_fn, backward_traces)
                )

            cs.last_computation_transformation_stop = time.time_ns()
            cs.last_traces = [computation_trc] + extraces
            cs.last_prologue_traces = [prologue_trc] + protraces
            cs.last_prologue = pro

        return inps, pro_to_epi, comp, epilogue, backward_fn

    @wraps(fn)
    def fn_(*args, **kwargs) -> Any:
        if is_tracing():
            _recursive_jit_call_warning()
            return fn(*args, **kwargs)

        # Updats call statistics
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        inps, pro_to_epi, comp, epilogue, backward_fn = get_computation_and_inputs(*args, **kwargs)
        cs.last_trace_host_execution_start = time.time_ns()

        result = comp(*inps)

        if backward_fn:
            # Run the compiled forward function
            data_for_autograd, (saved_tensors, saved_other) = result

            # Connect produced tensors with PyTorch's autograd graph
            ThunderFunction.apply(
                backward_fn,
                saved_tensors,
                saved_other,
                data_for_autograd["flat_output"],
                *data_for_autograd["flat_args"],
            )
            result = data_for_autograd["output"]

        if epilogue:
            result, comp_to_epi = result
            epilogue(*pro_to_epi, *comp_to_epi)

        cs.last_trace_host_execution_stop = time.time_ns()
        cs.last_computation_execution_stop = cs.last_trace_host_execution_stop

        cs.last_executed = comp
        cs.last_trace_cache_stop = time.time_ns()
        cs.last_trace_host_stop = time.time_ns()

        # Updates statistics
        cs.last_executed = comp
        cs.last_trace_host_stop = time.time_ns()

        return result

    if isinstance(fn, pytorch.nn.Module):
        fn_ = ThunderModule(fn, fn_)

    # Sets compile options and statistics attributes
    fn_._lc_cd = cd
    fn_._lc_cs = cs
    fn_._lc_transforms = additional_transforms[:]  ## transforms
    fn_._lc_post_optimization_transforms = []  ## post_optimization_transforms

    return fn_


def compile(
    fn: Callable,
    *,
    langctx: None | Any = None,
    executors_list: None | Sequence[Executor] = None,
    cache_mode: None | str | CACHE_OPTIONS = None,
    use_cudagraphs: bool = False,
    use_torch_compile: bool = False,
    disable_torch_autograd_support: bool = False,
    use_rematerialization: bool = False,
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
    **kwargs,
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors_list,
        cache_option=cache_mode,
        use_cudagraphs=use_cudagraphs,
        use_torch_compile=use_torch_compile,
        disable_torch_autograd_support=disable_torch_autograd_support,
        use_rematerialization=use_rematerialization,
        only_execute_prims=only_execute_prims,
        disable_preprocessing=disable_preprocessing,
        compile_options=kwargs,
    )

    cs = CompileStats()
    _fn = _create_callable(cd, cs)
    return _fn


def compile_data(fn) -> CompileData | None:
    return getattr(fn, "_lc_cd", None)


def compile_stats(fn) -> CompileStats | None:
    return getattr(fn, "_lc_cs", None)


# TODO We should remove compiledata.last_traces in favor of forward_last_traces and backward_last_traces
def last_traces(fn) -> list[TraceCtx] | tuple[list[TraceCtx], list[TraceCtx]]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.forward_last_traces is not None and cs.backward_last_traces is not None:
        return cs.forward_last_traces, cs.backward_last_traces
    if cs.last_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_traces


def last_prologue_traces(fn) -> TraceCtx:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_prologue_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_prologue_traces


def cache_option(fn) -> CACHE_OPTIONS:
    cd = compile_data(fn)
    if cd is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cd.cache_option


def cache_hits(fn) -> int:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_hits


def cache_misses(fn) -> int:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_misses


def list_transforms(fn) -> list:
    return fn._lc_transforms


def last_interpreted_instructions(fn: Callable) -> list[dis.Instruction]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_instructions is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_instructions


def last_interpreted_history(fn: Callable) -> list[dis.Instruction | str]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_history is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_history


# Prints how compiled options were used (or not)
def last_compile_options(fn: Callable, /) -> None:
    cd = compile_data(fn)
    cs = compile_stats(fn)

    # NOTE Different categories of compile options
    # Specified and Queried --- in cs.last_compile_reasons and cd.compile_options
    # Queried but not Specified --- in cs.last_compile_reasons but not in cd.compile_options (not printed)
    # Specified but not Queried --- in cd.compile_options but not in cs.last_compile_reasons

    specified: set = set(cd.compile_options.keys())
    queried: set = set(cs.last_compile_reasons.keys())

    # Prints used options
    print("Used compile options:")
    used = specified & queried

    if len(used) == 0:
        print("\tNo used options")

    for option in used:
        reasons = set(cs.last_compile_reasons[option])

        for reason in reasons:
            print(f"\t{option}. {reason}")

    # Prints unused options
    print("Unused compile options:")
    unused: set = specified - queried

    if len(unused) == 0:
        print("\tNo unused options")

    for option in unused:
        print(f"\t{option}")


# TODO (mruberry) Update this
def _grad_transform(trace):
    grad_fwd_trace = from_trace(trace)
    trace_tok = set_tracectx(grad_fwd_trace)
    all_residuals = []

    # Constructs grad fwd and records info
    # TODO: make recursive (or iterative, whatever)
    current_inputs = grad_fwd_trace.args
    for bsym in trace.bound_symbols:
        grad_defined = bsym.sym.grad_defined
        grad_ignored = bsym.sym.grad_ignored
        grad_fwd, grad_bwd = bsym.sym.grad_fwd, bsym.sym.grad_bwd

        if not grad_defined:
            raise NotImplementedError

        # Constructs the new grad_fwd symbol, which returns the primals and residuals
        if grad_fwd is None:
            fw_result = bsym.sym(*current_inputs)
            residuals = None
            all_residuals.append(residuals)
            current_inputs = fw_result if isinstance(fw_result, Sequence) else (fw_result,)
            continue

        fw_result, residuals = grad_fwd(*current_inputs)
        all_residuals.append(residuals)
        current_inputs = fw_result if isinstance(fw_result, Sequence) else (fw_result,)

    # Constructs bwd part of the program
    current_grads = (prims.full(o.shape, 1.0, device=o.device, dtype=o.dtype) for o in fw_result)

    for bsym, residuals in zip(reversed(trace.bound_symbols), reversed(all_residuals)):
        grad_fwd = bsym.sym.grad_fwd
        grad_bwd = bsym.sym.grad_bwd
        grad_defined = bsym.sym.grad_defined
        if not grad_defined:
            raise NotImplementedError(f"grad_bwd not defined for {bsym.sym}")

        if grad_fwd is None:
            continue
        current_grads = grad_bwd(*current_grads, *residuals)
        current_grads = (current_grads,) if not isinstance(current_grads, Sequence) else current_grads

    grad_fwd_trace.output = current_grads

    # Resets tracing context
    reset_tracectx(trace_tok)

    return grad_fwd_trace


# TODO Test nesting of grad and grad and grad and grad
# TODO Test nesting of a regular function + calling grad
def grad(fn):
    cfn = compile(fn)

    @wraps(cfn)
    def _fn(*args, **kwargs):
        original_result, original_trace = cfn(*args, **kwargs)
        original_trace = last_traces(cfn)

        gradir = _grad_transform(original_trace)

        return original_result, original_trace

    return _fn
