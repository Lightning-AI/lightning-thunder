from functools import wraps
from typing import Any
from collections import defaultdict, namedtuple
from collections.abc import Callable
from collections.abc import Sequence
from contextvars import ContextVar
import os
import dis
import time
import warnings

from looseversion import LooseVersion

from thunder.core.module import ThunderModule
from thunder.core.interpreter import InterpreterLogItem
from thunder.core.options import (
    CACHE_OPTIONS,
    SHARP_EDGES_OPTIONS,
    DebugOptions,
)
from thunder.core.trace import (
    TraceResults,
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
    is_tracing,
)

import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.transform_common import (
    dce,
    Transform,
    wrap_return_value_together_with_arguments,
    unwrap_return_value,
    remove_context_manager_prims_from_trace,
)
from thunder.core.functionalization import (
    check_inplace_to_views,
    functionalize_inplace_ops,
)
from thunder.core.recipe import Recipe, Lookaside
from thunder.common import (
    CompileData,
    CompileStats,
    trace,
    transform_for_execution,
    transform_to_torch_types,
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor
from thunder.core.compile_data import compile_data_and_stats, get_compile_data
from thunder.core.langctxs import LanguageContext
import thunder.core.langctxs as langctxs
from thunder.core.symbol import has_tags
from thunder.core.baseutils import run_once, check
from thunder.core.codeutils import Positions
from thunder.core.proxies import (
    Proxy,
    TensorProxy,
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
from thunder.core.interpreter import print_interpreter_log, print_to_log
from thunder.core.jit_ext import thunder_general_jit
from thunder.executors.torch_autograd import split_forward_backward, ThunderFunction

# NOTE This import is intentionally pytorch so that it thunder.torch doesn't import this
import torch as pytorch

import thunder.clang as clang
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map

# Imports executors (to populate default executors and make them accessible)
import thunder.executors.pythonex
import thunder.executors.torchex
import thunder.executors.nvfuserex

pythonex = extend.get_executor("python")
assert pythonex is not None


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
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
    # language aliases
    "torch",
    "numpy",
    "prims",
    # interface functions
    # TODO Extend this
    # TODO Add device aliases
    # TODO Add executor aliases
    "cudnn_executor",
    "sdpa_executor",
    "nvfuser_executor",
    "pytorch_executor",
    # debugging functions
    "DebugOptions",
    "set_execution_callback_file",
    "jit",
    "resolve_executors",
    "add_executor_lists",
    "get_executor",
    "get_all_executors",
    "get_default_executors",
    "get_always_executors",
    "compile_data",
    "compile_stats",
    "last_traces",
    "last_backward_traces",
    "cache_option",
    "cache_hits",
    "cache_misses",
    "list_transforms",
    "last_interpreter_log",
    "last_interpreted_instructions",
    "print_last_interpreter_log",
    "last_compile_options",
    "get_auto_registered_torch_op_names",
    "grad",
]


from thunder.__about__ import *  # import all


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
float8_e5m2 = dtypes.float8_e5m2
float8_e5m2fnuz = dtypes.float8_e5m2fnuz
float8_e4m3fn = dtypes.float8_e4m3fn
float8_e4m3fnuz = dtypes.float8_e4m3fnuz
float16 = dtypes.float16
float32 = dtypes.float32
float64 = dtypes.float64
complex32 = dtypes.complex32
complex64 = dtypes.complex64
complex128 = dtypes.complex128

#
# Promoted executor-related functions and objects
#

# TODO Add more of these functions
resolve_executors = extend.resolve_executors
add_executor_lists = extend.add_executor_lists
get_executor = extend.get_executor
get_all_executors = extend.get_all_executors
get_default_executors = extend.get_default_executors
get_always_executors = extend.get_always_executors

cudnn_executor: None | extend.Executor = extend.get_executor("cudnn")
sdpa_executor: None | extend.Executor = extend.get_executor("sdpa")
torchcompile_cat_executor: None | extend.Executor = extend.get_executor("torchcompile_cat")
apex_executor: None | extend.Executor = extend.get_executor("apex")
nvfuser_executor: None | extend.Executor = extend.get_executor("nvfuser")
pytorch_executor: None | extend.Executor = extend.get_executor("torch")

# Default executor list is [cudnn -> sdpa -> torchcompile_cat -> nvfuser -> torch -> python]
# Note that add_default_executor inserts executor at start of list, hence the reverse order below.
if nvfuser_executor:
    add_default_executor(nvfuser_executor)

if torchcompile_cat_executor and pytorch._dynamo.is_inductor_supported():
    add_default_executor(torchcompile_cat_executor)

if sdpa_executor:
    add_default_executor(sdpa_executor)

if cudnn_executor:
    add_default_executor(cudnn_executor)

if apex_executor:
    add_default_executor(apex_executor)

#
# Promoted debugging functions
#

# If set, Python programs will be written to this file before being executed, and if the
#   the file is modified then the modified version of the program will be compiled and executed, instead.
from thunder.core.trace import _set_execution_file

set_execution_callback_file = _set_execution_file


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


CacheEntry = namedtuple(
    "CacheEntry",
    [
        "prologue_fn",
        "prologue_traces",
        "computation_fn",
        "computation_traces",
        "epilogue_fn",
        "epilogue_traces",
        "backward_fn",
        "backward_traces",
        "return_none_instead_of_grads",
    ],
)


def compile(fn: Callable, recipe: Recipe | None):
    if recipe is None:
        return thunder.jit(fn)

    return recipe.apply(fn)


# This function will replace compile() (below) before RC1
# TODO RC1 Consider renaming compile_options to additional_compile_options
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any | LanguageContext = None,
    executors: None | Sequence[Executor | str] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    disable_torch_autograd: bool = False,  # TODO Revisit this UX for RC1
    transforms: list[Transform] | None = None,
    debug_options: DebugOptions | None = None,
    **compile_options,  # TODO RC1 Make this explicit -- dict of options
) -> Callable:
    """Just-in-time compile a callable (function or model).

    .. note::

        Thunder's support of PyTorch in-place support is experimental.
        Thunder functionalizes in-place ops and adds required tensor copies.
        The functionalization can be turned off with the kwarg of ``skip_inplace_functionalization``.
        See :func:`thunder.core.functionalization.functionalize_inplace_ops`
        for the details.

    Args:
        fn: A :class:`~torch.nn.Module` or a function to compile.
    Keyword Args:
        langctx: the language context, which language / library to emulate. default: "torch" for PyTorch compatibility.
        executors: list of executors to use. Defaults to the executors returned by :func:`thunder.extend.get_default_executors` and always amended by :func:`thunder.extend.get_always_executors`.
                   You can get a list of all available executors with :func:`thunder.get_all_executors`. You can also pass the name of an executor that's been registered, and it will be resolved with :func:`thunder.extend.get_executor`.
        sharp_edges: sharp edge detection action. What to do when thunder detects a construct that is likely to lead to errors. Can be ``"allow"``, ``"warn"``, ``"error"``. Defaults to ``"allow"``.
        cache: caching mode. default: ``"constant values"```

               - ``"no caching"`` - disable caching and always recompute,
               - ``"constant values"`` - require Tensors to be of the same shape, device, dtype etc., and integers and strings to match exactly,
               - ``"same input"`` - don't check, but just assume that a cached function works if it exists.

        transforms: optional list of transforms to be applied. It should be a list of instances of :class:`thunder.core.transforms.Transform`. Default: ``None``

        debug_options: optional :class:`thunder.DebugOptions` instance. See the doc string of :class:`DebugOptions` for supported debug options. Default: ``None``
    """

    if "executors_list" in compile_options:
        warnings.warn("outdated argument executors_list= in call, please use executors=")
        if executors is None:
            executors = compile_options.pop("executors_list")

    if "early_transforms" in compile_options:
        raise RuntimeError("early_transforms= has been absorbed by transforms=")

    if compile_options.get("use_cudagraphs") is not None:
        raise RuntimeError("use_cudagraphs is replaced by using thunder.transforms.CUDAGraphTransform")

    if transforms is None:
        transforms = []

    # Resolve names of executors
    executors = resolve_executors(executors)
    ad_hoc_executor = extend.AdHocExecutor()
    executors = (ad_hoc_executor, *executors)

    # TODO: verify that tutorials don't have false positives and enable warning by default
    # # Make sharp_edges == warn default if not supplied and if in the general jit
    # if interpretation is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON and sharp_edges is None:
    #     sharp_edges = SHARP_EDGES_OPTIONS.WARN

    executor_lookasides = {}
    for ex in executors:
        # TODO: sharp edge if lookasides are shadowed?
        executor_lookasides.update(ex._lookasides)

    # TODO RC1 Refine the compile data option to remove unused options
    # TODO: refine options
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors,
        cache_option=cache,
        sharp_edges=sharp_edges,
        using_jit=True,
        disable_torch_autograd_support=disable_torch_autograd,
        only_execute_prims=False,
        disable_preprocessing=True,
        compile_options=compile_options,
        executor_lookasides=executor_lookasides,
        debug_options=debug_options,
    )
    cs = CompileStats()

    def _alias_tensor_of_args_kwargs_dict(*args, **kwargs) -> dict[int, list[int]]:
        flat_args, _ = tree_flatten((args, kwargs))
        data_ptr_to_tensor_group_index = {}
        tensor_group_index_to_tensor_indices = defaultdict(list)
        for idx, t in enumerate(flat_args):
            if pytorch.is_tensor(t) and t.layout == pytorch.strided:
                data_ptr = t.untyped_storage().data_ptr()
                if data_ptr not in data_ptr_to_tensor_group_index:
                    data_ptr_to_tensor_group_index[data_ptr] = len(data_ptr_to_tensor_group_index)
                tgi = data_ptr_to_tensor_group_index[data_ptr]
                tensor_group_index_to_tensor_indices[tgi].append(idx)
        return tensor_group_index_to_tensor_indices

    def _alias_tensor_of_args_kwargs(*args, **kwargs) -> str:
        """If no aliases found, empty string, otherwise, aliases are comma separated, groups are hyphen separated."""

        alias_indices = []
        for k, v in _alias_tensor_of_args_kwargs_dict(*args, **kwargs).items():
            if len(v) > 1:
                s = ",".join(f"{i}" for i in v)
                alias_indices.append(s)
        if not alias_indices:
            return ""
        return "-".join(alias_indices)

    @langctxs.langctx(cd.langctx)
    @_with_cache_info_ctx
    def get_computation_and_inputs(*args, **kwargs):
        # set up a record of things in the current environment that impact caching / prologues
        # this could be replaced by the respective querying in the prologues
        cache_info = _get_cache_info()

        # default dtype (for factory functions)
        cache_info["default_dtype"] = pytorch.get_default_dtype()

        # default device (for factory functions)
        cache_info["default_device"] = pytorch.get_default_device()

        # autocast related operations
        is_autocast_enabled = False
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
            cache_info.update(autocast_thunder_dtype=str(autocast_thunder_dtype))
            device = "cuda" if pytorch.is_autocast_enabled() else "cpu"
            dtype = autocast_thunder_dtype
            cd.autocast_stack.push(device, dtype, is_autocast_enabled)

        cache_info["is_autocast_enabled"] = is_autocast_enabled

        is_ddp_enabled = getattr(fn, "use_ddp", False)
        is_fsdp_enabled = getattr(fn, "use_fsdp", False)
        no_grad_sync = False
        if is_ddp_enabled or is_fsdp_enabled:
            from thunder.distributed import get_skip_data_parallel_grad_sync

            no_grad_sync = get_skip_data_parallel_grad_sync()
        cache_info["no_grad_sync"] = no_grad_sync
        return_none_instead_of_grads = is_fsdp_enabled and no_grad_sync

        # NOTE(crcrpar): If a callable is free from in-place ops whose operand is args and/or their views
        # alaises wouldn't matter, thus it'd be better to nullify this entry in such cases.
        # It however would require the functionalized computation trace to interact with `cache_info`,
        # which seems to break the consistency of cache_info, leading to a failure in cache_info check.
        cache_info["alias_tensor_indices"] = _alias_tensor_of_args_kwargs(*args, **kwargs)

        # Store the `is_grad_enabled` state of PyTorch. This is used by vjp transform
        # to treat certain Symbols as constant.
        cache_info["is_grad_enabled"] = pytorch.is_grad_enabled()
        cd.is_grad_enabled = pytorch.is_grad_enabled()

        # TODO RC1 Add module and function checks to prologue (make it a compile option)

        # Checks cache
        cs.last_trace_cache_start = time.perf_counter_ns()
        if (cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES) or (cd.cache_option is CACHE_OPTIONS.SYMBOLIC_VALUES):
            for cache_entry in reversed(cs.interpreter_cache):
                with compile_data_and_stats(cd, cs):
                    (
                        pro,
                        pro_traces,
                        comp,
                        comp_traces,
                        epilogue,
                        epilogue_traces,
                        backward_fn,
                        backward_traces,
                        _return_none_instead_of_grads,
                    ) = cache_entry
                    try:
                        inps, pro_to_epi = pro(*args, **kwargs)
                    except Exception as _:
                        continue

                    # Updates cache statistics
                    cs.cache_hits += 1
                    cs.last_traces = comp_traces
                    cs.last_interpreted_instructions = None
                    cs.last_interpreter_log = None
                    cs.last_prologue_traces = pro_traces
                    cs.last_prologue = pro
                    cs.last_prologue_transformation_start = 0
                    cs.last_prologue_transformation_stop = 0
                    cs.last_computation_transformation_start = 0
                    cs.last_computation_transformation_stop = 0

                    return cache_entry, inps, pro_to_epi

        if cd.cache_option is CACHE_OPTIONS.SAME_INPUT:
            if len(cs.interpreter_cache):
                cache_entry = cs.interpreter_cache[0]
                (
                    pro,
                    pro_traces,
                    comp,
                    comp_traces,
                    epilogue,
                    epilogue_traces,
                    backward_fn,
                    backward_traces,
                ) = cache_entry

                inps, pro_to_epi = pro(*args, **kwargs)

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreter_log = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro

                return cache_entry, inps, pro_to_epi

        cs.cache_misses += 1
        cs.last_trace_cache_stop = time.perf_counter_ns()

        # Resets use of compile flags
        cs.last_compile_reasons = defaultdict(list)

        with compile_data_and_stats(cd, cs):
            # Acquires the trace OR inlines the trace into an existing trace and
            #   returns the (proxied) result of the operation
            cs.last_trace_tracing_start = time.perf_counter_ns()

            prologue_trc: TraceCtx
            computation_trc: TraceCtx
            jit_results: TraceResults = thunder_general_jit(
                fn,
                args,
                kwargs,
                ad_hoc_executor=ad_hoc_executor,
                sharp_edges=cd.sharp_edges,
            )
            prologue_trc = jit_results.prologue_trace
            computation_trc = jit_results.computation_trace
            epilogue_trc = jit_results.epilogue_trace
            last_interpreter_log = jit_results.interpreter_log

            prologue_traces = [prologue_trc]
            computation_traces = [computation_trc]

            computation_trc = wrap_return_value_together_with_arguments(computation_trc)
            computation_traces.append(computation_trc)

            computation_trc = remove_context_manager_prims_from_trace(computation_trc)
            computation_traces.append(computation_trc)

            orig_to_view_swap_map = check_inplace_to_views(computation_trc)
            if not compile_options.get("skip_inplace_functionalization", False):
                alias_tensor_indices = []
                if alias_tensor_indices_str := cache_info["alias_tensor_indices"]:
                    alias_tensor_indices: list[list[int]] = [
                        [int(i) for i in s.split(",")] for s in alias_tensor_indices_str.split("-")
                    ]
                computation_traces.extend(
                    functionalize_inplace_ops(
                        computation_trace=computation_trc,
                        orig_to_view_swap_map=orig_to_view_swap_map,
                        alias_tensor_indices=alias_tensor_indices,
                    )
                )
                computation_trc = computation_traces[-1]

            epilogue_traces = [epilogue_trc]

            cs.last_trace_tracing_stop = time.perf_counter_ns()

            # Makes the prologue callable
            cs.last_prologue_transformation_start = time.perf_counter_ns()

            transform: Transform
            for transform in transforms:
                thunder.core.utils.check_type(transform, Transform)
                new_prologue_trc, new_computation_trc, new_epilogue_trc = transform.transform_traces_pre_prologue(
                    prologue_trc, computation_trc, epilogue_trc, executors_list=cd.executors_list
                )
                # if the transform did anything in the transform_traces_pre_prologue step
                if (
                    new_prologue_trc is not prologue_trc
                    or new_computation_trc is not computation_trc
                    or new_epilogue_trc is not epilogue_trc
                ):
                    prologue_trc, computation_trc, epilogue_trc = (
                        new_prologue_trc,
                        new_computation_trc,
                        new_epilogue_trc,
                    )
                    prologue_traces.append(prologue_trc)
                    computation_traces.append(computation_trc)
                    if epilogue_trc is not None:
                        epilogue_traces.append(epilogue_trc)

            prologue_traces += transform_for_execution(
                prologue_trc,
                executors_list=(pythonex,),
                use_del_last_used=False,
            )
            prologue_trc = prologue_traces[-1]
            pro = prologue_trc.python_callable(include_decorators=False)
            pro = prologue_execution_timer(pro)

            epilogue_trc = transform_to_torch_types(epilogue_trc)
            epilogue = epilogue_trc.python_callable()

            cs.last_prologue_transformation_stop = time.perf_counter_ns()
            cs.last_prologue_traces = prologue_traces
            cs.last_prologue = pro
            cs.last_traces = computation_traces
            cs.last_epilogue_traces = epilogue_traces
            backward_traces = []
            cs.last_backward_traces = backward_traces
            cs.last_interpreter_log = last_interpreter_log
            cs.last_interpreted_instructions = (i for i in last_interpreter_log if isinstance(i, dis.Instruction))

            inps, pro_to_epi = pro(*args, **kwargs)

            computation_trc = dce(computation_trc)
            computation_traces.append(computation_trc)

            backward_trc = None
            if not cd.disable_torch_autograd_support:
                tensor_cls = (pytorch.Tensor, TensorProxy)
                requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in inps)

                if requires_grad:
                    # Currently split_forward_backward also includes
                    # transform_for_execution and various sorting of symbols,
                    # applying transform_for_execution after this would be
                    # breaking the order of operations
                    computation_trc, backward_trc = split_forward_backward(computation_trc, cd, cs, *inps)
                    # Note computation_trc and backward_trc have been appended to cs.last_(backward_)traces
                    # by split_forward_backward

            if backward_trc is None:
                from thunder.executors.passes import transform_for_execution as transform_for_execution_pass
                from thunder.executors.passes import _transform_for_operator_executor_execution
                from thunder.distributed.utils import maybe_sort_waits

                tmp_comp_trc = _transform_for_operator_executor_execution(computation_trc, cd.executors_list)
                is_transformed, tmp_comp_trc = maybe_sort_waits(tmp_comp_trc)
                if is_transformed:
                    computation_trc = tmp_comp_trc
                    computation_traces.append(computation_trc)

                extraces = transform_for_execution(
                    computation_trc,
                    executors_list=cd.executors_list,
                    use_del_last_used=False,
                )
                computation_traces.extend(extraces)
                computation_trc = computation_traces[-1]
                computation_trc = thunder.executors.passes.del_last_used(computation_trc)

            if not compile_options.get("disable_inplace_copy_check", False):
                thunder.core.transform_common._inplace_copy_sanity_check(computation_trc)
                computation_traces.append(computation_trc)

            for transform in transforms:
                # NOTE: `backward_trc` could be None.
                new_computation_trc = transform.transform_trace_post_optimization(
                    computation_trc, executors_list=cd.executors_list
                )
                if new_computation_trc is not computation_trc:
                    computation_trc = new_computation_trc
                    computation_traces.append(computation_trc)
                if backward_trc is not None:
                    new_backward_trc = transform.transform_trace_post_optimization(
                        backward_trc, executors_list=cd.executors_list
                    )
                    if new_backward_trc is not backward_trc:
                        backward_trc = new_backward_trc
                        backward_traces.append(backward_trc)

            if backward_trc is not None:
                backward_fn = backward_trc.python_callable()
            else:
                backward_fn = None
                # We do not have to return auxiliary tensors, which will only be useful in backward pass
                computation_trc = unwrap_return_value(computation_trc)
                computation_traces.append(computation_trc)

            computation_trc = transform_to_torch_types(computation_trc)
            comp = computation_trc.python_callable()

            # TODO RC1 Update the cache
            cache_entry = CacheEntry(
                pro,
                prologue_traces,
                comp,
                computation_traces,
                epilogue,
                epilogue_traces,
                backward_fn,
                backward_traces,
                return_none_instead_of_grads,
            )
            if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
                cs.interpreter_cache.append(cache_entry)

        return cache_entry, inps, pro_to_epi

    def host_execution_timer(fn):
        def wrapped(*args, **kwargs):
            cs.last_trace_host_execution_start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                cs.last_trace_host_execution_stop = time.perf_counter_ns()

        return wrapped

    def prologue_execution_timer(fn):
        def wrapped(*args, **kwargs):
            cs.last_prologue_execution_start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                cs.last_prologue_execution_stop = time.perf_counter_ns()

        return wrapped

    def decorate_computation_function(get_computation_and_inputs_fn, *decorators):
        def wrapped(*args, **kwargs):
            cache_entry, inps, pro_to_epi = get_computation_and_inputs_fn(*args, **kwargs)
            decorated_computation_fn = cache_entry.computation_fn
            for decorator in decorators:
                decorated_computation_fn = decorator(decorated_computation_fn)
            if decorators:
                cache_entry = cache_entry._replace(computation_fn=decorated_computation_fn)
            return cache_entry, inps, pro_to_epi

        return wrapped

    get_computation_and_inputs = decorate_computation_function(get_computation_and_inputs, host_execution_timer)
    cd.get_computation_and_inputs = get_computation_and_inputs

    def update_call_statistics(fn):
        def wrapped(*args, **kwargs):
            cs.calls += 1
            cs.last_trace_host_start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                cs.last_trace_host_stop = time.perf_counter_ns()

        return wrapped

    def maybe_connect_to_autograd(cache_entry, result):
        if cache_entry.backward_fn:
            # If the backward function is available, we need to connect the
            # resulting tensors to PyTorch's Autograd graph using the
            # ThunderFunction (which is a torch.autograd.Function subclass)
            data_for_autograd, (saved_tensors, saved_other) = result
            ThunderFunction.apply(
                cache_entry.return_none_instead_of_grads,
                cache_entry.backward_fn,
                saved_tensors,
                saved_other,
                data_for_autograd["flat_output"],
                *data_for_autograd["flat_args"],
            )
            result = data_for_autograd["output"]

        return result

    def call_epilogue(cache_entry, comp_result, pro_to_epi):
        result = cache_entry.epilogue_fn(*pro_to_epi, *comp_result)
        return result

    @wraps(fn)
    @update_call_statistics
    def fn_(*args, **kwargs) -> Any:
        if is_tracing():
            _recursive_jit_call_warning()
            return fn(*args, **kwargs)

        cache_entry, inps, pro_to_epi = get_computation_and_inputs(*args, **kwargs)

        result = cache_entry.computation_fn(*inps)
        result = maybe_connect_to_autograd(cache_entry, result)
        result = call_epilogue(cache_entry, result, pro_to_epi)

        cs.last_computation = cache_entry.computation_fn
        return result

    if isinstance(fn, pytorch.nn.Module):
        fn_ = ThunderModule(fn, fn_)
        cd._thunder_module_map[id(fn)] = fn_

    # Sets compile options and statistics attributes
    cd._get_computation_and_inputs = get_computation_and_inputs
    fn_._lc_cd = cd
    fn_._lc_cs = cs

    if isinstance(fn, pytorch.nn.Module):
        fn_._lc_transforms = []
        for transform in transforms:
            transform.transform_module(fn_)
            fn_._lc_transforms.append(transform)
    else:
        # todo: move to compile data
        fn_._lc_transforms = transforms[:]

    return fn_


def compile_data(fn) -> CompileData | None:
    """Obtains the compilation data from a JITed function.

    The compile data (:class:`thunder.common.CompileData`) contains information about how the JIT has been configured
    for compilation (including referencing the function or module that is being compiled).
    """
    return getattr(fn, "_lc_cd", None)


def compile_stats(fn) -> CompileStats | None:
    """Obtains the compilation statistics from a JITed function.

    The compilation statistics (:class:`thunder.common.CompileStats`) contain information about each compilation run -
    collected when a JITed function is called for the first time or with previously unseen state.
    This includes the cache of traces (pologues, computation, possibly backward and epilogue) and
    how they have been transformed and information about cache hits and misses and timings.
    """
    return getattr(fn, "_lc_cs", None)


def last_traces(fn) -> list[TraceCtx]:
    """Obtains the list of computation traces that have been produced for the last run of the function. This is a list
    of traces mirroring the progression of transformations being applied to the trace (at index 0) that has
    been acquired from interpreting the user program.

    If the function has forward and backward, the forward is returned.
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_traces


def last_backward_traces(fn) -> list[TraceCtx]:
    """Obtains the list of backward traces that have been produced for the last run of the function and the selected prologue."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_backward_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_backward_traces


def last_prologue_traces(fn) -> TraceCtx:
    """Obtains the list of prologue traces that have been produced for the last run of the function and the selected prologue."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_prologue_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_prologue_traces


def cache_option(fn) -> CACHE_OPTIONS:
    """Returns the cache options set when JITting the function."""
    cd = compile_data(fn)
    if cd is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cd.cache_option


def cache_hits(fn) -> int:
    """Returns the number of cache hits we found when running the function."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_hits


def cache_misses(fn) -> int:
    """Returns the number of cache misses we found when running the function."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_misses


def list_transforms(fn) -> list:
    """Returns the list of (explicit) transforms applied to the JITed function."""
    return fn._lc_transforms


def last_interpreter_log(fn: Callable) -> list[InterpreterLogItem]:
    """Returns the list of instructions and other information the interpreter encountered while tracing through the
    user program (on the last cache miss).
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreter_log is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreter_log


def last_interpreted_instructions(fn: Callable) -> list[dis.Instruction]:
    """Returns the list of instructions the interpreter encountered while tracing through the
    user program (on the last cache miss).
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_instructions is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return list(cs.last_interpreted_instructions)


def print_last_interpreter_log(
    fn: Callable,
    /,
    print_fn: Callable = print,
    use_colors: bool = True,
    indent: bool = True,
    max_depth: int | None = None,
    color_internals: bool = False,
    print_source_code: bool = True,
) -> None:
    """Prints a log of the last run of the interpreter for the given function.

    Args:
        fn: The function returned by :func:`thunder.jit` to print the last interpreter run log for. The function must have been called at least once first.
        print_fn: The function to use for printing. Defaults to builtin `print`.
        use_colors: Whether to use colors in the output. Defaults to `None`, which attempts to autodetect if the terminal supports ANSI color.
        indent: Whether to indent the output with function scope. Defaults to :obj:`True`.
        max_depth: The maximum indentation depth of the output. Doesn't print log items nested deeper than the max depth. Defaults to :obj:`None`, which means no limit.
        color_internals: Whether to color instructions implicitly interpreted by other instructions. Defaults to :obj:`False`, so that only the instructions in the user's code are highlighted in color.
        print_source_code: Whether to print the source line below each LineLogItem in the log. Defaults to :obj:`True`.
    """
    log = last_interpreter_log(fn)
    print_interpreter_log(
        log,
        print_fn=print_fn,
        use_colors=use_colors,
        indent=indent,
        max_depth=max_depth,
        color_internals=color_internals,
        print_source_code=print_source_code,
    )


def last_compile_options(fn: Callable, /) -> None:
    """Prints how compiled options were used (or not)"""

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


def get_auto_registered_torch_op_names(fn: Callable, /) -> set[str] | None:
    """Returns a set of auto-registered Torch operator names present in the given JIT-compiled function."""
    trc = last_traces(fn)[0]
    return {bsym.sym.id for bsym in trc.bound_symbols if has_tags(bsym, {prims.OpTags.AUTO_REGISTERED})}


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
