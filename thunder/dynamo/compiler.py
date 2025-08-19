from __future__ import annotations
from functools import partial
from looseversion import LooseVersion
from typing import TYPE_CHECKING
import warnings
from pathlib import Path
import copy

import torch

from thunder.dynamo.utils import (
    recompile_graph,
    remove_empty_autocast,
    CompilerType,
    get_split_reasons_string,
    thunder_options_to_str,
    ProfileStats,
    ThunderAoTOptimizer,
    default_filter,
    default_optimizer,
    input_to_example_input_meta,
)
from thunder.dynamo.splitter import _splitter
from thunder.dynamo.benchmark_utils import ThunderCompileSpecification
from thunder.transforms.extraction_only_prologue_transform import ExtractionOnlyPrologueTransform

if TYPE_CHECKING:
    from thunder.dynamo.utils import SubgraphInfo
    from thunder.core.transform_common import Transform
    from thunder.core.trace import TraceCtx as Trace
    from os import PathLike
    from collections.abc import Callable


_DEFAULT_THUNDER_FUSION_TYPE = "dataflow"

# Split Autograd is disabled by default as
# it can lead to race conditions when using thunderFX + TE + FSDP
# leading to NCCL hang-up due to collective mismatch.
# TODO(kshitij12345): Investigate more and understand if the bug is in PyTorch or elsewhere.
_DEFAULT_THUNDERFX_DISABLE_SPLIT_AUTOGRAD = True


def _add_prologue_pruning(options: dict):
    """
    Add a transform to prune prologue checks to the list of transforms in the given options dictionary.

    Args:
        options: The dictionary of options to modify
    """
    transforms: list[Transform] | None = options.get("transforms", None)
    if transforms is None:
        transforms = []
    transforms.append(ExtractionOnlyPrologueTransform())
    options["transforms"] = transforms


class ThunderCompiler:
    def __init__(self, **thunder_options):
        """
        A class that compiles a :class:`torch.fx.GraphModule` to a :class:`thunder.ThunderModule`.
        This class is meant to be used as a backend for the :func:`torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to :func:`thunder.jit`.

        Example:
            >>> import torch
            >>> from thunder.dynamo import ThunderCompiler
            >>> backend = ThunderCompiler()
            >>> x = torch.ones(2, requires_grad=True)
            >>> @torch.compile(backend=backend)
            ... def func(x):
            ...     x = torch.sin(x)
            ...     if x.sum() > 0:
            ...         return x + 1
            ...     else:
            ...         return x - 1
            >>> out = func(x)
        """
        from thunder import jit

        if LooseVersion(torch.__version__) < LooseVersion("2.4.0"):
            # NOTE: PyTorch 2.3 or lower has bug in `split_module` function used in splitter.
            # See https://github.com/Lightning-AI/lightning-thunder/pull/1075#issuecomment-2324918409
            err_msg = f"thunder.jit as torch.compile backend is only supported with PyTorch version 2.4 or later, found version {torch.__version__}"
            raise RuntimeError(err_msg)

        # Thunder-compiled functions should be readily available for inspection
        # and testing, so we will store them in a list[SubgraphInfo]. The order of the
        # functions in the list will be the same as the order in which they were
        # compiled.
        # Ref to the documentation of `SubgraphInfo` to know more about the information it contains.
        self.subgraph_infos: list[SubgraphInfo] = []

        thunder_options["fusion_type"] = thunder_options.get("fusion_type", _DEFAULT_THUNDER_FUSION_TYPE)
        # NOTE: Dynamo already adds guards for modules by default (see flag `torch._dynamo.config.guard_nn_modules`), so thunder can avoid adding extra metadata checks for parameters
        #       in prologue.
        _add_prologue_pruning(thunder_options)
        thunder_options["thunderfx_disable_split_autograd"] = thunder_options.get(
            "thunderfx_disable_split_autograd", _DEFAULT_THUNDERFX_DISABLE_SPLIT_AUTOGRAD
        )
        self.thunder_options = thunder_options
        self._thunder_jit = partial(jit, **thunder_options)
        self._torch_compile = torch.compile

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        gm = remove_empty_autocast(gm)

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        recompile_graph(gm)

        # The whole graph may not be supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        split_module, subgraph_info = _splitter(gm, self._thunder_jit, self._torch_compile, sample_args)
        self.subgraph_infos.append(subgraph_info)
        return split_module

    def save_reproducer_to_folder(
        self,
        reproducer_folder: str | PathLike,
        use_pytest_benchmark: bool = False,
        serialize_inputs=False,
    ):
        """
        Save the reproducer script for the GraphModule executed by Thunder to the specified ``reproducer_folder``.
        Each saved script is named as "graph[graph_id]_thunder_[module_id]", where:

                - ``graph_id`` indexes the graph generated by Dynamo, which is then passed to Thunder.
                - ``module_id`` indexes the submodule split by the :func:`thunder.dynamo.utils._splitter`.

        Args:
            reproducer_folder: The folder where the reproducer code will be written. Can be specified as an absolute or relative path.
            use_pytest_benchmark: Determines the type of script to create. When :obj:`False`, create a reproducer script.
                Otherwise, creats a benchmark script to compare the reproducer's performance with other backends, including Torch eager, torch.compile.
        """
        if not self.subgraph_infos:
            raise TypeError(f"{self} doesn't seem to have been called yet.")
        reproducer_folder = Path(reproducer_folder)
        reproducer_folder.mkdir(exist_ok=True, parents=True)
        thunder_options_str = thunder_options_to_str(self.thunder_options)
        thunder_ex_str = f"partial(thunder.jit, {thunder_options_str})" if thunder_options_str else "thunder.jit"

        for graph_idx, subgraph_info in enumerate(self.subgraph_infos):
            thunder_module_names = []
            for node in subgraph_info.split_graph_module.graph.nodes:
                target = node.target
                if isinstance(target, str) and target.startswith("thunder_"):
                    thunder_module_names.append(f"graph{graph_idx}_{target}")
            original_thunder_modules = (
                m
                for m, compiled_m in subgraph_info.submodule_to_compiled_functions.items()
                if compiled_m.compiler == CompilerType.THUNDER
            )
            example_inputs = subgraph_info.thunder_compiled_fns_example_inputs
            from thunder.dynamo.report import FXReport

            result = FXReport(original_thunder_modules, thunder_module_names)
            split_reason_str = get_split_reasons_string(subgraph_info)
            for subgraph_idx, report in enumerate(result.fx_graph_reports):
                has_cuda_args = any(
                    hasattr(arg, "device") and arg.device.type == "cuda" for arg in example_inputs[subgraph_idx]
                )
                import_str = ["import thunder", "from functools import partial"]
                if has_cuda_args:
                    # Since Thunder compile options don't clearly indicate required imports,
                    # we include commonly used transforms by default.
                    import_str.extend(
                        [
                            "from thunder.transforms.cudagraph import CUDAGraphTransform",
                            "from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform",
                        ]
                    )

                compile_fn = ThunderCompileSpecification(**self.thunder_options)
                if not use_pytest_benchmark:
                    report.write_repro(
                        reproducer_folder,
                        file_name=f"{report.graph_name}_repro.py",
                        compile_fn=compile_fn,
                        check_consistency=True,
                        serialize_inputs=serialize_inputs,
                        inputs=example_inputs[subgraph_idx],
                        extra_comment_str=split_reason_str,
                    )
                    continue

                executor_names_list = ["thunder", "torch_inductor", "eager"]
                executors = [thunder_ex_str, "torch_inductor", "None"]

                if has_cuda_args:
                    executor_names_list.append("thunder_cudagraph")
                    executors.append("partial(thunder.jit, transform=CUDAGraphTransform())")

                report.write_pytest_benchmark(
                    reproducer_folder,
                    f"{report.graph_name}_benchmark.py",
                    executor_names_list,
                    executor_str=executors,
                    import_str=import_str,
                    serialize_inputs=serialize_inputs,
                    inputs=example_inputs[subgraph_idx],
                    extra_comment_str=split_reason_str,
                )


def thunderfx(fn: Callable, /, **kwargs) -> Callable:
    """Compiles a callable (function or model) by using Thunder as the backend of :func:`torch.compile`.

    Args:
        fn: A :class:`~torch.nn.Module` or a function to compile.
    Keyword Args:
        **kwargs: a dictionary of options to pass to :func:`torch.compile` and :func:`thunder.jit`.
    Returns:
        The compiled callable
    """
    import thunder

    from thunder.dynamo.utils import get_torch_compile_kwargs

    torch_compile_kwargs = get_torch_compile_kwargs(**kwargs)
    thunder_jit_kwargs = {k: v for k, v in kwargs.items() if k not in torch_compile_kwargs}

    backend = ThunderCompiler(**thunder_jit_kwargs)
    compiled = torch.compile(fn, backend=backend, **torch_compile_kwargs)

    # We return this object instead of just the raw `compiled` Callable so that
    # we have a place to hang the `last_*traces` properties.
    class CompiledObject:
        def __init__(self, be, func: Callable):
            self._backend = be
            self._func = func

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

        @property
        def last_traces(self) -> list[Trace]:
            """
            Get the Thunder traces for all the forward subgraphs of a ThunderFX
            callable.

            .. note:: The object must have been invoked before calling this
                      function.
            """
            rv: list[Trace] = []
            if not self._backend.subgraph_infos:
                warnings.warn("Must invoke the function before using last_traces")
            for sinfo in self._backend.subgraph_infos:
                for th_fqn in sinfo.thunder_compiled_fns:
                    trcs = thunder.last_traces(th_fqn)
                    if trcs != []:
                        rv.append(trcs[-1])
                    del trcs
            return rv

        @property
        def last_backward_traces(self) -> list[Trace]:
            """
            Get the Thunder traces for all the backward subgraphs of a
            ThunderFX callable.

            .. note:: The object must have been invoked before calling this
                      function.
            """
            rv: list[Trace] = []
            if not self._backend.subgraph_infos:
                warnings.warn("last_backward_traces used before function invoked")
            for sinfo in self._backend.subgraph_infos:
                for th_fqn in sinfo.thunder_compiled_fns:
                    trcs_bw = thunder.last_backward_traces(th_fqn)
                    if trcs_bw != []:
                        rv.append(trcs_bw[-1])
            return rv

    c = CompiledObject(backend, compiled)
    return c


def thunder_profile(fn):
    """
    This function returns a profiling callable that wraps the torch compiled function and profiling
    statistics. The statistics are stored in a :class:`thunder.dynamo.utils.ProfileStats` object,
    which can be accessed through the `_tao.id_to_profile_stats` attribute of the returned callable.

    For an example, see :func:`thunder.dynamo.compiler.thunder_optimize`
    """
    torch.compiler.reset()
    tao: ThunderAoTOptimizer = ThunderAoTOptimizer()

    def dispatching_backend(gm, *args):
        if tao.is_profiling:
            idx: int = id(gm)
            # The gm needs to be recorded during Torch JIT compilation,
            # because input information is discarded once compilation is complete.
            # We need this input information to infer the input range for each subgraph in the splitter.
            profile_stats = ProfileStats(gm=copy.deepcopy(gm))
            tao.id_to_profile_stats[idx] = profile_stats
            tao.id_to_gm_map[idx] = gm

            def record_stats(*args):
                input_meta = input_to_example_input_meta(args)
                tao.id_to_profile_stats[idx].input_meta_to_called_times[input_meta] += 1
                return gm(*args)

            tao.dispatch_map[idx] = record_stats

            def _dispatch(*args):
                return tao.dispatch_map[idx](*args)

            return _dispatch

        # NOTE When not in profiling mode, this just returns the gm for eager execution
        return gm

    cfn = torch.compile(fn, backend=dispatching_backend)

    # Wraps the torch compiled callable so we can invalidate the profiling
    #   callable for UX clarity
    def profiling_callable(*args, **kwargs):
        if tao.is_profiling:
            return cfn(*args, **kwargs)

        raise AssertionError("No longer profiling")

    profiling_callable._compiled_fn = cfn
    profiling_callable._tao = tao
    return profiling_callable


def thunder_optimize(
    profiling_callable,
    *,
    gm_filter=default_filter,
    optimizer=default_optimizer,
):
    """
    Optimizes the given profiling callable based on the statistics collected during profiling.

    Args:
        profiling_callable: The callable returned by :func:`thunder.dynamo.compiler.thunder_profile`
        gm_filter: Function that selects which graph modules to optimize
        optimizer: Function that performs the actual optimization on each graph module

    Returns:
        The torch compiled function

    Example:
        >>> import torch
        >>> from thunder.dynamo import thunder_profile
        >>> def foo(x):
        ...     return x + 1
        >>> x = torch.randn(2, 2)
        >>> pfoo = thunder_profile(foo)
        >>> pfoo(x)  # Profile with first input
        >>> stats = pfoo._tao.id_to_profile_stats  # Access collected statistics
        >>> optfoo = thunder_optimize(pfoo)
        >>> optfoo(x)  # Optimized with first input
    """

    # 1) Filters the FX graphs based on their statistics
    indices_to_optimize: set[int] = gm_filter(profiling_callable)

    # 2) Replaces the FX graphs that are to be optimized with the optimized
    #      callables, and other graphs with the graph modules
    tao = profiling_callable._tao
    dispatch_map = tao.dispatch_map
    id_to_gm_map = tao.id_to_gm_map
    for idx, call in dispatch_map.items():
        if idx in indices_to_optimize:
            gm = id_to_gm_map[idx]
            profile_stats = tao.id_to_profile_stats[idx]
            try:
                dispatch_map[idx] = optimizer(gm, profile_stats)
            except NotImplementedError:
                # Executes the gm eagerly if the optimizer can't optimize it
                dispatch_map[idx] = gm
        else:
            dispatch_map[idx] = id_to_gm_map[idx]

    # 3) Marks the profiling period as over
    tao.is_profiling = False

    # 4) Returns the actual torch.compile'd callable to minimize calling latency
    return profiling_callable._compiled_fn
