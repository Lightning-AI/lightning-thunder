from __future__ import annotations
from functools import partial
from looseversion import LooseVersion
from typing import TYPE_CHECKING
import warnings
import inspect
from pathlib import Path

import torch

from thunder.dynamo.utils import (
    recompile_graph,
    remove_empty_autocast,
    CompilerType,
    get_split_reasons_string,
    thunder_options_to_str,
)
from thunder.dynamo.splitter import _splitter
from thunder.core.utils import check
from thunder.dynamo.benchmark_utils import ThunderCompileSpecification
from thunder.transforms.extraction_only_prologue_transform import ExtractionOnlyPrologueTransform

if TYPE_CHECKING:
    from thunder.dynamo.utils import SubgraphInfo
    from thunder.core.transform_common import Transform
    from os import PathLike
    from collections.abc import Callable
    from typing import List

    CompilerFn = Callable[[torch.fx.GraphModule, list[torch.Tensor]], (Callable, str)]
    CompilerStrategy = Callable[[torch.fx.GraphModule, list[torch.Tensor]], CompilerFn]


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


def _torch_inductor_compile(gm, example_inputs) -> torch.nn.GraphModule:
    class InductorModuleWrapper(torch.nn.Module):
        def __init__(self, optimized_callable):
            super().__init__()
            self.optimized_callable = optimized_callable

        def forward(self, *args):
            return self.optimized_callable(*args)

    return InductorModuleWrapper(torch._inductor.compile(gm, example_inputs)), "inductor"


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
    """Compiles a callable (function or model) by using Thunder as the backend of :func:`torch.compile`
    Args:
        fn: A :class:`~torch.nn.Module` or a function to compile.
    Keyword Args:
        **kwargs: a dictionary of options to pass to :func:`torch.compile` and :func:`thunder.jit`.
    Returns:
        The compiled callable
    """
    import thunder

    from thunder.dynamo.utils import get_thunder_jit_kwargs, get_torch_compile_kwargs

    thunder_jit_kwargs = get_thunder_jit_kwargs(**kwargs)
    torch_compile_kwargs = get_torch_compile_kwargs(**kwargs)

    rest_kwargs = {k: v for k, v in kwargs.items() if k not in thunder_jit_kwargs and k not in torch_compile_kwargs}
    check(
        not rest_kwargs,
        lambda: f"There are kwargs that are not supported by either thunder.jit or torch.compile: {rest_kwargs}",
    )

    overlap = [kwarg_name for kwarg_name in thunder_jit_kwargs if kwarg_name in torch_compile_kwargs]
    check(
        not overlap,
        lambda: f"There are overlapping kwargs between thunder.jit and torch.compile: {overlap}",
        ValueError,
    )

    backend = ThunderCompiler(**thunder_jit_kwargs)
    compiled = torch.compile(fn, backend=backend, **torch_compile_kwargs)

    # We return this object instead of just the raw `compiled` Callable so that
    # we have a place to hang the `last_*traces` properties.
    class CompiledObject:
        def __init__(self, be, func: Callable):
            self._backend = backend
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


from thunder.dynamo.benchmark_utils import TorchInductorSpecification, WallTime
from thunder.dynamo.utils import input_to_example_input_meta


def default_compile_strategy(gm, example_inputs) -> CompilerFn:
    from thunder.dynamo.report import FXGraphReport

    example_input_metadata = input_to_example_input_meta(example_inputs)
    report = FXGraphReport(gm, "gm", example_input_metadata)
    thunderjit = ThunderCompileSpecification()
    torchinductor = TorchInductorSpecification()
    m1 = report.run_benchmark(thunderjit, WallTime, reset_torch_dynamo=False)
    m2 = report.run_benchmark(torchinductor, WallTime, reset_torch_dynamo=False)
    if sum(m.median for m in m1 if m is not None) > sum(m.median for m in m2 if m is not None):
        print("use torch.compile")
        return _torch_inductor_compile
    else:
        print("use thunder.jit")
        from thunder import jit

        return lambda *args: (jit(args[0]), "thunder")


from thunder.dynamo.utils import (
    ProfileStats,
    update_node_and_submodule,
    _get_example_inputs_from_placeholder,
)


class ThunderAoTOptimizer:
    # Helper class that just keeps some data about the AoT optimization process
    # We could use better statistics than what's in this example!

    def __init__(
        self,
    ):
        self.is_profiling = True
        self.dispatch_map: dict = {}  # gm_id -> callable
        self.id_to_gm_map: dict = {}
        self.gm_to_profile_stats = {}


import copy
from thunder import jit


def thunder_profile(fn):
    torch.compiler.reset()
    tao: ThunderAoTOptimizer = ThunderAoTOptimizer()

    def record_stats_jit(gm, *args):
        print("profiling compile: ", id(gm))
        placeholders = list(n for n in gm.graph.nodes if n.op == "placeholder")
        example_input_metadata = list(
            map(partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders)
        )
        profile_stats = ProfileStats(gm=copy.deepcopy(gm), example_inputs=example_input_metadata, called_times=0)
        tao.gm_to_profile_stats[gm] = profile_stats

    def dispatching_backend(gm, *args):
        if tao.is_profiling:
            idx: int = id(gm)
            # the information needs to be recorded during torch jit compilation.
            # when compiling is over the input information is removed
            record_stats_jit(gm, *args)

            def record_stats_execution(*args):
                # Records statistics about the FX Graph
                # In this example, just appends the args and separately records the gm,
                #   but we should probably record the graph, the # of times the graph has
                #   been called, and the metadata of the args, and not record redundant
                #   input metadata
                # tao.stats[idx].append(args)
                print("profiling execution: ", id(gm))
                tao.id_to_gm_map[idx] = gm
                tao.gm_to_profile_stats[gm].called_times += 1
                return gm(*args)

            # record the gm and the callable
            tao.dispatch_map[idx] = record_stats_execution

            def _dispatch(*args):
                return tao.dispatch_map[idx](*args)

            return _dispatch

        # NOTE When not in profiling mode, this just returns the gm for eager
        #   execution
        return gm

    cfn = torch.compile(fn, backend=dispatching_backend)

    # Wraps the torch compiled callable so we can invalidate the profiling
    #   callable for UX clarity
    def profiling_callable(*args, **kwargs):
        if tao.is_profiling:
            return cfn(*args, **kwargs)

        raise AssertionError(f"No longer profiling")

    profiling_callable._compiled_fn = cfn
    profiling_callable._tao = tao
    return profiling_callable


def default_filter(fn) -> set[int]:
    # Default filtering function placeholder
    # Takes the FX graph statistics, and returns a subset of
    # FX graphs to optimize
    # TODO The statistics should be an attribute of the profiling fn passed to this

    # Currently this just returns every FX Graph index, but the proposal suggests
    #   that we should start by filtering FX Graphs that include symbolic values
    #   in their args
    from torch._inductor.utils import is_symbolic

    id_to_gm = fn._tao.id_to_gm_map

    choosen = set()
    for idx, gm in id_to_gm.items():
        dynamic = False
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for placeholder in placeholders:
            example_value = placeholder.meta.get("example_value", None)
            if example_value is None:
                continue

            if is_symbolic(example_value):
                dynamic = True
                break
        if not dynamic:
            choosen.add(idx)
    return choosen


def default_optimizer(gm, stats):
    # Default optimizing function placeholder
    # Takes a gm and its statistics, and returns a (possibly optimized) callable

    # Currently this just ignores the statistics and doesn't optimize anything
    # This is where the gm could be benchmarked with different executors, and
    #   a callable generated that uses the fastest executor
    # The real magic goes here!
    import copy

    cur_gm = remove_empty_autocast(stats.gm)
    recompile_graph(cur_gm)
    _, subgraph_info = _splitter(cur_gm, jit, torch.compile, _unused_sample_args=None)

    thunder_split_gm = subgraph_info.original_split_graph_module
    split_gm = copy.deepcopy(thunder_split_gm.split_graph_module)
    for node in split_gm.graph.nodes:
        if not node.name.startswith("submod"):
            continue
        graph_module = getattr(split_gm, node.name)
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        example_inputs = [_get_example_inputs_from_placeholder(p, only_metadata=False) for p in placeholders]
        if thunder_split_gm.is_thunder_supported_partition(node):
            compiler_fn = default_compile_strategy(graph_module, example_inputs)
            optimized_module, compiler_name = compiler_fn(graph_module, example_inputs)
            # Update the node name from "submod_*" to the optimizer specifed name for more user-friendly names
            update_node_and_submodule(split_gm, node, node.name.replace("submod", compiler_name), optimized_module)
            # print(split_gm.__repr__())
        else:  # For inductor
            optimized_module, compiler_name = _torch_inductor_compile(graph_module, example_inputs)
            # Update the node name from "submod_*" to "inductor_*" for more user-friendly names
            update_node_and_submodule(split_gm, node, node.name.replace("submod", "inductor"), optimized_module)

    recompile_graph(split_gm)
    print(split_gm.__repr__())
    print(thunder_split_gm.split_graph_module.__repr__())
    return split_gm


def thunder_optimize(
    profiling_callable,
    *,
    gm_filter=default_filter,
    optimizer=default_optimizer,
):
    # NOTE Assumes fn is a "profiling_callable" as returned from thunder_profile

    # Resets the dispatch table so that no more profiling occurs

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
            # dispatch_map[idx] = optimizer(gm, stats[idx])
            profile_stats = tao.gm_to_profile_stats[gm]
            dispatch_map[idx] = optimizer(gm, profile_stats)
        else:
            dispatch_map[idx] = gm

    # 3) Marks the profiling period as over
    tao.is_profiling = False

    # 4) Returns the actual torch.compile'd callable to minimize calling latency
    return profiling_callable._compiled_fn
