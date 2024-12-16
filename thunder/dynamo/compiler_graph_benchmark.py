from __future__ import annotations
from itertools import chain
from pytest_benchmark.fixture import BenchmarkFixture
from typing import TYPE_CHECKING
from looseversion import LooseVersion

import torch
from thunder.dynamo import ThunderCompiler
from thunder.dynamo.utils import _get_example_inputs_from_placeholder
from thunder.core.utils import check


if TYPE_CHECKING:
    from collections.abc import Callable

GRAPH_BY_GRAPH_BENCHMARK_PARAMS_KEYS = ("GraphID", "SplitModuleName", "executor")


class ThunderCompilerGraphBenchmarking(ThunderCompiler):
    _executors = (
        "eager",
        "inductor",
        "thunder",
    )

    def __init__(
        self,
        bench: BenchmarkFixture,
        executors: dict[str, Callable],
        **debug_options,
    ):
        """
        This class acts as a backend for the :func:`torch.compile` function, facilitating the benchmarking of each :class:`torch.fx.GraphModule` produced by Thunder dynamo splitter.
        Each :class:`torch.fx.GraphModule` instance is executed by the specified executors and benchmarked using `pytest-benchmark`.

        Args:
            bench: the BenchmarkFixture created by ``pytest_benchmark``
            executors: A dictionary of functors to compare.
                - Key: The name of the executor to be displayed in the test name.
                - Value: A callable representing the compile function to be applied to the GraphModule.
                    If the value is None, no compilation is performed, and the GraphModule runs in Torch eager mode.

        Example:
            .. code-block:: python

                # script.py
                import torch
                import thunder
                from thunder.dynamo.compiler_graph_benchmark import ThunderCompilerGraphBenchmarking

                def func(x):
                    x = torch.sin(x)
                    if x.sum() > 0:
                        return x + 1
                    else:
                        return x - 1

                def test_func(benchmark):
                    backend = ThunderCompilerGraphBenchmarking(benchmark, executors={"eager": None, "thunder": thunder.jit})
                    compiled = torch.compile(backend=backend)(func)
                    x = torch.ones(2, requires_grad=True).cuda()
                    compiled(x)

        Note:
            Ensure the pytest configuration file (`thunder/tests/conftest.py`) is present in the same directory as `script.py` to provide the grouping customization.

            To run the benchmark test and group the results by split module, execute the following command:
            `pytest script.py  --benchmark-group-by='graph-by-graph:param:GraphID,param:SplitModuleName'`

            In this example, Dynamo segments the graph into two subgraphs, each identified by the 'GraphID[id]' field in the test name.
            Each subgraph contains a single split module, processed by the Thunder-defined splitter,
            which corresponds to the 'SplitModuleName[split_module_name]' field.
            The currently active executor is indicated by the 'executor[executor_name]'.
            With `--benchmark-group-by='graph-by-graph:param:GraphID,param:SplitModuleName'`, the test cases are grouped based on GraphID and SplitModuleName,
            allowing for performance comparison between different executors (e.g., 'eager' vs. 'thunder').
        """
        super().__init__()
        self.bench = bench
        check(isinstance(executors, dict) and executors, lambda: f"'executors' must be a non-empty dictionary.")
        check(
            not any("-" in k for k in executors.keys()),
            lambda: f"Executor names cannot contain '-' as it conflicts with the 'benchmark-group-by' function. Please rename it using a different character.",
        )
        self.executors = executors
        self._get_debug_options(**debug_options)

        self.graph_idx = 0

    def _get_debug_options(self, **debug_options):
        self.post_graph = debug_options.get("post_graph", False)

    def run_bench(self, gm: torch.fx.GraphModule, name: str, *sample_args):
        from thunder.benchmarks import record_peak_allocated_memory, MAX_ALLOCATED_MEMORY_KEYWORD

        for ex_name, ex in self.executors.items():
            if ex is None:
                compiled_fn = gm
            else:
                try:
                    compiled_fn = ex(gm)
                except Exception as e:
                    raise RuntimeError(f"The input executor {ex_name} failed to compile {gm}") from e
            if self.post_graph:
                compiled_fn = self.post_graph(compiled_fn, sample_args)

            # This guard ensures compatibility with CPU-only PyTorch builds.
            if torch.cuda.is_available():
                with record_peak_allocated_memory(self.bench):
                    self.bench(compiled_fn, *sample_args)
            else:
                self.bench(compiled_fn, *sample_args)
            # BenchmarkFixture.stats is created each time bench is called (ref: https://github.com/pybenchmark/pytest-benchmark/blob/8c9a5faa1dd178b53ab7b2a66f5364a77e903d74/src/pytest_benchmark/fixture.py#L150)
            # Adds the graph number, split module name and executor suffix to the name string
            gid_key, module_name_key, ex_key = GRAPH_BY_GRAPH_BENCHMARK_PARAMS_KEYS
            self.bench.stats.name += f"-{gid_key}[{self.graph_idx}]-{module_name_key}[{name}]-{ex_key}[{ex_name}]"

            if torch.cuda.is_available():
                assert MAX_ALLOCATED_MEMORY_KEYWORD in self.bench.extra_info
                assert f"{self.bench.stats.name}_{MAX_ALLOCATED_MEMORY_KEYWORD}" not in self.bench.extra_info
                # NOTE: A benchmark can include multiple stats, but only one extra_info field is allowed per benchmark.
                # Therefore, we use the current stats name as a prefix to distinguish memory usage for each stats.
                self.bench.extra_info[f"{self.bench.stats.name}_{MAX_ALLOCATED_MEMORY_KEYWORD}"] = (
                    self.bench.extra_info.pop(MAX_ALLOCATED_MEMORY_KEYWORD)
                )

            # when the graph is segmented, the self.bench run multiple times, pybenchmark throws an error:
            # `FixtureAlreadyUsed("Fixture can only be used once. Previously it was used in %s mode." % self._mode)`
            # Ref: https://github.com/pybenchmark/pytest-benchmark/blob/8c9a5faa1dd178b53ab7b2a66f5364a77e903d74/src/pytest_benchmark/fixture.py#L115-L118
            # Here manually set the BenchmarkFixture._mode=None to avoid it
            self.bench._mode = None

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        split_module = super().__call__(gm, sample_args)

        def has_checkpoint_node(g):
            if g.find_nodes(op="call_function", target=torch.ops.higher_order.tag_activation_checkpoint):
                return True
            for n in g.nodes:
                if n.op == "call_module" and has_checkpoint_node(getattr(g.owning_module, n.target).graph):
                    return True
            return False

        if LooseVersion(torch.__version__) < LooseVersion("2.6.0"):
            # NOTE: PyTorch 2.6 changes the structure of GraphModule when using activation checkpointing.
            # It's hard to retrieve the example input tensor for the GraphModule contains checkpoint operator before PyTorch 2.6
            if has_checkpoint_node(split_module.graph):
                raise RuntimeError(
                    "The benchmarking of the Torch activation checkpointing is only supported with PyTorch version 2.6 or later."
                )

        compiled_functions_to_submodule = {
            v.compiled_fn: k for k, v in self.subgraph_infos[self.graph_idx].submodule_to_compiled_functions.items()
        }
        for node in split_module.graph.nodes:
            target = node.target
            # Benchmarks the modules produced by the splitter and are supported by Thunder.
            if isinstance(target, str) and target.startswith("thunder_"):
                check(
                    hasattr(split_module, target),
                    lambda: f"the submodule {target} does not exist in {split_module}",
                    ValueError,
                )
                cur_module = getattr(split_module, target)
                cur_nodes = cur_module.graph.nodes
                # Greates random input values for the current module based on the faketensor 'example_value' of the placeholder node
                placeholders = list(n for n in cur_nodes if n.op == "placeholder")
                args = list(map(_get_example_inputs_from_placeholder, placeholders))
                # Runs the benchmark on the original module with the generated random inputs
                self.run_bench(compiled_functions_to_submodule[cur_module], target, *args)
        self.graph_idx += 1
        return split_module
