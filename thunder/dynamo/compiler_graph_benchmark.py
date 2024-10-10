from __future__ import annotations
from itertools import chain
from pytest_benchmark.fixture import BenchmarkFixture
from typing import TYPE_CHECKING

import torch
from thunder.dynamo import ThunderCompiler
from thunder.dynamo.utils import _get_example_inputs_from_placeholder
from thunder.core.utils import check


if TYPE_CHECKING:
    from collections.abc import Sequence

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
        executors: Sequence[str],
        **thunder_options,
    ):
        """
        This class acts as a backend for the :func:`torch.compile` function, facilitating the benchmarking of each :class:`torch.fx.GraphModule` produced by Thunder dynamo splitter.
        Each :class:`torch.fx.GraphModule` instance is executed by the specified executors and benchmarked using `pytest-benchmark`.

        Args:
            bench: the BenchmarkFixture created by ``pytest_benchmark``
            executors: list of executors to compare. Supported executors include: 'eager', 'inductor', and 'thunder'. If None, defaults to all available executors.
            **thunder_options: a dictionary of options to pass to :func:`thunder.jit`. Besides all the arguments to :func:`thunder.jit`,
                                it accepts `torch_inductor_options` which are passed to :func:`torch.compile` if part of the graph
                                is not supported by thunder.

        Example:
            .. code-block:: python

                # script.py
                import torch
                import thunder
                from thunder.benchmarks import ThunderCompilerGraphBenchmarking

                def func(x):
                    x = torch.sin(x)
                    if x.sum() > 0:
                        return x + 1
                    else:
                        return x - 1

                def test_func(benchmark):
                    backend = ThunderCompilerGraphBenchmarking(benchmark, executors=["eager", "thunder"])
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
        super().__init__(**thunder_options)
        self.bench = bench
        if not executors:
            self.executors = ThunderCompilerGraphBenchmarking._executors
        else:
            check(
                all(ex in ThunderCompilerGraphBenchmarking._executors for ex in executors),
                lambda: f"ThunderCompilerGraphBenchmarking only supports the following executor names: {ThunderCompilerGraphBenchmarking._executors}  ",
            )
            self.executors = executors
        self.graph_idx = 0

    def run_bench(self, gm: torch.fx.GraphModule, name: str, *sample_args):
        from thunder.benchmarks.targets import record_peak_allocated_memory, MAX_ALLOCATED_MEMORY_KEYWORD

        for ex in self.executors:
            # Uses the already compiled module if it is compiled with the expected executor
            if name.startswith(ex):
                fn = self.subgraph_infos[self.graph_idx].submodule_to_compiled_functions[gm].compiled_fn
            else:
                if ex == "thunder":
                    fn = self._thunder_jit(gm)
                elif ex == "inductor":
                    fn = self._torch_compile(gm)
                else:
                    fn = gm
            with record_peak_allocated_memory(self.bench):
                self.bench(fn, *sample_args)
            # BenchmarkFixture.stats is created each time bench is called (ref: https://github.com/pybenchmark/pytest-benchmark/blob/8c9a5faa1dd178b53ab7b2a66f5364a77e903d74/src/pytest_benchmark/fixture.py#L150)
            # Adds the graph number, split module name and executor suffix to the name string
            gid_key, module_name_key, ex_key = GRAPH_BY_GRAPH_BENCHMARK_PARAMS_KEYS
            self.bench.stats.name += f"-{gid_key}[{self.graph_idx+1}]-{module_name_key}[{name}]-{ex_key}[{ex}]"
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
        compiled_functions_to_submodule = {
            v.compiled_fn: k for k, v in self.subgraph_infos[self.graph_idx].submodule_to_compiled_functions.items()
        }
        for node in split_module.graph.nodes:
            target = node.target
            # Benchmarks the modules produced by the splitter.
            if isinstance(target, str) and target.startswith(("thunder_", "inductor_")):
                check(
                    hasattr(split_module, target),
                    lambda: f"the submodule {target} does not exist in {split_module}",
                    ValueError,
                )
                cur_module = getattr(split_module, target)
                cur_nodes = cur_module.graph.nodes
                # Greates random input values for the current module based on the faketensor 'example_value' of the placeholder node
                placeholders = list(n for n in cur_nodes if n.op == "placeholder")
                args = chain(*map(_get_example_inputs_from_placeholder, placeholders))
                # Runs the benchmark on the original module with the generated random inputs
                self.run_bench(compiled_functions_to_submodule[cur_module], target, *args)
        self.graph_idx += 1
        return split_module
