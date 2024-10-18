import pytest
from collections import defaultdict
import pytest_benchmark
from thunder.dynamo.compiler_graph_benchmark import GRAPH_BY_GRAPH_BENCHMARK_PARAMS_KEYS


@pytest.hookimpl(hookwrapper=True)
def pytest_benchmark_group_stats(config, benchmarks, group_by):
    """
    The function customize the behavior for ThunderCompilerGraphBenchmarking.
    The custom grouping function is only invoked when the `--benchmark-group-by`
    option is set to 'graph-by-graph:param:GraphID,param:SplitModuleName'.
    For an example, refer to the comment section in `ThunderCompilerGraphBenchmarking`.

    Reference: https://pytest-benchmark.readthedocs.io/en/latest/hooks.html#pytest_benchmark.hookspec.pytest_benchmark_group_stats
    """
    prefix = "graph-by-graph:"
    outcome = yield
    if group_by.startswith(prefix):
        group_by = group_by[len(prefix) :]
        for bench in benchmarks:
            if bench["params"] is None:
                bench["params"] = {}
            # The benchs with the same `params`` share the same dict
            # We need to create a deepcopy of the original dictionary to add parameters specific to each graph.
            else:
                bench["params"] = bench["params"].copy()
            if bench["param"] is None:
                bench["param"] = ""

            name = bench["name"]
            gid, module_name, ex = name.split("-")[-3:]
            # Add the "GraphID", "SplitModuleName","executor" as params in benchmark
            gid_key, module_name_key, ex_key = GRAPH_BY_GRAPH_BENCHMARK_PARAMS_KEYS
            bench["params"].update({gid_key: gid, module_name_key: module_name, ex_key: ex})
            bench["param"] += f"-{gid}-{module_name}-{ex}"

    result = pytest_benchmark.plugin.pytest_benchmark_group_stats(config, benchmarks, group_by)
    outcome.force_result(result)
