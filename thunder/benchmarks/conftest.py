import pytest
from collections import defaultdict
import pytest_benchmark


@pytest.hookimpl(hookwrapper=True)
def pytest_benchmark_group_stats(config, benchmarks, group_by):
    param_keys = ("GraphID", "SplitModuleName", "executor")
    prefix = "graph-by-graph:"
    # import pdb;pdb.set_trace()
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
            bench["params"].update({"GraphID": gid, "SplitModuleName": module_name, "executor": ex})
            bench["param"] += f"-{gid}-{module_name}-{ex}"

    result = pytest_benchmark.plugin.pytest_benchmark_group_stats(config, benchmarks, group_by)
    outcome.force_result(result)
