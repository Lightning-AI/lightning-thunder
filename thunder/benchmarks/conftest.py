import os
import platform
import psutil
from typing import Any
import warnings
import importlib.util


def pytest_addoption(parser):
    # CLI option to specify where to store the benchmark results in asv format.
    # If not set or None, results won't be saved in asv.
    parser.addoption("--asv_bench_dir", action="store", default=os.getenv("THUNDER_BENCH_DIR"))


def pytest_sessionfinish(session, exitstatus):
    # Save result only if the pytest session was a benchmark.
    if hasattr(session.config, "_benchmarksession"):
        save_benchmark_results_asv(session.config)


def sanitize_params(benchmark_params: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Util function that takes a list of params and removes serialization information. E.g. given '<function torch_executor at 0xffffffffff>' returns 'torch_executor'."""
    sane_params = []
    for k, v in benchmark_params:
        if k == "executor":
            sane_params += [(k, str(v).split()[1])]
        else:
            sane_params += [(k, v)]
    return sane_params


def save_benchmark_results_asv(config):
    """Save the benchmark results after a pytest session in the asv format.
    User must specify the --asv_bench_dir flag to store the results.
    """

    bench_dir = config.option.asv_bench_dir

    if not importlib.util.find_spec("asv"):
        warnings.warn("asvdb is not available. Results won't be saved in asv format.")
        return

    if not bench_dir:
        warnings.warn("asv_bench_dir' is not set. Results won't be saved in asv format.")
        return

    from asvdb import utils, ASVDb, BenchmarkResult, BenchmarkInfo

    benchmarks = config._benchmarksession.benchmarks

    # Get system information to store alongside the results.
    uname = platform.uname()
    commit_hash, commit_time = utils.getCommitInfo()
    repo_name, current_branch = utils.getRepoInfo()
    python_version = platform.python_version()
    memory_size = str(psutil.virtual_memory().total)

    bench_info = BenchmarkInfo(
        machineName=uname.machine,
        osType=f"{uname.system} {uname.release}",
        pythonVer=python_version,
        commitHash=commit_hash,
        commitTime=commit_time,
        cpuType=uname.processor,
        arch=uname.machine,
        ram=memory_size,
    )

    # Create the asv result database.
    db = ASVDb(dbDir=bench_dir, repo=repo_name, branches=[current_branch])

    # Add all the benchmarks to the database.
    for bench in benchmarks:
        name = bench.name.split("[")[0]
        params_pairs = sanitize_params(bench.params.items())
        result = BenchmarkResult(
            funcName=name,
            argNameValuePairs=params_pairs,
            result=bench.stats.median * 1e6,
            unit="µseconds",
        )
        db.addResult(bench_info, result)
