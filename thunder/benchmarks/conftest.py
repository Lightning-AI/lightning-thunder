import warnings
import platform
import psutil
from typing import Any
import os

try:
    from asvdb import utils, BenchmarkInfo, BenchmarkResult, ASVDb
    HAS_ASVDB = True
except ImportError:
    warnings.warn("asvdb is not imported, results won't be saved as asv")
    HAS_ASVDB = False

if not (bench_dir := os.getenv("THUNDER_BENCH_DIR")):
    bench_dir = "./asv"

def pytest_addoption(parser):
    parser.addoption("--asv_bench_dir", action="store", default=bench_dir)

def sanitize_params(benchmark_params: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    sane_params = []
    for k, v in benchmark_params:
        if k == "executor":
            sane_params += [(k, str(v).split()[1])]
        else:
            sane_params += [(k, v)]
    return sane_params


def pytest_sessionfinish(session, exitstatus):
    if not HAS_ASVDB or not hasattr(session.config, "_benchmarksession"):
        return

    benchmarks = session.config._benchmarksession.benchmarks

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

    db = ASVDb(dbDir=session.config.option.asv_bench_dir, repo=repo_name, branches=[current_branch])

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
