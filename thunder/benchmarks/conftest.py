import os
from os import path
import platform
import psutil
from typing import Any
import warnings
import importlib.util
from pytest import hookimpl, TestReport, Item, Parser
import multiprocessing as mp
import subprocess

BENCHMARK_JSON_DIR = "benchmarks_reports"
FAILED_BENCHMARK_LOGS_DIR = "failed_benchmarks_logs"


def pytest_addoption(parser: Parser):
    # CLI option to specify where to store the benchmark results in asv format.
    # If not set or None, results won't be saved in asv.
    parser.addoption("--asv_bench_dir", action="store", default=os.getenv("THUNDER_BENCH_DIR"))

    parser.addoption("--isolate-benchmarks", action="store_true", default=False)


def launch_benchmark(target_file, target_name: str):
    target_filename = target_name.replace("/", "_")

    target_json = path.join(BENCHMARK_JSON_DIR, f"{target_filename}.json")
    target_log = path.join(FAILED_BENCHMARK_LOGS_DIR, f"{target_filename}.log")

    with open(target_log, "w") as target_log_file:
        subprocess.run(
            [
                "pytest",
                f"{target_file}::{target_name}",
                "-vs",
                "--benchmark-json",
                target_json,
            ],
            check=True,
            text=True,
            stderr=subprocess.STDOUT,
            stdout=target_log_file,
        )


def run_in_isolation(item: Item) -> TestReport:
    process = mp.Process(
        target=launch_benchmark,
        args=(
            item.location[0],
            item.name,
        ),
    )
    process.start()
    process.join()

    # Will mark skip as passed because pytest returns error only if there are failed tests.
    outcome = "failed" if process.exitcode != 0 else "passed"
    target_filename = item.name.replace("/", "_")

    if outcome == "passed":
        test_log = path.join(FAILED_BENCHMARK_LOGS_DIR, f"{target_filename}.log")
        os.remove(test_log)

    benchmark_json = path.join(BENCHMARK_JSON_DIR, f"{target_filename}.json")
    if outcome == "failed" or path.getsize(benchmark_json) == 0:
        os.remove(benchmark_json)

    return TestReport(item.nodeid, item.location, keywords=item.keywords, outcome=outcome, longrepr=None, when="call")


@hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: Item, nextitem: Item):
    # If the option was not passed, let pytest manage the run.
    if not item.config.getoption("--isolate-benchmarks"):
        return None

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    test_report = run_in_isolation(item)

    ihook.pytest_runtest_logreport(report=test_report)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_runtestloop(session):
    global BENCHMARK_JSON_DIR, FAILED_BENCHMARK_LOGS_DIR

    if not session.config.getoption("--isolate-benchmarks"):
        return None

    mp.set_start_method("spawn")

    from _pytest.terminal import TerminalReporter

    terminal: TerminalReporter = session.config.pluginmanager.get_plugin("terminalreporter")

    custom_report_dir = os.getenv("THUNDER_BENCH_DIR")
    BENCHMARK_JSON_DIR = custom_report_dir if custom_report_dir else BENCHMARK_JSON_DIR

    os.makedirs(BENCHMARK_JSON_DIR, exist_ok=True)
    os.makedirs(FAILED_BENCHMARK_LOGS_DIR, exist_ok=True)

    terminal.write_line(f"Saving failed benchmarks logs in {FAILED_BENCHMARK_LOGS_DIR}")
    terminal.write_line(f"Saving benchmarks reports in {BENCHMARK_JSON_DIR}")


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
