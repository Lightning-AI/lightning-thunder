repro_code_template = '''
"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}
"""
{torch_import_str}
{import_str}
import argparse

parser = argparse.ArgumentParser(description="Script for executing an FX graph with specified configurations.")

parser.add_argument(
    "--check_consistency",
    type=bool,
    default=False,
    help="Whether to check consistency (default: False)"
)
parser.add_argument(
    "--compute_type",
    type=str,
    choices=["forward", "forward+backward"],
    default="forward",
    help="Type of computation to perform (forward, forward+backward)"
)
args = parser.parse_args()
compute_type = args.compute_type
check_acc = args.check_consistency

def test_{graph_name}():
{dynamo_module}

{inputs}

    model = DynamoModule()
    from thunder.dynamo.report import run_repro1

    {custom_executor_str}
    result = run_repro1(compiled_model, compute_type, *inputs)
    if check_acc:
        eager_result = run_repro1(model, compute_type, *inputs)
        for (compute_t, eager_v), (_, cur_v) in zip(eager_result.items(), result.items()):
            torch.testing.assert_close(eager_v, cur_v, msg=lambda e : f'{{compute_t}}: {{e}}')


test_{graph_name}()
'''


benchmark_code_template = '''
"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}
"""
# NOTE: This script requires `pytest-benchmark==4.0.0` to be installed.
# To execute the script, run `pytest {graph_name}.py --benchmark-timer=torch.utils.benchmark.utils.timer.timer --benchmark-warmup=on --benchmark-group-by=param:compute_type`
# To check the peak allocated CUDA memory, use --benchmark-json=json_file_name and look at the "max_allocated_memory_MB" field in the json file
# To run tests for a specific compute_type, use the pytest `-k` option.
# For example:
#   - `-k "forward"` will run only the forward pass.
#
# Available options:
#   - compute_type: "forward", "backward"

import pytest
from thunder.benchmarks.targets import parametrize_compute_type_only_training, benchmark_for_compute_type
{torch_import_str}
{import_str}


@parametrize_compute_type_only_training
def test_{graph_name}(benchmark, compute_type):
{dynamo_module}

{inputs}

    model = DynamoModule()
    {custom_executor_str}
    {call_benchmark}
'''
