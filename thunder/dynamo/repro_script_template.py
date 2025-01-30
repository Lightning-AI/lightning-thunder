repro_code_template = '''
"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}

{extra_comment_str}
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
    from thunder.dynamo.report import run_repro

    executor={executor_str}
    if executor == None:
        compiled_model = model
    else:
        compiled_model = executor(model)
    result = run_repro(compiled_model, compute_type, *inputs)
    if check_acc:
        eager_result = run_repro(model, compute_type, *inputs)
        for (compute_t, eager_v), (_, cur_v) in zip(eager_result.items(), result.items()):
            torch.testing.assert_close(eager_v, cur_v, msg=lambda e : f'{{compute_t}}: {{e}}')


test_{graph_name}()
'''


benchmark_multi_exe_code_template = '''
"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}

{extra_comment_str}
"""
# NOTE: This script requires `pytest-benchmark==4.0.0` to be installed.
# To execute the script, run `pytest {graph_name}_benchmark.py --benchmark-timer=torch.utils.benchmark.utils.timer.timer --benchmark-warmup=on --benchmark-group-by=param:compute_type`
# To check the peak allocated CUDA memory, use --benchmark-json=json_file_name and look at the "max_allocated_memory_MB" field in the json file
# To run tests for a specific compute_type, use the pytest `-k` option.
# For example:
#   - `-k "forward"` will run only the forward pass.
#
# Available options:
#   - compute_type: "forward", "backward"

import pytest
from thunder.benchmarks.targets import parametrize_compute_type_only_training, benchmark_for_compute_type, ComputeType
{torch_import_str}
{import_str}

# NOTE: The reproducer function has already been processed by TorchDynamo.
# If we let it go through TorchDynamo again, it could be segmented further.
# To avoid this, we directly use Inductor here.
# See issue https://github.com/Lightning-AI/lightning-thunder/issues/1521
def torch_inductor(fn, inputs):
    from torch._inductor import compile as inductor_compile
    from torch.fx import symbolic_trace

    fx_graph = symbolic_trace(fn)
    return inductor_compile(fx_graph, inputs)

{executors}
{executor_names}
@pytest.mark.parametrize(
    "executor",
    executors,
    ids=executor_names,
)
{compute_type_decorator}
def test_{graph_name}(benchmark, executor, compute_type):
{dynamo_module}

{inputs}

    model = DynamoModule()
    if executor == None:
        compiled_model = model
    elif executor == torch_inductor:
        compiled_model = executor(model, inputs)
    else:
        compiled_model = executor(model)
    {call_benchmark}
'''
