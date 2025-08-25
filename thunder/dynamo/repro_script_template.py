FXGRAPH_CLASS_NAME = "DynamoModule"
INPUTS_NAME = "inputs"
CALLABLE_NAME = "model"
COMPILED_CALLABLE_NAME = "compiled_model"
THUNDER_IMPORT_STRS = """
from thunder.dev_utils import nvtx_profile_transform
"""

pytest_benchmark_multi_exe_code_template = f'''
# NOTE: This script requires `pytest-benchmark==4.0.0` to be installed.
# To execute the script, run `pytest {{graph_name}}_benchmark.py --benchmark-timer=torch.utils.benchmark.utils.timer.timer --benchmark-warmup=on --benchmark-group-by=param:compute_type`
# To check the peak allocated CUDA memory, use --benchmark-json=json_file_name and look at the "max_allocated_memory_MB" field in the json file
# To run tests for a specific compute_type, use the pytest `-k` option.
# For example:
#   - `-k "forward"` will run only the forward pass.
#
# Available options:
#   - compute_type: "forward", "backward"

import pytest
from thunder.benchmarks.targets import parametrize_compute_type_only_training, benchmark_for_compute_type, ComputeType
{{torch_import_str}}
{{import_str}}
{THUNDER_IMPORT_STRS}

# NOTE: The reproducer function has already been processed by TorchDynamo.
# If we let it go through TorchDynamo again, it could be segmented further.
# To avoid this, we directly use Inductor here.
# See issue https://github.com/Lightning-AI/lightning-thunder/issues/1521
def torch_inductor(fn, inputs):
    from torch._inductor import compile as inductor_compile
    from torch.fx import symbolic_trace

    fx_graph = symbolic_trace(fn)
    return inductor_compile(fx_graph, inputs)

{{executors}}
{{executor_names}}

{{dynamo_module}}

@pytest.mark.parametrize(
    "executor",
    executors,
    ids=executor_names,
)
{{compute_type_decorator}}
def test_{{graph_name}}(benchmark, executor, compute_type):
{{inputs}}

    model = DynamoModule()
    if executor is None:
        compiled_model = model
    elif executor == torch_inductor:
        compiled_model = executor(model, inputs)
    else:
        compiled_model = executor(model)
    {{call_benchmark}}

"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{{torch_env}}

Versions of Thunder related libraries:
{{thunder_pkgs}}

{{extra_comment_str}}
"""
'''


bsym_torch_compile_repro_template = '''
"""
{extra_comment_str}
"""
{python_func}

from thunder.executors.torch_compile import make_compiled as make_torch_compile_callable
import thunder.examine

inputs = {inputs}

jfn = thunder.jit({func_name})
jfn(*inputs)

trc = thunder.last_traces(jfn)[-1]
fusion_symbols = thunder.examine.get_fusion_symbols(trc)
assert len(fusion_symbols) == 1
bsym = fusion_symbols[0]

# NOTE: The nvFusion function cannot be compiled directly using `torch.compile`.
# It must first be processed by Thunder into BoundSymbols and compiled with `make_torch_compile_callable`.
# Additionally, it's recommended to visually verify that `bsym` matches the
# `nvFusion` function above by printing it using `print(bsym)`.
torch_compiled_callable = make_torch_compile_callable(bsym.subsymbols, bsym.flat_args, bsym.flat_outs)
'''

repro_bench_code_template = f"""
{{import_str}}
{THUNDER_IMPORT_STRS}

{{dynamo_module}}
def test_{{graph_name}}():
{{inputs}}

    model = {FXGRAPH_CLASS_NAME}()
"""

main_code = """
if __name__ == "__main__":
    test_{graph_name}()
"""

comment_str_template = '''
"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}

{extra_comment_str}
"""
'''
