from dataclasses import dataclass
from functools import wraps
from itertools import groupby, product
from typing import Callable, Sequence

import pytest

import torch

from litgpt import Config, GPT
from litgpt.config import configs

import thunder

from thunder.benchmarks import thunder_executor, torch_compile_executor, torch_executor
from thunder.common import CompileData, CompileStats
from thunder.core.compile_data import set_compile_data_and_stats

from thunder.core.jit_ext import thunder_general_jit
from thunder.core.langctxs import set_langctx
from thunder.core.trace import TraceCtx
from thunder.core.transforms import eval_trace
from thunder.executors.torch_compile import to_torch_translator

from thunder.tests.make_tensor import make_tensor

BATCH_SIZE = 2
CONFIG_NAMES = list(sorted((c["name"] for c in configs)))
# CONFIG_NAMES = ["Llama-2-7b-hf",]


def make_torch_traces_for_config(name: str):
    config = Config.from_name(name)
    # Two layers is enough to expose the fusing opportunity of the following network boundaries:
    # - Embedding layer -> 0th Transformer layer
    # - Last Transformer layer -> Output layer
    # - End of the Transformer layer -> Beginning of the Transformer layer
    config.n_layer = 2

    model = GPT(config).to(dtype=torch.bfloat16, device="cuda")
    input_shape = (BATCH_SIZE, config.block_size)
    x = torch.randint(0, config.vocab_size, input_shape, dtype=torch.int64, device="cuda")

    # Acquire the initial trace
    # We could use thunder.jit here, but we want to not execute the compiled function
    # and instead only get the initial trace before any transformations
    # jit_model = thunder.jit(model)
    # out = jit_model(x)
    # trace = thunder.last_traces(jit_model)[0]

    # We need to set up contexts that are usually set up by the thunder.jit decorator
    cd = CompileData(fn=model, disable_preprocessing=True, executor_lookasides={})
    cs = CompileStats()
    set_langctx(thunder.torch.torchctx)
    set_compile_data_and_stats(cd, cs)
    thunder._cache_info_ctx.set({})
    prologue, trace, epilogue = thunder_general_jit(
        model, (x,), {}, sharp_edges=thunder.core.options.SHARP_EDGES_OPTIONS.ALLOW
    )

    # Remove subsymbols for readability of the trace
    for bsym in trace.bound_symbols:
        bsym.subsymbols = []

    producers, consumers = thunder.core.utils.producers_and_consumers(trace)

    # Remove unpacking prims so that they can be identified as inputs of the first chunk
    trace.bound_symbols = [bsym for bsym in trace.bound_symbols if bsym.sym.id != thunder.prims.unpack_trivial.id]

    # Remove return prim so that it can be identified as the output of the last chunk
    assert trace.bound_symbols.pop().sym.id == thunder.prims.python_return.id

    # We want to split the trace into chunks of network between the scaled dot-product attention calls
    assert (
        len([bsym for bsym in trace.bound_symbols if bsym.sym.id == thunder.torch.scaled_dot_product_attention.id])
        == config.n_layer
    )

    # This is going to be our delimiter for splitting the trace into chunks
    thunder_sdpa = thunder.torch.scaled_dot_product_attention
    chunks = list(list(g) for k, g in groupby(trace.bound_symbols, key=lambda x: x.sym.id == thunder_sdpa.id) if not k)

    # Now we need to convert the chunks into a list of functions
    regions = [thunder.executors.utils.Region(producers, consumers, chunk) for chunk in chunks]

    # After this point, we will have a list of regions that are represented as regular PyTorch functions
    # We can acquire the Python functions by calling .python_callable() on each "torch_trace" object
    torch_traces = []
    for r in regions:
        # Here we construct a trace that will be used to compile the function
        region_trace = TraceCtx(None)
        region_trace.bound_symbols = list(r.bound_symbols)
        sorted_proxy_inputs = [v.proxy for v in sorted(r.inputs, key=lambda x: x.proxy.name)]
        sorted_proxy_outputs = [v.proxy for v in sorted(r.outputs, key=lambda x: x.proxy.name)]
        region_trace.args = sorted_proxy_inputs
        region_trace.kwargs = {}
        region_trace.bound_symbols.append(thunder.prims.python_return.bind(sorted_proxy_outputs, output=()))
        region_trace = thunder.executors.passes.dce(region_trace)

        def torch_interpreted_func(*args):
            return eval_trace(region_trace, *args, symbol_mapper=to_torch_translator)

        torch_trace = thunder.trace(inline_trace=False)(torch_interpreted_func, *sorted_proxy_inputs)

        # Remove subsymbols for readability of the trace
        for bsym in torch_trace.bound_symbols:
            bsym.subsymbols = []

        torch_traces.append(torch_trace)

    return torch_traces


def wrap_for_benchmark(fn):
    @wraps(fn)
    def fn_(*args, **kwargs):
        torch.cuda.synchronize()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return result

    return fn_


def forward_and_backward(torch_trace: TraceCtx, jit_fn: Callable):
    fn = torch_trace.python_callable(include_no_grad=False)
    jfn = jit_fn(fn)

    @wraps(jfn)
    def wrapper(*args, **kwargs):
        result = jfn(*args, **kwargs)
        if isinstance(result, Sequence):
            torch.autograd.backward(result, [torch.ones_like(x) for x in result])
        else:
            result.backward(torch.ones_like(result))
        return result

    return wrapper


to_executor_name = {
    torch_executor: "eager",
    torch_compile_executor: "inductor",
    thunder_executor: "thunder",
}


@dataclass
class TraceInfo:
    config_name: str
    region_idx: int
    trace: TraceCtx


litgpt_traces = [
    TraceInfo(name, i, trace) for name in CONFIG_NAMES for i, trace in enumerate(make_torch_traces_for_config(name))
]

# Now we have a list of torch_traces that are ready to be benchmarked
trace_executor_pairs = list(product(litgpt_traces, (torch_executor, torch_compile_executor, thunder_executor)))


@pytest.mark.parametrize(
    "info, executor",
    trace_executor_pairs,
    ids=[
        f"{info.config_name}_region{info.region_idx}_{to_executor_name[executor]}"
        for info, executor in trace_executor_pairs
    ],
)
def test_litgpt(benchmark, info, executor):
    torch_trace = info.trace

    def setup():
        args = []
        for a in torch_trace.args:
            torch_dtype = thunder.torch.to_torch_dtype(a.dtype)
            torch_device = thunder.core.devices.to_torch_device(a.device)
            is_float = isinstance(a.dtype, thunder.core.dtypes.floating)
            low = 0 if not is_float else None
            args.append(make_tensor(a.shape, dtype=torch_dtype, device=torch_device, requires_grad=is_float, low=low))
        return args, {}

    fn = forward_and_backward(torch_trace, executor)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)
