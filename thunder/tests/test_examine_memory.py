import pytest

import torch

import thunder
from thunder.core.pytree import tree_map
import thunder.torch as ltorch
from thunder.examine.memory_calculation import get_alloc_memory

from thunder.tests.framework import requiresCUDA, TorchExecutor
from thunder.tests.make_tensor import make_tensor


def measure_memory_usage(trace):
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats().get("requested_bytes.all.current", 0)

    def make_tensor_like_torch_dtype(p):
        return make_tensor(p.shape, dtype=ltorch.to_torch_dtype(p.dtype), device=p.device)

    args, kwargs = tree_map(make_tensor_like_torch_dtype, (trace.args, trace.kwargs))
    output = trace.python_callable()(*args, **kwargs)

    after = torch.cuda.memory_stats()["requested_bytes.all.current"]
    peak = torch.cuda.memory_stats()["requested_bytes.all.peak"]

    return {"peak": peak - before, "current": after - before, "output": output}


def measure_fw_and_bw_memory_usage(fw_trace, bw_trace):
    fw_results = measure_memory_usage(fw_trace)
    bw_results = measure_memory_usage(bw_trace)

    return {f"fw_{k}": v for k, v in fw_results.items()} | {f"bw_{k}": v for k, v in bw_results.items()}


# TODO: Test for nvFuserExecutor
# nvFuserExecutor is skipped for now, because nvFuser and eager execution treat allocation and broadcast differently.
# In the future, we need to update get_alloc_memory to support nvFuser and update tests accordingly.
@requiresCUDA
def test_view_ops():
    def test(func, *shapes):
        inputs = [make_tensor(shape, dtype=torch.float32, device="cuda", requires_grad=True) for shape in shapes]
        cfunc = TorchExecutor.make_callable(func, disable_preprocessing=False)
        cfunc(*inputs)

        fw_trace = thunder.last_traces(cfunc)[-1]
        bw_trace = thunder.last_backward_traces(cfunc)[-1]
        max_mem_fw = get_alloc_memory(fw_trace)
        max_mem_bw = get_alloc_memory(bw_trace)

        result = measure_fw_and_bw_memory_usage(fw_trace, bw_trace)
        assert max_mem_fw[0] == result["fw_peak"]
        assert sum(max_mem_fw[1].values()) == result["fw_current"]
        assert max_mem_bw[0] == result["bw_peak"]
        assert sum(max_mem_bw[1].values()) == result["bw_current"]

    def foo(a, b):  # [4] [4]
        a_1 = torch.unsqueeze(a, 0)  # [1,4]
        b_2 = torch.unsqueeze(b, 0)  # [1,4]
        return (a_1 + b_2,)

    test(foo, (4,), (4,))

    def bar(a, b):  # [4] [2,2]
        a_1 = torch.unsqueeze(a, 0)  # [1,4]
        a_2 = torch.unsqueeze(a_1, 1)  # [1,1,4]
        a_3 = a_2.expand(2, 3, 4)  # [2,3,4]

        b_1 = torch.reshape(b, (4,))  # [4]
        b_2 = torch.unsqueeze(b_1, 0)  # [1,4]
        b_3 = torch.unsqueeze(b_2, 1)  # [1,1,4]
        b_4 = b_3.expand(2, 3, 4)  # [2,3,4]

        result1 = a_2 + b_3
        result2 = b_4 + a_3
        return result1, result2

    test(bar, (4,), (2, 2))

    def bar1(a, b, c):  # [4], [1,4,4], [4,1,4]
        a_1 = torch.unsqueeze(a, 0)  # [1,4]
        a_2 = torch.unsqueeze(a_1, 1)  # [1,1,4]
        a_3 = a_2.expand(1, 4, 4)
        a_4 = a_2.expand(4, 1, 4)
        return b + a_3, c + a_4

    test(bar1, (4,), (1, 4, 4), (4, 1, 4))

    def bar2(a, b):  # [5,2], [2,2]
        a_1, a_2, a_3 = torch.split(a, 2)
        c = a_1 + b
        d = a + a
        return c, d, a_2, a_3  # We have to use all the outputs of torch.split due to #1043

    test(bar2, (5, 2), (2, 2))


@requiresCUDA
def test_nanogpt_block():
    import thunder.tests.nanogpt_model as nanogpt_model

    config = nanogpt_model.GPTConfig(dropout=0)
    block = nanogpt_model.Block(config).to(dtype=torch.float32, device="cuda")
    cblock = TorchExecutor.make_callable(block)
    inp = make_tensor((2, config.block_size, config.n_embd), dtype=torch.float32, device="cuda", requires_grad=True)
    cblock(inp)

    fw_trace = thunder.last_traces(cblock)[-1]
    bw_trace = thunder.last_backward_traces(cblock)[-1]
    max_mem_fw = get_alloc_memory(fw_trace)
    max_mem_bw = get_alloc_memory(bw_trace)

    # Actual memory usage may vary depending on hardware and cuBLAS settings.
    # We are checking the estimated memory against a fixed value for consistency.
    assert max_mem_fw[0] == 381754368
    assert sum(max_mem_fw[1].values()) == 375462912
    assert max_mem_bw[0] == 437292032
    assert sum(max_mem_bw[1].values()) == 40934400
