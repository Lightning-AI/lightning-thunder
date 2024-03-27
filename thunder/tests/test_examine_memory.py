from contextlib import contextmanager
from functools import partial

import pytest

import torch

import thunder
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.examine.memory_caculation import get_alloc_memory

from thunder.tests.framework import instantiate, nvFuserTestExecutor, TorchTestExecutor
from thunder.tests.make_tensor import make_tensor


@contextmanager
def runtime_allocated_memory(dev):
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        yield
    finally:
        memory_states = torch.cuda.memory_stats(dev)
        alloc = memory_states["allocated_bytes.all.peak"]
        req = memory_states["requested_bytes.all.peak"]
        print(f"**peak allocated/required memory: {alloc}, {req}")


def get_return_memory(bsym):
    assert bsym.sym.name == "return"
    return_tensors_name = set()
    res = 0
    for x in bsym.flat_proxy_args:
        if isinstance(x, TensorProxy) and x.name not in return_tensors_name:
            res += x.numel * x.dtype.bytes
            return_tensors_name.add(x.name)
    return res


@instantiate(dtypes=(thunder.float32,), devicetypes=(devices.DeviceType.CUDA,))
def test_view_ops(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((4,), device=device, dtype=torch_dtype, requires_grad=True)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype, requires_grad=True)

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

    cbar = executor.make_callable(bar, disable_preprocessing=False)
    with runtime_allocated_memory(device):
        cbar(a, b)

    fw_traces = thunder.last_traces(cbar)
    fwd_extrace = fw_traces[-1]
    max_mem_fwd = get_alloc_memory(fwd_extrace)
    assert max_mem_fwd[0] == 144
    assert sum(max_mem_fwd[1].values()) == get_return_memory(fwd_extrace.bound_symbols[-1])  # 144
    bw_traces = thunder.last_backward_traces(cbar)
    bw_extrace = bw_traces[-1]
    max_mem_bw = get_alloc_memory(bw_extrace)
    assert max_mem_bw[0] == 144
    assert sum(max_mem_bw[1].values()) == get_return_memory(bw_extrace.bound_symbols[-1])  # 32

    def bar1(a, b, c):  # [4], [1,4,4], [4,1,4]
        a_1 = torch.unsqueeze(a, 0)  # [1,4]
        a_2 = torch.unsqueeze(a_1, 1)  # [1,1,4]
        a_3 = a_2.expand(1, 4, 4)
        a_4 = a_2.expand(4, 1, 4)
        return b + a_3, c + a_4

    a = make_tensor((4,), device=device, dtype=torch_dtype)
    b = make_tensor((1, 4, 4), device=device, dtype=torch_dtype)
    c = make_tensor((4, 1, 4), device=device, dtype=torch_dtype)
    cbar = executor.make_callable(bar1, disable_preprocessing=False)
    with runtime_allocated_memory(device):
        cbar(a, b, c)

    traces = thunder.last_traces(cbar)
    extrace = traces[-1]
    alloc_mem = get_alloc_memory(extrace)
    if isinstance(executor, nvFuserTestExecutor):
        assert alloc_mem[0] == 272
        assert sum(alloc_mem[1].values()) == get_return_memory(extrace.bound_symbols[-1])  # 128
    if isinstance(executor, TorchTestExecutor):
        assert alloc_mem[0] == 208
        assert sum(alloc_mem[1].values()) == get_return_memory(extrace.bound_symbols[-1])  # 128

    def bar2(a, b):  # [5,2], [2,2]
        a_1, a_2, a_3 = torch.split(a, 2)
        c = a_1 + b
        d = a + a
        return c, a_2, d

    a = make_tensor((5, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)
    cbar = executor.make_callable(bar2, disable_preprocessing=False)

    with runtime_allocated_memory(device):
        cbar(a, b)

    traces = thunder.last_traces(cbar)
    extrace = traces[-1]
    alloc_mem = get_alloc_memory(extrace)
    if isinstance(executor, nvFuserTestExecutor):
        assert alloc_mem[0] == 144
        assert sum(alloc_mem[1].values()) == get_return_memory(extrace.bound_symbols[-1])  # 72
    if isinstance(executor, TorchTestExecutor):
        assert alloc_mem[0] == 112
        assert sum(alloc_mem[1].values()) == get_return_memory(extrace.bound_symbols[-1])  # 72


@instantiate(dtypes=(thunder.float32,), devicetypes=(devices.DeviceType.CUDA,))
def test_nanogpt_block(executor, device, dtype):
    import thunder.tests.nanogpt_model as nanogpt_model

    tdtype = ltorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    config = nanogpt_model.GPTConfig(dropout=0)
    block = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
    cblock = executor.make_callable(block)

    with runtime_allocated_memory(device):
        inp = make((2, config.block_size, config.n_embd))
        result = cblock(inp)
    with runtime_allocated_memory(device):
        result.backward(torch.ones_like(result))
    fw_extrace = thunder.last_traces(cblock)[-1]
    bw_extrace = thunder.last_backward_traces(cblock)[-1]
    fw_alloc_mem = get_alloc_memory(fw_extrace)
    bw_alloc_mem = get_alloc_memory(bw_extrace)

    if isinstance(executor, nvFuserTestExecutor):
        assert fw_alloc_mem[0] == 267426816
        # t67 is the expand result of ln_2_weight, and they are both return values in trace
        # but for calculation we assume they share memory, so expect to subtract the size of t67
        expected_return_calculated_mem = get_return_memory(fw_extrace.bound_symbols[-1]) - 4 * 2 * 1024 * 768
        assert expected_return_calculated_mem == sum(fw_alloc_mem[1].values())

        assert bw_alloc_mem[0] == 361783296
        assert sum(bw_alloc_mem[1].values()) == get_return_memory(bw_extrace.bound_symbols[-1])
    if isinstance(executor, TorchTestExecutor):
        assert fw_alloc_mem[0] == 362863616
        # same reason as above, expect to -t38+t37-t65-t67
        expected_return_calculated_mem = (
            get_return_memory(fw_extrace.bound_symbols[-1]) - 23 * 1024 * 1024 - 4 * 2 * 1024 * 768 * 2
        )
        assert expected_return_calculated_mem == sum(fw_alloc_mem[1].values())
        assert bw_alloc_mem[0] == 412109824
        assert sum(bw_alloc_mem[1].values()) == get_return_memory(bw_extrace.bound_symbols[-1])
