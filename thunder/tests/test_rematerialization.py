import pytest
import torch

import thunder
from thunder import torch as ttorch

from thunder.core import utils
from thunder.core.transforms import inline, value_and_grad
from thunder.tests.framework import instantiate, NOTHING, nvFuserExecutor
from thunder.tests.make_tensor import make_tensor, make_tensor_like


@inline
@value_and_grad
def func(t0):
    t1 = ttorch.exp(t0)
    t2 = ttorch.exp(t1)
    t3 = ttorch.exp(t2)
    t4 = ttorch.matmul(t3, t3)  # Fusion breaks here
    return t4


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_producer_symbols(executor, device, _):
    # We will try to find a subgraph for rematerializing __c and __d
    t0 = torch.randn(2, 2, dtype=torch.float32, device="cuda")
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 4

    # Let's consider the last nvFuser region
    nvfuser_symbol = nvfuser_symbols[-1]
    # nvfuser_symbol is the following:
    # (__n,) = nvFusion3(__c, __d, __e, __h, __j)
    # __k = prims.add(__h, __j)
    # __l = prims.mul(__k, __e)
    # __m = prims.mul(__l, __d)
    # __n = prims.mul(__m, __c)

    # We will try to find a subgraph for rematerializing __c and __d
    assert "__c" in map(lambda x: x.name, nvfuser_symbol.args)
    assert "__d" in map(lambda x: x.name, nvfuser_symbol.args)
    c_proxy = next(filter(lambda x: x.name == "__c", nvfuser_symbol.args))
    d_proxy = next(filter(lambda x: x.name == "__d", nvfuser_symbol.args))

    # We need to find the producer of __c and __d that is not in subsymbols of nvfuser_symbol
    # We will search for the producer of __c and __d in the flattened trace
    flattened_trace = next(filter(lambda x: str(x._provenance) == "# Constructed by Flatten", traces))

    # Get the producers of __c and __d
    # We should stop at __a, which is the input to the recomputed region
    a_proxy = flattened_trace.bound_symbols[0].output[0]
    assert a_proxy.name == "__a"
    stop_proxies = [a_proxy]

    recomputed_producers = utils.find_producer_symbols(flattened_trace, (c_proxy, d_proxy), stop_proxies)
    assert len(recomputed_producers) == 2
    assert c_proxy.name in map(lambda x: x.output.name, recomputed_producers)
    assert d_proxy.name in map(lambda x: x.output.name, recomputed_producers)
    assert a_proxy.name in map(lambda x: x.args[0].name, recomputed_producers)
