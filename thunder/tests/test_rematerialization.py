import pytest

import torch

import thunder
from thunder import torch as ttorch

from thunder.core import utils
from thunder.core.rematerialization import (
    apply_rematerialization_for_consumer,
    apply_rematerialization_for_producer,
    find_cut,
    find_nvfuser_producer_consumer_pairs,
    find_cut,
)
from thunder.core.transforms import inline, value_and_grad
from thunder.examine import get_fusions
from thunder.tests.framework import instantiate, NOTHING, nvFuserExecutor
from thunder.tests.make_tensor import make_tensor


@inline
@value_and_grad
def func(t0):
    t1 = ttorch.exp(t0)
    t2 = ttorch.exp(t1)
    t3 = ttorch.exp(t2)
    t4 = ttorch.matmul(t3, t3)  # Fusion breaks here
    return t4


@inline
@value_and_grad
def func_with_dropout(t0):
    t1 = ttorch.exp(t0)
    t2 = ttorch.exp(t1)
    t3 = ttorch.dropout(t2, p=0.1)
    t4 = ttorch.exp(t3)
    t5 = ttorch.matmul(t4, t4)  # Fusion breaks here
    return t5


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_producer_symbols(executor, device, _):
    # We will try to find a subgraph for rematerializing __c and __d
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    fusions = get_fusions(trace)

    assert len(fusions) == 2

    # TODO Update this to use the fusions returned from get_fusions
    nvfuser_symbols = tuple(filter(lambda x: x.sym.is_fusion, trace.bound_symbols))

    # Let's consider the last nvFuser region
    nvfuser_symbol = nvfuser_symbols[-1]
    # nvfuser_symbol is the following:
    # (__n,) = nvFusion3(__c, __d, __e, __h, __j)
    # __k = prims.add(__h, __j)
    # __l = prims.mul(__k, __e)
    # __m = prims.mul(__l, __d)
    # __n = prims.mul(__m, __c)

    # We will try to find a subgraph for rematerializing __c and __d
    assert "t2" in map(lambda x: x.name, nvfuser_symbol.args)
    assert "t3" in map(lambda x: x.name, nvfuser_symbol.args)
    c_proxy = next(filter(lambda x: x.name == "t2", nvfuser_symbol.args))
    d_proxy = next(filter(lambda x: x.name == "t3", nvfuser_symbol.args))

    # We need to find the producer of __c and __d that is not in subsymbols of nvfuser_symbol
    # We will search for the producer of __c and __d in the flattened trace
    flattened_trace = next(filter(lambda x: str(x._provenance).startswith("# Constructed by Flatten"), traces))

    # Get the producers of __c and __d
    # We should stop at __a, which is the input to the recomputed region
    a_proxy = flattened_trace.bound_symbols[0].output.collection()[0]
    assert a_proxy.name == "t0"
    stop_proxies = [a_proxy]

    recomputed_producers = utils.find_producer_symbols(flattened_trace, (c_proxy, d_proxy), stop_proxies)
    assert len(recomputed_producers) == 2
    assert c_proxy.name in map(lambda x: x.output.name, recomputed_producers)
    assert d_proxy.name in map(lambda x: x.output.name, recomputed_producers)
    assert a_proxy.name in map(lambda x: x.args[0].name, recomputed_producers)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_apply_rematerialization_producer(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]

    cut = ("t0", "t4")
    assert cut[0] in map(lambda x: x.name, producer.args)
    assert cut[1] in map(lambda x: x.name, producer.output)

    external_producer_outputs = ("t1", "t4")
    new_producer = apply_rematerialization_for_producer(trace, producer, consumer, cut)
    assert new_producer.sym.name == producer.sym.name
    assert new_producer.args == producer.args
    assert new_producer.subsymbols == producer.subsymbols
    assert len(new_producer.output) == 2
    assert cut[1] in (x.name for x in new_producer.output)
    assert external_producer_outputs[0] in (x.name for x in new_producer.output)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_apply_rematerialization_consumer(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]

    cut = ("t0", "t4")
    assert cut[0] in map(lambda x: x.name, producer.args)
    assert cut[1] in map(lambda x: x.name, producer.output)
    assert cut[1] in map(lambda x: x.name, consumer.args)

    new_consumer = apply_rematerialization_for_consumer(producer, consumer, cut)
    assert new_consumer.sym.name == consumer.sym.name
    assert cut[0] in map(lambda x: x.name, new_consumer.args)
    assert cut[1] in map(lambda x: x.name, new_consumer.args)
    assert new_consumer.output == consumer.output
    assert len(new_consumer.args) <= len(consumer.args)
    assert len(new_consumer.subsymbols) > len(consumer.subsymbols)

    # Check that the new consumer has the following structure:
    assert new_consumer.subsymbols[0].sym.name == "exp"
    assert new_consumer.subsymbols[0].args[0].name == cut[0]

    assert new_consumer.subsymbols[1].sym.name == "exp"
    assert new_consumer.subsymbols[1].args[0].name == new_consumer.subsymbols[0].output.name

    # The rest of the subsymbols should be the same as in the original consumer
    assert new_consumer.subsymbols[2:] == consumer.subsymbols


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_nvfuser_producer_consumer_pairs(executor, device, _):
    n_fusion_regions = 7

    @inline
    @value_and_grad
    def func(t0):
        for _ in range(n_fusion_regions):
            t1 = ttorch.exp(t0)
            t2 = ttorch.exp(t1)
            t3 = ttorch.exp(t2)
            t4 = ttorch.matmul(t3, t3)  # Fusion breaks here
            t0 = t4
        return t4

    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    pairs = find_nvfuser_producer_consumer_pairs(trace)
    assert len(pairs) == n_fusion_regions

    # Test that each producer is unique
    producers = set(map(lambda x: x[0], pairs))
    assert len(producers) == n_fusion_regions

    # Test that each consumer is unique
    consumers = set(map(lambda x: x[1], pairs))
    assert len(consumers) == n_fusion_regions

    # Check that pairs are valid, i.e. the consumer really consumes the output
    # of the producer
    for producer, consumer in pairs:
        producer_output_names = map(lambda x: x.name, producer.output)
        consumer_input_names = map(lambda x: x.name, consumer.args)
        assert set(producer_output_names).intersection(set(consumer_input_names))


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_cut(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]
    cut = find_cut(trace, producer, consumer)
    assert cut == ("t0", "t4")


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_cut_dropout(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func_with_dropout, disable_preprocessing=True)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]
    cut = find_cut(trace, producer, consumer)
    # Note t5 is the boolean mask for dropout. It should be chosen over the t6
    # that is the float32 mask. See this issue for the original problem:
    # https://github.com/Lightning-AI/lightning-thunder/issues/706
    assert cut == ("t0", "t5", "t9")


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_cut_one_producer_op_no_args(executor, device, _):
    # TODO: This test will fail once
    # https://github.com/Lightning-AI/lightning-thunder/issues/601 is fixed
    # You can remove this test once the issue is fixed
    # This test would fail on the first assert statement
    def func(x, device):
        t1 = torch.full((3, 3), 1.0, device=device)
        t2 = x @ x.transpose(-2, -1)
        t3 = torch.cos(t2 + t1)
        t4 = torch.sin(t3)
        return t4

    t0 = make_tensor(3, 3, dtype=torch.float32, device=device)
    compiled_func = thunder.compile(func)
    _ = compiled_func(t0, device)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]
    cut = find_cut(trace, producer, consumer)
    assert not cut


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_rematerialization(executor, device, _):
    n_fusion_regions = 7

    @inline
    @value_and_grad
    def func(t0):
        for _ in range(n_fusion_regions):
            t1 = ttorch.cos(t0)
            t2 = ttorch.sin(t1)
            t3 = ttorch.cos(t2)
            t4 = ttorch.matmul(t3, t3)  # Fusion breaks here
            t0 = t4
        return t4

    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)

    # Result with rematerialization and without rematerialization should match
    result_with_remat = thunder.compile(
        func,
        disable_preprocessing=True,
        use_rematerialization=True,
    )(t0)
    assert not isinstance(result_with_remat, Exception)

    result_without_remat = thunder.compile(
        func,
        disable_preprocessing=True,
        use_rematerialization=False,
    )(t0)

    torch.testing.assert_close(result_with_remat, result_without_remat)
