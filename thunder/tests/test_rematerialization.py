import torch

import thunder
from thunder import torch as ttorch

from thunder.core import utils
from thunder.core.rematerialization import (
    apply_rematerialization_for_consumer,
    apply_rematerialization_for_producer,
    find_nvfuser_producer_consumer_pairs,
    find_cut,
)
from thunder.core.transforms import inline, value_and_grad
from thunder.tests.framework import instantiate, NOTHING, nvFuserExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.examine import get_fusions


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
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
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
    assert "__2" in map(lambda x: x.name, nvfuser_symbol.args)
    assert "__3" in map(lambda x: x.name, nvfuser_symbol.args)
    c_proxy = next(filter(lambda x: x.name == "__2", nvfuser_symbol.args))
    d_proxy = next(filter(lambda x: x.name == "__3", nvfuser_symbol.args))

    # We need to find the producer of __c and __d that is not in subsymbols of nvfuser_symbol
    # We will search for the producer of __c and __d in the flattened trace
    flattened_trace = next(filter(lambda x: str(x._provenance) == "# Constructed by Flatten", traces))

    # Get the producers of __c and __d
    # We should stop at __a, which is the input to the recomputed region
    a_proxy = flattened_trace.bound_symbols[0].output[0]
    assert a_proxy.name == "__0"
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
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]

    cut = ("__0", "__4")
    assert cut[0] in map(lambda x: x.name, producer.args)
    assert cut[1] in map(lambda x: x.name, producer.output)

    external_producer_outputs = ("__1", "__4")
    new_producer = apply_rematerialization_for_producer(trace, producer, consumer, cut)
    assert new_producer.sym.name == producer.sym.name
    assert new_producer.args == producer.args
    assert new_producer.subsymbols == producer.subsymbols
    assert len(new_producer.output) == 4
    assert cut[1] in (x.name for x in new_producer.output)
    assert external_producer_outputs[0] in (x.name for x in new_producer.output)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_apply_rematerialization_consumer(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]

    cut = ("__0", "__4")
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
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
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
    _, traces = thunder.compile_with_info(func, disable_preprocessing=True)(t0)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[-1]
    cut = find_cut(trace, producer, consumer)
    assert cut == ("__0", "__4")
