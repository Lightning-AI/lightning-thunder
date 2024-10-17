import pytest
from functools import partial, wraps

import torch

import thunder
import thunder.examine
from thunder import torch as ttorch

from thunder.core import utils, devices
from thunder.core.rematerialization import (
    apply_rematerialization_for_consumer,
    apply_rematerialization_for_producer,
    find_cut,
    find_external_producer_outputs,
    find_filtered_producer_consumer_pairs,
    find_nvfuser_producer_consumer_pairs,
)
from thunder.core import prims
from thunder.core.transforms import value_and_grad
from thunder.core.trace import TraceCtx
from thunder.examine import get_fusions
from thunder.tests.framework import instantiate, NOTHING, nvFuserExecutor, TorchExecutor, requiresCUDA
from thunder.tests.make_tensor import make_tensor
import thunder.torch as ltorch


@value_and_grad
def func(t0):
    t1 = ttorch.exp(t0)
    t2 = ttorch.exp(t1)
    t3 = ttorch.exp(t2)
    t4 = ttorch.matmul(t3, t3)  # Fusion breaks here
    return t4


@value_and_grad
def func_with_dropout(t0):
    t1 = ttorch.exp(t0)
    t2 = ttorch.exp(t1)
    t3 = ttorch.dropout(t2, p=0.1)
    t4 = ttorch.exp(t3)
    t5 = ttorch.matmul(t4, t4)  # Fusion breaks here
    return t5


def disable_rematerialization_in_nvfuser_fusion(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from thunder.executors.nvfuserex_impl import ex

        ex._use_rematerialization = False
        try:
            return func(*args, **kwargs)
        finally:
            ex._use_rematerialization = True

    return wrapper


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
@disable_rematerialization_in_nvfuser_fusion
def test_find_producer_symbols(executor, device, _):
    # We will try to find a subgraph for rematerializing __c and __d
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    initial_trace = thunder.trace()(func, t0)
    compiled_func = thunder.jit(initial_trace.python_callable())
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
    flattened_trace = traces[0]

    # Get the producers of __c and __d
    # We should stop at __a, which is the input to the recomputed region
    a_proxy = flattened_trace.bound_symbols[0].output
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
    initial_trace = thunder.trace()(func, t0)
    compiled_func = thunder.jit(initial_trace.python_callable())
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]

    cut = ("t0", "t4")
    assert cut[0] in map(lambda x: x.name, producer.args)
    assert cut[1] in map(lambda x: x.name, producer.output)

    external_producer_outputs = tuple(x for x in producer.output if x.name in ("t1", "t4"))
    new_producer = apply_rematerialization_for_producer(external_producer_outputs, producer, cut)
    assert new_producer.sym.name == producer.sym.name
    assert new_producer.args == producer.args
    assert new_producer.subsymbols == producer.subsymbols
    assert len(new_producer.output) == 2
    assert cut[1] in (x.name for x in new_producer.output)
    assert external_producer_outputs[0].name in (x.name for x in new_producer.output)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
@disable_rematerialization_in_nvfuser_fusion
def test_apply_rematerialization_consumer(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    initial_trace = thunder.trace()(func, t0)
    compiled_func = thunder.jit(initial_trace.python_callable())
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    # Let's consider a pair of the first and the last nvFuser regions
    # We call them producer and consumer because the last region consumes some
    # outputs of the first region
    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[1]

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

    assert new_consumer.subsymbols[2].sym.name == "exp"
    assert new_consumer.subsymbols[2].args[0].name == new_consumer.subsymbols[0].output.name

    # The subsymbols are reordered according to the topological order.
    # The rest of the subsymbols should be the same as in the original consumer, in this case the symbols except at position 0 and 2.
    assert (new_consumer.subsymbols[1], *new_consumer.subsymbols[3:]) == tuple(consumer.subsymbols)

    # Test case when duplicated symbols appear in producer and consumer
    duplicated_sym = [sym for sym in producer.subsymbols if sym.output.name == "t3"]
    assert len(duplicated_sym) == 1
    from dataclasses import replace

    # Both producer and consumer subsymbols contain `t3 = prims.exp(t2)`
    consumer_with_duplicated_syms_as_producer = replace(
        consumer, args=tuple(a for a in consumer.args if a.name != "t3")
    )
    consumer_with_duplicated_syms_as_producer.subsymbols = (duplicated_sym[0], *consumer.subsymbols)
    new_consumer_case2 = apply_rematerialization_for_consumer(producer, consumer_with_duplicated_syms_as_producer, cut)
    # The new_consumer_case2 generated with duplicated subsymbols between producer and consumer is the same as the previous new_consumer
    assert tuple(new_consumer.subsymbols) == tuple(new_consumer_case2.subsymbols)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
@disable_rematerialization_in_nvfuser_fusion
def test_apply_rematerialization_consumer_early_exit(executor, device, _):
    @value_and_grad
    def foo(t0):
        t1 = ttorch.exp(t0)
        t2 = ttorch.matmul(t1, t1)
        return t2

    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    initial_trace = thunder.trace()(foo, t0)
    compiled_func = thunder.jit(initial_trace.python_callable())
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[1]

    # Create a cut that has t0 as extra information and
    # that contains all arguments(t2) from consumer.
    cut = ("t0", "t2")
    new_consumer = apply_rematerialization_for_consumer(producer, consumer, cut)

    # Check that the new consumer is the old consumer
    assert id(new_consumer) == id(consumer)
    assert tuple(new_consumer.subsymbols) == tuple(consumer.subsymbols)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_nvfuser_producer_consumer_pairs(executor, device, _):
    n_fusion_regions = 7

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
    initial_trace = thunder.trace()(func, t0)
    compiled_func = thunder.jit(initial_trace.python_callable())
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    pairs = find_nvfuser_producer_consumer_pairs(trace)
    assert len(pairs) == n_fusion_regions

    # Test that the order of pairs follows the order of nvFuser regions in the
    # trace
    nvfuser_regions = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    producers_trace_order = tuple(filter(lambda x: x in map(lambda x: x[0], pairs), nvfuser_regions))
    consumers_trace_order = tuple(filter(lambda x: x in map(lambda x: x[1], pairs), nvfuser_regions))
    # Consumers order is reversed because the consumer of the first nvFuser
    # block is the last nvFuser block and so on
    assert consumers_trace_order == tuple(map(lambda x: x[1], reversed(pairs)))
    assert producers_trace_order == tuple(map(lambda x: x[0], pairs))

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
    executors=(TorchExecutor,),
)
def test_find_filtered_producer_consumer_pairs_multiple_consumers(executor, device, _):
    from thunder.core import prims

    find_pairs = partial(
        find_filtered_producer_consumer_pairs, filter_func=lambda bsym: bsym.sym.id == prims.PrimIDs.ADD
    )

    def func(t0):
        t1 = prims.exp(t0)
        t2 = prims.add(t0, t0)  # one filtered producer
        t3 = prims.add(t2, t0)  # first filtered consumer
        t4 = prims.add(t2, t1)  # second filtered consumer
        return t3, t4

    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    compiled_func = executor.make_callable(func)
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[0]
    pairs = find_pairs(trace)
    assert len(pairs) == 2

    # Check the order of pairs
    first_consumer = trace.bound_symbols[-3]
    second_consumer = trace.bound_symbols[-2]
    assert pairs[0][1] == first_consumer
    assert pairs[1][1] == second_consumer

    # Check that pairs are valid, i.e. the consumer really consumes the output
    # of the producer
    for producer, consumer in pairs:
        producer_output_names = map(lambda x: x.name, utils.sequencify(producer.output))
        consumer_input_names = map(lambda x: x.name, consumer.args)
        assert set(producer_output_names).intersection(set(consumer_input_names))


@instantiate(dtypes=NOTHING, executors=(nvFuserExecutor,), devicetypes=(devices.DeviceType.CUDA,))
def test_find_fusion_producer_consumer_pairs_multiple_producers(executor, device, _):
    from torch.nn.functional import layer_norm, linear

    def func(
        t0: "f32[2, 2, 2, 2]",
        t1: "f32[4, 4]",
        x: "f32[2, 2, 4]",
        t2: "f32[4]",
        t3: "f32[2, 4]",
        t4: "f32[4, 2]",
        t5: "f32[4]",
        t6: "f32[2, 4]",
    ):
        y_1: "f32[2, 2, 4]" = t0.view(2, 2, 4)
        linear_1: "f32[2, 2, 4]" = linear(y_1, t1, None)
        x_1: "f32[2, 2, 4]" = x + linear_1
        layer_norm_1: "f32[2, 2, 4]" = layer_norm(x_1, (4,), t2, None, 1e-05)
        linear_2: "f32[2, 2, 2]" = linear(layer_norm_1, t3, None)
        linear_3: "f32[2, 2, 4]" = linear(linear_2, t4, None)
        x_2: "f32[2, 2, 4]" = x_1 + linear_3
        layer_norm_2: "f32[2, 2, 4]" = layer_norm(x_2, (4,), t5, None, 1e-05)
        linear_4: "f32[2, 2, 2]" = linear(layer_norm_2, t6, None)
        split_1 = linear_4.split(4, dim=2)
        return split_1

    make = partial(make_tensor, device=device, dtype=torch.float32, requires_grad=True)
    t0 = make((2, 2, 2, 2))
    t1 = make((4, 4))
    x = make(2, 2, 4)
    t2 = make(4)
    t3 = make((2, 4))
    t4 = make((4, 2))
    t5 = make(4)
    t6 = make((2, 4))

    from thunder.executors.torch_compile import torch_compile_cat_ex

    try:
        compiled_func = thunder.jit(func, executors=(torch_compile_cat_ex, thunder.nvfuser_executor))
        _ = compiled_func(
            t0,
            t1,
            x,
            t2,
            t3,
            t4,
            t5,
            t6,
        )
    except Exception as e:
        pytest.fail(f"Rematerialization fails (issue #1038): {e}")


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_cut(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    intial_trace = thunder.trace()(func, t0)
    compiled_func = thunder.jit(intial_trace.python_callable())
    _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[1]
    ext_external_producer_outputs = find_external_producer_outputs(utils.consumers(trace), (), producer, consumer)
    cut = find_cut(ext_external_producer_outputs, producer, consumer)
    assert cut == ("t0", "t4")


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_find_cut_dropout(executor, device, _):
    t0 = make_tensor(2, 2, dtype=torch.float32, device=device)
    from unittest.mock import patch, MagicMock

    # mock the replace_uniform transform to return the input trace
    replace_uniform_mock = MagicMock(side_effect=lambda trc: trc)

    with patch("thunder.core.rematerialization.replace_uniform", new=replace_uniform_mock):
        intial_trace = thunder.trace()(func_with_dropout, t0)
        compiled_func = thunder.jit(intial_trace.python_callable())
        _ = compiled_func(t0)
    traces = thunder.last_traces(compiled_func)
    trace = traces[-1]
    nvfuser_symbols = tuple(filter(lambda x: x.sym.name.startswith("nvFusion"), trace.bound_symbols))
    assert len(nvfuser_symbols) == 2

    producer = nvfuser_symbols[0]
    consumer = nvfuser_symbols[1]
    ext_producer_outputs = find_external_producer_outputs(utils.consumers(trace), (), producer, consumer)
    cut = find_cut(ext_producer_outputs, producer, consumer)
    assert cut[0] == producer.args[0].name
    # Note cut[1] is the boolean mask for dropout. It should
    # be chosen over the float32 mask. See this issue: "The Recomputation
    # Algorithm on Dropout choses a float32 mask to save"
    producer_output_names = tuple(o.name for o in producer.output)
    assert cut[1] in producer_output_names
    assert cut[2] in producer_output_names


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_rematerialization(executor, device, _):
    n_fusion_regions = 7

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
    initial_trace = thunder.trace()(func, t0)
    result_with_remat = thunder.jit(initial_trace.python_callable())(t0)
    assert not isinstance(result_with_remat, Exception)

    result_without_remat = disable_rematerialization_in_nvfuser_fusion(thunder.jit(initial_trace.python_callable()))(t0)

    torch.testing.assert_close(result_with_remat, result_without_remat)


@requiresCUDA
def test_rematerialization_name_collision():
    # This test is to verify that we don't have name collision in forward and backward trace
    # (which used to produce an error in remat pass)
    def forward(x):
        return x.softmax(dim=1, dtype=torch.float)

    jforward = thunder.jit(forward)

    x = torch.randn([32768, 8], dtype=torch.bfloat16, device="cuda", requires_grad=True)

    actual = jforward(x)
    expected = forward(x)

    torch.testing.assert_close(actual, expected)

    grad_output = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, grad_output)
    expected_grad = torch.autograd.grad(expected, x, grad_output)

    torch.testing.assert_close(actual_grad, expected_grad)


@requiresCUDA
def test_not_rematerialize_matmul():
    nn = torch.nn

    class MLP(nn.Module):
        def __init__(self, embedding_size: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(embedding_size, embedding_size * 4),
                nn.GELU(),
                nn.Linear(embedding_size * 4, embedding_size),
                nn.Dropout(),
            )

        def forward(self, x):
            return self.layers(x)

    batch_size, sequence_length, embedding_size = 2, 3, 4
    inp = torch.randn(batch_size, sequence_length, embedding_size, device="cuda", requires_grad=True)

    model = MLP(embedding_size)
    model.to("cuda")

    # At the time of writing, linear and matmul are not fused into nvFuser
    # regions by default therefore, we should enable them separately
    jmodel = thunder.jit(model, nv_enable_linear=True, nv_enable_matmul=True)
    jmodel(inp)

    def assert_subsymbol_count(trace: TraceCtx, /, num_linears: int, num_matmuls: int):
        fusions = thunder.examine.get_fusion_symbols(trace)
        assert len(fusions) == 1

        subsymbol_ids = [subsymbol.sym.id for subsymbol in fusions[0].subsymbols]
        assert subsymbol_ids.count(prims.PrimIDs.LINEAR) == num_linears
        assert subsymbol_ids.count(prims.PrimIDs.MATMUL) == num_matmuls

    fw_trace = thunder.last_traces(jmodel)[-1]
    assert_subsymbol_count(fw_trace, num_linears=2, num_matmuls=0)

    bw_trace = thunder.last_backward_traces(jmodel)[-1]
    assert_subsymbol_count(bw_trace, num_linears=0, num_matmuls=4)
