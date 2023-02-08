import sys
from functools import partial, wraps
import math
import types

import pytest
import torch
import torch.nn as nn
from torch import add as tadd
from torch.testing import assert_close, make_tensor

import thunder.langs.torch as ttorch
import thunder.core.script.frontend
import thunder.core.script.passes
import thunder.core.script.python_ir

from . import nanogpt_model
from .framework import Executor, executors, NOTHING, nvFuser, requiresCUDA


def sample_add_fn(x, y):
    return tadd(x, y)


class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 5)
        self.b = torch.nn.Linear(5, 4)

    def forward(self, x: torch.Tensor, flag: bool = True):
        # while flag:
        #    x = 2 * x
        if flag:
            return self.a(x)
        return 2 * x


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_acquisition_compile():
    model = M1()
    gr = thunder.core.script.frontend.acquire_method(model.forward)

    # TODO (t-vi): should these be called automatically? yes.
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a, True), fn(model, a, True))
    assert_close(model(a, False), fn(model, a, False))


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_torch_to_thunder():
    gr = thunder.core.script.frontend.acquire_method(sample_add_fn)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.torch_to_thunder(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)

    traced_fn = thunder.make_traced(thunder_fn, executor="torch")
    a = torch.randn((2, 2), device="cpu", dtype=torch.float32)
    b = torch.randn((2, 2), device="cpu", dtype=torch.float32)

    res = traced_fn(a, b)
    expected = sample_add_fn(a, b)
    assert_close(res, expected)


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_sequential():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 3),
    )

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a), fn(model, a))


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_nanogpt_basic():
    model = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randint(0, 255, (5, 5))
    torch.manual_seed(5)
    res, _ = fn(model, x, None)
    torch.manual_seed(5)
    expected, _ = model.forward(x)

    assert_close(res, expected)


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_split_block():
    def foo(a, b):
        c = a + b
        d = a + c
        return d

    gr = thunder.core.script.frontend.acquire_method(foo, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.split_block(gr, gr.blocks[0], gr.blocks[0].nodes[1])
    dot = thunder.core.script.graph.make_dot(gr, add_names=True)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(5)
    b = torch.randn(5)
    assert_close(fn(a, b), foo(a, b))


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_inline_submodule():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(5, 10)
            self.l2 = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.l2(torch.tanh(self.l1(x)))

    m = MLP()
    gr = thunder.core.script.frontend.acquire_method(m.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)

    nodes_to_inline = [gr.blocks[0].nodes[0], gr.blocks[0].nodes[2]]
    for n in nodes_to_inline:
        thunder.core.script.passes.inline_method_call(gr, n)

    assert len(gr.blocks) > 1

    thunder.core.script.passes.merge_blocks_where_possible(gr)

    assert len(gr.blocks) == 1

    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randn(5, 5)
    assert_close(fn(m, x), m(x))

    # explicitly check for things to have been inlined?


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
def test_inline_submodule_and_convert_to_thunder():
    model = nanogpt_model.MLP(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(model.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.graph.check_graph(gr)
    thunder.core.script.passes.inline_submodule_calls(gr)
    thunder.core.script.graph.check_graph(gr)
    thunder.core.script.passes.merge_blocks_where_possible(gr)
    thunder.core.script.graph.check_graph(gr)
    thunder.core.script.passes.torch_to_thunder(gr)
    thunder.core.script.graph.check_graph(gr)

    fn = thunder.core.script.python_ir.generate_function(gr)

    ### now trace fn and check things work...


def test_nanogpt_inlining_unrolling():
    m = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(m.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)

    ## Check on the graph
    thunder.core.script.graph.check_graph(gr)

    # these will likely change specialization, more inlining, ...
    # but lets check when it happens
    assert len(gr.blocks) == 5
    assert sum(len(bl.nodes) for bl in gr.blocks) == 579

    # has everything been inlined/unrolled?
    funcs = {
        thunder.core.script.passes.find_and_evaluate_method_through_phi_parent(n.inputs[0])  # for function calls
        or n.inputs[0].name  # for Tensor methods (but we don't check that)
        or n.inputs[0].node.i.opname  # for the oddball assertion instantiation
        for n in gr.nodes()
        if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION", "CALL_FUNCTION_KW"}
    }
    allowed_funcs = {
        float,
        math.sqrt,
        ## This might eventually go (i.e. be inlined as well)...
        nanogpt_model.new_gelu,
        ## PyTorch functions
        torch.arange,
        torch.nn.functional.cross_entropy,
        torch.nn.functional.dropout,
        torch.nn.functional.embedding,
        torch.nn.functional.layer_norm,
        torch.nn.functional.linear,
        torch.nn.functional.softmax,
        ## these should be Tensor methods
        "contiguous",
        "masked_fill",
        "size",
        "split",
        "transpose",
        "unsqueeze",
        "view",
        ## there is an oddball (handled above) from instantiating the AssertionError
        "LOAD_ASSERTION_ERROR",
    }
    assert not funcs - allowed_funcs

    fn = thunder.core.script.python_ir.generate_function(gr)
    x = torch.randint(0, 255, (5, 5))

    torch.manual_seed(5)
    o = fn(m, x, None)
    torch.manual_seed(5)

    o2 = m.forward(x)

    assert_close(o[0], o2[0])


def test_nanogpt_functionalization():
    m = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(m.forward, verbose=False)
    thunder.core.script.frontend.make_ssa(gr)
    thunder.core.script.frontend.make_single_return(gr)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)
    additional_param_names = thunder.core.script.passes.module_to_function(gr)
    thunder.core.script.graph.check_graph(gr)

    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randint(0, 255, (5, 5))

    sd = m.state_dict()
    additional_params = [sd[n.replace("[", "").replace("]", "")] for n in additional_param_names]

    torch.manual_seed(5)
    o = fn(x, None, *additional_params)
    torch.manual_seed(5)

    o2 = m.forward(x)

    assert_close(o[0], o2[0])


def bar(a, b):
    return torch.nn.functional.linear(a, b)


def foo(a, c_fc_weight, c_proj_weight):
    b = bar(a, c_fc_weight)
    # c = new_gelu(b)
    # d = torch.nn.functional.linear(c, c_proj_weight)
    # e = torch.nn.functional.dropout(d)
    # return b
    return b


@pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 11),
    reason="requires python3.10",
)
@requiresCUDA
def test_inlining_function_and_convert_to_thunder():
    def convert_to_thunder(fn):
        global gr
        gr = thunder.core.script.frontend.acquire_method(fn, verbose=False)

        thunder.core.script.frontend.make_ssa(gr)
        thunder.core.script.frontend.make_single_return(gr)

        thunder.core.script.passes.inline_submodule_calls(gr)
        thunder.core.script.passes.inline_method_call(gr, gr.blocks[0].nodes[0])
        thunder.core.script.passes.merge_blocks_where_possible(gr)
        thunder.core.script.graph.check_graph(gr)
        thunder.core.script.passes.torch_to_thunder(gr)
        thunder.core.script.graph.check_graph(gr)

        thunder_fn = thunder.core.script.python_ir.generate_function(gr)

        return thunder_fn

    n = 4
    a = make_tensor((n, n), dtype=torch.float32, device="cuda")
    c_fc_weight = make_tensor((4 * n, n), dtype=torch.float32, device="cuda")
    c_proj_weight = make_tensor((n, 4 * n), dtype=torch.float32, device="cuda")
    thunder_foo = convert_to_thunder(foo)

    thunder_fn = thunder.make_traced(thunder_foo, executor="nvfuser")

    torch_result = foo(a, c_fc_weight, c_proj_weight)
    thunder_result = thunder_fn(a, c_fc_weight, c_proj_weight)

    assert_close(torch_result, thunder_result)


@executors(dtypes=(thunder.float32,))
def test_preprocess_option(executor, device, dtype):
    def foo(a, b):
        return torch.add(a, b)

    tdtype = ttorch.torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((2, 2), device=device, dtype=tdtype)

    thunder_fn = thunder.make_traced(foo, executor=executor, _preprocess=True)

    thunder_result = thunder_fn(a, b)
    torch_result = foo(a, b)
    assert_close(thunder_result, torch_result)


def _nanogpt_mlp_helper(device, dtype, thunder_fn, torch_fn):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    n = 4
    a = make((n, n))
    c_fc_weight = make((4 * n, n))
    c_proj_weight = make((n, 4 * n))

    thunder_result = thunder_fn(a, c_fc_weight, c_proj_weight)
    torch_result = torch_fn(a, c_fc_weight, c_proj_weight)

    assert_close(thunder_result, torch_result)


# TODO: enable the following tests


@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional_simplified(executor, device, dtype):
    def nanogpt_mlp_functional_simplified(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        d = torch.nn.functional.linear(b, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = thunder.make_traced(nanogpt_mlp_functional_simplified, executor=executor, _preprocess=True)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional_simplified)


@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional_inlined(executor, device, dtype):
    def nanogpt_mlp_functional_inlined(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        c = 0.5 * b * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (b + 0.044715 * torch.pow(b, 3.0))))
        d = torch.nn.functional.linear(c, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = thunder.make_traced(nanogpt_mlp_functional_inlined, executor=executor, _preprocess=True)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional_inlined)


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional(executor, device, dtype):
    def nanogpt_mlp_functional(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        c = new_gelu(b)
        d = torch.nn.functional.linear(c, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = thunder.make_traced(nanogpt_mlp_functional, executor=executor, _preprocess=True)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional)


# TODO: enable me by converting torch inputs to Thunder inputs when proxying
# TODO: once this test works, also test acquiring the function from a collection
# @executors(dtypes=(thunder.float32,))
# def test_fn_input(executor, device, dtype):
#     make = partial(make_tensor, device=device, dtype=dtype)

#     def foo(fn, *args):
#         return fn(*args)

#     thunder_fn = thunder.make_traced(foo, executor=executor, _preprocess=True)

#     fn = torch.add
#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(fn, a, b)
#     torch_result = foo(fn, a, b)

#     assert_close(thunder_result, torch_result)

# TODO: FIXME
# @executors(dtypes=(thunder.float32,))
# def test_local_translation(executor, device, dtype):
#     make = partial(make_tensor, device=device, dtype=dtype)

#     def foo(a, b):

#         def _convert(x):
#             return torch.add(x, 1)

#         a, b = tuple(_convert(x) for x in (a, b))

#         return a, b

#     thunder_fn = thunder.make_traced(foo, executor=executor, _preprocess=True)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(fn, a, b)
#     torch_result = foo(fn, a, b)

#     assert_close(thunder_result, torch_result)

# @executors(dtypes=(thunder.float32,))
# def test_local_wrapped_translation(executor, device, dtype):
#     make = partial(make_tensor, device=device, dtype=dtype)

#     def foo(a, b):

#         @wraps(torch.add)
#         def _convert(x):
#             return torch.add(x, 1)

#         a, b = tuple(_convert(x) for x in (a, b))

#         return a, b

#     thunder_fn = thunder.make_traced(foo, executor=executor, _preprocess=True)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(fn, a, b)
#     torch_result = foo(fn, a, b)

#     assert_close(thunder_result, torch_result)


# @executors(dtypes=(thunder.float32,))
# def test_lambda_translation(executor, device, dtype):
#     make = partial(make_tensor, device=device, dtype=dtype)

#     def foo(a, b):
#         return map(lambda a: torch.add(a, 1), (a, b))

#     thunder_fn = thunder.make_traced(foo, executor=executor, _preprocess=True)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(fn, a, b)
#     torch_result = foo(fn, a, b)

#     assert_close(thunder_result, torch_result)


# @executors(dtypes=(thunder.float32,))
# def test_nanogpt_mlp(executor, device, dtype):

#     def new_gelu(x):
#         return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

#     n = 4
#     class MLP(nn.Module):

#         def __init__(self):
#             super().__init__()
#             n = 4
#             self.c_fc = nn.Linear(n, 4 * n)
#             self.c_proj = nn.Linear(4 * n, n)
#             self.dropout = nn.Dropout(p=0.0)

#         def forward(self, a):
#             b = self.c_fc(a)
#             c = new_gelu(b)
#             d = self.c_proj(c)
#             e = self.dropout(d)
#             return e

#     tdtype = ttorch.torch_dtype(dtype)

#     mlp = MLP()
#     mlp.to(device, dtype=tdtype)

#     thunder_fn = thunder.make_traced(mlp, executor=executor, _preprocess=True)

#     make = partial(make_tensor, dtype=tdtype, device=device)

#     n = 4
#     a = make((n, n))

#     thunder_result = thunder_fn(a)
#     torch_result = mlp(a)

#     assert_close(thunder_result, torch_result)
