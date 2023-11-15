import inspect
import math
import operator
import sys
import traceback
from contextlib import nullcontext as does_not_raise
from functools import partial

import pytest
import torch
from lightning_utilities import compare_version
from torch import add as tadd
from torch.testing import assert_close, make_tensor

import thunder.core.script.frontend
import thunder.core.script.passes
import thunder.core.script.python_ir
import thunder.core.script.python_ir_data
import thunder.torch as ltorch
from thunder.core.script.noinline import noinline, invoke_noinline
from thunder.core.symbol import Symbol
from thunder.core.utils import enable_debug_asserts
from thunder.tests import nanogpt_model, lit_llama_model, lit_gpt_model
from thunder.tests.framework import instantiate, requiresCUDA, requiresNVFuser, IN_CI

from thunder import pytorch_executor, nvfuser_executor

torchex = [pytorch_executor]
nvfuserex = [nvfuser_executor, pytorch_executor]

enable_debug_asserts()


def skipif_not_pytorch_2_1(f):
    return pytest.mark.skipif(
        # use_base_version=True allows checking against versions with nightly modifiers
        compare_version("torch", operator.lt, "2.1.0", use_base_version=True),
        reason=f"requires PyTorch >= 2.1, got {torch.__version__=}",
    )(f)


def skipif_not_python_3_10(f):
    return pytest.mark.skipif(
        not thunder.core.script.python_ir_data.SUPPORTS_PREPROCESSING,
        reason=f"requires python3.10, got {sys.version_info=}",
    )(f)


def skipif_windows(f):
    return pytest.mark.skipif(
        sys.platform == "win32",
        reason=f"has trouble on windows (e.g. memory)",
    )(f)


def skipif_not_CI(f):
    return pytest.mark.skipif(not IN_CI, reason=f"Not in CI.")(f)


def skipif_CI(f):
    return pytest.mark.skipif(IN_CI, reason=f"In CI.")(f)


def _helper_get_func_calls(gr):
    return {
        thunder.core.script.passes.find_and_evaluate_method_through_phi_parent(n.inputs[0])  # for function calls
        or n.inputs[0].name  # for Tensor methods (but we don't check that)
        or n.inputs[0].node.i.opname  # for the oddball assertion instantiation
        for n in gr.nodes()
        if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION", "CALL_FUNCTION_KW"}
    }


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


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@skipif_not_python_3_10
def test_acquisition_compile():
    model = M1()
    gr = thunder.core.script.frontend.acquire_method(model.forward)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a, True), fn(model, a, True))
    assert_close(model(a, False), fn(model, a, False))

    # Test kwargs
    assert_close(model(a, flag=False), fn(model, a, flag=False))
    assert_close(model(x=a, flag=True), fn(model, x=a, flag=True))


@skipif_not_python_3_10
def test_torch_to_thunder():
    gr = thunder.core.script.frontend.acquire_method(sample_add_fn)
    thunder.core.script.passes.torch_to_thunder(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)

    traced_fn = thunder.compile(thunder_fn, executors_list=torchex)
    a = torch.randn((2, 2), device="cpu", dtype=torch.float32)
    b = torch.randn((2, 2), device="cpu", dtype=torch.float32)

    res = traced_fn(a, b)
    expected = sample_add_fn(a, b)
    assert_close(res, expected)


@skipif_not_python_3_10
def test_sequential():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 3),
    )

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(2, 3)
    assert_close(model(a), fn(model, a))


@skipif_not_python_3_10
def test_thunder_compile():
    model0 = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 3),
    )
    models = [model0, torch.nn.Sequential(model0)]
    for model in models:
        tom = thunder.compile(model)

        # several times for caching
        a = torch.randn(2, 3)
        for _ in range(3):
            assert_close(model(a), tom(a))

    tfn = thunder.compile(new_gelu)
    for _ in range(3):
        assert_close(new_gelu(a), tfn(a))


@skipif_not_python_3_10
def test_nanogpt_basic():
    model = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(model.forward)
    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randint(0, 255, (5, 5))
    torch.manual_seed(5)
    res, _ = fn(model, x, None)
    torch.manual_seed(5)
    expected, _ = model.forward(x)

    assert_close(res, expected)


@skipif_not_python_3_10
def test_split_block():
    def foo(a, b):
        c = a + b
        d = a + c
        return d

    gr = thunder.core.script.frontend.acquire_method(foo)
    thunder.core.script.passes.split_block(gr, gr.blocks[0], gr.blocks[0].nodes[1])
    fn = thunder.core.script.python_ir.generate_function(gr)

    a = torch.randn(5)
    b = torch.randn(5)
    assert_close(fn(a, b), foo(a, b))


# there could be more versions of this
def fn1(a, /, b, c=3, *args, d=5, **kwargs):
    return f"{a=}, {b=}, {c=}, {d=}, {args=}, {kwargs=}"


@skipif_not_python_3_10
def test_arg_handling():
    gr = thunder.core.script.frontend.acquire_method(fn1)
    generated_fn = thunder.core.script.python_ir.generate_function(gr)

    # structureal tests
    assert inspect.signature(fn1) == inspect.signature(generated_fn)

    # also test errors?
    for args, kwargs in (
        ((2, 3, 4, 5, 7), dict(abc=3)),
        ((2, 3, 4), dict(abc=3, d=2)),
        ((1, 2), dict()),
    ):
        assert fn1(*args, **kwargs) == generated_fn(*args, **kwargs)


@skipif_not_python_3_10
def test_inline_submodule():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(5, 10)
            self.l2 = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.l2(torch.tanh(self.l1(x)))

    m = MLP()
    gr = thunder.core.script.frontend.acquire_method(m.forward)

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


@skipif_not_pytorch_2_1
@skipif_not_python_3_10
@pytest.mark.parametrize(
    "name",
    (
        # TODO FIXME https://github.com/Lightning-AI/lightning-thunder/issues/1408
        # "gpt-neox-like",
        "llama1-like",
        "long-context-like",
        "llama2-like",
        "falcon-7b-like",
        "falcon-40b-like",
        "codellama2-like",
    ),
)
@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_litgpt_variants(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    x = torch.randint(0, 200, (5, 5), device=device)
    config = lit_gpt_model.Config.from_name(name)

    with device:
        reference = lit_gpt_model.GPT(config)
    expected_logits = reference(x)
    expected_logits.sum().backward()

    with device:
        model = lit_gpt_model.GPT(config)
    model.load_state_dict(reference.state_dict())
    tom = thunder.compile(model, executors_list=nvfuserex if device.type == "cuda" else torchex)
    actual_logits = tom(x)
    assert_close(actual_logits, expected_logits)

    if device.type == "cuda" in config._mlp_class == "GptNeoxMLP":
        # https://github.com/Lightning-AI/lightning-thunder/issues/1187
        with pytest.raises(RuntimeError, match=r"out->getValType\(\) == ValType::TensorView"):
            actual_logits.sum().backward()
        return
    else:
        actual_logits.sum().backward()

    for param1, param2 in zip(reference.parameters(), tom.parameters()):
        assert param1 is not param2
        assert param1.grad is not None
        torch.testing.assert_close(param1.grad, param2.grad, rtol=1e-2, atol=1e-2)


@skipif_not_python_3_10
def test_llama_block_compile():
    m = lit_llama_model.Block(lit_llama_model.LLaMAConfig.from_name("7B"))
    m2 = lit_llama_model.Block(lit_llama_model.LLaMAConfig.from_name("7B"))
    m2.load_state_dict(m.state_dict())
    tom = thunder.compile(m2, executors_list=torchex, disable_torch_autograd_support=True)
    inp = torch.randn(1, 8, 4096)
    expected_result = m(inp)

    thunder_result = tom(inp)
    assert_close(expected_result, thunder_result)
    assert_close(m.attn.rope_cache, m2.attn.rope_cache)

    # TODO: enable:
    # inp = torch.randn(1, 8, 4096, requires_grad=True)
    expected_result = m(inp)
    thunder_result = tom(inp)
    assert_close(expected_result, thunder_result)

    # go = torch.ones(expected_result.shape, dtype=expected_result.dtype, device=expected_result.device)

    # expected_gr, = torch.autograd.grad(expected_result, inp, go)
    # thunder_gr, = torch.autograd.grad(thunder_result, inp, go)
    # assert_close(expected_result, thunder_result)


@skipif_not_python_3_10
def test_llama_block_inlining():
    m = lit_llama_model.Block(lit_llama_model.LLaMAConfig.from_name("7B"))

    gr = thunder.core.script.frontend.acquire_method(m.forward)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)
    thunder.core.script.passes.strongly_inline_functions(gr)

    ## Check on the graph
    thunder.core.script.graph.check_graph(gr)

    # has everything been inlined/unrolled?
    funcs = _helper_get_func_calls(gr)
    allowed_funcs = {
        ## PyTorch functions
        torch.arange,
        torch.cos,
        torch.mean,
        torch.outer,
        torch.rsqrt,
        torch.sin,
        torch.stack,
        torch.nn.functional.silu,
        torch.nn.functional.scaled_dot_product_attention,
        torch.nn.functional.linear,
        ## these should be Tensor methods
        "contiguous",
        "flatten",
        "float",
        "size",
        "split",
        "transpose",
        "view",
        "type_as",
        "half",
    }
    disallowed = funcs - allowed_funcs
    unseen = allowed_funcs - funcs
    assert (not disallowed) and (not unseen)


@skipif_not_python_3_10
def test_nanogpt_inlining_unrolling():
    m = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(m.forward)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)

    ## Check on the graph
    thunder.core.script.graph.check_graph(gr)

    # these will likely change specialization, more inlining, ...
    # but lets check when it happens
    assert len(gr.blocks) == 80
    assert sum(len(bl.nodes) for bl in gr.blocks) == 669

    # has everything been inlined/unrolled?
    funcs = _helper_get_func_calls(gr)
    allowed_funcs = {
        float,
        math.sqrt,
        ## PyTorch functions
        torch.arange,
        torch.nn.functional.cross_entropy,
        torch.nn.functional.dropout,
        torch.nn.functional.embedding,
        torch.nn.functional.layer_norm,
        torch.nn.functional.linear,
        torch.nn.functional.softmax,
        torch.nn.functional.scaled_dot_product_attention,
        torch.nn.functional.gelu,
        ## these should be Tensor methods
        "contiguous",
        "masked_fill",
        "size",
        "split",
        "transpose",
        "view",
        ## there is an oddball (handled above) from instantiating the AssertionError
        "LOAD_ASSERTION_ERROR",
        # This isinstance check comes from an update to allow the nanogpt model to work in a torch.nn.Sequential module
        isinstance,
    }
    assert not (delta := funcs ^ allowed_funcs), delta

    fn = thunder.core.script.python_ir.generate_function(gr)
    x = torch.randint(0, 255, (5, 5))

    torch.manual_seed(5)
    o = fn(m, x, None)
    torch.manual_seed(5)

    o2 = m.forward(x)

    assert_close(o[0], o2[0])


@skipif_not_python_3_10
def test_exception_source_line():
    m = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Linear(3, 3),  # intentionally broken
    )
    tom = thunder.compile(m, executors_list=torchex)
    x = torch.randn(2, 3)
    try:
        tom(x)
        assert False, "expected Exception to be thrown"
    except RuntimeError as e:
        tb = e.__traceback__
        while "thunder-generated" not in tb.tb_frame.f_code.co_filename:
            tb = tb.tb_next

        (tb_str,) = traceback.format_tb(tb, limit=1)
        assert "F.linear" in tb_str


@skipif_not_python_3_10
@skipif_windows
def test_nanogpt_functionalization():
    m = nanogpt_model.GPT(nanogpt_model.GPTConfig)

    gr = thunder.core.script.frontend.acquire_method(m.forward)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)
    additional_param_names, additional_param_values, _ = thunder.core.script.passes.module_to_function(gr)
    thunder.core.script.graph.check_graph(gr)

    fn = thunder.core.script.python_ir.generate_function(gr)

    x = torch.randint(0, 255, (5, 5))

    sd = m.state_dict(keep_vars=True)
    additional_params = [sd[n.replace("[", "").replace("]", "")] for n in additional_param_names]

    assert len(additional_param_names) == len(additional_param_values)
    assert all(a is b for a, b in zip(additional_params, additional_param_values))

    torch.manual_seed(5)
    o = fn(*additional_params, x, None)
    torch.manual_seed(5)

    o2 = m.forward(x)

    assert_close(o[0], o2[0])


class ModuleTestFunctionalizationConditional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 0.2
        self.b = True

    def forward(self, x):
        p = self.a if self.b else 0.1
        return torch.nn.functional.dropout(x, p=p)


@skipif_not_python_3_10
def test_functionalization_conditional():
    m = ModuleTestFunctionalizationConditional()
    tom = thunder.compile(m)
    a = torch.randn(5, 5)
    torch.manual_seed(123)
    expected = m(a)
    torch.manual_seed(123)
    actual = tom(a)
    assert_close(actual, expected)


@skipif_not_python_3_10
@skipif_windows
def test_nanogpt_tom():
    m = nanogpt_model.GPT(nanogpt_model.GPTConfig(dropout=0.0))
    tom = thunder.compile(m, executors_list=torchex, disable_torch_autograd_support=True)
    x = torch.randint(0, 255, (5, 5))

    torch.manual_seed(5)
    o = tom(x)
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


@skipif_not_python_3_10
@requiresNVFuser
def test_inlining_function_and_convert_to_thunder():
    def convert_to_thunder(fn):
        gr = thunder.core.script.frontend.acquire_method(fn)

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

    thunder_fn = thunder.compile(thunder_foo, executors_list=nvfuserex)

    torch_result = foo(a, c_fc_weight, c_proj_weight)
    thunder_result = thunder_fn(a, c_fc_weight, c_proj_weight)

    assert_close(torch_result, thunder_result)


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_builtin_convert_to_thunder(executor, device, dtype):
    def foo(a, b):
        return sorted(a), b

    a = []
    b = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))
    thunder_fn = executor.make_callable(foo, disable_preprocessing=False)

    thunder_result = thunder_fn(a, b)
    torch_result = foo(a, b)
    assert_close(thunder_result, torch_result)


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_preprocess_option(executor, device, dtype):
    def foo(a, b):
        return torch.add(a, b)

    tdtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((2, 2), device=device, dtype=tdtype)

    thunder_fn = executor.make_callable(foo, disable_preprocessing=False)

    thunder_result = thunder_fn(a, b)
    torch_result = foo(a, b)
    assert_close(thunder_result, torch_result)


def _nanogpt_mlp_helper(device, dtype, thunder_fn, torch_fn):
    tdtype = ltorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    n = 4
    a = make((n, n))
    c_fc_weight = make((4 * n, n))
    c_proj_weight = make((n, 4 * n))

    thunder_result = thunder_fn(a, c_fc_weight, c_proj_weight)
    torch_result = torch_fn(a, c_fc_weight, c_proj_weight)

    assert_close(thunder_result, torch_result)


# TODO: enable the following tests


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional_simplified(executor, device, dtype):
    def nanogpt_mlp_functional_simplified(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        d = torch.nn.functional.linear(b, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = executor.make_callable(nanogpt_mlp_functional_simplified, disable_preprocessing=False)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional_simplified)


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional_inlined(executor, device, dtype):
    def nanogpt_mlp_functional_inlined(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        c = 0.5 * b * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (b + 0.044715 * torch.pow(b, 3.0))))
        d = torch.nn.functional.linear(c, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = executor.make_callable(nanogpt_mlp_functional_inlined, disable_preprocessing=False)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional_inlined)


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional(executor, device, dtype):
    def nanogpt_mlp_functional(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        c = new_gelu(b)
        d = torch.nn.functional.linear(c, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    def nanogpt_mlp_functional_kw(a, c_fc_weight, c_proj_weight):
        b = torch.nn.functional.linear(a, c_fc_weight)
        c = new_gelu(x=b)
        d = torch.nn.functional.linear(c, c_proj_weight)
        e = torch.nn.functional.dropout(d, p=0.0)
        return e

    thunder_fn = executor.make_callable(nanogpt_mlp_functional, disable_preprocessing=False)
    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional)

    # see if everything works and is inlined
    allowed_funcs = {math.sqrt, ltorch.pow, ltorch.linear, ltorch.dropout, ltorch.tanh}
    thunder_fn = executor.make_callable(nanogpt_mlp_functional_kw, disable_preprocessing=False)
    funcs = _helper_get_func_calls(thunder_fn._pfn._gr)
    assert not (funcs ^ allowed_funcs)

    _nanogpt_mlp_helper(device, dtype, thunder_fn, nanogpt_mlp_functional_kw)
    funcs = _helper_get_func_calls(thunder_fn._pfn._gr)
    assert not (funcs ^ allowed_funcs)


@skipif_not_python_3_10
def test_clone_graph():
    gr = thunder.core.script.frontend.acquire_method(new_gelu)

    thunder.core.script.passes.inline_submodule_calls(gr)
    thunder.core.script.passes.merge_blocks_where_possible(gr)
    thunder.core.script.graph.check_graph(gr)
    thunder.core.script.passes.torch_to_thunder(gr)
    thunder.core.script.graph.check_graph(gr)

    gr2, _ = gr.clone()

    s2 = str(gr2)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)

    assert s2 == str(gr2)  # working with gr does not change gr2

    thunder_fn2 = thunder.core.script.python_ir.generate_function(gr2)

    import dis

    s = "\n".join([i._disassemble() for i in dis.get_instructions(thunder_fn)])
    s2 = "\n".join([i._disassemble() for i in dis.get_instructions(thunder_fn2)])
    assert s == s2


# Ref: https://github.com/lightning-AI/lightning-thunder/issues/386
def test_raise_nonlocals():
    f = lambda x: x

    def func(a):
        return f(a)

    with pytest.raises(RuntimeError) as excinfo:
        thunder.preprocess(func, is_module=False)
    assert "nonlocal variables are not supported but" in str(excinfo.value)
    assert "return f(a)" in str(excinfo.value)


# Ref: https://github.com/Lightning-AI/lightning-thunder/issues/667
def test_nonlocal_closure():
    def func():
        x = 0

        def inner():
            return x

        return inner

    # Once we support nonlocal variables, this should work.
    # This is a regression test to make sure it doesn't crash instead.
    with pytest.raises(RuntimeError) as excinfo:
        thunder.preprocess(func, is_module=False)
    assert "nonlocal variables are not supported but" in str(excinfo.value)


def add1(a, b):
    return a + b


def add2(a, b):
    return a + b


@noinline
def add_noinline(a, b):
    return a + b


@skipif_not_python_3_10
def test_noinline():
    def fn1():
        a = torch.arange(10)
        b = torch.arange(10)
        add1(a, b)

    def fn2():
        a = torch.arange(10)
        b = torch.arange(10)
        add_noinline(a, b)

    assert add1 not in thunder.compile(fn1)._pfn.__code__.co_consts
    noinline(add1)
    assert add1 in thunder.compile(fn1)._pfn.__code__.co_consts
    assert add_noinline in thunder.compile(fn2)._pfn.__code__.co_consts


@skipif_not_python_3_10
def test_invoke_noinline():
    def fn1():
        a = torch.arange(10)
        b = torch.arange(10)
        add2(a, b)

    def fn2():
        a = torch.arange(10)
        b = torch.arange(10)
        invoke_noinline(add2)(a, b)

    assert add2 not in thunder.compile(fn1)._pfn.__code__.co_consts
    assert add2 in thunder.compile(fn2)._pfn.__code__.co_consts


@skipif_not_python_3_10
def test_clang_noinline():
    def func(a, b):
        return thunder.clang.add(a, b)

    pfn = thunder.preprocess(func, is_module=False)
    assert "clang" in pfn.__code__.co_names
    assert "add" in pfn.__code__.co_names


@skipif_not_python_3_10
def test_torch_noinline():
    def func(a, b):
        return thunder.torch.add(a, b)

    pfn = thunder.preprocess(func, is_module=False)
    assert "torch" in pfn.__code__.co_names
    assert "add" in pfn.__code__.co_names


@skipif_not_python_3_10
def test_numpy_noinline():
    def func(a, b):
        return thunder.numpy.add(a, b)

    pfn = thunder.preprocess(func, is_module=False)
    assert "numpy" in pfn.__code__.co_names
    assert "add" in pfn.__code__.co_names


@skipif_not_python_3_10
def test_prims_noinline():
    def func(a, b):
        return thunder.core.prims.add(a, b)

    pfn = thunder.preprocess(func, is_module=False)
    assert "core" in pfn.__code__.co_names
    assert "prims" in pfn.__code__.co_names
    assert "add" in pfn.__code__.co_names


@skipif_not_python_3_10
def test_symbol_noinline():
    def func():
        return Symbol(name="to", meta=None, _module=torch.Tensor)

    pfn = thunder.preprocess(func, is_module=False)
    assert Symbol in pfn.__code__.co_consts


# TODO: enable me by converting torch inputs to Thunder inputs when proxying
# TODO: once this test works, also test acquiring the function from a collection
# @instantiate(dtypes=(thunder.float32,))
# def test_fn_input(executor, device, dtype):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

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
# @instantiate(dtypes=(thunder.float32,))
# def test_local_translation(executor, device, dtype):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

#     def foo(a, b):

#         def _convert(x):
#             return torch.add(x, 1)

#         a, b = tuple(_convert(x) for x in (a, b))

#         return a, b

#     thunder_fn = thunder.make_traced(foo, executor=executor, disable_preprocessing=False)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(a, b)
#     torch_result = foo(a, b)

#     assert_close(thunder_result, torch_result)

# @instantiate(dtypes=(thunder.float32,))
# def test_local_wrapped_translation(executor, device, dtype):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

#     def foo(a, b):

#         @wraps(torch.add)
#         def _convert(x):
#             return torch.add(x, 1)

#         a, b = tuple(_convert(x) for x in (a, b))

#         return a, b

#     thunder_fn = thunder.make_traced(foo, executor=executor, disable_preprocessing=False)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(a, b)
#     torch_result = foo(a, b)

#     assert_close(thunder_result, torch_result)


@skipif_not_python_3_10
@instantiate(dtypes=(thunder.float32,))
def test_local_aliased_translation(executor, device, dtype):
    tdtype = ltorch.to_torch_dtype(dtype)
    make = partial(make_tensor, device=device, dtype=tdtype)

    def foo(a, b):
        fn = torch.nn.functional.linear
        return fn(a, b)

    thunder_fn = executor.make_callable(foo, disable_preprocessing=False)

    shape = (2, 2)
    a = make(shape)
    b = make(shape)

    thunder_result = thunder_fn(a, b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)


@skipif_not_python_3_10
def test_unused_arg():
    def foo(a):
        return 1 + 2

    gr = thunder.core.script.frontend.acquire_method(foo)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = foo(1)
    actual = thunder_fn(1)
    assert_close(actual, expected)


@skipif_not_python_3_10
def test_unresolved_global_error():
    def f():
        return unresolved_global_variable  # pyright: ignore

    with pytest.raises(ValueError, match="global variable"):
        thunder.compile(f)


@skipif_not_python_3_10
def test_partial():
    def foo(a, b):
        return a + b

    def foo2(a=3, b=4):
        return a + b

    def foo3(*args, **kwargs):
        return args, kwargs

    pfoo = partial(foo, 1)
    gr = thunder.core.script.frontend.acquire_method(pfoo)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo(2)
    actual = thunder_fn(2)
    assert_close(actual, expected)

    pfoo2 = partial(foo2, 1)
    gr = thunder.core.script.frontend.acquire_method(pfoo2)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo2(2)
    actual = thunder_fn(2)
    assert_close(actual, expected)

    pfoo2 = partial(foo2, a=1)
    gr = thunder.core.script.frontend.acquire_method(pfoo2)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo2()
    actual = thunder_fn()
    assert_close(actual, expected)

    pfoo = partial(foo, a=1)
    gr = thunder.core.script.frontend.acquire_method(pfoo2)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo(b=1)
    actual = thunder_fn(b=1)
    assert_close(actual, expected)

    pfoo2 = partial(foo2, a=1, c=1)
    with pytest.raises(TypeError, match="'c'"):
        gr = thunder.core.script.frontend.acquire_method(pfoo2)

    # test varargs
    pfoo3 = partial(foo3, 1)
    gr = thunder.core.script.frontend.acquire_method(pfoo3)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo3(2)
    actual = thunder_fn(2)
    assert expected == actual

    pfoo3 = partial(foo3, a=1)
    gr = thunder.core.script.frontend.acquire_method(pfoo3)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo3(b=1)
    actual = thunder_fn(b=1)
    assert expected == actual

    pfoo_nested = partial(partial(foo3, 1, a=1, b=2), a=1, c=2)
    gr = thunder.core.script.frontend.acquire_method(pfoo_nested)
    thunder.core.script.graph.check_graph(gr)
    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    expected = pfoo_nested(b=1)
    actual = thunder_fn(b=1)
    assert expected == actual


@skipif_not_python_3_10
def test_foreign_store_attr():
    def fn(o):
        o.hi = "hello"
        return 2

    class A:
        pass

    o = A()
    tfn = thunder.compile(fn)
    tfn(o)


class TestStoreAttr1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_x = None

    def forward(self, x):
        if self.first_x is None:
            self.first_x = x
        return 2 * x


@skipif_not_python_3_10
def test_store_attr_mod1():
    m = TestStoreAttr1()

    tom = thunder.compile(m)
    x = torch.randn(5, 5)
    m(x)
    tom(x)
    m(x * 2)
    tom(x * 2)
    assert_close(m.first_x, tom._model.first_x)


class TestStoreAttr2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_x = None

    def forward(self, x):
        if self.first_x is None:
            # this is not allowed (we don't want to deal with the ordering issues)
            self.first_x = x
            self.first_x.xx = x
        return 2 * x


@skipif_not_python_3_10
def test_store_attr_mod2():
    m = TestStoreAttr2()
    with pytest.raises(RuntimeError, match="members of.*members"):
        tom = thunder.compile(m)


@skipif_not_python_3_10
def test_codeobject_debug():
    def f1(x):
        return torch.sin(x), x

    def f2(y):
        return torch.sin(y), y

    cf1 = thunder.compile(f1)
    cf2 = thunder.compile(f2)

    d1 = thunder.core.script.python_ir_data.debug_compare_functions(f1, f2, show=True)
    d2 = thunder.core.script.python_ir_data.debug_compare_functions(cf1, cf2, show=True)

    # Asserting the current behavior.
    # Copying over the co_varnames seems like a bit much.
    # Do we want to enforce the co_name to match? Currently it's always "_fn".
    assert d1 == {"co_varnames": ({"x"}, {"y"}), "co_name": ("f1", "f2")}
    assert d2 == {}


def tuple_fold(x):
    t = (x, 2)
    a, _ = t
    return a


def test_tuple_fold():
    tom = thunder.compile(tuple_fold)
    assert tuple_fold(1) == tom(1)


def tuple_fold_ex(x):
    t = (x, 2, 3, 4, 5)
    a, b, c, *_ = t
    return a, b, c


@skipif_not_python_3_10
def test_tuple_fold_ex():
    tom = thunder.compile(tuple_fold_ex)
    assert tuple_fold_ex(1) == tom(1)


def tuple_index(x):
    t = (x, 2)
    a = t[0]
    b = t[-1]
    return a, t[1], b


def test_tuple_index():
    tom = thunder.compile(tuple_index)
    assert tuple_index(1) == tom(1)


def nested_tuple_fold(x):
    t = ("aaa", ((x, 1), 2))
    (y, a), b = t[1]
    return y, t


def test_nested_tuple_fold():
    tom = thunder.compile(nested_tuple_fold)
    assert nested_tuple_fold(1) == tom(1)


def noop(*args):
    return args


def collision():
    return noop(True, 1)


def test_collision():
    tom = thunder.compile(collision)
    a0, b0 = tom()
    a1, b1 = collision()
    assert type(a0) is type(a1) and a0 == a1
    assert type(b0) is type(b1) and b0 == b1


class ConditionallyDefinedModule(torch.nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.a = torch.nn.Linear(1, 1)
        if not cond:
            self.b = torch.nn.Linear(1, 1)
        self.cond = cond

    def forward(self, x):
        x = self.a(x)
        if not self.cond:
            x = x + self.b(x)
        return x


@skipif_not_python_3_10
def test_conditionally_defined_module():
    m = ConditionallyDefinedModule(True)
    tom = thunder.compile(m)
    x = torch.tensor([3.0])
    expected = m(x)
    result = tom(x)
    assert_close(expected, result)
    m.cond = False
    tom = thunder.compile(m)
    with pytest.raises(AttributeError, match="ConditionallyDefinedModule.*'b'"):
        tom(x)


class ConditionallyDefinedAttribute(torch.nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.a = torch.nn.Linear(1, 1)
        if not cond:
            self.b = torch.nn.Parameter(torch.randn())
        self.cond = cond

    def forward(self, x):
        x = self.a(x)
        if not self.cond:
            x = x + self.b
        return x


@skipif_not_python_3_10
def test_conditionally_defined_attribute():
    m = ConditionallyDefinedAttribute(True)
    tom = thunder.compile(m)
    x = torch.tensor([3.0])
    expected = m(x)
    result = tom(x)
    assert_close(expected, result)
    m.cond = False
    tom = thunder.compile(m)
    with pytest.raises(AttributeError, match="ConditionallyDefinedAttribute.*'b'"):
        tom(x)


# @instantiate(dtypes=(thunder.float32,))
# def test_local_acquired_translation(executor, device, dtype):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

#     def foo(a, b):

#         fn = getattr(torch.nn.functional, "linear")
#         return fn(a, b)

#     thunder_fn = thunder.make_traced(foo, executor=executor, disable_preprocessing=False)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(a, b)
#     torch_result = foo(a, b)

#     assert_close(thunder_result, torch_result)

# @instantiate(dtypes=(thunder.float32,))
# def test_lambda_translation(executor, device, dtype):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

#     def foo(a, b):
#         return map(lambda a: torch.add(a, 1), (a, b))

#     thunder_fn = thunder.make_traced(foo, executor=executor, disable_preprocessing=False)

#     shape = (2, 2)
#     a = make(shape)
#     b = make(shape)

#     thunder_result = thunder_fn(a, b)
#     torch_result = foo(a, b)

#     assert_close(thunder_result, torch_result)


# @instantiate(dtypes=(thunder.float32,))
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

#     tdtype = ltorch.torch_dtype(dtype)

#     mlp = MLP()
#     mlp.to(device, dtype=tdtype)

#     thunder_fn = thunder.make_traced(mlp, executor=executor, disable_preprocessing=False)

#     make = partial(make_tensor, dtype=tdtype, device=device)

#     n = 4
#     a = make((n, n))

#     thunder_result = thunder_fn(a)
#     torch_result = mlp(a)

#     assert_close(thunder_result, torch_result)


@skipif_not_python_3_10
def test_method_with_kwarg():
    class Model(torch.nn.Module):
        def forward(self, x):
            return self.thing(x, b=1)

        def thing(self, a, b):
            return a + b

    m = Model()
    tom = thunder.compile(m)
    x = torch.tensor(3)
    actual = tom(x)
    assert actual.item() == 4


@skipif_not_python_3_10
def test_functionalization_callmethod_nonmethod():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_cache = None

        def forward(self, x, use_kv_cache=False):
            if use_kv_cache:
                x = self.kv_cache(x)
            return x

    class Attention2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_cache = torch.nn.Linear(1, 4)
            noinline(self.some_fw)

        def forward(self, x, use_kv_cache=False):
            x = self.some_fw(x)
            return x

        def some_fw(self, x):
            return x + 2

    model = Attention()
    x = torch.tensor([3.0])
    tom = thunder.compile(model)
    assert_close(model(x), tom(x))

    # by calling a no-inline method we generate a non-removable reference to self
    # to check that we don't mess with things that should be using CALL_METHOD
    model = Attention2()
    x = torch.tensor([3.0])
    with pytest.raises(RuntimeError, match="could not eliminate self"):
        tom = thunder.compile(model)


class Conv1d(torch.nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, weight=self.weight, bias=self.bias)


def test_nn_modules():
    modules_to_test = [
        (Conv1d, (3, 3, 3), {}, lambda: (torch.randn(2, 3, 4),)),
        (torch.nn.Conv1d, (3, 3, 3), {}, lambda: (torch.randn(2, 3, 4),)),
        (torch.nn.Conv2d, (3, 3, 3), {}, lambda: (torch.randn(2, 3, 4, 4),)),
        (torch.nn.Conv3d, (3, 3, 3), {}, lambda: (torch.randn(2, 3, 4, 4, 4),)),
    ]
    for cls, clsargs, clskwargs, posinpgen in modules_to_test:
        m = cls(*clsargs, **clskwargs)
        m.requires_grad_(False)
        tom = thunder.compile(m)
        torch.manual_seed(1234)
        inps = posinpgen()
        torch.manual_seed(1234)
        expected = m(*inps)
        torch.manual_seed(1234)
        result = tom(*inps)
        assert_close(expected, result)


def test_modulelist_additional_return_names():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = None

        def forward(self, x):
            self.w = x
            return self.w

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(2))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    x = torch.randn(1)
    model = MyModel()
    model = thunder.compile(model)
    # the test was only failing if these are present
    assert model._additional_return_names
    out = model(x)
    assert out.shape == (1,)


@skipif_not_python_3_10
@requiresCUDA
def test_non_hashable():
    def foo(a, b):
        return thunder.prims.device_put(a, thunder.devices.device_from_string("cuda"))

    cfoo = thunder.compile(foo)
    a = torch.randn((2, 2), device="cpu", dtype=torch.float32, requires_grad=False)
    b = cfoo(a, 1)
    assert_close(a.to("cuda"), b)


@skipif_not_python_3_10
def test_raises_exception_opcode():
    # previously this left the opcode at CALL_FUNCTION_KW and segfaulted
    def foo(a, b):
        return torch.this_does_not_exist((2, 2), device="cpu", dtype=torch.bfloat16)

    cfoo = thunder.compile(foo)
    with pytest.raises(AttributeError, match="no attribute.*this_does_not_exist"):
        cfoo(1, 1)
