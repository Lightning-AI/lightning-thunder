from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product
from contextlib import nullcontext

import operator
import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

from lightning_utilities import compare_version

import thunder
from thunder.core.interpreter import is_jitting, InterpreterError

from thunder.tests import litgpt_model
from thunder.tests.framework import version_between
import thunder.clang as clang
from thunder.core.options import INTERPRETATION_OPTIONS, CACHE_OPTIONS
import thunder.torch as ltorch
import thunder.core.prims as prims
from thunder import pytorch_executor, nvfuser_executor
from thunder.executors.sdpaex import sdpa_ex
from thunder.core.jit_ext import JITSharpEdgeError
from thunder.core.transforms import PostOptimizationTransform

#
# Test suite for the general jit
#

# TODO RC1 Merge this test file with test_jit.py

torchex = [pytorch_executor]
nvfuserex = [nvfuser_executor, pytorch_executor]


def skipif_python_3_11_plus(f):
    if sys.version_info >= (3, 11):
        return pytest.mark.skip(f, reason=f"not yet implemented for Python 3.11+, got {sys.version_info=}")
    return f


def skipif_not_pytorch_2_1(f):
    return pytest.mark.skipif(
        # use_base_version=True allows checking against versions with nightly modifiers
        compare_version("torch", operator.lt, "2.1.0", use_base_version=True),
        reason=f"requires PyTorch >= 2.1, got {torch.__version__=}",
    )(f)


def test_jitting_through_opaque_torch_symbols_error():
    def no_error(x):
        # randn_like is in ltorch
        return torch.randn_like(x)

    def should_error(x):
        # rand_like is not yet in ltroch
        return torch.rand_like(x)

    x = torch.rand(1)

    jno_error = thunder.jit(no_error)
    jno_error(x)

    jshould_error = thunder.jit(should_error)
    with pytest.raises(NotImplementedError):
        jshould_error(x)


def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors_closure():
    def foo(a, b):
        c = a + b

        def bar():
            return torch.add(c, 1)

        return bar()

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors_closure_external():
    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    def bar(b):
        return torch.add(a, b)

    def foo():
        bar(b)

    jbar = thunder.jit(bar)
    actual = jbar(b)
    expected = bar(b)
    assert_close(actual, expected)

    jfoo = thunder.jit(foo)
    actual = jfoo()
    expected = foo()
    assert_close(actual, expected)


def test_args_len():
    def foo(a, b=1):
        return a + b

    jfoo = thunder.jit(foo)

    assert foo(2) == jfoo(2)
    assert foo(2, 3) == jfoo(2, 3)
    assert foo(2, b=2) == jfoo(2, b=2)


def test_intermediate_torch_operations():
    def foo(a, b):
        c = a + b
        d = torch.sub(c, b)
        e = torch.mul(d, a)
        f = torch.matmul(e, c)
        g = [e, f]
        return torch.cat(g)

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_cache_basic():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    # Tests rank changing
    a = torch.randn((2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    # Tests dtype changing
    a = torch.randn((2, 2), device="cpu", dtype=torch.bfloat16)
    b = torch.randn((2, 2), device="cpu", dtype=torch.bfloat16)

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 2

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3

    # Tests shape changing
    a = torch.randn((2, 1), device="cpu", dtype=torch.bfloat16)
    b = torch.randn((2, 1), device="cpu", dtype=torch.bfloat16)

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 3

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 4


def test_cache_always_trace():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo, cache=CACHE_OPTIONS.NO_CACHING)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 0


def test_cache_equality_constraint():
    x, y = torch.randn(2, 2)

    def fn(b):
        if b:
            return x
        else:
            return y

    jfn = thunder.jit(fn)

    assert_close(fn(True), jfn(True))
    assert_close(fn(False), jfn(False))

    assert thunder.cache_misses(jfn) == 2
    assert thunder.cache_hits(jfn) == 0

    assert_close(fn(True), jfn(True))
    assert_close(fn(False), jfn(False))

    assert thunder.cache_misses(jfn) == 2
    assert thunder.cache_hits(jfn) == 2


def test_nn_parameter():
    a = torch.nn.Parameter(torch.randn(2, 3))
    b = torch.tensor(2)

    def fn(a):
        return b * a

    jfn = thunder.jit(fn)

    expected = fn(a)
    actual = jfn(a)
    assert_close(expected, actual)


def test_nn_module():
    m = torch.nn.Linear(3, 4)
    m2 = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 3),
    )
    a = torch.randn(2, 3)

    tom = thunder.jit(m)
    assert isinstance(tom, thunder.ThunderModule)
    # attributes are forwarded
    assert isinstance(tom.weight, torch.Tensor)
    expected = m(a)
    actual = tom(a)
    assert_close(expected, actual)

    tom2 = thunder.jit(m2)
    assert isinstance(tom2, thunder.ThunderModule)
    # `ThunderModule` is not subscriptable even though it compiles a `Sequential`
    with pytest.raises(TypeError, match="not subscriptable"):
        assert isinstance(tom2[1].weight, torch.Tensor)
    # saving is the same
    torch.testing.assert_close(tom2.state_dict(), m2.state_dict(), rtol=0, atol=0)
    # loading works
    tom2.load_state_dict(m2.state_dict(), strict=True, assign=True)
    expected = m2(a)
    actual = tom2(a)
    assert_close(expected, actual)

    def fn(a):
        return m(a)

    jfn = thunder.jit(fn)
    expected = fn(a)
    actual = jfn(a)
    assert_close(expected, actual)

    def fn2(a):
        return m2(a)

    jfn2 = thunder.jit(fn2)
    expected = fn2(a)
    actual = jfn2(a)
    assert_close(expected, actual)


def test_compile_within_jit():
    def model(a, b, c):
        return a @ b + c

    def jit_me(a, b, c):
        cfn = torch.compile(model)
        return cfn(a, b, c)

    jcfn = thunder.jit(jit_me)

    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    z = torch.randn(2, 2)

    with pytest.raises(NotImplementedError) as exc_info:
        jcfn(x, y, z)

    assert "Using torch.compile within a function to be JIT-compiled by Thunder is not supported." in str(
        exc_info.value
    )


def test_add_numbers():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder.jit(foo)

    # TODO Add test for bool
    # see issue "Binary addition on booleans should promote to an integer"
    cases = (
        (2, 3),
        (2.1, 3.4),
        (complex(1, 1), complex(-1, 2)),
    )

    for a, b in cases:
        actual = jfoo(a, b)
        expected = a + b

        assert_close(actual, expected)


def test_binary_add_tensor_number():
    # Tests using torch.add
    def foo(a):
        return torch.add(a, 3)

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)

    # Tests using addition operator
    def foo(a):
        return a + 4

    jfoo = thunder.jit(foo)

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)


def test_binary_add_numbers():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)

    # TODO Add test for bool
    # see issue "Binary addition on booleans should promote to an integer"
    cases = (
        (2, 3),
        (2.1, 3.4),
        (complex(1, 1), complex(-1, 2)),
    )

    for a, b in cases:
        actual = jfoo(a, b)
        expected = foo(a, b)

        assert_close(actual, expected)


_test_add_global_global = 2


@pytest.mark.xfail(reason='"disallow global reads and writes (temporarily)"', raises=BaseException)
def test_global_fails():
    def foo():
        return _test_add_global_global

    jfoo = thunder.jit(foo)

    with pytest.raises(NotImplementedError):
        jfoo()


@pytest.mark.xfail(
    reason='"Raise an error when a program attempts to write to a nonlocal that was captured from outside the interpreter"',
    raises=BaseException,
)
def test_nonlocal_outside_interpreter_fails():
    def foo():
        x = 3

        def bar():
            nonlocal x
            x = 4

        jbar = thunder.jit(bar)

        jbar()

        return x

    with pytest.raises(NotImplementedError):
        foo()


def test_lookaside_bool():
    def foo(a, b, i):
        if bool(i):
            return a + b
        return a - b

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b, 0)
    actual = jfoo(a, b, 0)
    assert_close(expected, actual)

    expected = foo(a, b, 1)
    actual = jfoo(a, b, 1)
    assert_close(expected, actual)


# see https://github.com/Lightning-AI/lightning-thunder/issues/95
def test_get_default_dtype():
    def foo():
        return torch.get_default_dtype()

    assert foo() == thunder.jit(foo)()


@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_proxy_no_multiple_renames(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    def f(a, b, c, d):
        def g(b, a):
            return b + a

        def h(d, c):
            return d + c

        return g(a, b) + h(d, c)

    jf = thunder.jit(f)

    a = torch.rand(1, device=device)
    b = torch.rand(1, device=device)
    c = torch.rand(1, device=device)
    d = torch.rand(1, device=device)

    expected = f(a, b, c, d)
    actual = jf(a, b, c, d)
    assert_close(expected, actual)

    comp_trace = thunder.last_traces(jf)[-1]
    args_names = tuple(p.name for p in comp_trace.args)
    # Once proxies got their names in `f`, these should not change in `g` and `h`
    assert args_names == ("a", "b", "c", "d")


def test_litgpt():
    from thunder.benchmarks import LitGPTBenchmark
    from thunder.tests.litgpt_model import Config

    cfg: Config = Config.from_name("gpt-neox-like")
    bench = LitGPTBenchmark(config=cfg, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    module = bench.fn()
    jfn = thunder.jit(module)

    args, kwargs = bench.make_batch()

    # the transforms of thunder_backward introduce numerical differences in bfloat16 mode that cause flakiness
    for p in module.parameters():
        p.requires_grad_(False)

    reference = module(*args, **kwargs)
    result = jfn(*args, **kwargs)
    assert_close(result, reference)

    args, kwargs = bench.make_batch()
    result = jfn(*args, **kwargs)
    assert_close(result, module(*args, **kwargs))

    assert thunder.cache_misses(jfn) == 1
    assert thunder.cache_hits(jfn) == 1


def test_nanogpt_block():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = thunder.jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt_attn():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn()
    module = module.attn

    # the transforms of thunder_backward introduce numerical differences in bfloat16 mode that cause flakiness
    for p in module.parameters():
        p.requires_grad_(False)

    args, kwargs = bench.make_batch()

    jfn = thunder.jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs), atol=3e-5, rtol=1e-5)


def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn().mlp

    args, kwargs = bench.make_batch()

    jfn = thunder.jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt():
    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0, n_layer=2)
    config.update(**_nanogpt_configs["test"])
    bench = NanoGPTBenchmark(config=config, device="cpu")
    module = bench.fn()

    # the transforms of thunder_backward introduce numerical differences in bfloat16 mode that cause flakiness
    for p in module.parameters():
        p.requires_grad_(False)

    args, kwargs = bench.make_batch()
    jfn = thunder.jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


@skipif_not_pytorch_2_1
@pytest.mark.parametrize(
    "name",
    (
        "gpt-neox-like",
        "llama1-like",
        "long-context-like",
        "llama2-like",
        "falcon-7b-like",
        "falcon-40b-like",
        "codellama2-like",
        pytest.param(
            "mixtral-like",
            marks=pytest.mark.xfail(raises=(NotImplementedError, TypeError), reason="topk and where", strict=True),
        ),
    ),
)
@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda", "meta"),
)
def test_litgpt_variants(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    x = torch.randint(0, 200, (5, 5), device=device)
    config = litgpt_model.Config.from_name(name)

    with device:
        reference = litgpt_model.GPT(config)
    expected_logits = reference(x)

    expected_logits.sum().backward()

    with device:
        model = litgpt_model.GPT(config)
    model.load_state_dict(reference.state_dict())
    tom = thunder.jit(model, executors=nvfuserex if device.type == "cuda" else torchex)
    actual_logits = tom(x)
    assert_close(actual_logits, expected_logits)

    actual_logits.sum().backward()

    for param1, param2 in zip(reference.parameters(), model.parameters()):
        assert param1 is not param2
        assert param1.grad is not None
        torch.testing.assert_close(param1.grad, param2.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    version_between(torch.__version__, min_ver="2.5.0a0", max_ver="2.5.0a99"),
    reason="https://github.com/Lightning-AI/lightning-thunder/issues/669",
)
@skipif_not_pytorch_2_1
@pytest.mark.parametrize(
    "name",
    (
        # TODO this seems flaky on CI - the cause is unclear
        # "gpt-neox-like",
        "llama1-like",
        "long-context-like",
        "llama2-like",
        "falcon-7b-like",
        "falcon-40b-like",
        "codellama2-like",
        pytest.param(
            "mixtral-like",
            marks=pytest.mark.xfail(raises=(NotImplementedError, TypeError), reason="topk and where", strict=True),
        ),
    ),
)
@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_litgpt_variants_kvcache(name, device):
    import torch._dynamo  # this monkeypatches torch.manual_seed

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    x = torch.randint(0, 200, (1, 2), device=device)
    config = litgpt_model.Config.from_name(name)

    with device:
        model = litgpt_model.GPT(config)
        model.max_seq_length = 3

    for p in model.parameters():
        p.requires_grad_(False)

    executors = nvfuserex if device.type == "cuda" else torchex
    executors = [sdpa_ex] + executors

    def sample(logits):
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    # the reference is 2 regular forward without the kv cache
    logits_1 = model(x)
    token_1 = sample(logits_1)
    logits_2 = model(torch.cat((x, token_1), dim=-1))

    with device:
        model.set_kv_cache(batch_size=1)
    tom = thunder.jit(model, executors=executors)  # , disable_torch_autograd_support=True

    # kv cache prefill
    thunder_logits_1 = tom(x, torch.tensor([0, 1], device=device))
    thunder_token_1 = sample(thunder_logits_1)
    # 1 token generation
    thunder_logits_2 = tom(thunder_token_1, torch.tensor([2], device=device))

    assert_close(logits_1, thunder_logits_1)
    assert_close(logits_2[:, -1:], thunder_logits_2)


@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_tom_overrides_proxy(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    x = torch.randint(0, 200, (5, 5), device=device)
    config = litgpt_model.Config.from_name("llama2-like")

    with device:
        reference = litgpt_model.GPT(config)
    expected_logits = reference(x)

    expected_logits.sum().backward()

    with device:
        model = litgpt_model.GPT(config)
    model.load_state_dict(reference.state_dict())
    tom = thunder.jit(model, executors=nvfuserex if device.type == "cuda" else torchex)

    # we manually replace tensors here, early transforms (like distributed) will do this
    for k, v in tom._overrides_parameters.items():
        tom._overrides_parameters[k] = torch.nn.Parameter(v.detach().clone())

    actual_logits = tom(x)
    assert_close(actual_logits, expected_logits)

    actual_logits.sum().backward()

    grads_expected = {k: t.grad for k, t in reference.named_parameters()}
    grads_actual = {k: t.grad for k, t in tom.named_parameters()}

    # on the original model, we have no grads
    for param in model.parameters():
        assert param.grad is None

    assert len(grads_expected) == len(grads_actual)

    for k, v in grads_expected.items():
        torch.testing.assert_close(v, grads_actual[k], rtol=1e-2, atol=1e-2)

    # after deleting overrides, we expect the tom to have the same named params as the original model
    tom._overrides_parameters.clear()

    params_expected = {k: t.grad for k, t in model.named_parameters()}
    params_actual = {k: t.grad for k, t in tom.named_parameters()}

    assert len(params_expected) == len(params_actual)

    for k, v in params_expected.items():
        assert v is params_actual[k]


@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_cache_symbolic_values_basic(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def foo(a, scalar):
        return (a * scalar).sum(scalar)

    jfoo = thunder.jit(foo, cache="symbolic values")

    a = torch.randn((2, 2, 2), device=device)
    b = 1

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    b = 2

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1


def test_post_optimization_transform():
    def foo(a, b, c):
        return a * a + b * c

    class MyTransform(PostOptimizationTransform):
        def transform_trace(self, trace, executors_list=None):
            # Transform that adds a comment before any `add` BoundSymbol.
            commented_trace = thunder.core.trace.from_trace(trace)

            bsyms = []
            for bsym in trace.bound_symbols:
                if bsym.sym.name == "add":
                    op_name = bsym.sym.name
                    comment_bsym = prims.comment.bind(f"Executing {op_name}", output=None)
                    bsyms.append(comment_bsym)

                bsyms.append(bsym)

            commented_trace.bound_symbols = bsyms
            return commented_trace

    jfoo = thunder.jit(foo, post_optimization_transforms=[MyTransform()])

    a = torch.randn(3, 3, requires_grad=True)
    b = torch.randn(3, 3)
    c = torch.randn(3, 3)
    _ = jfoo(a, b, c)
    exec_trc = thunder.last_traces(jfoo)[-1]

    comment_bsyms = list(filter(lambda bsym: bsym.sym.id == thunder.prims.PrimIDs.COMMENT, exec_trc.bound_symbols))
    assert any(map(lambda bsym: bsym.args[0].startswith("Executing"), comment_bsyms))

    bwd_trc = thunder.last_backward_traces(jfoo)[-1]
    comment_bsyms = list(filter(lambda bsym: bsym.sym.id == thunder.prims.PrimIDs.COMMENT, bwd_trc.bound_symbols))
    assert any(map(lambda bsym: bsym.args[0].startswith("Executing"), comment_bsyms))


@pytest.mark.parametrize(
    "cache_option",
    (
        thunder.CACHE_OPTIONS.CONSTANT_VALUES,
        thunder.CACHE_OPTIONS.SYMBOLIC_VALUES,
        thunder.CACHE_OPTIONS.NO_CACHING,
        thunder.CACHE_OPTIONS.SAME_INPUT,
    ),
)
def test_device_as_input(cache_option):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(3, 3)
    devices_to_check = ("cuda:0", "cpu")

    def foo(x, device):
        return x.to(device)

    ctx = nullcontext()
    if cache_option is thunder.CACHE_OPTIONS.SAME_INPUT:
        ctx = pytest.raises(NotImplementedError)

    jfoo = thunder.jit(foo, cache=cache_option)

    for device in devices_to_check:
        expected_device = torch.device(device)
        with ctx:
            actual_device = jfoo(x, expected_device).device
            assert actual_device == expected_device


def test_cache_symbolic_values_constraints():
    def foo(scalar):
        if scalar > 0:
            return scalar
        return 0

    jfoo = thunder.jit(foo, cache="symbolic values")

    expected = foo(1.5)
    actual = jfoo(1.5)

    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    expected = foo(2.0)
    actual = jfoo(2.0)

    assert_close(expected, actual)
    # even though we should be able to re-use the cache, we cannot do it now. Because constraints are propagated to inputs being static number.
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 0

    expected = foo(1.5)
    actual = jfoo(1.5)

    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    expected = foo(-0.3)
    actual = jfoo(-0.3)

    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 1

    def bar(t):
        if t[0].item() > 5:
            return t + 1.0
        return t

    with pytest.raises(
        thunder.core.interpreter.InterpreterError, match="conversion to bool is not allowed on dynamic proxy"
    ):
        jbar = thunder.jit(bar, cache="symbolic values")
        t = torch.randn(4, device="cpu")
        jbar(t)


def test_cache_symbolic_values_torch_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def foo(dev, idx):
        # NOTE dtype needs to be explicit, see issue: https://github.com/Lightning-AI/lightning-thunder/issues/621
        return torch.ones(1, device=torch.device(dev, idx), dtype=torch.float32)

    jfoo = thunder.jit(foo, cache="symbolic values")
    expected = foo("cuda", 0)
    actual = jfoo("cuda", 0)

    assert_close(expected, actual)


def test_load_original_state_dict():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("param", torch.nn.Parameter(torch.randn(3)))
            self.register_buffer("buffer", torch.randn(3))

        def forward(self, x):
            return x

    m = Model()

    thunder_module = thunder.jit(Model())
    thunder_module.load_original_state_dict(m.state_dict())

    # Check the updated values
    # We can't directly compare state_dict - https://github.com/Lightning-AI/lightning-thunder/issues/647
    torch.testing.assert_close(thunder_module._overrides_parameters["param"], m.param)
    torch.testing.assert_close(thunder_module._overrides_buffers["buffer"], m.buffer)


@pytest.mark.parametrize("prefix", ("", "foo"), ids=("prefix=", "prefix=foo"))
@pytest.mark.parametrize("recurse", (True, False), ids=("recurse=True", "recurse=False"))
@pytest.mark.parametrize(
    "remove_duplicate",
    (False, True),
    ids=("remove_duplicate=False", "remove_duplicate=True"),
)
def test_named_params_and_named_buffers(prefix, recurse, remove_duplicate):

    buffer_tensor = torch.tensor([1.0])

    class SubMod(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.register_buffer("buffer", buffer_tensor)

        def forward(self, x):
            return x

    class MyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fc1 = torch.nn.Linear(1, 1)
            self.register_buffer("buffer", buffer_tensor)
            self.register_buffer("buffer2", buffer_tensor)
            self.sub_module = torch.nn.Sequential(
                torch.nn.Linear(1, 1), SubMod(), torch.nn.Sequential(torch.nn.Linear(1, 1))
            )

        def forward(self):
            names_params_buffers = []
            for name, param in self.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
                names_params_buffers.append((name, param))
            for name, buffer in self.named_buffers(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
                names_params_buffers.append((name, buffer))
            return names_params_buffers

    m = MyModel()
    expected = dict(m())

    jm = thunder.jit(m)
    actual = dict(jm())

    torch.testing.assert_close(actual, expected)


def test_isinstance_parameter():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x):
            weight = self.fc.weight

            # Verify that `thunder.jit` correctly picks this branch.
            if isinstance(weight, torch.nn.Parameter):
                return x + 1

            return x

    m = Model()
    x = torch.ones(
        1,
    )
    expected = m(x)
    actual = thunder.jit(m)(x)

    torch.testing.assert_close(actual, expected)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x):
            weight = self.fc.weight

            # Verify that `thunder.jit` correctly picks this branch.
            if isinstance(weight, (torch.nn.Parameter, type(None))):
                return x + 1

            return x

    m = Model()
    x = torch.ones(
        1,
    )
    expected = m(x)
    actual = thunder.jit(m)(x)

    torch.testing.assert_close(actual, expected)
