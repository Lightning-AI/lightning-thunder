from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product

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
import thunder.clang as clang
from thunder.core.options import INTERPRETATION_OPTIONS, CACHE_OPTIONS
import thunder.torch as ltorch
import thunder.core.prims as prims
from thunder import pytorch_executor, nvfuser_executor
from thunder.executors.sdpaex import sdpa_ex
from thunder.core.jit_ext import JITSharpEdgeError


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
        pytest.param("mixtral-like", marks=pytest.mark.xfail(raises=TypeError, reason="topk", strict=True)),
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
        pytest.param("mixtral-like", marks=pytest.mark.xfail(raises=TypeError, reason="topk", strict=True)),
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
