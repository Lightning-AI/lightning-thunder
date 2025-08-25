from functools import partial
from contextlib import nullcontext
import weakref

import operator
import sys

import pytest
import torch
from torch.testing import assert_close

from lightning_utilities import compare_version

import thunder

from thunder.tests.framework import requiresCUDA, IS_WINDOWS
from thunder.core.options import CACHE_OPTIONS
import thunder.core.prims as prims
from thunder import pytorch_executor, nvfuser_executor
from thunder.executors.sdpaex import sdpa_ex
from thunder.core.transforms import Transform


thunder_jit = partial(thunder.jit, debug_options=thunder.DebugOptions(check_traces=2))

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

    def should_error(x, y):
        # rand_like is not yet in ltroch
        return torch.allclose(x, y)

    x = torch.rand(1)

    jno_error = thunder_jit(no_error)
    jno_error(x)

    jshould_error = thunder_jit(should_error)
    with pytest.raises(NotImplementedError):
        jshould_error(x, x)


def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    jfoo = thunder_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder_jit(foo)

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

    jfoo = thunder_jit(foo)

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

    jbar = thunder_jit(bar)
    actual = jbar(b)
    expected = bar(b)
    assert_close(actual, expected)

    jfoo = thunder_jit(foo)
    actual = jfoo()
    expected = foo()
    assert_close(actual, expected)


def test_args_len():
    def foo(a, b=1):
        return a + b

    jfoo = thunder_jit(foo)

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

    jfoo = thunder_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_cache_basic():
    def foo(a, b):
        return a + b

    jfoo = thunder_jit(foo)

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

    jfoo = thunder_jit(foo, cache=CACHE_OPTIONS.NO_CACHING)

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

    jfn = thunder_jit(fn)

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

    jfn = thunder_jit(fn)

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

    tom = thunder_jit(m)
    assert isinstance(tom, thunder.ThunderModule)
    # attributes are forwarded
    assert isinstance(tom.weight, torch.Tensor)
    expected = m(a)
    actual = tom(a)
    assert_close(expected, actual)

    tom2 = thunder_jit(m2)
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

    jfn = thunder_jit(fn)
    expected = fn(a)
    actual = jfn(a)
    assert_close(expected, actual)

    def fn2(a):
        return m2(a)

    jfn2 = thunder_jit(fn2)
    expected = fn2(a)
    actual = jfn2(a)
    assert_close(expected, actual)


def test_compile_within_jit():
    def model(a, b, c):
        return a @ b + c

    def jit_me(a, b, c):
        cfn = torch.compile(model)
        return cfn(a, b, c)

    jcfn = thunder_jit(jit_me)

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

    jfoo = thunder_jit(foo)

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

    jfoo = thunder_jit(foo)

    a = torch.randn((2, 2), device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)

    # Tests using addition operator
    def foo(a):
        return a + 4

    jfoo = thunder_jit(foo)

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)


def test_binary_add_numbers():
    def foo(a, b):
        return a + b

    jfoo = thunder_jit(foo)

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


def test_finfo():
    def foo(a):
        return torch.finfo(a.dtype)

    jfoo = thunder_jit(foo)
    a = torch.randn((2, 2), device="cpu")
    actual = jfoo(a)
    expected = foo(a)
    assert actual == expected

    def bar(a):
        return torch.finfo(a.dtype).min

    jbar = thunder_jit(bar)
    a = torch.randn((2, 2), device="cpu")
    actual = jbar(a)
    expected = bar(a)
    assert_close(actual, expected)


_test_add_global_global = 2


@pytest.mark.xfail(reason='"disallow global reads and writes (temporarily)"', raises=BaseException)
def test_global_fails():
    def foo():
        return _test_add_global_global

    jfoo = thunder_jit(foo)

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

        jbar = thunder_jit(bar)

        jbar()

        return x

    with pytest.raises(NotImplementedError):
        foo()


def test_lookaside_bool():
    def foo(a, b, i):
        if bool(i):
            return a + b
        return a - b

    jfoo = thunder_jit(foo)

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

    assert foo() == thunder_jit(foo)()


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

    jf = thunder_jit(f)

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
    jfn = thunder_jit(module)

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

    jfn = thunder_jit(module)
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

    jfn = thunder_jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs), atol=3e-5, rtol=1e-5)


def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn().mlp

    args, kwargs = bench.make_batch()

    jfn = thunder_jit(module)
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
    jfn = thunder_jit(module)
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
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    x = torch.randint(0, 200, (5, 5), device=device)
    config = Config.from_name(name)

    with device:
        reference = GPT(config)
    expected_logits = reference(x)

    expected_logits.sum().backward()

    with device:
        model = GPT(config)
    model.load_state_dict(reference.state_dict())
    tom = thunder_jit(model, executors=nvfuserex if device.type == "cuda" else torchex)
    actual_logits = tom(x)
    assert_close(actual_logits, expected_logits)

    # small check that we do not leak internal var names
    assert "tos" not in str(thunder.last_prologue_traces(tom)[0])

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
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT
    import torch._dynamo  # this monkeypatches torch.manual_seed

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if IS_WINDOWS:
        pytest.skip("slow on windows")

    device = torch.device(device)
    x = torch.randint(0, 200, (1, 2), device=device)
    config = Config.from_name(name)

    with device:
        model = GPT(config)
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

    # TODO: make check_trace mode work
    tom = thunder_jit(model, executors=executors)  # , disable_torch_autograd_support=True

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
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    x = torch.randint(0, 200, (5, 5), device=device)
    config = Config.from_name("llama2-like")

    with device:
        reference = GPT(config)
    expected_logits = reference(x)

    expected_logits.sum().backward()

    with device:
        model = GPT(config)
    model.load_state_dict(reference.state_dict())
    tom = thunder_jit(model, executors=nvfuserex if device.type == "cuda" else torchex)

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


def test_cache_symbolic_values_basic():
    def foo(a, scalar):
        return (a * scalar).sum(scalar)

    jfoo = thunder_jit(foo, cache="symbolic values")

    a = torch.randn((2, 2, 2), device="cpu")
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

    class MyTransform(Transform):
        def transform_trace_post_optimization(self, trace, executors_list=None):
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

    jfoo = thunder_jit(foo, transforms=[MyTransform()])

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

    jfoo = thunder_jit(foo, cache=cache_option)

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

    jfoo = thunder_jit(foo, cache="symbolic values")

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
        jbar = thunder_jit(bar, cache="symbolic values")
        t = torch.randn(4, device="cpu")
        jbar(t)


def test_cache_symbolic_values_torch_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def foo(dev, idx):
        # NOTE dtype needs to be explicit, see issue: https://github.com/Lightning-AI/lightning-thunder/issues/621
        return torch.ones(1, device=torch.device(dev, idx), dtype=torch.float32)

    jfoo = thunder_jit(foo, cache="symbolic values")
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

    thunder_module = thunder_jit(Model())
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

    jm = thunder_jit(m)
    actual = dict(jm())

    torch.testing.assert_close(actual, expected)


def test_isinstance_parameter():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x):
            weight = self.fc.weight

            # Verify that `thunder_jit` correctly picks this branch.
            if isinstance(weight, torch.nn.Parameter):
                return x + 1

            return x

    m = Model()
    x = torch.ones(
        1,
    )
    expected = m(x)
    actual = thunder_jit(m)(x)

    torch.testing.assert_close(actual, expected)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x):
            weight = self.fc.weight

            # Verify that `thunder_jit` correctly picks this branch.
            if isinstance(weight, (torch.nn.Parameter, type(None))):
                return x + 1

            return x

    m = Model()
    x = torch.ones(
        1,
    )
    expected = m(x)
    actual = thunder_jit(m)(x)

    torch.testing.assert_close(actual, expected)


def test_cache_symbolic_values_reshape():
    a = torch.randn((4, 8, 6), device="cpu")

    def foo(t, batch_size):
        return t.reshape(batch_size, -1).sum(-1)

    jfoo = thunder_jit(foo, cache="symbolic values")
    expected = foo(a, 32)
    actual = jfoo(a, 32)

    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    expected = foo(a, 16)
    actual = jfoo(a, 16)

    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1


@pytest.mark.filterwarnings("ignore:Please use `torch.vmap`")
def test_custom_autograd_function():
    from torch.autograd.gradcheck import GradcheckError
    from torch.testing._internal.common_utils import gradcheck

    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        # this is wrong on purpose.
        @staticmethod
        def backward(ctx, grad_output) -> torch.Tensor:
            return grad_output

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x) -> torch.Tensor:
            return MyFunction.apply(x)

    x = torch.randn((2, 2), dtype=torch.float64, requires_grad=True)
    model = Model().to(dtype=torch.float64)
    jitted = thunder_jit(model)

    with pytest.raises(GradcheckError):
        gradcheck(jitted, (x,))
    with pytest.raises(GradcheckError):
        gradcheck(model, (x,))

    class MyLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
            ctx.shape = shape
            ctx.save_for_backward(x, weight)
            ctx.pretty_attr = 100
            ctx.scaler = 1.0
            return torch.matmul(x, weight.t())

        @staticmethod
        def backward(ctx, grad_output):
            (x, weight) = ctx.saved_tensors
            assert weight.shape == ctx.shape  # really bogus, just to use ctx.shape
            scaler2 = ctx.shape[0] / ctx.shape[1]
            return torch.matmul(grad_output, weight) * ctx.scaler, torch.matmul(grad_output.t(), x) / scaler2, None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):
            return MyLinear.apply(x, self.l1.weight, self.l1.weight.shape)

    x = torch.randn((2, 2), dtype=torch.float64, requires_grad=True)
    model = Model().to(dtype=torch.float64)
    jitted = thunder_jit(model)
    gradcheck(jitted, (x,), check_batched_grad=False)

    jitted.zero_grad()
    x = torch.randn((2, 2), dtype=torch.float64, requires_grad=True)
    out = jitted(x)
    out.backward(torch.rand_like(out))
    assert jitted.l1.weight.grad is not None


@requiresCUDA
def test_complex_backward_custom_autograd():
    # This tests that backward tags are correctly propagated to the subsymbols of the custom autograd function.
    # Without this propagation, operations for the backward pass could be fused with forward pass operations.
    class CustomBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_at_output):
            x = torch.randn_like(grad_at_output)
            index = torch.randint(0, 1, (0, x.shape[-1]), device="cuda", dtype=torch.int64)
            # the descent to scatter_add_'s subsymbols doesn't happen until _transform_for_operator_executor_execution
            x.scatter_add_(index=index, src=grad_at_output, dim=-1)
            return grad_at_output * x

    def f(x):
        y = CustomBackward.apply(x)
        # the in-place op introduces a fusion break
        x.add_(1)
        z = x + x
        return y, z

    jf = thunder_jit(f, fusion_type="dataflow")

    x = torch.ones(2, 3, device="cuda", requires_grad=True)

    # This should not raise an error about variables referenced before assignment.
    jf(x)


@pytest.mark.filterwarnings("ignore:Please use torch.vmap")
def test_autograd_function_apply():
    # see https://github.com/Lightning-AI/lightning-thunder/issues/1248#issuecomment-2388655917
    # for why `torch.foo` instead of `torch.Tensor.foo`
    def forward(ctx, x):
        saved_for_backward = (x,)
        return torch.sin(x), saved_for_backward

    def backward(ctx, grad_output, *saved_tensors):
        (x,) = saved_tensors
        return grad_output * torch.cos(x)

    def my_sin(x):
        return torch.ops.higher_order.autograd_function_apply(
            forward,
            backward,
            x,
            args_tensor_mask=[True],
            non_differentiable_idx=[],
        )

    jitted = thunder_jit(my_sin)
    x = torch.randn((2, 2), requires_grad=True)
    x_ref = x.clone().detach().requires_grad_()

    y = jitted(x)
    y_ref = my_sin(x_ref)
    torch.testing.assert_close(y, y_ref)

    grad = torch.rand_like(y)
    actual_grad = torch.autograd.grad(y, x, grad)
    expect_grad = torch.autograd.grad(y_ref, x_ref, grad)
    torch.testing.assert_close(actual_grad, expect_grad)

    def wrong_backward(ctx, grad_output, *saved_tensors):
        (x,) = saved_tensors
        return grad_output * x.sin()

    def my_sin_with_wrong_backward(x):
        return torch.ops.higher_order.autograd_function_apply(
            forward,
            wrong_backward,
            x,
            args_tensor_mask=[True],
            non_differentiable_idx=[],
        )

    jitted = thunder_jit(my_sin_with_wrong_backward)
    x = torch.randn((2, 2), requires_grad=True)
    x_ref = x.clone().detach().requires_grad_()

    y = jitted(x)
    y_ref = my_sin_with_wrong_backward(x_ref)
    actual_grad = torch.autograd.grad(y, x, grad)
    expect_grad = torch.autograd.grad(y_ref, x_ref, grad)
    torch.testing.assert_close(actual_grad, expect_grad)

    from torch.autograd.gradcheck import GradcheckError
    from torch.testing._internal.common_utils import gradcheck

    with pytest.raises(GradcheckError):
        gradcheck(jitted, (x,))


def test_autograd_function_apply_with_no_grad():
    # This case is using `torch` operations
    def forward(_, x):
        saved_for_backward = (x,)

        with torch.no_grad():
            sin = torch.sin(x)
        return sin, saved_for_backward

    def backward(_, grad_output, *saved_tensors):
        return grad_output * 2

    def my_sin(x):
        res = torch.ops.higher_order.autograd_function_apply(
            forward,
            backward,
            x,
            args_tensor_mask=[True],
            non_differentiable_idx=[],
        )
        return res

    jitted = thunder_jit(my_sin)
    x = torch.randn((2, 2), requires_grad=True)

    out = jitted(x)
    grad = torch.rand_like(out)
    out.backward(grad)

    # Verify that `backward` was applied.
    torch.testing.assert_close(x.grad, grad * 2)

    # This is using `thunder` operations
    # NOTE - This takes a different codepath compared to above.
    def forward(_, x):  # noqa: F811
        saved_for_backward = (x,)
        thunder.torch._set_grad_enabled_with_warning(False)
        sin = thunder.torch.sin(x)
        thunder.torch._set_grad_enabled_with_warning(True)
        return sin, saved_for_backward

    def backward(_, grad_output, *saved_tensors):  # noqa: F811
        # NOTE - This is incorrect on purpose
        return grad_output * 2

    def fn(x):
        res = thunder.torch.autograd_function_apply(
            forward,
            backward,
            x,
            args_tensor_mask=[True],
            non_differentiable_idx=[],
        )
        return res

    jitted = thunder_jit(fn)
    x = torch.randn((2, 2), requires_grad=True)

    out = jitted(x)
    grad = torch.rand_like(out)
    out.backward(grad)

    # Verify that `backward` was applied.
    torch.testing.assert_close(x.grad, grad * 2)


def test_autograd_function_empty_forward():
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            return x

        @staticmethod
        def backward(self, grad_x):
            return 2 * grad_x

    def fn(x):
        # TODO: there still is a bug when the result is directly returned
        return Fn.apply(x) * 3

    a = torch.randn(2)
    jfn = thunder_jit(fn)

    ref = fn(a)
    out = jfn(a)

    assert_close(out, ref)

    a = torch.randn(2, requires_grad=True)
    go = torch.randn_like(a)

    ref = fn(a)
    out = jfn(a)
    (grad,) = torch.autograd.grad(out, a, go)
    (grad_ref,) = torch.autograd.grad(ref, a, go)

    assert_close(out, ref)
    assert_close(grad, grad_ref)


@requiresCUDA  # I have not found a good other object to use
def test_cpp_property():
    def fn():
        return torch.cuda.get_device_properties(0).major

    assert fn() == thunder_jit(fn)()


def test_failing_prologue_in_last_prologue_traces():
    # we know that this will fail in the prologue
    i = 0

    def fn():
        nonlocal i
        i += 1
        return i

    jfn = thunder_jit(fn)
    with pytest.raises(RuntimeError, match="Expected 1 to be equal to and have the type of 0"):
        jfn()

    # make sure that we have prologue traces in the last_prologue_traces
    assert len(thunder.last_prologue_traces(jfn)) > 0


@pytest.mark.parametrize(
    "device",
    ("cpu", "cuda"),
)
def test_matmul_nd_times_2d_runs_2d_gemm(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def f(x, y):
        return x @ y

    jf = thunder_jit(f)

    x = torch.rand(2, 3, 4, device=device)
    y = torch.rand(4, 5, device=device)

    thunder_res = jf(x, y)
    torch_res = f(x, y)
    assert_close(thunder_res, torch_res)

    trace = thunder.last_traces(jf)[-1]

    # Extract prims.matmul
    matmul_prim_bsym = None
    for bsym in trace.bound_symbols:
        if bsym.sym.name == "matmul":
            for subbsym in bsym.subsymbols:
                if subbsym.sym.name == "matmul":
                    for subsubbsym in subbsym.subsymbols:
                        if subsubbsym.sym.id == prims.PrimIDs.MATMUL:
                            matmul_prim_bsym = subsubbsym
                            break
                    break
            break

    # Check that prim.matmul outputs a 2d tensor
    assert matmul_prim_bsym is not None
    assert matmul_prim_bsym.output.ndim == 2


def test_tag_static_memory_location():
    # not much sense, but hey.
    m = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))
    jm = thunder_jit(m)
    jm(torch.randn(2, 2))
    lt = thunder.last_traces(jm)[-1]

    # input should not be tagged static
    assert thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION not in lt.args[0].tags
    # parameters and buffers should be tagged static
    assert all(thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION in a.tags for a in lt.args[1:])

    # outputs of operations should not be tagged static
    for bsym in lt.bound_symbols:
        if bsym.sym == thunder.core.prims.unpack_trivial:
            continue
        for a in bsym.flat_outs:
            if isinstance(a, thunder.Proxy):
                assert thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION not in a.tags
    assert str(thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION) == "ProxyTag.STATIC_MEMORY_LOCATION"


def test_args_order():
    @thunder_jit
    def fn(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
        # do not skip alias update process
        a9 += 1
        # do not drop arguments by dce
        return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10

    args = [torch.zeros(()) for _ in range(11)]
    args[0] = args[1] = torch.zeros((2,))
    fn(*args)

    assert [a.name for a in thunder.last_traces(fn)[-1].args] == [f"a{i}" for i in range(11)]


def test_cache_symbolic_values_dynamic_shape():
    def foo(a):
        return a.relu()

    jfoo = thunder_jit(foo, cache="symbolic values")

    a = torch.randn((2, 2, 2), device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    a = torch.randn((3, 4, 5), device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1


def test_cache_symbolic_values_reshape_numel():
    def foo(a):
        a = torch.reshape(a, [a.numel()])
        return a.relu()

    jfoo = thunder_jit(foo, cache="symbolic values")

    a = torch.randn(2, 3, 8, requires_grad=True, device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)


def test_cache_symbolic_values_slice():
    def foo(a):
        a = a[..., : a.shape[-1]]
        return a.relu()

    jfoo = thunder_jit(foo, cache="symbolic values")

    a = torch.randn(2, 3, 8, requires_grad=True, device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)


def test_cache_symbolic_values_dict():
    def foo(a, v):
        return a[v].relu()

    jfoo = thunder_jit(foo, cache="symbolic values")

    a = {
        2: torch.randn(2, 3, 8, requires_grad=True, device="cpu"),
        5: torch.randn(4, 8, requires_grad=True, device="cpu"),
    }

    actual = jfoo(a, 2)
    expected = foo(a, 2)

    assert_close(actual, expected)

    b = {
        "a": torch.randn(2, 8, requires_grad=True, device="cpu"),
        "b": torch.randn(7, requires_grad=True, device="cpu"),
    }

    actual = jfoo(b, "b")
    expected = foo(b, "b")

    assert_close(actual, expected)


def test_specific_dataclass_returns():
    import transformers

    def fn(x):
        return transformers.modeling_outputs.BaseModelOutputWithPast(last_hidden_state=x)

    jfn = thunder_jit(fn)
    x = torch.randn(2, 2)
    expected = fn(x)
    res = jfn(x)
    assert expected.last_hidden_state is x
    assert res.last_hidden_state is x


def test_modulelist_idx():
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.ModuleList(
                [
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                ]
            )

        def forward(self, x):
            for m in self.l[:-1]:
                x = m(x)
            x = self.l[-1](x)
            return x

    m = MyModel()
    jm = thunder_jit(m)
    x = torch.randn(2, 4)
    expected = m(x)
    res = jm(x)
    assert_close(res, expected)


def test_partial_method():
    import functools

    def capture(*args, **kwargs):
        return args, kwargs

    class A:
        nothing = functools.partialmethod(capture)
        positional = functools.partialmethod(capture, 1)
        keywords = functools.partialmethod(capture, a=2)
        both = functools.partialmethod(capture, 3, b=4)
        spec_keywords = functools.partialmethod(capture, self=1, func=2)

        nested = functools.partialmethod(positional, 5)

        over_partial = functools.partialmethod(functools.partial(capture, c=6), 7)

        static = functools.partialmethod(staticmethod(capture), 8)
        cls = functools.partialmethod(classmethod(capture), d=9)

    a = A()

    test_cases = [
        lambda: a.nothing(),
        lambda: a.nothing(5),
        lambda: a.nothing(c=6),
        lambda: a.nothing(5, c=6),
        lambda: a.positional(),
        lambda: a.positional(5),
        lambda: a.positional(c=6),
        lambda: a.positional(5, c=6),
        lambda: a.keywords(),
        lambda: a.keywords(5),
        lambda: a.keywords(c=6),
        lambda: a.keywords(5, c=6),
        lambda: a.both(),
        lambda: a.both(5),
        lambda: a.both(c=6),
        lambda: a.both(5, c=6),
        lambda: A.both(a, 5, c=6),
        lambda: a.spec_keywords(),
        lambda: a.nested(),
        lambda: a.nested(6),
        lambda: a.nested(d=7),
        lambda: a.nested(6, d=7),
        lambda: A.nested(a, 6, d=7),
        # These don't work because the unpacking does not work.
        # lambda: a.over_partial(),
        # lambda: a.over_partial(5),
        # lambda: a.over_partial(d=8),
        # lambda: a.over_partial(5, d=8),
        # lambda: A.over_partial(a, 5, d=8),
        lambda: A.static(),
        lambda: A.static(5),
        lambda: A.static(d=8),
        lambda: A.static(5, d=8),
        lambda: A.cls(),
        lambda: A.cls(5),
        lambda: A.cls(c=8),
        lambda: A.cls(5, c=8),
        lambda: a.static(),
        lambda: a.static(5),
        lambda: a.static(d=8),
        lambda: a.static(5, d=8),
        lambda: a.cls(),
        lambda: a.cls(5),
        lambda: a.cls(c=8),
        lambda: a.cls(5, c=8),
        lambda: a.keywords(a=3),
        lambda: A.keywords(a, a=3),
    ]

    for fn in test_cases:
        jfn = thunder.jit(fn)
        print(fn(), jfn())
        assert fn() == jfn()


@requiresCUDA
def test_jit_compile_data_cycle_leak():
    memory_at_start = torch.cuda.memory_allocated()

    def _allocate_model_in_function():
        model = torch.nn.Linear(256, 256, device="cuda")

        tfn = thunder.jit(model)
        cd = tfn._lc_cd
        return weakref.ref(cd), weakref.ref(model)

    refs = _allocate_model_in_function()

    assert torch.cuda.memory_allocated() == memory_at_start
    for ref in refs:
        assert ref() is None
