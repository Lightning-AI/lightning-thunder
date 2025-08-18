import pytest
import warnings
import itertools
import os
import subprocess
import sys
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from looseversion import LooseVersion
from unittest.mock import patch
import weakref
import re
from hypothesis import strategies as st
from hypothesis import given, settings
from hypothesis import HealthCheck
import copy

from thunder import dtypes
from thunder.dynamo import thunderfx
from thunder.dynamo.utils import CompilerType
from thunder.dynamo.compiler_graph_benchmark import ThunderCompilerGraphBenchmarking
from thunder import last_traces
from thunder.core.symbol import Symbol
from thunder.tests.bf16 import device_supports_bf16
from thunder.tests.framework import (
    instantiate,
    NOTHING,
    DynamoThunderExecutor,
    IS_WINDOWS,
    requiresCUDA,
)
from thunder.tests.make_tensor import make_tensor
from thunder.dynamo.report import (
    thunderfx_pytest_benchmark_report,
    fx_report,
    analyze_thunder_splits,
    save_failing_repros,
    get_thunder_fxgraph_reports,
)
from thunder.dynamo.benchmark_utils import (
    ThunderCompileSpecification,
    TorchCompileSpecification,
    TorchEagerSpecification,
    WallTime,
    KernelTime,
    WallTimeWithMemoryUsage,
    BoundSymbolNvfuserSpecification,
    BoundSymbolTorchCompileSpecification,
)


# This will be applied to all tests in this file.
@pytest.fixture(scope="function", autouse=True)
def reset_torch_dynamo():
    # From torch.compile docs - https://pytorch.org/docs/stable/generated/torch.compile.html
    # > Multiple compiled results can be associated with a frame up to torch._dynamo.config.cache_size_limit, which defaults to 8; at which point we will fall back to eager.
    #
    # Without this fixture, if a function frame is compiled multiple times
    # potentially due to matrix of inputs then it will hit cache_size_limit
    # and fallback to eager.
    #
    # [0/8] torch._dynamo hit config.cache_size_limit (8)
    # [0/8]    function: 'func' (lightning-thunder/thunder/tests/test_dynamo.py:26)
    # [0/8]    last reason: 0/0:
    # [0/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
    torch._dynamo.reset()


def run_script(file_name, cmd):
    cmd = cmd + [file_name]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, f"Script {file_name} failed: {result}"


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.skipif(
            condition=IS_WINDOWS,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_basic(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

    def func(x):
        x = torch.sin(x)
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

    compiled = thunderfx(func, dynamic=dynamic)
    out = compiled(x)

    # out should have grad_fn and its name should be ThunderFunctionBackward
    assert out.grad_fn is not None
    assert out.grad_fn.name() == "ThunderFunctionBackward"

    # We record the GraphModules that was compiled by ThunderCompiler
    backend = compiled._backend
    assert len(backend.subgraph_infos) == 2  # 2 due to data-dependent flow

    for subgraph_info in backend.subgraph_infos:
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
        for thunder_fn in subgraph_info.thunder_compiled_fns:
            assert last_traces(thunder_fn)  # Verify that we can fetch last_traces


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_basic_splitter(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.ones(2, 2, device=device, dtype=dtype, requires_grad=True)

    def func(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        y = torch.sinc(x) + torch.cos(x)
        return y + 1

    cfunc = thunderfx(func, dynamic=dynamic)
    expected = torch.compile(func, dynamic=False)(x)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    backend = cfunc._backend
    assert len(backend.subgraph_infos) == 1
    assert len(backend.subgraph_infos[0].submodule_to_compiled_functions) > 1  # Verify that the subgraph was split.
    assert any(
        "automatic torch fallback" in split_reason.info for split_reason in backend.subgraph_infos[0].split_reasons
    )  # Verify that we had a split because we detected an `automatic registered operator`
    targets = (node.target for node in backend.subgraph_infos[0].split_graph_module.graph.nodes)
    assert any(target.startswith("thunder_") for target in targets)  # Verify that the submodules have name `thunder_*`


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_splitter_autocast_ctx(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.rand(2, 2, device=device, dtype=dtype, requires_grad=True)

    def func(x):
        x = x + 2
        with torch.autocast("cpu"):
            y = torch.log(x)
            return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=False)(x)

    cfunc = thunderfx(func, dynamic=dynamic)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    backend = cfunc._backend
    assert len(backend.subgraph_infos) == 1
    assert len(backend.subgraph_infos[0].split_reasons) == 0
    compiled_functions = tuple(backend.subgraph_infos[0].submodule_to_compiled_functions.values())
    assert all(compiled_fn.compiler == CompilerType.THUNDER for compiled_fn in compiled_functions)
    assert not any(compiled_fn.compiler == CompilerType.TORCH_INDUCTOR for compiled_fn in compiled_functions)


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_splitter_autocast_ctx_with_graph_break(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.rand(2, 2, device=device, dtype=dtype, requires_grad=True)

    def func(x):
        x = x + 2
        with torch.autocast(device):
            y = torch.sin(x)
            torch._dynamo.graph_break()
            return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=dynamic)(x)
    cfunc = thunderfx(func, dynamic=dynamic)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    backend = cfunc._backend
    # 2 subgraphs due to graph-break
    assert len(backend.subgraph_infos) == 2
    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0
        compiled_functions = tuple(subgraph_info.submodule_to_compiled_functions.values())
        assert all(compiled_fn.compiler == CompilerType.THUNDER for compiled_fn in compiled_functions)
        assert not any(compiled_fn.compiler == CompilerType.TORCH_INDUCTOR for compiled_fn in compiled_functions)


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_splitter_autocast_ctx_with_split(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.rand(2, 2, device=device, dtype=dtype, requires_grad=True)

    def func(x):
        x = x + 2
        with torch.autocast(device):
            y = torch.sin(x)

            #  torch.sinc has automatic fallback registered,
            # so that operation will be given to inductor.
            y = torch.sinc(y)
            return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=dynamic)(x)
    cfunc = thunderfx(func, dynamic=dynamic)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    backend = cfunc._backend
    assert len(backend.subgraph_infos) == 1  # no graph break in dynamo

    subgraph_info = backend.subgraph_infos[0]
    assert len(subgraph_info.split_reasons) > 1  # Split due to `torch.sinc`
    compiled_functions = tuple(subgraph_info.submodule_to_compiled_functions.values())
    assert any(compiled_fn.compiler == CompilerType.THUNDER for compiled_fn in compiled_functions)
    assert any(compiled_fn.compiler == CompilerType.TORCH_INDUCTOR for compiled_fn in compiled_functions)
    assert any(
        "automatic torch fallback" in split_reason.info for split_reason in subgraph_info.split_reasons
    )  # Verify that we had a split because we detected an `automatic registered operator`


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_splitter_autograd_function(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    # Workaround for "RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered"
    # https://github.com/pytorch/pytorch/issues/124565
    if device != "cpu":
        torch.empty(1, device="cuda", requires_grad=True).backward()

    class Sin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sin(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * torch.cos(x) * 100

    def func(x):
        y = torch.cos(x) + Sin.apply(x)
        return torch.matmul(x, y)

    x = torch.ones(2, device=device, dtype=dtype, requires_grad=True)
    expected = torch.compile(func, dynamic=dynamic)(x)

    cfunc = thunderfx(func, dynamic=dynamic)
    actual = cfunc(x)

    backend = cfunc._backend
    assert len(backend.subgraph_infos) == 1  # no graph break in dynamo
    subgraph_info = backend.subgraph_infos[0]
    assert len(subgraph_info.split_reasons) == 0  # no split
    assert len(subgraph_info.thunder_compiled_fns) == 1
    jfunc = subgraph_info.thunder_compiled_fns[0]
    trc = last_traces(jfunc)[0]
    assert any(
        isinstance(bsym.sym.id, str) and bsym.sym.id.startswith("higher_order_autograd_function_apply")
        for bsym in trc.bound_symbols
    )

    # Verify forward pass
    torch.testing.assert_close(actual, expected)

    # Verify backward pass
    g = torch.rand_like(actual)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
)
def test_force_skip_lazy_graph_module(executor, device: str, dtype: dtypes.dtype):
    with torch.fx._lazy_graph_module._force_skip_lazy_graph_module():
        x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

        def func(x):
            x = torch.sin(x)
            return x + 2

        cfunc = thunderfx(func)
        out = cfunc(x)

        # out should have grad_fn and its name should be ThunderFunctionBackward
        assert out.grad_fn is not None
        assert out.grad_fn.name() == "ThunderFunctionBackward"

        backend = cfunc._backend
        # We record the GraphModules that was compiled by ThunderCompiler
        assert len(backend.subgraph_infos) == 1

        for subgraph_info in backend.subgraph_infos:
            assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
            assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
            for thunder_fn in subgraph_info.thunder_compiled_fns:
                assert last_traces(thunder_fn)  # Verify that we can fetch last_traces


@instantiate(
    dtypes=NOTHING, executors=[DynamoThunderExecutor], decorators=(pytest.mark.parametrize("cat_kwarg", (True, False)),)
)
def test_cat_no_split(executor, device: str, dtype: dtypes.dtype, cat_kwarg):
    # fx.Node for `torch.cat` receives `torch.fx.immutable_collections.immutable_list` as Node.args.
    # This test verifies that we don't cause a split because of this.
    x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

    if not cat_kwarg:

        def func(x):
            x = torch.cat([x, x])
            return x + 2

    else:

        def func(x):
            x = torch.cat(tensors=[x, x])
            return x + 2

    cfunc = thunderfx(func)
    out = cfunc(x)

    backend = cfunc._backend
    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 1

    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0  # Verify there were no splits
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
        for thunder_fn in subgraph_info.thunder_compiled_fns:
            assert last_traces(thunder_fn)  # Verify that we can fetch last_traces


@instantiate(dtypes=NOTHING, executors=[DynamoThunderExecutor])
def test_method_only_registrations(executor, device: str, dtype: dtypes.dtype):
    # In thunder, some operations are registered only as methods and put in a different map (accessible via torchctx).
    # This test is to verify that we consider those methods as supported in `thunder` and don't cause a split because of them.

    def func(x):
        y = x.float()
        return y.sin()

    x = torch.randn(3, 3, device=device, dtype=dtype)
    cfunc = thunderfx(func)
    o = cfunc(x)

    backend = cfunc._backend
    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 1

    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0  # Verify there were no splits
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
        for thunder_fn in subgraph_info.thunder_compiled_fns:
            assert last_traces(thunder_fn)  # Verify that we can fetch last_traces


@instantiate(dtypes=NOTHING, executors=[DynamoThunderExecutor])
def test_where_nonzero_overload(executor, device: str, dtype: dtypes.dtype):
    # Verify that `torch.where(cond)` leads to graph break and `torch.where(cond, x, y)`
    # is correctly passed to `thunder`.

    def func(x):
        y = x[torch.where(x > 0.5)]  # This will lead to graph-break
        y = torch.where(y > 1, y, 0)
        return y.sin()

    x = torch.randn(3, 3, device=device, dtype=dtype, requires_grad=True)
    cfunc = thunderfx(func)
    actual = cfunc(x)
    expected = torch.compile(func, backend="eager")(x)

    backend = cfunc._backend
    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 2  # There were 2 graphs.

    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0  # Verify there were no splits in the subgraph.
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
        for thunder_fn in subgraph_info.thunder_compiled_fns:
            assert last_traces(thunder_fn)  # Verify that we can fetch last_traces

    torch.testing.assert_close(actual, expected)

    g = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)


@instantiate(
    dtypes=(dtypes.float32,),
    executors=(DynamoThunderExecutor,),
    decorators=(
        pytest.mark.skip(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1821"),
        pytest.mark.parametrize(
            "optim",
            (
                torch.optim.SGD,
                torch.optim.Adam,
                torch.optim.AdamW,
            ),
            ids=(
                "sgd",
                "adam",
                "adamw",
            ),
        ),
        pytest.mark.skipif(
            IS_WINDOWS,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
@requiresCUDA
def test_thundercompiler_optim_step(executor, device, dtype, optim):
    from thunder.tests.distributed.helper import ToyModel

    if not device_supports_bf16(device):
        pytest.skip(f"{device} does not support bf16")

    tdtype = dtypes.to_torch_dtype(dtype)
    model = ToyModel().to(device=device, dtype=tdtype)
    optimizer = optim(model.parameters())
    jitted_step = executor.make_callable(optimizer.step)

    ref_model = ToyModel().to(device=device, dtype=tdtype)
    ref_model.load_state_dict(model.state_dict())
    ref_optimizer = optim(ref_model.parameters())
    ref_optimizer.load_state_dict(optimizer.state_dict())

    for i in range(2):
        x = make_tensor((1, ToyModel.N_IN), dtype=tdtype, device=device)
        x_ref = x.clone().detach()

        y = model(x)
        y.mean().backward()
        jitted_step()
        optimizer.zero_grad()

        y_ref = ref_model(x_ref)
        y_ref.mean().backward()
        ref_optimizer.step()
        ref_optimizer.zero_grad()

        # There could be numerical error, see https://github.com/NVIDIA/Fuser/issues/2664
        torch.testing.assert_close(
            tuple(model.parameters()),
            tuple(ref_model.parameters()),
            msg=lambda s: f"{i + 1}-iter {s}",
        )


@instantiate(dtypes=NOTHING, executors=[DynamoThunderExecutor])
def test_no_grad_ctx_manager(executor, device: str, dtype: dtypes.dtype):
    def func(x):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = x @ x
        return y + x

    x = torch.randn(3, 3, device=device, dtype=dtype, requires_grad=True)
    cfunc = thunderfx(func)
    actual = cfunc(x)
    expected = torch.compile(func, backend="eager")(x)

    backend = cfunc._backend
    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 1

    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0  # Verify there were splits in the subgraph.
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)

    torch.testing.assert_close(actual, expected)

    g = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)


@instantiate(dtypes=NOTHING, executors=[DynamoThunderExecutor])
def test_no_grad_enabled_grad_nested_ctx_manager(executor, device: str, dtype: dtypes.dtype):
    def func(x):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = x @ x

            with torch.enable_grad():
                z = x.sin()
        return y + x + z

    x = torch.randn(3, 3, device=device, dtype=dtype, requires_grad=True)
    cfunc = thunderfx(func)
    actual = cfunc(x)
    expected = torch.compile(func, backend="eager")(x)

    backend = cfunc._backend
    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 1

    for subgraph_info in backend.subgraph_infos:
        assert len(subgraph_info.split_reasons) == 0  # Verify there were splits in the subgraph.
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)

    torch.testing.assert_close(actual, expected)

    g = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_empty_autocast():
    autocast_ops = (torch.amp.autocast_mode._enter_autocast, torch.amp.autocast_mode._exit_autocast)

    def _call_thunder_backend(fn, args):
        jf = thunderfx(f)
        jf(*args)
        return jf._backend

    # autocast region is removed
    def f():
        with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
            pass
        return

    backend = _call_thunder_backend(f, ())
    assert all(node.target not in autocast_ops for node in backend.subgraph_infos[0].split_graph_module.graph.nodes)

    # Both autocast regions are removed
    def f(x):
        with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
            pass
        y = x @ x
        with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
            pass
        return y

    x = torch.randn(3, 3)
    backend = _call_thunder_backend(f, (x,))

    all_nodes = itertools.chain(
        backend.subgraph_infos[0].split_graph_module.graph.nodes,
        backend.subgraph_infos[0].split_graph_module.thunder_0.graph.nodes,
    )
    assert all(node.target not in autocast_ops for node in all_nodes)

    # First autocast region is removed and second isn't
    def f(x):
        with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
            pass
        y = x @ x
        with torch.autocast(dtype=torch.bfloat16, device_type="cpu"):
            y = y @ y
        return y

    x = torch.randn(3, 3)
    backend = _call_thunder_backend(f, (x,))
    all_nodes = itertools.chain(
        backend.subgraph_infos[0].split_graph_module.graph.nodes,
        backend.subgraph_infos[0].split_graph_module.thunder_0.graph.nodes,
    )
    assert sum(node.target in autocast_ops for node in all_nodes) == 2


# Sample command to run the benchmark using ThunderCompilerGraphBenchmarking
# pytest thunder/tests/test_dynamo.py -k test_ThunderCompilerGraphBenchmarking_groupby --benchmark-group-by='graph-by-graph:param:GraphID,param:SplitModuleName'
# For more details, see :class:`thunder.dynamo.compiler_graph_benchmark.ThunderCompilerGraphBenchmarking`
# NOTE: The conftest.py file customizes the benchmark grouping behavior for ThunderCompilerGraphBenchmarking.
# It must be located in the same folder as the test file to ensure the configuration.
@requiresCUDA
def test_ThunderCompilerGraphBenchmarking_LitGTMLPBenchmark(benchmark):
    import thunder

    backend = ThunderCompilerGraphBenchmarking(
        benchmark, executors={"thunder": thunder.jit, "inductor": torch.compile, "eager": None}
    )
    from thunder.benchmarks import LitGPTMLPBenchmark, Benchmark

    bench: Benchmark = LitGPTMLPBenchmark(
        config="Llama-2-7b-hf",
        batchdims=(2,),
        device="cuda:0",
        requires_grad=True,
    )

    args, kwargs = bench.make_batch()
    # Using torch.compile here fails with "TypeError: cannot pickle '_io.TextIOWrapper' object" in
    # https://github.com/Lightning-AI/pytorch-lightning/blob/828fd998961f6a60f92c35254bb94d6e049ad069/src/lightning/fabric/wrappers.py#L421
    fn = torch._dynamo.optimize(backend=backend)(bench.fn())
    fn(*args, **kwargs)


@requiresCUDA
def test_ThunderCompilerGraphBenchmarking_groupby(benchmark):
    def f(x, y):
        x = torch.sin(x)
        if x.sum() > 0:
            x = x.exp()
            y = torch.sinc(x) + torch.cos(y)
            return y
        else:
            y = y.exp()
            x = torch.sinc(y) + torch.cos(x)
            return x

    import thunder

    backend = ThunderCompilerGraphBenchmarking(benchmark, executors={"thunder": thunder.jit, "inductor": torch.compile})
    compiled = torch.compile(backend=backend)(f)
    x = torch.ones(2).cuda()
    y = torch.ones(2, requires_grad=True).cuda()
    compiled(x, y)


@requiresCUDA
def test_ThunderCompilerGraphBenchmarking_post_graph(benchmark):
    def f(x):
        return torch.sin(x)

    import thunder
    from functools import partial

    x = torch.randn((2, 2), device="cuda").requires_grad_()
    post_gp = partial(torch.cuda.make_graphed_callables, num_warmup_iters=1, allow_unused_input=True)
    backend = ThunderCompilerGraphBenchmarking(
        benchmark, executors={"inductor": torch.compile, "thunder": thunder.jit}, post_graph=post_gp
    )
    compiled = torch.compile(backend=backend)(f)
    compiled(x)


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("2.6.0"),
    reason="The checkpoint function becomes a submodule of the module containing `tag_activation_checkpoint` in PyTorch 2.6.0.",
)
@requiresCUDA
def test_ThunderCompilerGraphBenchmarking_checkpoint(benchmark):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)

        def forward(self, x):
            x = torch.utils.checkpoint.checkpoint(self.layer1, x)
            x = F.relu(x)
            return x

    x = torch.randn(5, 10).cuda().requires_grad_()
    model = SimpleModel().cuda().train()

    backend = ThunderCompilerGraphBenchmarking(benchmark, executors={"inductor": torch.compile, "thunderfx": thunderfx})
    # Using torch.compile here fails with "TypeError: cannot pickle '_io.TextIOWrapper' object" in
    # https://github.com/Lightning-AI/pytorch-lightning/blob/828fd998961f6a60f92c35254bb94d6e049ad069/src/lightning/fabric/wrappers.py#L421
    jf = torch._dynamo.optimize(backend=backend)(model)
    out = jf(x)


@requiresCUDA
@pytest.mark.filterwarnings(r"ignore:`torch\.cpu\.amp\.autocast\((.*?)\)` is deprecated.*:FutureWarning")
def test_checkpoint_converter():
    import torch.utils.checkpoint as checkpoint

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 20)

        def forward(self, x):
            x = torch.sin(x)
            x = checkpoint.checkpoint(self.layer1, x)
            x = checkpoint.checkpoint(self.layer2, x)
            x = F.relu(x)
            return x

    # Input tensor
    x = torch.randn(5, 10).cuda().requires_grad_()
    x_ref = x.detach().requires_grad_()

    model = SimpleModel().cuda().train()
    ref_model = SimpleModel().cuda().train()
    ref_model.load_state_dict(model.state_dict())

    jf = thunderfx(model)

    ref_out = ref_model(x_ref)
    out = jf(x)
    torch.testing.assert_close(ref_out, out)

    g = torch.randn_like(out)
    out.backward(g)

    ref_g = g.clone()
    ref_out.backward(ref_g)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(tuple(model.parameters()), tuple(ref_model.parameters()))


@requiresCUDA
def test_checkpoint_converter_submodule():
    import torch.utils.checkpoint as checkpoint

    class SubModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Sequential(nn.ReLU(), nn.Linear(10, 10))

        def forward(self, x):
            return self.lin(x)

    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sub_mod = SubModule()

        def forward(self, x):
            x = torch.sin(x)
            x = checkpoint.checkpoint(self.sub_mod, x)
            return x

    x = torch.randn(5, 10, device="cuda", requires_grad=True)
    model = SimpleModel().cuda()
    jf = thunderfx(model)
    out = jf(x)
    backend = jf._backend

    subgraph_info = backend.subgraph_infos[0]
    split_m = subgraph_info.split_graph_module

    def find_target_module(model, target_module_name):
        if hasattr(model, target_module_name):
            return getattr(model, target_module_name)
        for submodule in model.children():
            cur = find_target_module(submodule, target_module_name)
            if cur is not None:
                return cur
        return None

    submodule_name = "wrap_body_0"
    # 2.6.0a0+git9ca749d, split_m:
    # GraphModule(
    #     (thunder_1): ThunderModule(
    #         (_model): GraphModule(
    #             (wrap_body_0): GraphModule()
    #         )
    #     )
    # )
    #
    # torch 2.4.0
    # GraphModule(
    #     (wrap_body_0): GraphModule()
    #     (thunder_1): ThunderModule(
    #         (_model): GraphModule()
    #     )
    # )
    submodule = find_target_module(split_m, submodule_name)
    assert submodule is not None
    for n in submodule.graph.nodes:
        if n.op == "call_function":
            assert isinstance(n.target, Symbol)


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("use_pytest_benchmark", (True, False), ids=("benchmark", "repro")),),
)
def test_dynamo_reproducer_2graph(executor, device: str, dtype: dtypes.dtype, use_pytest_benchmark, tmp_path):
    if IS_WINDOWS and use_pytest_benchmark:
        pytest.skip(
            "Skipping on Windows because this uses torch.compile (see https://github.com/Lightning-AI/lightning-thunder/issues/1326)"
        )

    from thunder import nvfuser_executor
    from thunder.transforms import ConstantFolding

    def func(x):
        x = torch.sin(x)
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

    if device.startswith("cuda"):
        cfunc = thunderfx(
            func,
            transforms=[
                ConstantFolding(),
            ],
            executors=[nvfuser_executor],
            cache="constant values",
            langctx=None,
        )
    else:
        cfunc = thunderfx(func, executors=None)
    # Test non-contiguous input tensor
    x = make_tensor((4, 4), low=3, high=10, dtype=torch.int64, device=device, noncontiguous=True)

    out = cfunc(x)
    cfunc._backend.save_reproducer_to_folder(tmp_path, use_pytest_benchmark=use_pytest_benchmark)

    suffix = "_benchmark" if use_pytest_benchmark else "_repro"
    s1 = f"{tmp_path}/graph0_thunder_0{suffix}.py"
    s2 = f"{tmp_path}/graph1_thunder_0{suffix}.py"
    assert os.path.exists(s1)
    assert os.path.exists(s2)
    cmd = [sys.executable]
    if use_pytest_benchmark:
        cmd = cmd + ["-m", "pytest"]
    cmd1 = cmd + [s1]
    cmd2 = cmd + [s2]
    result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert result1.returncode == 0, f"Reproducer {s1} failed: {result1}"
    assert result2.returncode == 0, f"Reproducer {s2} failed: {result2}"


@requiresCUDA
@pytest.mark.parametrize("use_pytest_benchmark", (True, False), ids=("benchmark", "repro"))
def test_dynamo_reproducer_submodules(use_pytest_benchmark, tmp_path):
    from thunder.tests.distributed.helper import ToyModel
    import torch.nn as nn

    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sub_mod = ToyModel()
            self.seq = nn.Sequential(self.sub_mod, nn.ReLU())

        def forward(self, x):
            x = torch.sin(x)
            x = self.seq(x)
            return x

    x = torch.randn(1, ToyModel.N_IN, device="cuda", requires_grad=True)
    model = SimpleModel().cuda()
    jf = thunderfx(model)
    out = jf(x)
    jf._backend.save_reproducer_to_folder(tmp_path, use_pytest_benchmark=use_pytest_benchmark)

    suffix = "_benchmark" if use_pytest_benchmark else "_repro"
    s1 = f"{tmp_path}/graph0_thunder_0{suffix}.py"
    assert os.path.exists(s1)
    cmd = [sys.executable]
    if use_pytest_benchmark:
        cmd = cmd + ["-m", "pytest"]
    cmd1 = cmd + [s1]
    result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result1.returncode == 0, f"Reproducer {s1} failed: {result1}"


def test_deepcopy_graph_module():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1

    m = MyModule()
    gm = torch.fx.symbolic_trace(m)
    n = gm.graph.find_nodes(op="output")
    gm.graph.erase_node(n[0])
    import thunder

    _, subgraph_info = thunder.dynamo.splitter._splitter(gm, thunder.jit, thunder.jit, [])
    original_split_gm = subgraph_info.original_split_graph_module.split_graph_module
    assert original_split_gm.graph.find_nodes(op="output")
    for subm in original_split_gm.children():
        assert subm.graph.find_nodes(op="output")
    import copy

    # No assertion error
    copy_gm = copy.deepcopy(original_split_gm)


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("use_pytest_benchmark", (True, False), ids=("benchmark", "repro")),),
)
@given(file_indices=st.lists(st.integers(min_value=0, max_value=2), min_size=2, max_size=2, unique=True))
@settings(max_examples=1, deadline=None)
def test_dynamo_reproducer_split(
    executor, device: str, dtype: dtypes.dtype, use_pytest_benchmark, tmp_path, file_indices
):
    if IS_WINDOWS and use_pytest_benchmark:
        pytest.skip(
            "Skipping on Windows because this uses torch.compile (see https://github.com/Lightning-AI/lightning-thunder/issues/1326)"
        )

    x = torch.ones(2, 2, device=device, dtype=dtype, requires_grad=True)

    def func(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        y = torch.sinc(x) + torch.cos(x)
        y = y + torch.sinc(x)
        return y + 1

    cfunc = thunderfx(func)
    actual = cfunc(x)
    cfunc._backend.save_reproducer_to_folder(tmp_path, use_pytest_benchmark)

    suffix = "_benchmark" if use_pytest_benchmark else "_repro"
    s1 = f"{tmp_path}/graph0_thunder_0{suffix}.py"
    s2 = f"{tmp_path}/graph0_thunder_2{suffix}.py"
    s3 = f"{tmp_path}/graph0_thunder_4{suffix}.py"
    cmd = [sys.executable]
    if use_pytest_benchmark:
        cmd = cmd + ["-m", "pytest"]

    all_files = [s1, s2, s3]
    selected_files = [all_files[i] for i in file_indices]
    for file in selected_files:
        run_script(file, cmd)


@requiresCUDA
def test_thunderfx():
    def foo(x):
        return torch.sin(x) + torch.cos(x)

    x = torch.randn(4, 4, device="cuda", requires_grad=True)
    cfoo = thunderfx(foo)
    cfoo(x)
    thunder_compiled_fns = cfoo._backend.subgraph_infos[0].thunder_compiled_fns
    assert len(thunder_compiled_fns) == 1
    assert last_traces(thunder_compiled_fns[0])

    from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform

    cfoo = thunderfx(foo, dynamic=True, transforms=[NvtxProfileTransform()])
    cfoo(x)
    thunder_compiled_fns = cfoo._backend.subgraph_infos[0].thunder_compiled_fns
    assert len(thunder_compiled_fns) == 1
    trc = last_traces(thunder_compiled_fns[-1])[-1]
    assert any(bsym.sym.id == "nvtx_range_push" for bsym in trc.bound_symbols)

    def fn(x, w):
        return x @ w

    x = torch.randn(4, 4, device="cuda", requires_grad=True)
    w = torch.randn(4, 4, device="cuda", requires_grad=True)
    # Tests the compile_options in thunder.jit
    cfn = thunderfx(fn, nv_enable_matmul=True)
    cfn(x, w)
    trc = cfn.last_traces[-1]
    assert any(bsym.sym.name == "nvFusion0" for bsym in trc.bound_symbols)


def test_thunderfx_last_traces():
    def foo(x):
        return torch.sin(x) + torch.cos(x)

    x = torch.randn((4, 4), requires_grad=True)
    cfoo = thunderfx(foo)
    cfoo(x)
    assert cfoo.last_traces != []
    assert cfoo.last_backward_traces != []

    # Call it w/o invoking the function first.
    dfoo = thunderfx(foo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert dfoo.last_traces == []
        assert "Must invoke" in str(w[0].message)
        assert dfoo.last_backward_traces == []
        assert "before function invoked" in str(w[1].message)


def test_get_example_input_tensor_metadata():
    from thunder.dynamo.utils import (
        _get_example_input_tensor_metadata,
        arg_like_tensor,
        arg_like,
        _create_random_tensor_from_tensor_metadata,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    int_tensor = torch.arange(1, 11, dtype=torch.int)
    meta_int_tensor = _get_example_input_tensor_metadata(int_tensor)
    assert meta_int_tensor.dtype == torch.int and meta_int_tensor.min_val >= 1 and meta_int_tensor.max_val < 11

    fake_mode = FakeTensorMode()
    fake_tensor = fake_mode.from_tensor(torch.empty(4, 4, dtype=torch.float32))
    meta_fake_tensor = _get_example_input_tensor_metadata(fake_tensor)
    assert meta_fake_tensor.shape == (4, 4) and meta_fake_tensor.dtype == torch.float32

    t0 = torch.randn((5, 10), device="meta")
    meta_t0 = _get_example_input_tensor_metadata(t0)
    assert meta_t0.min_val == None and meta_t0.max_val == None and meta_t0.device.type == "meta"
    t0_str = arg_like_tensor(meta_t0)
    assert (
        t0_str
        == "torch.testing.make_tensor((5, 10), dtype=torch.float32,  device='meta', requires_grad=False, low=None, high=None,),"
    )

    t1 = torch.randn(20).as_strided((2, 3), (4, 2), storage_offset=2).requires_grad_()
    meta_t1 = _get_example_input_tensor_metadata(t1)
    assert meta_t1.strides == (4, 2)
    assert meta_t1.storage_offset() == 2
    assert meta_t1.shape == (2, 3)
    assert meta_t1.storage_shape == (11,)
    t1_str = arg_like_tensor(meta_t1)
    p1 = r"""^torch\.testing\.make_tensor\(\(11,\), dtype=torch\.float32,\s*device='cpu',\s*requires_grad=True,\s*low=[-+]?[0-9]*\.?[0-9]+,\s*high=[-+]?[0-9]*\.?[0-9]+,\)\.as_strided\(\(2, 3\), \(4, 2\), 2\),$"""
    assert re.fullmatch(p1, t1_str), "The string does not match the expected format!"

    # Tests for nested inputs
    inputs = [
        [
            torch.randn((), dtype=torch.bfloat16, device="cpu", requires_grad=False),
            torch.randn((), dtype=torch.bfloat16, device="cpu", requires_grad=False),
        ],
        torch.randn(24512, dtype=torch.bfloat16, device="cpu", requires_grad=False).as_strided(
            (128, 1, 128), (192, 24576, 1), 0
        ),
    ]
    str_out = arg_like(inputs)
    out = eval(str_out)[0]
    assert len(out) == len(inputs) and len(out[0]) == len(inputs[0])
    assert out[0][0].shape == inputs[0][0].shape
    assert out[0][1].shape == inputs[0][1].shape
    assert out[1].shape == inputs[1].shape and out[1].stride() == inputs[1].stride()

    # Tests for contiguous tensor with storage_offset
    t2 = torch.randn(1024).as_strided((1, 1, 64), (576, 576, 1), storage_offset=512)
    meta_t2 = _get_example_input_tensor_metadata(t2)
    assert meta_t2.shape == (1, 1, 64) and meta_t2.stride() == (576, 576, 1) and meta_t2.storage_offset() == 512
    t2_str = arg_like_tensor(meta_t2)
    p2 = r"""^torch\.testing\.make_tensor\(\(576,\), dtype=torch\.float32,\s*device='cpu',\s*requires_grad=False,\s*low=[-+]?[0-9]*\.?[0-9]+,\s*high=[-+]?[0-9]*\.?[0-9]+,\)\.as_strided\(\(1, 1, 64\), \(576, 576, 1\), 512\),$"""
    assert re.fullmatch(p2, t2_str), "The string does not match the expected format!"
    t2_tensor = _create_random_tensor_from_tensor_metadata(meta_t2)
    assert t2_tensor.shape == (1, 1, 64) and t2_tensor.stride() == (576, 576, 1) and t2_tensor.storage_offset() == 512


def test_thunderfx_meta_tensor():
    def foo(x):
        y = torch.sin(x)
        return y

    t0 = torch.randn((5, 10), device="meta")

    thfoo = thunderfx(foo)
    out = thfoo(t0)
    assert out.device.type == "meta"


@requiresCUDA
def test_report_thunderfx_pytest_benchmark_report(tmp_path, capsys):
    def foo(x):
        return x.sin()

    x = torch.randn(4, 4, device="cuda")
    thunderfx_pytest_benchmark_report(foo, x, folder_path=tmp_path, check_consistency=True)
    captured = capsys.readouterr()
    msg = captured.out
    assert not captured.err
    assert "Verifying consistency" in msg
    assert "Analyzing performance through benchmarking" in msg
    assert "The input callable can be successfully executed by ThunderFX." in msg
    assert "Max allocated CUDA memory usage:" in msg

    with patch("torch.compile", side_effect=Exception("compilation raises exception")):
        thunderfx_pytest_benchmark_report(foo, x, folder_path=tmp_path, check_consistency=False)
        captured = capsys.readouterr()
        assert not captured.err
        assert "Failed to run the function using ThunderFX" in captured.out
        assert "Failed to save reproducer" in captured.out


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(
        pytest.mark.parametrize("use_benchmark", (True, False), ids=("benchmark", "repro")),
        pytest.mark.xfail(
            condition=IS_WINDOWS,
            strict=True,
            reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
        ),
    ),
)
def test_fxreport(executor, device: str, dtype: dtypes.dtype, use_benchmark, tmp_path):
    def foo(x, y):
        return x + y

    from thunder import jit

    with pytest.raises(
        ValueError,
        match=r"fx_report requires the original \(uncompiled\) callable and cannot be used on the Thunder-compiled function.",
    ):
        fx_report(jit(foo))

    x = torch.randn(4, 4, device=device, requires_grad=True)
    y = torch.randn(4, 4, device=device, requires_grad=True)
    results = fx_report(foo, dynamic=True)(x, y)
    for r in results.fx_graph_reports:
        r.write_eager_repro(tmp_path, use_benchmark=use_benchmark)
        r.write_thunder_repro(tmp_path, use_benchmark=use_benchmark)
        r.write_inductor_repro(tmp_path, use_benchmark=use_benchmark)
        my_exe = "partial(thunder.jit, executors=[pytorch_executor])"
        my_imports = [
            "import thunder",
            "from thunder import pytorch_executor",
            "from functools import partial",
        ]
        if use_benchmark:
            r.write_pytest_benchmark(
                tmp_path, f"{r.graph_name}_mythunder_benchmark.py", ["mythunder"], [my_exe], my_imports
            )

    cmd = [sys.executable]
    if use_benchmark:
        cmd = cmd + ["-m", "pytest"]
    py_files = list(tmp_path.glob("*.py"))
    num_of_files = 4 if use_benchmark else 3
    assert len(py_files) == num_of_files

    for file in py_files:
        run_script(file, cmd)


def test_leak_on_unsupported_thunder_operator():
    # This test is to check the fix for a previous leak
    # which was caused by holding onto the
    # exception object in split_reason.

    def unsupported_op_fn(w1) -> torch.Tensor:
        topk_ids = torch.tensor([[0, 1]])
        # This operation is not supported by thunder and get's passed to inductor.
        return torch.sinc(w1) + 1

    def call_thunderfx_on_leaking_fn():
        w1 = torch.randn(16, 16, 32, dtype=torch.bfloat16)
        fn = thunderfx(unsupported_op_fn)
        fn(w1)

        # There should be two thunder traces because of the split caused by indexing.
        # In future when thunder supports this indexing, the test will fail here
        # as we won't be checking for the leak in case of unsupported operation.
        assert len(fn.last_traces) == 2

        return weakref.ref(w1)

    t_weak_ref = call_thunderfx_on_leaking_fn()
    assert t_weak_ref() is None


@requiresCUDA
@given(file_indices=st.lists(st.integers(min_value=0, max_value=15), min_size=2, max_size=2, unique=True))
@settings(max_examples=2, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_thunder_specific_reports(tmp_path, file_indices):
    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    def foo(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        torch._dynamo.graph_break()
        y = torch.sinc(x) + torch.cos(x)
        return y + 1

    results = fx_report(foo)(x)
    for idx, fx_graph_report in enumerate(results.fx_graph_reports):
        thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report)
        thunder_fx_graph_report.write_thunder_repro(tmp_path)
        for thunder_split_report in thunder_fx_graph_report.subgraph_reports:
            split_folder = tmp_path / str(idx)
            thunder_split_report.write_eager_repro(split_folder)
            thunder_split_report.write_thunder_repro(split_folder)
            thunder_split_report.write_inductor_repro(split_folder)
            thunder_split_report.create_fusion_reports()
            for nvf in thunder_split_report.fusion_reports:
                nvf.write_nvfuser_repro(split_folder / "nvfusion")
                nvf.write_inductor_repro(split_folder / "nvfusion")

    cmd = [sys.executable]
    py_files = list(tmp_path.rglob("*.py"))
    assert len(py_files) == 16

    selected_files = [py_files[i] for i in file_indices]
    for file in selected_files:
        run_script(file, cmd)


@requiresCUDA
def test_WallTime_KernelTime():
    from nvfuser import FusionDefinition, DataType

    def nvfuser_fusion_id2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
        )
        T1 = fd.define_tensor(
            shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
        )
        T2 = fd.ops.cos(T0)
        T3 = fd.ops.add(T1, T2)
        S4 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T5 = fd.ops.add(T3, S4)
        fd.add_output(T5)

    with FusionDefinition() as fd:
        nvfuser_fusion_id2(fd)

    inputs = [
        torch.testing.make_tensor((2, 2), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((2, 2), dtype=torch.float32, device="cuda:0"),
    ]

    WallTime.time(stmt="fd.execute(inputs)", globals={"fd": fd, "inputs": inputs})
    KernelTime.time(stmt="fd.execute(inputs)", globals={"fd": fd, "inputs": inputs})

    m = WallTimeWithMemoryUsage.time(stmt="fd.execute(inputs)", globals={"fd": fd, "inputs": inputs})
    torch.cuda.reset_peak_memory_stats()
    fd.execute(inputs)
    max_mem = torch.cuda.max_memory_allocated()
    assert max_mem == m.max_allocated_memory


@requiresCUDA
def test_ThunderCompileSpecification():
    from thunder.executors.torch_compile import torch_compile_cat_ex
    from thunder import nvfuser_executor

    def foo(x):
        return x + x

    x = torch.randn(2, 2, device="cuda")
    thunderjit1 = ThunderCompileSpecification()
    str1 = thunderjit1.to_source("foo")

    thunderjit2 = ThunderCompileSpecification(
        executors=[torch_compile_cat_ex, nvfuser_executor],
        cache="constant values",
        langctx=None,
        record_history=False,
    )
    str2 = thunderjit2.to_source("foo")
    o1 = thunderjit1.compile(foo)(x)
    o2 = thunderjit2.compile(foo)(x)
    assert o1.equal(o2)
    assert str1 == "thunder.jit(foo)"
    assert (
        str2
        == "thunder.jit(foo, executors=[thunder.extend.get_executor('torchcompile_cat'),thunder.extend.get_executor('nvfuser')],cache='constant values',langctx=None,record_history=False,)"
    )


@requiresCUDA
@given(file_indices=st.lists(st.integers(min_value=0, max_value=15), min_size=2, max_size=2, unique=True))
@settings(max_examples=2, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_reports_repro(tmp_path, file_indices):
    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    def foo(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        torch._dynamo.graph_break()
        y = torch.sinc(x) + torch.cos(x)
        return y + 1

    results = fx_report(foo)(x)
    thunderjit = ThunderCompileSpecification()
    torchcompile = TorchCompileSpecification(dynamic=True)
    torcheager = TorchEagerSpecification()
    for idx, fx_graph_report in enumerate(results.fx_graph_reports):
        thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report)
        thunder_fx_graph_report.write_repro(tmp_path, thunderjit, check_consistency=True)
        for thunder_split_report in thunder_fx_graph_report.subgraph_reports:
            split_folder = tmp_path / str(idx)
            split_name = thunder_split_report.graph_name
            thunder_split_report.write_repro(split_folder, torchcompile, file_name=f"{split_name}_torchcompile.py")
            thunder_split_report.write_repro(split_folder, torcheager, file_name=f"{split_name}_eager.py")
            thunder_split_report.write_repro(
                split_folder, thunderjit, check_consistency=True, file_name=f"{split_name}_thunder.py"
            )
            thunder_split_report.run_repro(thunderjit, check_consistency=True)
            thunder_split_report.create_fusion_reports()
            for nvf in thunder_split_report.fusion_reports:
                nvf.write_nvfuser_repro(split_folder / "nvfusion")
                nvf.write_inductor_repro(split_folder / "nvfusion")
                nvf.run_repro(BoundSymbolNvfuserSpecification())
                nvf.run_repro(BoundSymbolTorchCompileSpecification())

    cmd = [sys.executable]
    py_files = list(tmp_path.rglob("*.py"))
    assert len(py_files) == 16

    selected_files = [py_files[i] for i in file_indices]
    for file in selected_files:
        run_script(file, cmd)


@requiresCUDA
@given(file_indices=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=1, unique=True))
@settings(max_examples=2, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_reports_benchmark(tmp_path, file_indices):
    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    def foo(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        torch._dynamo.graph_break()
        y = torch.sinc(x) + torch.cos(x)
        return y

    results = fx_report(foo)(x)
    thunderjit = ThunderCompileSpecification()
    torchcompile = TorchCompileSpecification()
    torcheager = TorchEagerSpecification()
    assert len(results.fx_graph_reports) == 2  # 2 Dynamo graphs
    fx_graph_report = results.fx_graph_reports[0]
    thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report)
    assert len(thunder_fx_graph_report.subgraph_reports) == 1  # exp
    thunder_split_report = thunder_fx_graph_report.subgraph_reports[0]
    split_name = thunder_split_report.graph_name
    thunder_split_report.write_benchmark(
        tmp_path,
        torchcompile,
        WallTimeWithMemoryUsage(min_run_time=0.01, max_run_time=4.0, threshold=0.08),
        file_name=f"{split_name}_torchcompile.py",
    )
    thunder_split_report.write_benchmark(tmp_path, torcheager, WallTime, file_name=f"{split_name}_eager.py")
    thunder_split_report.write_benchmark(tmp_path, thunderjit, WallTime, file_name=f"{split_name}_jit.py")
    thunder_split_report.create_fusion_reports()
    assert len(thunder_split_report.fusion_reports) == 2  # fwd, bwd
    nvf = thunder_split_report.fusion_reports[0]
    nvf.write_nvfuser_benchmark(tmp_path, WallTime)
    nvf.write_inductor_benchmark(tmp_path, WallTime)
    nvf.run_benchmark(BoundSymbolNvfuserSpecification(), WallTime(min_run_time=0.01, max_run_time=4.0, threshold=0.08))
    nvf.run_benchmark(BoundSymbolTorchCompileSpecification(), WallTime)

    cmd = [sys.executable]
    py_files = list(tmp_path.rglob("*.py"))
    assert len(py_files) == 5

    selected_files = [py_files[i] for i in file_indices]
    for file in selected_files:
        run_script(file, cmd)


@requiresCUDA
def test_TorchInductorSpecification(tmp_path):
    from thunder.dynamo.benchmark_utils import TorchInductorSpecification

    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    def foo(x):
        return torch.sinc(x) + torch.cos(x)

    results = fx_report(foo)(x)
    assert len(results.fx_graph_reports) == 1  # 1 Dynamo graphs
    fx_graph_report = results.fx_graph_reports[0]
    thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report)
    assert len(thunder_fx_graph_report.subgraph_reports) == 1  # cos
    thunder_split_report = thunder_fx_graph_report.subgraph_reports[0]

    torchinductor = TorchInductorSpecification()
    thunder_split_report.run_benchmark(torchinductor, WallTime)
    thunder_split_report.run_repro(torchinductor)
    thunder_split_report.write_benchmark(tmp_path, torchinductor, WallTime)
    thunder_split_report.write_repro(tmp_path, torchinductor, file_name="repro.py")

    cmd = [sys.executable]
    py_files = list(tmp_path.rglob("*.py"))
    assert len(py_files) == 2
    for file in py_files:
        run_script(file, cmd)


@requiresCUDA
def test_save_failing_repros(tmp_path):
    from thunder.dynamo.benchmark_utils import TorchEagerSpecification

    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    def foo(x):
        return torch.sin(x) + torch.cos(x)

    # Tests for dynamo fx graphreports
    results = fx_report(foo)(x)
    with patch("thunder.dynamo.report.FXGraphReport.run_repro", side_effect=Exception("run_Repro raises exception")):
        save_failing_repros(results.fx_graph_reports, TorchCompileSpecification(), tmp_path)
    assert os.path.exists(tmp_path / "graph0.py")

    # Tests for thunder split reports
    thunder_fxgraph_reports = get_thunder_fxgraph_reports(foo)(x)
    assert len(thunder_fxgraph_reports) == 1
    with patch("thunder.dynamo.report.FXGraphReport.run_repro", side_effect=Exception("run_Repro raises exception")):
        save_failing_repros(thunder_fxgraph_reports[0].subgraph_reports, ThunderCompileSpecification(), tmp_path)
    assert os.path.exists(tmp_path / "graph0_thunder_0.py")

    # Tests for check_consistency
    def wrapped_fn(x):
        return foo(x) + 1

    class _BadCompileSpecification(TorchEagerSpecification):
        def compile(self, fn, **kwargs):
            return wrapped_fn

    results = fx_report(foo)(x)
    save_failing_repros(
        results.fx_graph_reports, _BadCompileSpecification(), tmp_path / "consistency", check_consistency=False
    )
    assert not os.path.exists(tmp_path / "consistency" / "graph0.py")

    save_failing_repros(
        results.fx_graph_reports, _BadCompileSpecification(), tmp_path / "consistency", check_consistency=True
    )
    assert os.path.exists(tmp_path / "consistency" / "graph0.py")


@requiresCUDA
def test_autograd_function_fx_report(tmp_path):
    class Sin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sin(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * torch.cos(x) * 100

    def func(x):
        return torch.cos(x) + Sin.apply(x)

    x = torch.ones(2, 2, device="cuda", requires_grad=True)

    if LooseVersion(torch.__version__) < LooseVersion("2.6.0"):
        with pytest.raises(
            RuntimeError,
            match="The Reporting Tool for Torch higher-order operators is supported only in PyTorch version 2.6 or later.",
        ):
            results = fx_report(func)(x)
    else:
        results = fx_report(func)(x)
        assert len(results.fx_graph_reports) == 1  # 1 Dynamo graph
        fx_graph_report = results.fx_graph_reports[0]
        thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report)
        assert len(thunder_fx_graph_report.subgraph_reports) == 1  # no split
        thunder_split_report = thunder_fx_graph_report.subgraph_reports[0]

        thunder_split_report.run_repro(ThunderCompileSpecification())
        thunder_split_report.run_benchmark(ThunderCompileSpecification(), WallTime)
        thunder_split_report.write_benchmark(tmp_path, ThunderCompileSpecification(), WallTime)
        thunder_split_report.write_repro(tmp_path, ThunderCompileSpecification(), file_name="repro.py")

        cmd = [sys.executable]
        py_files = list(tmp_path.rglob("*.py"))
        assert len(py_files) == 2
        for file in py_files:
            run_script(file, cmd)


@requiresCUDA
def test_aot_optimize():
    from thunder.dynamo import thunder_profile, thunder_optimize

    def foo(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
        x = x.exp()
        torch._dynamo.graph_break()
        y = torch.sinc(x) + torch.cos(x)
        return y + 1

    x = torch.ones(2, 2, device="cuda", requires_grad=True)
    y = torch.ones(4, 4, device="cuda", requires_grad=True)
    z = torch.ones(8, 4, device="cuda", requires_grad=True)
    pfoo = thunder_profile(foo)
    pfoo(x)
    # 2 graphs because of the graph break
    assert len(pfoo._tao.id_to_profile_stats) == 2
    # 2 more dynamic graphs
    pfoo(y)
    assert len(pfoo._tao.id_to_profile_stats) == 4
    # uses the dynamic graphs instead of the static ones
    pfoo(x)
    pfoo(z)
    pfoo(x)
    pfoo(y)

    stats = list(pfoo._tao.id_to_profile_stats.values())
    # the last two are the dymamic graphs' stats
    # input x,y are called 2 times, z is called 1 time for the dynamic graphs
    input_meta_dict = stats[-1].input_meta_to_called_times
    for k, v in input_meta_dict.items():
        if k[-1].shape == (2, 2):
            assert v == 2
        elif k[-1].shape == (4, 4):
            assert v == 2
        elif k[-1].shape == (8, 4):
            assert v == 1
        else:
            assert False

    # filters the dynamic graphs, only the static ones are optimized
    optfoo = thunder_optimize(pfoo)
    # fallbacks to eager for the dynamic graphs
    optfoo(x)

    with pytest.raises(AssertionError, match="No longer profiling"):
        pfoo(x)


def test_spliter_bwd():
    def fn(x, idx, val):
        x = x.clone()
        x[idx] = val
        return x

    x = torch.randn(1, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.rand(1, 4, 5) > 0.5
    nz = torch.count_nonzero(idx)
    val = torch.randn(nz, dtype=torch.bfloat16, requires_grad=True)

    cfn = thunderfx(fn)
    cfn(x, idx, val)
    reason = cfn._backend.subgraph_infos[0].split_reasons
    assert len(reason) == 1
    assert "Failed while running meta for node with name: setitem" in reason[0].info
    assert "boolean advanced indexing" in reason[0].exception


@pytest.mark.skipif(
    IS_WINDOWS,
    reason="torch.compile Windows support is still WIP - https://github.com/pytorch/pytorch/issues/122094",
)
def test_get_proxy_inputs_from_node_symtype_hint():
    def fn(x, idx):
        return torch.select(x, 0, idx)

    x = torch.randn(4, 4)
    idx = 0
    cfn = thunderfx(fn, dynamic=True)
    cfn(x, idx)

    assert cfn._backend.subgraph_infos[0].split_reasons == []


@requiresCUDA
def test_spliter_einops():
    einops = pytest.importorskip("einops")

    def f(input, expr):
        return einops.rearrange(input, expr)

    fc = thunderfx(f)
    input = torch.randn(2, 3, 4, 5, device="cuda")
    out = fc(input, "b c h w -> b (c h w)")
    expected_out = f(input, "b c h w -> b (c h w)")

    assert len(fc._backend.subgraph_infos[0].split_reasons) == 0
    torch.testing.assert_close(out, expected_out)


@requiresCUDA
def test_thunderfx_with_intermediate_output_marked_as_non_differentiable():
    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fc = torch.nn.Linear(2, 2)
            self.register_buffer("buf", None, False)

        def forward(self, x):
            if self.buf is None:
                self.buf = torch.ones(1, 2, device=x.device)
            return self.fc(x) + self.buf

    with torch.device("cuda"):
        org_m = Model()
        thunder_m = copy.deepcopy(org_m)
        m = thunderfx(thunder_m)

        x = torch.randn(2, 2)

        # First iteration
        expected_output = org_m(x).sum()
        actual_output = m(x).sum()

        torch.testing.assert_close(actual_output, expected_output)
        actual_output.backward()
        expected_output.backward()
        torch.testing.assert_close(org_m.fc.weight.grad, thunder_m.fc.weight.grad)

        # Second iteration
        x = torch.randn(2, 2)
        expected_output = org_m(x).sum()
        actual_output = m(x).sum()

        torch.testing.assert_close(actual_output, expected_output)
        actual_output.backward()
        expected_output.backward()
        torch.testing.assert_close(org_m.fc.weight.grad, thunder_m.fc.weight.grad)


def test_thunderfx_node_with_no_example_value():
    def test_fn(x):
        y = x + 10
        z = y.tolist()[0]
        return z + 2

    x = torch.tensor([1, 2, 3, 4, 5])
    actual = thunderfx(test_fn)(x)
    expected = test_fn(x)
    torch.testing.assert_close(actual, expected)
