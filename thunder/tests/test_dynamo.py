import torch.fx
from thunder.tests.framework import instantiate, NOTHING, DynamoThunderExecutor
from thunder import dtypes
from thunder.dynamo import ThunderCompiler
from thunder import last_traces

import torch
import pytest


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),),
)
def test_basic(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    backend = ThunderCompiler()
    x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

    @torch.compile(backend=backend, dynamic=dynamic)
    def func(x):
        x = torch.sin(x)
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1

    out = func(x)

    # out should have grad_fn and its name should be ThunderFunctionBackward
    assert out.grad_fn is not None
    assert out.grad_fn.name() == "ThunderFunctionBackward"

    # We record the GraphModules that was compiled by ThunderCompiler
    assert len(backend.subgraph_infos) == 2  # 2 due to data-dependent flow

    for subgraph_info in backend.subgraph_infos:
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert len(subgraph_info.thunder_compiled_fns)  # There was atleast one function compiled with thunder.
        for thunder_fn in subgraph_info.thunder_compiled_fns:
            assert last_traces(thunder_fn)  # Verify that we can fetch last_traces


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),),
)
def test_basic_splitter(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.ones(2, 2, device=device, dtype=dtype, requires_grad=True)

    backend = ThunderCompiler()

    def func(x):
        # torch.sinc has automatic fallback registered.
        x = x.exp()
        y = torch.sinc(x) + torch.cos(x)
        return y + 1

    cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
    expected = torch.compile(func, dynamic=False)(x)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    assert len(backend.subgraph_infos) == 1
    assert len(backend.subgraph_infos[0].submodule_to_compiled_functions) > 1  # Verify that the subgraph was split.
    assert any(
        "automatic torch fallback" in split_reason.info for split_reason in backend.subgraph_infos[0].split_reasons
    )


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),),
)
def test_splitter_unsupported_ctx(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.rand(2, 2, device=device, dtype=dtype, requires_grad=True)

    backend = ThunderCompiler()

    def func(x):
        x = x + 2
        with torch.autocast("cpu"):
            y = torch.log(x)
            return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=False)(x)

    cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    assert len(backend.subgraph_infos) == 1
    assert len(backend.subgraph_infos[0].submodule_to_compiled_functions) > 1  # Verify that the subgraph was split.
    assert any(
        "didn't have any mapping in thunder" in split_reason.info
        for split_reason in backend.subgraph_infos[0].split_reasons
    )


@instantiate(
    dtypes=NOTHING,
    executors=[DynamoThunderExecutor],
    decorators=(pytest.mark.parametrize("dynamic", (True, False, None), ids=("dynamic", "static", "auto")),),
)
def test_splitter_unsupported_ctx_with_graph_break(executor, device: str, dtype: dtypes.dtype, dynamic: bool | None):
    x = torch.rand(2, 2, device=device, dtype=dtype, requires_grad=True)

    backend = ThunderCompiler()

    def func(x):
        x = x + 2
        with torch.autocast("cpu"):
            y = torch.sin(x)
            torch._dynamo.graph_break()
            return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=False)(x)
    cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
    actual = cfunc(x)

    g = torch.rand_like(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, g)
    expected_grad = torch.autograd.grad(expected, x, g)
    torch.testing.assert_close(actual_grad, expected_grad)

    # 2 subgraphs due to graph-break
    assert len(backend.subgraph_infos) == 2

    for subgraph_info in backend.subgraph_infos:
        # Verify that for each subgraph we had split due to `autocast` being enabled.
        assert any(
            "didn't have any mapping in thunder" in split_reason.info for split_reason in subgraph_info.split_reasons
        )
