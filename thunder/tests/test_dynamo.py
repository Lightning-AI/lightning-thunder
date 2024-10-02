import torch.fx
from thunder.tests.framework import instantiate, NOTHING, DynamoThunderExecutor, IS_WINDOWS
from thunder import dtypes
from thunder.dynamo import ThunderCompiler
from thunder import last_traces

import torch
import pytest


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

    backend = ThunderCompiler()

    def func(x):
        # torch.sinc has automatic fallback registered,
        # so that operation will be given to inductor.
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
    targets = (node.target for node in backend.subgraph_infos[0].split_graph_module.graph.nodes)
    assert any(target.startswith("thunder_") for target in targets)  # Verify that the submodules have name `thunder_*`
    assert any(
        target.startswith("inductor_") for target in targets
    )  # Verify that the submodules have name `inductor_*`


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
    x = torch.ones(2, device=device, dtype=dtype, requires_grad=True)

    class Sin(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.sin(x)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * torch.cos(x)

    def func(x):
        y = torch.cos(x) + Sin.apply(x)
        return torch.matmul(x, y)

    expected = torch.compile(func, dynamic=dynamic)(x)

    backend = ThunderCompiler()
    cfunc = torch.compile(func, backend=backend, dynamic=dynamic)
    actual = cfunc(x)

    targets = (node.target for node in backend.subgraph_infos[0].split_graph_module.graph.nodes)
    assert any(target.startswith("thunder_") for target in targets)
    assert any(target.startswith("inductor_") for target in targets)

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
        backend = ThunderCompiler()
        x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

        @torch.compile(backend=backend)
        def func(x):
            x = torch.sin(x)
            return x + 2

        out = func(x)

        # out should have grad_fn and its name should be ThunderFunctionBackward
        assert out.grad_fn is not None
        assert out.grad_fn.name() == "ThunderFunctionBackward"

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
    backend = ThunderCompiler()
    x = torch.ones(2, dtype=dtype, device=device, requires_grad=True)

    if not cat_kwarg:

        @torch.compile(backend=backend)
        def func(x):
            x = torch.cat([x, x])
            return x + 2

    else:

        @torch.compile(backend=backend)
        def func(x):
            x = torch.cat(tensors=[x, x])
            return x + 2

    out = func(x)

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

    backend = ThunderCompiler()

    @torch.compile(backend=backend)
    def func(x):
        y = x.float()
        return y.sin()

    x = torch.randn(3, 3, device=device, dtype=dtype)
    o = func(x)

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
    backend = ThunderCompiler()

    def func(x):
        y = x[torch.where(x > 0.5)]  # This will lead to graph-break
        y = torch.where(y > 1, y, 0)
        return y.sin()

    x = torch.randn(3, 3, device=device, dtype=dtype, requires_grad=True)
    actual = torch.compile(func, backend=backend)(x)
    expected = torch.compile(func, backend="eager")(x)

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
