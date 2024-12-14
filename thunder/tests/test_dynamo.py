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
    version_between,
)
from thunder.tests.make_tensor import make_tensor


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

    cfunc = thunderfx(func, dynamic=dynamic)
    actual = cfunc(x)

    backend = cfunc._backend
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
        pytest.mark.skipif(
            version_between(torch.__version__, min_ver="2.6.0dev0", max_ver="2.6.0a99"),
            reason="https://github.com/Lightning-AI/lightning-thunder/issues/1471",
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
            msg=lambda s: f"{i+1}-iter {s}",
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
        assert len(subgraph_info.split_reasons) > 1  # Verify there were splits in the subgraph.
        assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
        assert any("has been manually disabled" in split_reason.info for split_reason in subgraph_info.split_reasons)

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
def test_ThunderCompilerGraphBenchmarking_LlamaMLPBenchmark(benchmark):
    import thunder

    backend = ThunderCompilerGraphBenchmarking(
        benchmark, executors={"thunder": thunder.jit, "inductor": torch.compile, "eager": None}
    )
    from thunder.benchmarks import LlamaMLPBenchmark, Benchmark

    bench: Benchmark = LlamaMLPBenchmark(
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
            return y + 1
        else:
            y = y.exp()
            x = torch.sinc(y) + torch.cos(x)
            return x - 1

    import thunder

    backend = ThunderCompilerGraphBenchmarking(
        benchmark, executors={"thunder": thunder.jit, "inductor": torch.compile, "eager": None}
    )
    compiled = torch.compile(backend=backend)(f)
    x = torch.ones(2, requires_grad=True).cuda()
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

    from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform
    from thunder import nvfuser_executor
    from thunder.transforms.cudagraph import CUDAGraphTransform

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
                NvtxProfileTransform(),
                CUDAGraphTransform(),
            ],
            executors=[nvfuser_executor],
            cache="constant values",
            langctx=None,
            record_history=False,
        )
    else:
        cfunc = thunderfx(func, executors=None)
    # Test non-contiguous input tensor
    x = make_tensor((4, 4), low=3, high=10, dtype=torch.int64, device=device, noncontiguous=True)

    out = cfunc(x)
    cfunc._backend.save_reproducer_to_folder(tmp_path, use_pytest_benchmark=use_pytest_benchmark)

    s1 = f"{tmp_path}/graph0_thunder_0.py"
    s2 = f"{tmp_path}/graph1_thunder_0.py"
    assert os.path.exists(s1)
    assert os.path.exists(s2)
    cmd = [sys.executable]
    if use_pytest_benchmark:
        cmd = cmd + ["-m", "pytest"]
    cmd1 = cmd + [s1]
    cmd2 = cmd + [s2]
    result1 = subprocess.run(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    result2 = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

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

    s1 = f"{tmp_path}/graph0_thunder_0.py"
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
    original_split_gm = subgraph_info.original_split_graph_module
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
def test_dynamo_reproducer_split(executor, device: str, dtype: dtypes.dtype, use_pytest_benchmark, tmp_path):
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

    def check(file_name, cmd):
        assert os.path.exists(file_name)
        cmd = cmd + [file_name]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, f"Reproducer {file_name} failed: {result}"

    s1 = f"{tmp_path}/graph0_thunder_0.py"
    s2 = f"{tmp_path}/graph0_thunder_2.py"
    s3 = f"{tmp_path}/graph0_thunder_4.py"
    cmd = [sys.executable]
    if use_pytest_benchmark:
        cmd = cmd + ["-m", "pytest"]
    for fname in [s1, s2, s3]:
        check(fname, cmd)


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
