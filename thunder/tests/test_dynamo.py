import torch.fx
from thunder.tests.framework import instantiate, NOTHING, DynamoThunderExecutor
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
    assert len(backend.thunder_to_gm) == 2
    thunder_func, gm = list(backend.thunder_to_gm.items())[0]
    assert isinstance(gm, torch.fx.GraphModule)

    # This shouldn't be empty
    assert last_traces(thunder_func)
