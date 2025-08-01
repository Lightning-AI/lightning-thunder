import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.tests.framework import requiresCUDA

transformer_engine_module = pytest.importorskip(
    "transformer_engine", reason="transformer_engine was not found, skipping the tests."
)

from thunder.executors.transformer_engine_v2ex import transformer_engine_v2_ex, TransformerEngineTransformV2
from transformer_engine.common import recipe
import transformer_engine.pytorch as te

# FP8 is supported on compute arch 8.9 onwards.
# MXFP8 is supported on compute arch 10.0 onwards.
# Skip the tests if current hardware is not supported.
is_fp8_supported, msg_fp8 = te.fp8.check_fp8_support()
is_mxfp8_supported, msg_mxfp8 = te.fp8.check_mxfp8_support()
if not is_fp8_supported:
    pytest.skip(msg_fp8, allow_module_level=True)

hybrid_fp8_delayed_scaling_recipe = recipe.DelayedScaling()
mxfp8_e4m3_recipe = recipe.MXFP8BlockScaling()

# `None` is used to test the default recipe.
recipes = (None, hybrid_fp8_delayed_scaling_recipe, mxfp8_e4m3_recipe)
recipe_ids = ("default", "delayed_scaling", "mxfp8_e4m3")


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
def test_te_linear_forward_backward(fp8_recipe: recipe.Recipe):
    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    # Test Description:
    # Verify that `torch.nn.functional.linear` is replaced with `te_linear_*`
    # and the output as well as the gradients match for thunder compiled code.
    dtype = torch.bfloat16
    device = "cuda"

    # TE inputs (3D input)
    x_te = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 2048, params_dtype=dtype)

    # thunder inputs
    x = x_te.detach().clone()
    x.requires_grad_(True)
    w1 = te_linear1.weight.detach().clone()
    w1.requires_grad_(True)
    w2 = te_linear2.weight.detach().clone()
    w2.requires_grad_(True)

    def fn(x, w1, w2):
        o = torch.nn.functional.linear(x, w1)
        return torch.nn.functional.linear(o + x, w2)

    cfn = thunder.jit(fn, executors=[transformer_engine_v2_ex], transforms=[TransformerEngineTransformV2()])

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        thunder_result = cfn(x, w1, w2)

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        inter_result = te_linear1(x_te)
        te_result = te_linear2(inter_result + x_te)

    # Verifies the result is close to TE
    assert_close(thunder_result, te_result)

    grad_output = torch.randn_like(te_result)
    te_result.backward(grad_output)
    thunder_result.backward(grad_output)

    assert_close(x.grad, x_te.grad)
    assert_close(w1.grad, te_linear1.weight.grad)
    assert_close(w2.grad, te_linear2.weight.grad)

    # Verifies te_linear was called
    forward_trace = thunder.last_traces(cfn)
    backward_trace = thunder.last_backward_traces(cfn)

    assert any(bsym.sym.name.startswith("te_functional_linear") for bsym in forward_trace[-1].bound_symbols)
    assert any(bsym.sym.name.startswith("te_functional_linear_bwd") for bsym in backward_trace[-1].bound_symbols)
    # and only two
    assert 2 == len(
        tuple(filter(lambda bsym: bsym.sym.name.startswith("te_functional_linear"), forward_trace[-1].bound_symbols))
    )


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
def test_te_linear_forward_backward_multiple_iteration(fp8_recipe: recipe.Recipe):
    if not fp8_recipe:
        pytest.skip(
            "When recipe is None a new recipe is created for each iteration. This makes the results not numerically comparable."
        )

    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    # Test Description:
    # In this test, we verify whether a model using TransformerEngine Linear
    # and transformer_engine executor converge to same state.
    # Since, the FP8 operations are stateful, we want to verify that
    # our output matches over multiple iterations (where state handling comes into picture)
    dtype = torch.bfloat16
    device = "cuda"
    # Running more iterations leads to `nan` for both eager and thunder
    # with BlockScaling.
    # Potentially because we are training on dummy data and task
    iterations = 6

    # TE inputs
    input_shape = (768, 4096)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 2048, params_dtype=dtype)

    def clone_params(*params):
        return tuple(param.detach().clone() for param in params)

    # Parameters for thunder to optimize
    w1, w2, b1, b2 = clone_params(te_linear1.weight, te_linear2.weight, te_linear1.bias, te_linear2.bias)

    target_value = torch.randint(42, (768,), dtype=torch.int64, device=device)

    inputs = tuple(torch.rand(*input_shape, device=device, dtype=dtype) for _ in range(iterations))

    def train_model(model, optimizer):
        # Run for `iterations`.
        for iter_n in range(iterations):
            x = inputs[iter_n]
            result = model(x)
            loss = torch.nn.functional.cross_entropy(result, target_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def te_model(x):
        # Enable autocasting for the forward pass
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return te_linear2(te_linear1(x))

    te_sgd_optimizer = torch.optim.SGD(list(te_linear1.parameters()) + list(te_linear2.parameters()))

    train_model(te_model, te_sgd_optimizer)

    def fn(x, w1, w2, b1, b2):
        o = torch.nn.functional.linear(x, w1, b1)
        return torch.nn.functional.linear(o, w2, b2)

    cfn = thunder.jit(fn, executors=[transformer_engine_v2_ex], transforms=[TransformerEngineTransformV2()])

    # Enable grad on thunder params.
    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return cfn(x, w1, w2, b1, b2)

    train_model(thunder_model, thunder_sgd_optimizer)

    # Verify that the weights and biases converge to same value after few iterations.
    assert_close(w1, te_linear1.weight)
    assert_close(w2, te_linear2.weight)
    assert_close(b1, te_linear1.bias)
    assert_close(b2, te_linear2.bias)


@requiresCUDA
def test_te_linear_forward_backward_multiple_iteration_multiple_recipes():
    # This test is used to verify parity with TE library when it comes to changing recipes during runtime, regardless if that is the intended use or not.

    recipes = [recipe.DelayedScaling()]
    supports_mxfp8, _ = te.fp8.check_mxfp8_support()

    if supports_mxfp8:
        recipes += [recipe.MXFP8BlockScaling()]

    if len(recipes) < 2:
        pytest.skip("platform does not support two different recipes")

    dtype = torch.bfloat16
    device = "cuda"
    # Running more iterations leads to `nan` for both eager and thunder
    # with BlockScaling.
    # Potentially because we are training on dummy data and task
    iterations = 6

    # TE inputs
    input_shape = (768, 4096)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 2048, params_dtype=dtype)

    def clone_params(*params):
        return tuple(param.detach().clone() for param in params)

    # Parameters for thunder to optimize
    w1, w2, b1, b2 = clone_params(te_linear1.weight, te_linear2.weight, te_linear1.bias, te_linear2.bias)

    target_value = torch.randint(42, (768,), dtype=torch.int64, device=device)

    inputs = tuple(torch.rand(*input_shape, device=device, dtype=dtype) for _ in range(iterations))

    def train_model(model, optimizer):
        # Run for `iterations`.
        for iter_n in range(iterations):
            te_recipe = recipes[iter_n % 2]
            x = inputs[iter_n]
            result = model(x, te_recipe)
            loss = torch.nn.functional.cross_entropy(result, target_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def te_model(x, fp8_recipe):
        # Enable autocasting for the forward pass
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return te_linear2(te_linear1(x))

    te_sgd_optimizer = torch.optim.SGD(list(te_linear1.parameters()) + list(te_linear2.parameters()))

    train_model(te_model, te_sgd_optimizer)

    def fn(x, w1, w2, b1, b2):
        o = torch.nn.functional.linear(x, w1, b1)
        return torch.nn.functional.linear(o, w2, b2)

    cfn = thunder.jit(fn, executors=[transformer_engine_v2_ex], transforms=[TransformerEngineTransformV2()])

    # Enable grad on thunder params.
    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x, fp8_recipe):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return cfn(x, w1, w2, b1, b2)

    train_model(thunder_model, thunder_sgd_optimizer)

    # Verify that the weights and biases converge to same value after few iterations.
    assert_close(w1, te_linear1.weight)
    assert_close(w2, te_linear2.weight)
    assert_close(b1, te_linear1.bias)
    assert_close(b2, te_linear2.bias)


@requiresCUDA
def test_te_linear_invalid_inputs():
    def assert_not_transformed(x, w):
        def fn(x, w):
            return torch.nn.functional.linear(x, w)

        cfn = thunder.jit(fn, executors=[transformer_engine_v2_ex], transforms=[TransformerEngineTransformV2()])
        cfn(x, w)
        trace = thunder.last_traces(cfn)[-1]
        assert not any(bsym.sym.name.startswith("te_functional_linear") for bsym in trace.bound_symbols)

    # CPU is not supported.
    device = "cpu"
    x = torch.randn(16, 16, device=device)
    w = torch.randn(16, 16, device=device)
    assert_not_transformed(x, w)

    # Input shapes are not supported by TE.
    device = "cuda"
    x = torch.randn(16, 4, device=device)
    w = torch.randn(16, 4, device=device)
    assert_not_transformed(x, w)


@requiresCUDA
def test_te_with_autocast():
    from thunder.transforms.autocast import autocast

    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(64, 64, device=device, requires_grad=True)
    w = torch.randn(64, 64, device=device, requires_grad=True)

    cfunc = thunder.jit(
        autocast(foo, dtype=thunder.dtypes.bfloat16),
        executors=[transformer_engine_v2_ex],
        transforms=[TransformerEngineTransformV2()],
        disable_preprocessing=True,
    )
    cfunc(x, w)

    fwd_traces = thunder.last_traces(cfunc)
    # Verify that we have replaced `prims.linear` with `te_linear`
    assert any(bsym.sym.name.startswith("te_functional_linear") for bsym in fwd_traces[-1].bound_symbols)


# NOTE: strict=False as it passes on Blackwell.
@pytest.mark.xfail(strict=False, raises=(RuntimeError, TypeError), reason="Retain graph is not supported by TE")
@requiresCUDA
def test_te_with_retain_graph():
    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(16, 16, device=device, requires_grad=True)
    w = torch.randn(16, 16, device=device, requires_grad=True)

    cfunc = thunder.jit(
        foo,
        executors=[transformer_engine_v2_ex],
        transforms=[TransformerEngineTransformV2()],
    )
    out = cfunc(x, w)

    # Retain graph is not supported correctly by TE
    # https://github.com/NVIDIA/TransformerEngine/issues/990
    out.backward(torch.randn_like(out), retain_graph=True)
    out.backward(torch.randn_like(out))


@requiresCUDA
def test_te_trace_metadata_propagation():
    # This test is to verify that we correctly propagate metadata `_include_te_fp8_autocast` on
    # trace using `from_trace`. `_include_te_fp8_autocast` is used to enable wrapping forward trace with `fp8_autocast`.
    def foo(x, w):
        return torch.nn.functional.linear(x, w)

    device = "cuda"
    x = torch.randn(64, 64, device=device, requires_grad=True)
    w = torch.randn(64, 64, device=device, requires_grad=True)

    class MyNoopTransform(thunder.core.transforms.Transform):
        def transform_trace_post_optimization(self, computation_trace, **kwargs):
            new_trace = thunder.core.trace.from_trace(computation_trace)
            new_trace.bound_symbols = computation_trace.bound_symbols
            return new_trace

    cfunc = thunder.jit(
        foo,
        executors=[transformer_engine_v2_ex],
        transforms=[
            TransformerEngineTransformV2(),
            MyNoopTransform(),
        ],
    )
    out = cfunc(x, w)

    fwd_traces = thunder.last_traces(cfunc)

    # Verify that we have `te_linear` in the trace.
    assert any(bsym.sym.name.startswith("te_functional_linear") for bsym in fwd_traces[-1].bound_symbols)


def test_te_grad_computation_with_intermediate():
    # Test for issue - https://github.com/Lightning-AI/lightning-thunder/issues/1966
    def fn(x, w):
        # Due to autocast, trace becomes something like this
        # t4 = prims.convert_element_type(x, dtypes.bfloat16)  # t4: "cuda:0 bf16[32, 32]"
        # t5 = prims.convert_element_type(w, dtypes.bfloat16)  # t5: "cuda:0 bf16[32, 32]"
        # t6 = prims.linear(t4, t5, None)  # t6: "cuda:0 bf16[32, 32]"
        with torch.autocast("cuda", torch.bfloat16):
            return torch.nn.functional.linear(x, w)

    with torch.device("cuda"):
        x = torch.randn(32, 32, requires_grad=True)
        w = torch.randn(32, 32, requires_grad=True)

        tfn = thunder.jit(fn, executors=[transformer_engine_v2_ex], transforms=[TransformerEngineTransformV2()])

        o = tfn(x, w)
        o.sum().backward()

        assert w.grad is not None


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
def test_te_trace_correctness(fp8_recipe: recipe.Recipe):
    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(32, 32, device=device, requires_grad=True)
    w = torch.randn(32, 32, device=device, requires_grad=True)

    cfunc = thunder.jit(
        foo,
        executors=[transformer_engine_v2_ex],
        transforms=[TransformerEngineTransformV2()],
    )

    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        out = cfunc(x, w)

    fwd_trace = thunder.last_traces(cfunc)[-1]
    fwd_trace_pyctx = fwd_trace.python_ctx()
    from thunder.core.utils import OrderedSet

    fwd_trace_names = OrderedSet(map(lambda x: x.sym.name, fwd_trace.bound_symbols))
    fwd_te_trace_op_names = list(
        reversed(
            (
                "get_te_fp8_recipe",
                "get_te_fp8_state",
                "get_te_fp8_quantizers",
                "te_functional_linear_fwd",
                "te_fp8_amax_and_scale_update",
            )
        )
    )

    for name in fwd_trace_names:
        if fwd_te_trace_op_names and fwd_te_trace_op_names[-1] in name:
            # Check that the state is in the trace context
            assert name in fwd_trace_pyctx.keys()
            fwd_te_trace_op_names.pop()

    # If all the elements appear in order in the trace then the list is empty
    assert len(fwd_te_trace_op_names) == 0

    # Same check but now for the backward trace
    bwd_trace = thunder.last_backward_traces(cfunc)[-1]
    bwd_trace_pyctx = bwd_trace.python_ctx()
    bwd_trace_names = OrderedSet(map(lambda x: x.sym.name, bwd_trace.bound_symbols))
    # No get_te_fp8_recipe in this list beacuse the transform made sure it's carried over from the forward
    bwd_te_trace_op_names = list(
        reversed(
            ("get_te_fp8_state", "get_te_fp8_quantizers", "te_functional_linear_bwd", "te_fp8_amax_and_scale_update")
        )
    )

    for name in bwd_trace_names:
        if bwd_te_trace_op_names and bwd_te_trace_op_names[-1] in name:
            # Check that the state is in the trace context
            assert name in bwd_trace_pyctx.keys()
            bwd_te_trace_op_names.pop()

    # If all the elements appear in order in the trace then the list is empty
    assert len(bwd_te_trace_op_names) == 0
