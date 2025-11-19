from functools import partial

import pytest
import torch
from looseversion import LooseVersion

import thunder
from thunder.tests.framework import requiresCUDA

# NOTE: On SM120/121, TE defaults to using Float8BlockScaling
#       which is currently unsupported in thunder, we skip the tests for these SM architectures.
from thunder.tests.utils import skip_on_sm120_and_sm121, is_sm120_orsm121

transformer_engine_module = pytest.importorskip(
    "transformer_engine", reason="transformer_engine was not found, skipping the tests."
)

from thunder.executors.transformer_engineex import transformer_engine_ex, TransformerEngineTransform

import transformer_engine
from transformer_engine.common import recipe
import transformer_engine.pytorch as te

# FP8 is supported on compute arch 8.9 onwards.
# MXFP8 is supported on compute arch 10.0 onwards.
# NVFP4 is available from TE 2.8 onwards and supported only for arch 10.0 onwards.
# Skip the tests if current hardware is not supported.
is_fp8_supported, msg_fp8 = te.fp8.check_fp8_support()
is_mxfp8_supported, msg_mxfp8 = te.fp8.check_mxfp8_support()
is_nvfp4_supported = False
is_nvfp4_available = LooseVersion(transformer_engine.__version__) >= LooseVersion("2.8")
if is_nvfp4_available:
    is_nvfp4_supported, msg_nvfp4 = te.fp8.check_nvfp4_support()

# check_fp8_support is a subset of check_nvfp4_support, therefore even NVFP4 tests can be skipped if fp8 is not available
# https://github.com/NVIDIA/TransformerEngine/blob/dfe5b7dfc2288afc5d2f247709b1e0328af331e4/transformer_engine/pytorch/fp8.py#L58
if not is_fp8_supported:
    pytest.skip(msg_fp8, allow_module_level=True)

hybrid_fp8_delayed_scaling_recipe = recipe.DelayedScaling()
mxfp8_e4m3_recipe = recipe.MXFP8BlockScaling()

# `None` is used to test the default recipe.
recipes = [None, hybrid_fp8_delayed_scaling_recipe, mxfp8_e4m3_recipe]
recipe_ids = ["default", "delayed_scaling", "mxfp8_e4m3"]

if is_nvfp4_available:
    # Disable randomness otherwise the results will differ.
    nvfp4_e2m1_recipe = recipe.NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)

    recipes += [nvfp4_e2m1_recipe]
    recipe_ids += ["nvfp4_e2m1"]


# Returns the estimated numerical error for a given TE recipe based on:
# https://github.com/NVIDIA/TransformerEngine/blob/7e593c3be96b3eebc384da1a2ab307727065c9ab/tests/pytorch/utils.py#L68-L118
def te_assert_close(actual, expected, te_recipe=None, **kwargs):
    tolerances = {}

    if te_recipe is not None:
        te_format = getattr(te_recipe, "fp8_format", None) or getattr(te_recipe, "fp4_format", None)
        if is_nvfp4_available and te_format == recipe.Format.E2M1:
            # Tolerances for NVFP4 Float4 E2M1
            tolerances = dict(rtol=0.25, atol=0.125)
        elif te_format in (recipe.Format.HYBRID, recipe.Format.E5M2):
            # In the case where the recipe used hybrid or E5M2 formats, use the most relaxed one.
            tolerances = dict(rtol=0.25, atol=0.125)
        else:
            # Tolerances for FP8 recipes with only Float8 E4M3
            tolerances = dict(rtol=0.125, atol=0.0675)

    kwargs.update(tolerances)
    return torch.testing.assert_close(actual, expected, **kwargs)


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_te_linear_forward_backward(fp8_recipe: recipe.Recipe):
    if fp8_recipe and fp8_recipe.mxfp8() and not is_mxfp8_supported:
        pytest.skip(msg_mxfp8)
    elif fp8_recipe and is_nvfp4_available and fp8_recipe.nvfp4() and not is_nvfp4_supported:
        pytest.skip(msg_nvfp4)

    if is_sm120_orsm121 and fp8_recipe is None:
        pytest.skip("On SM120/121, default recipe is Float8BlockScaling which is not supported")

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

    cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        thunder_result = cfn(x, w1, w2)

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        inter_result = te_linear1(x_te)
        te_result = te_linear2(inter_result + x_te)

    # Verifies the result is close to TE
    te_assert_close(thunder_result, te_result, te_recipe=fp8_recipe)

    grad_output = torch.randn_like(te_result)
    te_result.backward(grad_output)
    thunder_result.backward(grad_output)

    te_assert_close(x.grad, x_te.grad, te_recipe=fp8_recipe)
    te_assert_close(w1.grad, te_linear1.weight.grad, te_recipe=fp8_recipe)
    te_assert_close(w2.grad, te_linear2.weight.grad, te_recipe=fp8_recipe)

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
@skip_on_sm120_and_sm121
def test_te_linear_forward_backward_multiple_iteration(fp8_recipe: recipe.Recipe):
    if fp8_recipe is None:
        pytest.skip(
            "When recipe is None a new recipe is created for each iteration. This makes the results not numerically comparable."
        )

    if fp8_recipe.mxfp8() and not is_mxfp8_supported:
        pytest.skip(msg_mxfp8)

    if fp8_recipe and is_nvfp4_available and fp8_recipe.nvfp4() and not is_nvfp4_supported:
        pytest.skip(msg_nvfp4)

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

    cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    # Enable grad on thunder params.
    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return cfn(x, w1, w2, b1, b2)

    train_model(thunder_model, thunder_sgd_optimizer)

    # Verify that the weights and biases converge to same value after few iterations.
    te_assert_close(w1, te_linear1.weight, te_recipe=fp8_recipe)
    te_assert_close(w2, te_linear2.weight, te_recipe=fp8_recipe)
    te_assert_close(b1, te_linear1.bias, te_recipe=fp8_recipe)
    te_assert_close(b2, te_linear2.bias, te_recipe=fp8_recipe)


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

    cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    # Enable grad on thunder params.
    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x, fp8_recipe):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return cfn(x, w1, w2, b1, b2)

    train_model(thunder_model, thunder_sgd_optimizer)

    # Verify that the weights and biases converge to same value after few iterations
    # And take the first recipe to set tolerances.
    te_assert_close(w1, te_linear1.weight, te_recipe=recipes[0])
    te_assert_close(w2, te_linear2.weight, te_recipe=recipes[0])
    te_assert_close(b1, te_linear1.bias, te_recipe=recipes[0])
    te_assert_close(b2, te_linear2.bias, te_recipe=recipes[0])


@requiresCUDA
def test_te_linear_invalid_inputs():
    def assert_not_transformed(x, w):
        def fn(x, w):
            return torch.nn.functional.linear(x, w)

        cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])
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
@skip_on_sm120_and_sm121
def test_te_with_autocast():
    from thunder.transforms.autocast import autocast

    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(64, 64, device=device, requires_grad=True)
    w = torch.randn(64, 64, device=device, requires_grad=True)

    cfunc = thunder.jit(
        autocast(foo, dtype=thunder.dtypes.bfloat16),
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform()],
        disable_preprocessing=True,
    )
    cfunc(x, w)

    fwd_traces = thunder.last_traces(cfunc)
    # Verify that we have replaced `prims.linear` with `te_linear`
    assert any(bsym.sym.name.startswith("te_functional_linear") for bsym in fwd_traces[-1].bound_symbols)


# NOTE: strict=False as it passes on Blackwell.
@pytest.mark.xfail(strict=False, raises=(RuntimeError, TypeError), reason="Retain graph is not supported by TE")
@requiresCUDA
@skip_on_sm120_and_sm121
def test_te_with_retain_graph():
    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(16, 16, device=device, requires_grad=True)
    w = torch.randn(16, 16, device=device, requires_grad=True)

    cfunc = thunder.jit(
        foo,
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform()],
    )
    out = cfunc(x, w)

    # Retain graph is not supported correctly by TE
    # https://github.com/NVIDIA/TransformerEngine/issues/990
    out.backward(torch.randn_like(out), retain_graph=True)
    out.backward(torch.randn_like(out))


@requiresCUDA
@skip_on_sm120_and_sm121
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
        executors=[transformer_engine_ex],
        transforms=[
            TransformerEngineTransform(),
            MyNoopTransform(),
        ],
    )
    cfunc(x, w)

    fwd_traces = thunder.last_traces(cfunc)

    # Verify that we have `te_linear` in the trace.
    assert any(bsym.sym.name.startswith("te_functional_linear") for bsym in fwd_traces[-1].bound_symbols)


@skip_on_sm120_and_sm121
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

        tfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

        o = tfn(x, w)
        o.sum().backward()

        assert w.grad is not None


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_te_trace_correctness(fp8_recipe: recipe.Recipe):
    if fp8_recipe and fp8_recipe.mxfp8() and not is_mxfp8_supported:
        pytest.skip(msg_mxfp8)
    elif fp8_recipe and is_nvfp4_available and fp8_recipe.nvfp4() and not is_nvfp4_supported:
        pytest.skip(msg_nvfp4)

    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(32, 32, device=device, requires_grad=True)
    w = torch.randn(32, 32, device=device, requires_grad=True)

    cfunc = thunder.jit(
        foo,
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform()],
    )

    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        cfunc(x, w)

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


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@pytest.mark.parametrize("compile_path", ["jit", "ThunderFX"])
@skip_on_sm120_and_sm121
def test_te_activation_checkpointing_trace(fp8_recipe: recipe.Recipe, compile_path: str):
    if fp8_recipe is None:
        pytest.skip(
            "When recipe is None a new recipe is created for each iteration. This makes the results not numerically comparable."
        )

    if fp8_recipe.mxfp8() and not is_mxfp8_supported:
        pytest.skip(msg_mxfp8)

    if is_nvfp4_available and fp8_recipe.nvfp4() and not is_nvfp4_supported:
        pytest.skip(msg_nvfp4)

    checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    def fn_to_checkpoint(x, y):
        a = torch.nn.functional.linear(x, y)
        a = torch.sin(a)
        return a

    def fn(x, w, w2):
        a = checkpoint_fn(fn_to_checkpoint, x, w)
        a = torch.nn.functional.linear(a, w2)
        return a

    if compile_path == "jit":
        cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])
    else:
        from thunder.dynamo import thunderfx

        cfn = thunderfx(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    device = "cuda"
    x = torch.randn(64, 64, device=device, requires_grad=True)
    w = torch.randn(64, 64, device=device, requires_grad=True)
    w2 = torch.randn(64, 64, device=device, requires_grad=True)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        cfn(x, w, w2)

    fwd_trace = thunder.last_traces(cfn)[-1] if compile_path == "jit" else cfn.last_traces[-1]

    from thunder.core.vjp_utils import get_saved_for_backward_tensors

    saved_tensors = {p.name for p in get_saved_for_backward_tensors(fwd_trace)}
    # only the first linear is checkpointed, so only the first two are saved for backward and the two inputs
    assert len(saved_tensors) == 4
    # make sure that only two outputs from the second linear in the forward are passed to the backward
    if compile_path == "jit":
        assert len(saved_tensors - {"x", "w"}) == 2
    else:
        assert len(saved_tensors - {"l_x_", "l_w_"}) == 2


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@pytest.mark.parametrize("compile_path", ["jit", "ThunderFX"])
@pytest.mark.filterwarnings("ignore::FutureWarning")  # Coming from TE v2.3
@skip_on_sm120_and_sm121
def test_te_activation_checkpointing_correctness(fp8_recipe: recipe.Recipe, compile_path: str):
    if fp8_recipe is None:
        pytest.skip(
            "When recipe is None a new recipe is created for each iteration. This makes the results not numerically comparable."
        )

    if fp8_recipe.mxfp8() and not is_mxfp8_supported:
        pytest.skip(msg_mxfp8)

    if is_nvfp4_available and fp8_recipe.nvfp4() and not is_nvfp4_supported:
        pytest.skip(msg_nvfp4)

    dtype = torch.bfloat16

    device = "cuda"
    iterations = 6

    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # Before starting, reset the state manager.
    FP8GlobalStateManager.reset()

    checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    input_shape = (768, 4096)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 2048, params_dtype=dtype)

    def clone_params(*params):
        return tuple(param.detach().clone() for param in params)

    w1, w2, b1, b2 = clone_params(te_linear1.weight, te_linear2.weight, te_linear1.bias, te_linear2.bias)

    target_value = torch.randint(42, (768,), dtype=torch.int64, device=device)
    inputs = tuple(torch.rand(*input_shape, device=device, dtype=dtype, requires_grad=True) for _ in range(iterations))

    def train_model(model, optimizer, loss_hist):
        for iter_n in range(iterations):
            x = inputs[iter_n]
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                result = model(x)
            loss = torch.nn.functional.cross_entropy(result, target_value)
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def te_model(x):
        a = te_linear1(x)
        a = torch.sin(a)
        return te_linear2(a)

    te_sgd_optimizer = torch.optim.SGD(list(te_linear1.parameters()) + list(te_linear2.parameters()))

    te_loss_hist = []
    train_model(te_model, te_sgd_optimizer, te_loss_hist)

    # TE does not expose the scales for MXFP8
    if fp8_recipe.delayed():
        te_scales = []
        te_amax_hist = []
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            for buffer_key, _ in FP8GlobalStateManager.global_amax_buffer.items():
                # needs to clone the tensors because TE will use in-place copy to modify it any time autocast is called
                te_scales += [t.detach().clone() for t in FP8GlobalStateManager.global_scale_buffer[buffer_key]]
                te_amax_hist += [
                    t.detach().clone() for t in FP8GlobalStateManager.global_amax_history_buffer[buffer_key]
                ]

        # Make sure that the global state manager has been reset and
        # that there are only the buffers we need and not more
        assert len(te_scales) == 4
        assert len(te_amax_hist) == 4

    def fn_to_checkpoint(x, w1, b1):
        a = torch.nn.functional.linear(x, w1, b1)
        a = torch.sin(a)
        return a

    def fn(x, w1, w2, b1, b2):
        o = checkpoint_fn(fn_to_checkpoint, x, w1, b1)
        return torch.nn.functional.linear(o, w2, b2)

    if compile_path == "jit":
        cfn = thunder.jit(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])
    else:
        from thunder.dynamo import thunderfx

        cfn = thunderfx(fn, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x):
        return cfn(x, w1, w2, b1, b2)

    thunder_loss_hist = []
    train_model(thunder_model, thunder_sgd_optimizer, thunder_loss_hist)

    for loss, te_loss in zip(thunder_loss_hist, te_loss_hist):
        te_assert_close(loss, te_loss, te_recipe=fp8_recipe)

    te_assert_close(w1, te_linear1.weight, te_recipe=fp8_recipe)
    te_assert_close(w2, te_linear2.weight, te_recipe=fp8_recipe)
    te_assert_close(b1, te_linear1.bias, te_recipe=fp8_recipe)
    te_assert_close(b2, te_linear2.bias, te_recipe=fp8_recipe)

    # TE does not expose the scales for MXFP8
    if fp8_recipe.delayed():
        if compile_path == "jit":
            fwd_trc_py_ctx = thunder.last_traces(cfn)[-1].python_ctx()
        else:
            fwd_trc_py_ctx = cfn.last_traces[-1].python_ctx()

        if compile_path == "jit":
            bwd_trc_py_ctx = thunder.last_backward_traces(cfn)[-1].python_ctx()
        else:
            bwd_trc_py_ctx = cfn.last_backward_traces[-1].python_ctx()

        th_scales = []
        th_amax_hist = []

        for k in fwd_trc_py_ctx.keys():
            if "get_te_fp8_state" in k:
                th_scales += [fwd_trc_py_ctx[k].state.scale]
                th_amax_hist += [fwd_trc_py_ctx[k].state.amax_history]

        th_bwd_scales = []
        th_bwd_amax_hist = []
        for k in bwd_trc_py_ctx.keys():
            if "get_te_fp8_state" in k:
                th_bwd_scales += [bwd_trc_py_ctx[k].state.scale]
                th_bwd_amax_hist += [bwd_trc_py_ctx[k].state.amax_history]

        th_scales.extend(reversed(th_bwd_scales))
        th_amax_hist.extend(reversed(th_bwd_amax_hist))

        # check the scales are the same but for last dimension which is always on in TE
        for te_scale, th_scale in zip(te_scales, th_scales):
            te_assert_close(te_scale[:-1], th_scale, te_recipe=fp8_recipe)

        # check that amax history is the same as TE
        for te_amax, th_amax in zip(te_amax_hist, th_amax_hist):
            te_assert_close(te_amax[:, :-1], th_amax, te_recipe=fp8_recipe)


@requiresCUDA
@pytest.mark.skipif(
    LooseVersion(transformer_engine.__version__) < LooseVersion("2.9"),
    reason="need TE >= 2.9 for quantizer location",
)
def test_te_inference_8bit():
    from thunder.transforms.te_inference import TEInference8BitTransform

    with torch.device("cuda"):
        m = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        ).requires_grad_(False)
        m2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        ).requires_grad_(False)
        a = torch.randn(16, 1024, device="cuda")

    quant_transform = TEInference8BitTransform()
    te_inference_executor = quant_transform.get_executor()
    quant_transform2 = TEInference8BitTransform()
    jm = thunder.jit(
        m, transforms=[quant_transform], executors=(te_inference_executor, *thunder.get_default_executors())
    )
    jm2 = thunder.jit(
        m2, transforms=[quant_transform2], executors=(te_inference_executor, *thunder.get_default_executors())
    )

    actual = jm(a)
    expected = m(a)
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)

    jm2.load_original_state_dict(m.state_dict())

    actual2 = jm2(a)
    torch.testing.assert_close(actual, actual2)
