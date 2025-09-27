import pytest
import torch
import torch.nn as nn

import thunder
from thunder.tests.framework import requiresCUDA


# NOTE: On SM120/121, TE defaults to using Float8BlockScaling
#       which is currently unsupported in thunder, we skip the tests for these SM architectures.
from thunder.tests.utils import skip_on_sm120_and_sm121, is_sm120_orsm121

transformer_engine_module = pytest.importorskip(
    "transformer_engine", reason="transformer_engine was not found, skipping the tests."
)

from thunder.executors.transformer_engineex import transformer_engine_ex, TransformerEngineTransform
from thunder.dynamo import ThunderCompiler
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
@skip_on_sm120_and_sm121
def test_te_reporter_linear_forward_backward(fp8_recipe: recipe.Recipe):
    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    if is_sm120_orsm121 and fp8_recipe is None:
        pytest.skip("On SM120/121, default recipe is Float8BlockScaling which is not supported")

    # Test Description:
    # Verify that the TEStateReporter correctly captures and reports TransformerEngine
    # FP8 state information during forward pass execution, including global context,
    # recipe summaries, and forward state summaries.

    dtype = torch.bfloat16
    device = "cuda"

    # Inputs (3D input)
    x = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(4096, 4096, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.randn(2048, 4096, device=device, dtype=dtype))

        def forward(self, x):
            o = torch.nn.functional.linear(x, self.w1)
            added = o + x
            return torch.nn.functional.linear(added, self.w2)

    model = Module()

    jmodel = thunder.jit(model, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        y = jmodel(x)

    # Validate TE reporter populated as expected
    assert hasattr(jmodel, "te_reporter"), "ThunderModule should expose te_reporter"
    rep = jmodel.te_reporter

    # Global context is captured
    assert rep.global_ctx is not None, "Global context should be populated"
    assert "fp8_available" in rep.global_ctx
    assert "mxfp8_available" in rep.global_ctx
    assert "fp8_block_scaling_available" in rep.global_ctx

    # Recipes captured; type should be one of known TE recipe classes
    assert len(rep.recipe_summaries) >= 1
    recipe_types = {rs.get("type") for rs in rep.recipe_summaries}
    known_types = {"DelayedScaling", "Float8BlockScaling", "MXFP8BlockScaling", "Float8CurrentScaling"}
    assert recipe_types & known_types, f"Unexpected recipe types collected: {recipe_types}"

    # If a specific recipe is requested, ensure it's reflected
    if isinstance(fp8_recipe, recipe.DelayedScaling):
        assert "DelayedScaling" in recipe_types
    if isinstance(fp8_recipe, recipe.MXFP8BlockScaling):
        assert "MXFP8BlockScaling" in recipe_types

    # Forward states and quantizers should be recorded; no backward states without backward pass
    assert len(rep.state_summaries_forward) == 2
    assert all(ss.get("mode") in (None, "forward") for ss in rep.state_summaries_forward)
    assert any(ss.get("num_quantizers") in (1, 2) for ss in rep.state_summaries_forward)
    assert len(rep.state_summaries_backward) == 0
    assert len(rep.quantizer_summaries) == 4
    assert all("cls" in qs and "dtype" in qs for qs in rep.quantizer_summaries)

    # Rendered report contains key sections
    report_txt = rep.render_report()
    assert "Global Context:" in report_txt
    assert "Recipes (" in report_txt
    assert "Forward States (" in report_txt
    assert "Quantizers (" in report_txt

    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    report_txt = rep.render_report()
    # After backward pass, backward states should be recorded and reported
    assert len(rep.state_summaries_forward) == 2  # Forward states not changed
    assert len(rep.state_summaries_backward) == 2
    assert all(ss.get("mode") in (None, "backward") for ss in rep.state_summaries_backward)
    assert "Backward States (" in report_txt


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_te_reporter_linear_forward_backward_multiple_iteration(fp8_recipe: recipe.Recipe):
    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    if is_sm120_orsm121 and fp8_recipe is None:
        pytest.skip("On SM120/121, default recipe is Float8BlockScaling which is not supported")

    # Test Description:
    # Run multiple forward/backward iterations under a single recipe configuration and
    # verify that the TE reporter does not grow with the iteration count. The recipe
    # list should contain one unique entry, and state/quantizer summaries should reflect
    # the two linear call sites exactly once per direction, independent of iterations.

    dtype = torch.bfloat16
    device = "cuda"

    # Inputs and model
    x = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(4096, 4096, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.randn(2048, 4096, device=device, dtype=dtype))

        def forward(self, x):
            o = torch.nn.functional.linear(x, self.w1)
            added = o + x
            return torch.nn.functional.linear(added, self.w2)

    model = Module()

    jmodel = thunder.jit(model, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    num_iters = 10
    for _ in range(num_iters):
        # Forward under FP8 autocast
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            y = jmodel(x)
        # Backward with unit upstream gradient
        y.backward(torch.ones_like(y))

    # Validate reporter after multiple iterations
    assert hasattr(jmodel, "te_reporter")
    rep = jmodel.te_reporter

    # Global context present
    assert rep.global_ctx is not None

    # Recipes captured
    assert len(rep.recipe_summaries) == 1

    # Forward/backward states recorded (may be cached, so at least one each)
    assert len(rep.state_summaries_forward) == 2
    assert len(rep.state_summaries_backward) == 2

    # Quantizers observed at least once
    assert len(rep.quantizer_summaries) == 6

    # Report reflects sections
    rpt = rep.render_report()
    assert "Forward States (" in rpt
    assert "Backward States (" in rpt


@requiresCUDA
def test_te_reporter_linear_forward_backward_multiple_recipies_iteration():
    # Test Description:
    # Alternate between two different recipes across iterations and ensure the reporter
    # records both recipe configurations exactly once each. Verify forward/backward states
    # and quantizers reflect both linear call sites per recipe, independent of iteration count.

    recipes = [recipe.DelayedScaling()]
    supports_mxfp8, _ = te.fp8.check_mxfp8_support()

    if supports_mxfp8:
        recipes += [recipe.MXFP8BlockScaling()]

    if len(recipes) < 2:
        pytest.skip("platform does not support two different recipes")

    dtype = torch.bfloat16
    device = "cuda"

    # Inputs and model
    x = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(4096, 4096, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.randn(2048, 4096, device=device, dtype=dtype))

        def forward(self, x):
            o = torch.nn.functional.linear(x, self.w1)
            added = o + x
            return torch.nn.functional.linear(added, self.w2)

    model = Module()
    iters = 10

    def train_model(model):
        for iter_n in range(iters):
            te_recipe = recipes[iter_n % 2]
            y = model(x, te_recipe)
            y.backward(torch.ones_like(y))

    jmodel = thunder.jit(model, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    def thunder_model(x, fp8_recipe):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return jmodel(x)

    train_model(thunder_model)

    rep_str = jmodel.te_reporter
    assert len(rep_str.recipe_summaries) == len(recipes)
    assert len(rep_str.state_summaries_forward) == 4
    assert len(rep_str.state_summaries_backward) == 4
    assert len(rep_str.quantizer_summaries) == 12


@requiresCUDA
def test_te_reporter_linear_forward_backward_same_recipe_not_reported_twice():
    # Test Description:
    # Alternate between two separate DelayedScaling instances that are equivalent in configuration.
    # Ensure the reporter treats them as the same effective recipe and does not duplicate entries
    # across iterations. Forward/backward states should reflect the two linear call sites once each,
    # and quantizers should be counted once per site, independent of iteration count.

    delayed_scaling_recipe_a = recipe.DelayedScaling()
    delayed_scaling_recipe_b = recipe.DelayedScaling()

    dtype = torch.bfloat16
    device = "cuda"

    # Inputs and model
    x = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(4096, 4096, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.randn(2048, 4096, device=device, dtype=dtype))

        def forward(self, x):
            o = torch.nn.functional.linear(x, self.w1)
            added = o + x
            return torch.nn.functional.linear(added, self.w2)

    model = Module()

    def train_model(model):
        # Run for `iterations`.
        for iter_n in range(3):
            y = model(x, delayed_scaling_recipe_a if iter_n % 2 == 0 else delayed_scaling_recipe_b)

            y.backward(torch.ones_like(y))

    jmodel = thunder.jit(model, executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])

    def thunder_model(x, fp8_recipe=None):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return jmodel(x)

    train_model(thunder_model)

    rep_str = jmodel.te_reporter
    assert len(rep_str.recipe_summaries) == 1
    assert len(rep_str.state_summaries_forward) == 4
    assert len(rep_str.state_summaries_backward) == 4
    assert len(rep_str.quantizer_summaries) == 12


@requiresCUDA
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_te_reporter_with_torch_compile_and_thunder_backend(fp8_recipe: recipe.Recipe):
    # Test Description:
    # Use torch.compile with Thunder as backend (ThunderCompiler) to run the model
    # under FP8 autocast. Verify that TE runtime states are exported and available
    # from the Thunder-compiled subgraphs via `te_reporter`, and that forward/backward
    # summaries match expectations (iteration-invariant).

    if fp8_recipe and not (fp8_recipe.delayed() or is_mxfp8_supported):
        pytest.skip(msg_mxfp8)

    if is_sm120_orsm121 and fp8_recipe is None:
        pytest.skip("On SM120/121, default recipe is Float8BlockScaling which is not supported")

    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(4096, 64, device=device, dtype=dtype, batch_first=True)
            self.norm1 = nn.LayerNorm(4096, device=device, dtype=dtype)
            self.norm2 = nn.LayerNorm(4096, device=device, dtype=dtype)
            self.mlp = nn.Sequential(
                nn.Linear(4096, 16384, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(16384, 4096, device=device, dtype=dtype),
            )

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)
            return x

    model = Module()

    # Compile with torch.compile using Thunder as backend
    backend = ThunderCompiler(executors=[transformer_engine_ex], transforms=[TransformerEngineTransform()])
    compiled_model = torch.compile(model, backend=backend)

    # Run one forward/backward under FP8 autocast
    def train_model(model):
        iters = 10
        for _ in range(iters):
            with te.fp8_autocast(fp8_recipe=fp8_recipe):
                y = model(x)
            y.backward(torch.ones_like(y))

    train_model(compiled_model)

    print(compiled_model.__class__)

    # Collect TE reporters from Thunder-compiled subgraphs
    reporters = []
    for sinfo in backend.subgraph_infos:
        if sinfo.thunder_compiled_fns:
            for fn in sinfo.thunder_compiled_fns:
                if hasattr(fn, "te_reporter"):
                    reporters.append(fn.te_reporter)

    # We expect at least one Thunder subgraph using TE
    assert len(reporters) >= 1

    # Aggregate counts across subgraphs
    total_recipes = sum(len(r.recipe_summaries) for r in reporters)
    total_fw_states = sum(len(r.state_summaries_forward) for r in reporters)
    total_bw_states = sum(len(r.state_summaries_backward) for r in reporters)
    total_quantizers = sum(len(r.quantizer_summaries) for r in reporters)

    # Recipe presence
    assert total_recipes >= 1
    # Two linear call sites leading to two forward and two backward states in total
    assert total_fw_states == 2
    assert total_bw_states == 2
    # Quantizers (2 per forward, 1 per backward site leading to 6 total)
    assert total_quantizers == 6
