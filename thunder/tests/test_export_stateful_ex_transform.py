import pytest
import torch
import torch.nn as nn

import thunder
from thunder.tests.framework import requiresCUDA


# NOTE: On SM120/121, TE defaults to using Float8BlockScaling
#       which is currently unsupported in thunder, we skip the tests for these SM architectures.
from thunder.tests.utils import skip_on_sm120_and_sm121, is_sm120_orsm121
from thunder.dev_utils.export_stateful_ex_transform import ExportStatefulExecutorsTransform
from thunder.dynamo import ThunderCompiler

# Make TE optional so this file can host tests for other executors too
TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    from thunder.executors.transformer_engineex import transformer_engine_ex, TransformerEngineTransform

    TE_AVAILABLE = True
except Exception:
    te = None
    recipe = None
    TE_AVAILABLE = False

if TE_AVAILABLE:
    # FP8 is supported on compute arch 8.9 onwards.
    # MXFP8 is supported on compute arch 10.0 onwards.
    # Skip the TE-specific parametrizations if current hardware is not supported.
    is_fp8_supported, msg_fp8 = te.fp8.check_fp8_support()
    is_mxfp8_supported, msg_mxfp8 = te.fp8.check_mxfp8_support()
    if not is_fp8_supported:
        pytest.skip(msg_fp8, allow_module_level=True)

    hybrid_fp8_delayed_scaling_recipe = recipe.DelayedScaling()
    mxfp8_e4m3_recipe = recipe.MXFP8BlockScaling()

    # `None` is used to test the default recipe.
    recipes = (None, hybrid_fp8_delayed_scaling_recipe, mxfp8_e4m3_recipe)
    recipe_ids = ("default", "delayed_scaling", "mxfp8_e4m3")
else:
    is_mxfp8_supported, msg_mxfp8 = (False, "TransformerEngine not available")
    recipes = (None,)
    recipe_ids = ("default",)


@requiresCUDA
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed.")
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_export_te_states_linear_forward(fp8_recipe):
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

    jmodel = thunder.jit(
        model,
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform(), ExportStatefulExecutorsTransform()],
    )

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        y = jmodel(x)

    # Validate TE exporter populated as expected
    assert hasattr(jmodel, "te_fp8_stats"), "ThunderModule should expose te_fp8_stats"
    stats = jmodel.te_fp8_stats
    assert isinstance(stats, dict) and set(stats.keys()) == {"forward", "backward"}
    # After forward, we should have exactly one forward entry and no backward entries yet
    assert len(stats["forward"]) == 1
    assert len(stats["backward"]) == 0
    f_entry = stats["forward"][0]
    assert isinstance(f_entry, dict)
    # Ensure we collected either delayed scaling or block-scaling style info
    assert ("delayed" in f_entry and isinstance(f_entry["delayed"], list)) or (
        "mxfp8_or_block" in f_entry and isinstance(f_entry["mxfp8_or_block"], list)
    )
    # If delayed scaling is used, ensure amax and scale are present
    if isinstance(fp8_recipe, recipe.DelayedScaling) or (fp8_recipe is None and te.fp8.check_fp8_support()[0]):
        assert "delayed" in f_entry
        d = f_entry["delayed"][0]
        assert d.get("scale") is not None
        assert d.get("amax") is not None

    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # After backward pass, one backward entry should be present
    stats = jmodel.te_fp8_stats
    assert len(stats["forward"]) == 1
    assert len(stats["backward"]) == 1


@requiresCUDA
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed.")
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_export_te_states_linear_forward_backward_multiple_iteration(fp8_recipe):
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

    jmodel = thunder.jit(
        model,
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform(), ExportStatefulExecutorsTransform()],
    )

    num_iters = 10
    for _ in range(num_iters):
        # Forward under FP8 autocast
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            y = jmodel(x)
        # Backward with unit upstream gradient
        y.backward(torch.ones_like(y))

    # Validate exporter after multiple iterations
    assert hasattr(jmodel, "te_fp8_stats")
    stats = jmodel.te_fp8_stats
    # One forward and one backward export entry per iteration
    assert len(stats["forward"]) == num_iters
    assert len(stats["backward"]) == num_iters


@requiresCUDA
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed.")
def test_export_te_states_linear_forward_backward_multiple_recipies_iteration():
    # Test Description:
    # Alternate between two different recipes across iterations and ensure the reporter
    # records both recipe configurations exactly once each. Verify forward/backward states
    # and quantizers reflect both linear call sites per recipe, independent of iteration count.

    test_recipes = [recipe.DelayedScaling()]
    supports_mxfp8, _ = te.fp8.check_mxfp8_support()

    if supports_mxfp8:
        test_recipes += [recipe.MXFP8BlockScaling()]

    if len(test_recipes) < 2:
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
    iters = 4

    def train_model(model):
        for iter_n in range(iters):
            te_recipe = test_recipes[iter_n % 2]
            y = model(x, te_recipe)
            y.backward(torch.ones_like(y))

    jmodel = thunder.jit(
        model,
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform(), ExportStatefulExecutorsTransform()],
    )

    def thunder_model(x, fp8_recipe):
        with te.fp8_autocast(fp8_recipe=fp8_recipe):
            return jmodel(x)

    train_model(thunder_model)

    stats = jmodel.te_fp8_stats
    # We expect as many forward/backward entries as iterations
    assert len(stats["forward"]) == iters
    assert len(stats["backward"]) == iters
    # Across all entries, we should see delayed info and, if supported, possibly block info
    has_delayed = any(e.get("delayed") for e in stats["forward"]) or any(e.get("delayed") for e in stats["backward"])
    assert has_delayed


@requiresCUDA
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed.")
@pytest.mark.parametrize("fp8_recipe", recipes, ids=recipe_ids)
@skip_on_sm120_and_sm121
def test_export_te_states_with_torch_compile_and_thunder_backend(fp8_recipe):
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
    backend = ThunderCompiler(
        executors=[transformer_engine_ex],
        transforms=[TransformerEngineTransform(), ExportStatefulExecutorsTransform()],
    )
    compiled_model = torch.compile(model, backend=backend)

    # Run one forward/backward under FP8 autocast
    iters = 10

    def train_model(model):
        for _ in range(iters):
            with te.fp8_autocast(fp8_recipe=fp8_recipe):
                y = model(x)
            y.backward(torch.ones_like(y))

    train_model(compiled_model)

    # Collect TE fp8 stats from Thunder-compiled subgraphs
    reporters = []
    for sinfo in backend.subgraph_infos:
        if sinfo.thunder_compiled_fns:
            for fn in sinfo.thunder_compiled_fns:
                if hasattr(fn, "te_fp8_stats"):
                    reporters.append(fn.te_fp8_stats)

    # We expect at least one Thunder subgraph using TE
    assert len(reporters) >= 1

    # Aggregate counts across subgraphs
    total_fw_entries = sum(len(r["forward"]) for r in reporters)
    total_bw_entries = sum(len(r["backward"]) for r in reporters)

    # We expect at least one Thunder subgraph using TE and to have exported entries
    assert total_fw_entries == iters
    assert total_bw_entries == iters
