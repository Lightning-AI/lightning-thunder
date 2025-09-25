from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

import thunder
import torch

import transformer_engine as te

import transformer_engine_torch
from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    RecipeState,
    get_fp8_torch_dtype,
)


def summarize_recipe(recipe: Recipe) -> dict[str, Any]:
    """Create a compact, serializable summary of a TE FP8 recipe.

    The summary captures the recipe class name and a small set of key
    configuration fields depending on the recipe family (DelayedScaling,
    Float8CurrentScaling, MXFP8BlockScaling, Float8BlockScaling). For delayed
    and current-scaling variants, the effective FP8 torch dtypes for forward
    and backward are also included.

    Args:
        recipe: A TransformerEngine `Recipe` instance from `transformer_engine.common.recipe`.

    Returns:
        A dictionary with fields describing the recipe.
    """
    summary: dict[str, Any] = {
        "type": recipe.__class__.__name__,
        "fp8_format": getattr(recipe, "fp8_format", None),
    }

    if recipe.delayed():
        summary.update(
            {
                "margin": getattr(recipe, "margin", None),
                "amax_history_len": getattr(recipe, "amax_history_len", None),
                "amax_compute_algo": getattr(recipe, "amax_compute_algo", None),
                "scaling_factor_compute_algo": getattr(recipe, "scaling_factor_compute_algo", None),
                "reduce_amax": getattr(recipe, "reduce_amax", None),
                "fp8_dpa": getattr(recipe, "fp8_dpa", None),
                "fp8_mha": getattr(recipe, "fp8_mha", None),
                # Effective FP8 dtypes per pass
                "fwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, True)),
                "bwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, False)),
            }
        )
    elif recipe.float8_current_scaling():
        summary.update(
            {
                "fp8_quant_fwd_inp": getattr(recipe, "fp8_quant_fwd_inp", None),
                "fp8_quant_fwd_weight": getattr(recipe, "fp8_quant_fwd_weight", None),
                "fp8_quant_bwd_grad": getattr(recipe, "fp8_quant_bwd_grad", None),
                "fp8_gemm_fprop": getattr(recipe, "fp8_gemm_fprop", None),
                "fp8_gemm_dgrad": getattr(recipe, "fp8_gemm_dgrad", None),
                "fp8_gemm_wgrad": getattr(recipe, "fp8_gemm_wgrad", None),
                "fp8_dpa": getattr(recipe, "fp8_dpa", None),
                "fp8_mha": getattr(recipe, "fp8_mha", None),
                "fwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, True)),
                "bwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, False)),
            }
        )
    elif recipe.mxfp8():
        summary.update(
            {
                "margin": getattr(recipe, "margin", None),
                "fwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, True)),
                "bwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, False)),
            }
        )
    elif recipe.float8_block_scaling():
        summary.update(
            {
                "x_block_scaling_dim": getattr(recipe, "x_block_scaling_dim", None),
                "w_block_scaling_dim": getattr(recipe, "w_block_scaling_dim", None),
                "grad_block_scaling_dim": getattr(recipe, "grad_block_scaling_dim", None),
                "fp8_quant_fwd_inp": getattr(recipe, "fp8_quant_fwd_inp", None),
                "fp8_quant_fwd_weight": getattr(recipe, "fp8_quant_fwd_weight", None),
                "fp8_quant_bwd_grad": getattr(recipe, "fp8_quant_bwd_grad", None),
                "fp8_gemm_fprop": getattr(recipe, "fp8_gemm_fprop", None),
                "fp8_gemm_dgrad": getattr(recipe, "fp8_gemm_dgrad", None),
                "fp8_gemm_wgrad": getattr(recipe, "fp8_gemm_wgrad", None),
                "fwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, True)),
                "bwd_fp8_torch_dtype": str(get_fp8_torch_dtype(recipe, False)),
            }
        )

    return summary


def summarize_state(state: RecipeState) -> dict[str, Any]:
    """Summarize a runtime FP8 `RecipeState` object.

    Captures the state class, mode (forward/backward/None), dtype, number of
    quantizers, and optionally basic tensor shape/device information for scale
    and amax history tensors when present.

    Args:
        state: A `RecipeState` produced by TransformerEngine during execution.

    Returns:
        A dictionary with essential metadata about the state for reporting.
    """
    out: dict[str, Any] = {
        "cls": state.__class__.__name__,
        "mode": getattr(state, "mode", None),
        "dtype": str(getattr(state, "dtype", None)),
        "num_quantizers": getattr(state, "num_quantizers", None),
    }
    scale = getattr(state, "scale", None)
    if isinstance(scale, torch.Tensor):
        out["scale_shape"] = tuple(scale.shape)
        out["scale_device"] = str(scale.device)
    amax_hist = getattr(state, "amax_history", None)
    if isinstance(amax_hist, torch.Tensor):
        out["amax_history_shape"] = tuple(amax_hist.shape)
        out["amax_history_device"] = str(amax_hist.device)
    return out


def summarize_quantizer(quantizer: Any) -> dict[str, Any]:
    """Summarize an FP8 quantizer instance.

    Extracts commonly useful fields across different quantizer implementations
    (rowwise/columnwise usage, internal flag, dtype) and, when available,
    additional configuration such as amax reduction info. Tensor shape/device
    metadata for `scale` and `amax` is included if present.

    Args:
        quantizer: A quantizer-like object from TransformerEngine runtime.

    Returns:
        A dictionary describing the quantizer in a compact, readable form.
    """
    base: dict[str, Any] = {
        "cls": quantizer.__class__.__name__,
        "rowwise_usage": getattr(quantizer, "rowwise_usage", None),
        "columnwise_usage": getattr(quantizer, "columnwise_usage", None),
        "internal": getattr(quantizer, "internal", None),
        "dtype": str(getattr(quantizer, "dtype", None)),
    }
    # Optional attributes by quantizer class
    if hasattr(quantizer, "with_amax_reduction"):
        base["with_amax_reduction"] = getattr(quantizer, "with_amax_reduction")
        base["amax_reduction_group"] = str(getattr(quantizer, "amax_reduction_group", None))
    if hasattr(quantizer, "force_pow_2_scales"):
        base["force_pow_2_scales"] = getattr(quantizer, "force_pow_2_scales")
    if hasattr(quantizer, "amax_epsilon"):
        base["amax_epsilon"] = getattr(quantizer, "amax_epsilon")
    # Shapes (when available)
    for attr in ("scale", "amax"):
        tensor = getattr(quantizer, attr, None)
        if isinstance(tensor, torch.Tensor):
            base[f"{attr}_shape"] = tuple(tensor.shape)
            base[f"{attr}_device"] = str(tensor.device)
    return base


def build_global_context() -> dict[str, Any]:
    """Collect global FP8 runtime context and environment details.

    Queries `FP8GlobalStateManager` and related sources to produce a stable
    snapshot of the current FP8 configuration and availability, along with
    environment metadata such as CUDA/cuBLASLt versions, device compute
    capability, world size, and package versions. The result is intended to be
    recorded once per session for reporting correlation.

    Returns:
        A dictionary of global context fields suitable for rendering in reports.
    """
    fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
    fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()
    with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()
    high_precision_init_val = FP8GlobalStateManager.with_high_precision_init_val()
    fp8_group = FP8GlobalStateManager.get_fp8_group()
    autocast_depth = FP8GlobalStateManager.FP8_AUTOCAST_DEPTH
    graph_capturing = FP8GlobalStateManager.fp8_graph_capturing()

    # Availability and reasons
    fp8_avail, reason_no_fp8 = FP8GlobalStateManager.is_fp8_available()
    mxfp8_avail, reason_no_mx = FP8GlobalStateManager.is_mxfp8_available()
    fp8blk_avail, reason_no_blk = FP8GlobalStateManager.is_fp8_block_scaling_available()

    # Versions / device
    cuda_version = getattr(torch.version, "cuda", None)
    cublaslt_version = transformer_engine_torch.get_cublasLt_version()
    device_cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

    # Dist info
    world_size = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = torch.distributed.get_world_size(group=fp8_group)
        except (RuntimeError, ValueError, TypeError):
            world_size = torch.distributed.get_world_size()

    # Package versions
    te_version = getattr(te, "__version__", None) if te is not None else None
    thunder_version = None
    thunder_version = getattr(thunder, "__version__", None)

    return {
        "fp8_enabled": fp8_enabled,
        "fp8_calibration": fp8_calibration,
        "with_fp8_parameters": with_fp8_params,
        "high_precision_init_val": high_precision_init_val,
        "fp8_group": str(fp8_group),
        "world_size": world_size,
        "autocast_depth": autocast_depth,
        "graph_capturing": graph_capturing,
        "fp8_available": fp8_avail,
        "reason_no_fp8": reason_no_fp8,
        "mxfp8_available": mxfp8_avail,
        "reason_no_mxfp8": reason_no_mx,
        "fp8_block_scaling_available": fp8blk_avail,
        "reason_no_fp8_block_scaling": reason_no_blk,
        "cuda_version": cuda_version,
        "cublaslt_version": cublaslt_version,
        "device_compute_capability": device_cc,
        "te_version": te_version,
        "thunder_version": thunder_version,
    }


@dataclass
class TEStateReporter:
    """Accumulates TE runtime summaries and renders a report."""

    global_ctx: dict[str, Any] | None = None
    recipe_summaries: list[dict[str, Any]] = field(default_factory=list)
    state_summaries_forward: list[dict[str, Any]] = field(default_factory=list)
    state_summaries_backward: list[dict[str, Any]] = field(default_factory=list)
    quantizer_summaries: list[dict[str, Any]] = field(default_factory=list)
    shape_policy: dict[str, Any] = field(default_factory=lambda: {"mxfp8_block": MXFP8_BLOCK_SCALING_SIZE})
    seen_fw_states: set[tuple[int, int]] = field(default_factory=set)
    seen_bw_states: set[tuple[int, int]] = field(default_factory=set)
    seen_quantizers: set[tuple[int, int]] = field(default_factory=set)

    def update_from_runtime(
        self,
        *,
        holder,
        recipe: Recipe | None = None,
        states: Sequence[RecipeState] | None = None,
        mode: str | None = None,
        quantizers: Sequence[Any] | None = None,
    ) -> None:
        """Update the reporter with data observed during runtime.

        This method is called one or more times during forward/backward passes
        to incrementally collect summaries. The first invocation also captures
        the global context snapshot.

        Args:
            holder: The holder object (TERecipe, TERecipeState, or TEQuantizerState)
                   that owns the runtime data being reported.
            recipe: Optional recipe active for the current autocast session.
            states: Optional sequence of `RecipeState` objects observed.
            mode: Optional mode string ("forward" or "backward") indicating the
                 execution phase when states are captured.
            quantizers: Optional sequence of quantizer objects observed.
        """
        if self.global_ctx is None:
            self.global_ctx = build_global_context()

        # Collect recipe summaries only when called from the main recipe holder (TERecipe class).
        # This avoids duplicate recipe entries when the same recipe is referenced by quantizer
        # or state holders, ensuring we track each unique recipe configuration exactly once.
        if recipe is not None and not quantizers:
            summary = summarize_recipe(recipe)
            if summary not in self.recipe_summaries:
                self.recipe_summaries.append(summary)

        # Each trace execution can contain multiple forward and backward states for different recipes.
        # We track unique combinations of (holder_id, recipe_id) to avoid duplicate state summaries
        # while ensuring we capture all distinct recipe configurations used during runtime.
        if states:
            if mode == "forward":
                if (id(holder), id(recipe)) not in self.seen_fw_states:
                    self.seen_fw_states.add((id(holder), id(recipe)))
                    self.state_summaries_forward.extend(summarize_state(s) for s in states)
            elif mode == "backward":
                if (id(holder), id(recipe)) not in self.seen_bw_states:
                    self.seen_bw_states.add((id(holder), id(recipe)))
                    self.state_summaries_backward.extend(summarize_state(s) for s in states)

        # Quantizers are reused across multiple trace executions but their behavior depends on the active recipe.
        # While the quantizer object instances remain the same, different recipes can affect their configuration
        # and internal state. We track unique combinations of (holder_id, recipe_id) to ensure we capture
        # quantizer summaries for each distinct recipe configuration, avoiding both duplicates and missed
        # configurations when recipes change during runtime.
        if quantizers:
            if (id(holder), id(recipe)) not in self.seen_quantizers:
                self.seen_quantizers.add((id(holder), id(recipe)))
                self.quantizer_summaries.extend(summarize_quantizer(q) for q in quantizers)

    def render_report(self) -> str:
        """Render a human-readable multi-section report of collected data.

        The report includes global context, recipes, forward/backward state
        summaries, quantizer summaries, and shape policy information.

        Returns:
            A formatted string suitable for console logging or test output.
        """
        lines: list[str] = []

        def add(line: str = "") -> None:
            lines.append(line)

        # Global Context
        ctx = self.global_ctx or {}
        add("Global Context:")
        add(f"  • FP8 Enabled: {ctx.get('fp8_enabled')}")
        add(f"  • FP8 Calibration: {ctx.get('fp8_calibration')}")
        add(f"  • FP8 Parameters: {ctx.get('with_fp8_parameters')}")
        add(f"  • High Precision Init: {ctx.get('high_precision_init_val')}")
        add(f"  • FP8 Group: {ctx.get('fp8_group')}")
        add(f"  • World Size: {ctx.get('world_size')}")
        add(f"  • Autocast Depth: {ctx.get('autocast_depth')}")
        add(f"  • Graph Capturing: {ctx.get('graph_capturing')}")
        add("")
        add("  Availability:")
        add(f"    - FP8: {ctx.get('fp8_available')}")
        add(f"    - MXFP8: {ctx.get('mxfp8_available')}")
        add(f"    - FP8 Block Scaling: {ctx.get('fp8_block_scaling_available')}")
        if not ctx.get("fp8_block_scaling_available", True):
            add(f"      Reason: {ctx.get('reason_no_fp8_block_scaling')}")
        add("")
        add("  Versions:")
        add(f"    - CUDA: {ctx.get('cuda_version')}  cuBLASLt: {ctx.get('cublaslt_version')}")
        add(f"    - Compute Capability: {ctx.get('device_compute_capability')}")
        add(f"    - TransformerEngine: {ctx.get('te_version')}  Thunder: {ctx.get('thunder_version')}")
        add("")

        # Recipes
        add(f"Recipes ({len(self.recipe_summaries)}):")
        for idx, rs in enumerate(self.recipe_summaries, 1):
            add(f"  [{idx}] {rs.get('type')} - {rs.get('fp8_format')}")
            # Print a compact subset
            for key in (
                "margin",
                "amax_history_len",
                "amax_compute_algo",
                "reduce_amax",
                "fp8_dpa",
                "fp8_mha",
                "fwd_fp8_torch_dtype",
                "bwd_fp8_torch_dtype",
                "x_block_scaling_dim",
                "w_block_scaling_dim",
                "grad_block_scaling_dim",
            ):
                if rs.get(key) is not None:
                    add(f"      {key}: {rs.get(key)}")
        add("")

        # States
        add(f"Forward States ({len(self.state_summaries_forward)}):")
        for idx, ss in enumerate(self.state_summaries_forward, 1):
            add(f"  [{idx}] Mode: {ss.get('mode')}  DType: {ss.get('dtype')}  Quantizers: {ss.get('num_quantizers')}")
            if ss.get("scale_shape") is not None:
                add(f"      Scale: {ss.get('scale_shape')} on {ss.get('scale_device')}")
            else:
                add("      Note: no per-tensor scale (likely MXFP8/blockwise)")
            if ss.get("amax_history_shape") is not None:
                add(f"      Amax History: {ss.get('amax_history_shape')} on {ss.get('amax_history_device')}")
        add("")

        # Backward States (if any)
        if self.state_summaries_backward:
            add(f"Backward States ({len(self.state_summaries_backward)}):")
            for idx, ss in enumerate(self.state_summaries_backward, 1):
                add(
                    f"  [{idx}] Mode: {ss.get('mode')}  DType: {ss.get('dtype')}  Quantizers: {ss.get('num_quantizers')}"
                )
                if ss.get("scale_shape") is not None:
                    add(f"      Scale: {ss.get('scale_shape')} on {ss.get('scale_device')}")
                if ss.get("amax_history_shape") is not None:
                    add(f"      Amax History: {ss.get('amax_history_shape')} on {ss.get('amax_history_device')}")
            add("")

        # Quantizers
        add(f"Quantizers ({len(self.quantizer_summaries)}):")
        for idx, qs in enumerate(self.quantizer_summaries, 1):
            add(
                f"  [{idx}] {qs.get('cls')} - {qs.get('dtype')}\n"
                f"      Rowwise: {qs.get('rowwise_usage')}\n"
                f"      Columnwise: {qs.get('columnwise_usage')}\n"
                f"      Internal: {qs.get('internal')}"
            )
            if qs.get("with_amax_reduction") is not None:
                add(f"      Amax Reduction: {qs.get('with_amax_reduction')} group={qs.get('amax_reduction_group')}")
            for attr in ("scale", "amax"):
                if qs.get(f"{attr}_shape") is not None:
                    add(f"      {attr.capitalize()}: {qs.get(f'{attr}_shape')} on {qs.get(f'{attr}_device')}")
        add("")

        # Shape policy
        add("Shape Policy:")
        add(f"  • mxfp8_block: {self.shape_policy.get('mxfp8_block')}")

        return "\n".join(lines)


__all__ = ["TEStateReporter"]
