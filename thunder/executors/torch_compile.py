import operator
import time
from collections.abc import Callable, Hashable
from typing import Any

import torch
from lightning_utilities import compare_version

from thunder.core import prims, utils
from thunder.core.proxies import Proxy, TensorProxy, unvariableify, Variable
from thunder.core.rematerialization import rematerialize
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import from_trace, tracectx, TraceCtx, TraceProvenance
from thunder.core.transform_common import dce
from thunder.core.pytree import tree_flatten
from thunder.executors.passes import (
    update_fusion_call_ctx,
    transform_for_execution,
)
from thunder.executors.utils import Region
from thunder.extend import FusionExecutor, register_executor, ImplInfo
from thunder.core.compile_data import get_compile_option
from thunder.executors.torchex import ex as pytorch_ex


_TORCH_GREATER_EQUAL_2_3 = compare_version("torch", operator.ge, "2.3.0", use_base_version=True)


def to_torch_translator(bsym: BoundSymbol) -> Callable:
    """Translates a BoundSymbol to a corresponding traceable by Thunder and
    executable by PyTorch callable.

    Args:
        bsym: The BoundSymbol to translate.

    Returns:
        A callable that can be executed by PyTorch after being traced by Thunder.
    """

    def _to_torch(*args, **kwargs) -> Any:
        impl_info = pytorch_ex.implmap.get(bsym.sym.id)
        torch_op = None
        if impl_info is not None:
            torch_op = impl_info.symbol
            if impl_info.execution_transform is not None:
                return impl_info.execution_transform(*args, **kwargs)

        if torch_op is None:
            torch_op = pytorch_ex.opmap.get(bsym.sym.name)

        # this should be really rare, but type_as has this,
        # ideally we would be also handling more subsymbols here
        if torch_op is None and len(bsym.subsymbols) == 1:
            torch_op = pytorch_ex.opmap.get(bsym.subsymbols[0].sym.name)

        if torch_op is None:
            raise RuntimeError("op not found for {bsym.sym.name}")

        return torch_op(*args, **kwargs)

    return _to_torch


def make_compiled(
    bsyms: list[BoundSymbol], sorted_unique_inputs: list[Proxy], sorted_unique_outputs: list[Proxy]
) -> Callable:
    from thunder.executors.torchex import no_autocast
    from thunder.core.codeutils import SigInfo

    # Here we construct a trace that will be used to compile the function
    # TODO: maybe we should have a utility that does this properly
    region_trace = TraceCtx(None)
    region_trace.args = sorted_unique_inputs
    region_trace.kwargs = {}
    region_trace.names = {a.name for a in region_trace.args}
    with tracectx(region_trace):
        for a in sorted_unique_inputs:
            prims.unpack_trivial(a, name=a.name)

    region_trace.bound_symbols += list(bsyms)
    region_trace.bound_symbols.append(prims.python_return.bind(sorted_unique_outputs, output=None))
    for bsym in region_trace.bound_symbols:
        if bsym.sym == prims.unpack_trivial:
            continue
        for o in bsym.flat_outs:
            if o is not None:
                region_trace.add_name(o.name)
        for sbsym in bsym.subsymbols:
            for o in sbsym.flat_outs:
                if o is not None and o.name not in region_trace.names:
                    region_trace.add_name(o.name)

    # maybe make this the default if no sig info is present?
    region_trace._siginfo = SigInfo("to_be_compiled")
    region_trace._siginfo.args = [(a.name, None) for a in region_trace.args]

    torchex_trace = transform_for_execution(region_trace, executors_list=(pytorch_ex,))
    trace_callable = torchex_trace.python_callable(include_decorators=False)

    torch_compile_fullgraph: None | bool = get_compile_option(
        "torch_compile_fullgraph", "Whether to enable `fullgraph` from `torch.compile`. Defaults to `True`."
    )
    if torch_compile_fullgraph is None:
        torch_compile_fullgraph = True
    compiled_func = torch.compile(trace_callable, fullgraph=torch_compile_fullgraph)
    # For each of `@torch.no_grad(), and `torch.autocast(device_type="cpu"|"cuda")` torch.compile
    # create caches with a guard for the wrapped function. Since the torch.compile caches are per code object, not
    # frame, all the dynamic copies of these context managers share the same code cache.
    # Since Thunder generates many traces, all of them annotated with these context managers, we must put these context
    # managers outside the `torch.compile` region
    compiled_func = no_autocast(compiled_func)
    compiled_func = torch.no_grad()(compiled_func)

    def compiled_func_wrapper(*args):
        if _TORCH_GREATER_EQUAL_2_3:
            return compiled_func(*args)

        orig = getattr(torch._dynamo.eval_frame.guarded_backend_cache, "skip_backend_check_for_run_only_mode", None)
        try:
            # Dynamo doesn't recreate a guard for the compiled function called from the backward thread. This is a
            # problem because the guard is created with the forward thread ID, and the guard is not valid
            # for the backward thread. Issue filed: https://github.com/pytorch/pytorch/issues/114674
            torch._dynamo.eval_frame.guarded_backend_cache.skip_backend_check_for_run_only_mode = True
            return compiled_func(*args)
        finally:
            if orig is not None:
                torch._dynamo.eval_frame.guarded_backend_cache.skip_backend_check_for_run_only_mode = orig

    return compiled_func_wrapper


class TorchCompileExecutor(FusionExecutor):
    def __init__(self, name: Hashable, required_ops: set | None = None):
        super().__init__(name, version=torch.__version__)
        self.required_ops = required_ops

    def fuse(self, region: Region, fusion_counter: int) -> BoundSymbol:
        def keyfn(x: Variable) -> str:
            return x.proxy.name

        sorted_unique_inputs: list[Proxy] = [unvariableify(x) for x in region.inputs]
        sorted_unique_outputs: list[Proxy] = [unvariableify(x) for x in region.outputs]

        compiled: Callable = make_compiled(region.bound_symbols, sorted_unique_inputs, sorted_unique_outputs)

        fusion_name = f"TorchCompile{fusion_counter}"

        ctx = {fusion_name: compiled}

        fusion_sym = Symbol(fusion_name, meta=None, is_fusion=True, executor=self)
        fusion_bsym = BoundSymbol(
            fusion_sym, sorted_unique_inputs, {}, sorted_unique_outputs, region.bound_symbols, _call_ctx=ctx
        )

        return fusion_bsym

    def fusion_pass(self, trace: TraceCtx) -> TraceCtx:
        start_time_ns: int = time.perf_counter_ns()

        fusedtrace: TraceCtx = from_trace(trace)

        producers, consumers = utils.producers_and_consumers(trace)
        from thunder.executors.data_dependent_partition import fuse_bound_symbols, Node

        def _should_fuse(a: Node, b: Node):
            def _can_fuse_node(n: Node):
                if len(n.group_bsyms) > 1:
                    return True
                bsym: BoundSymbol = n.group_bsyms[0]
                return self.can_fuse(bsym)

            return _can_fuse_node(a) and _can_fuse_node(b)

        bound_symbol_groups = fuse_bound_symbols(trace, _should_fuse)

        fused_bsyms = []
        # Counts how many fusions (per executor) have been constructed
        fusion_counter: int = 0
        for bsyms in bound_symbol_groups:
            if len(bsyms) == 1:
                bsym: BoundSymbol = bsyms[0]
                if not self.can_fuse(bsym):
                    fused_bsyms.append(bsym)
                    continue

            # TODO: this could use `get_fuel()` like nvfuserex does
            if self.required_ops is None or any(bsym.sym.id in self.required_ops for bsym in bsyms):
                region = Region(producers, consumers, bsyms)
                fusion_bsym: BoundSymbol = self.fuse(region, fusion_counter)
                fusion_counter += 1
                fused_bsyms.append(fusion_bsym)
            else:
                fused_bsyms.extend(bsyms)

        fusedtrace.bound_symbols = fused_bsyms

        fusedtrace = rematerialize(fusedtrace)
        fusedtrace = dce(fusedtrace)
        fusedtrace = update_fusion_call_ctx(fusedtrace)

        end_time_ns: int = time.perf_counter_ns()
        elapsed_time_ns: int = end_time_ns - start_time_ns
        elapsed_time_millis: int = elapsed_time_ns // 1000000
        fusedtrace.set_provenance(TraceProvenance(f"Fusion (took {elapsed_time_millis} milliseconds)"))

        return fusedtrace


def cuda_device_checker(*args, **kwargs):
    # We only want to compile if all the TensorProxy arguments are on the GPU
    flat_args, _ = tree_flatten((args, kwargs))
    flat_tensorproxy_args = [x for x in flat_args if isinstance(x, TensorProxy)]
    for arg in flat_tensorproxy_args:
        if arg.device.type != "cuda":
            return False
    return True


# NOTE: [torch_compile_cat_ex vs torch_compile_ex]
# The former only relies on `torch.compile` for the operators where it shines the most and is meant to be used
# together with the nvfuser executor. Its current goal is only to fuse RoPE but the set of ops fused will change as each
# of the fusion backends evolve.
# The latter will try to `torch.compile` all the torch operators and is meant to be used without the nvfuser_executor
# since they would be competing over fusion opportunities. The advantage over simply doing `torch.compile` is that you
# still get all of Thunder's advantages, like enabling custom executors (e.g. with custom triton kernels) before it.
required_ops = {
    "torch.cat",
    prims.cat.id,
}
torch_compile_cat_ex = TorchCompileExecutor(name="torchcompile_cat", required_ops=required_ops)
register_executor(torch_compile_cat_ex)
# TODO: Carefully enable more ops checking that they do improve performance
supported_ops = {
    "torch.split",
    "torch.sum",
    prims.add.id,
    prims.broadcast_in_dim.id,
    prims.cat.id,
    prims.convert_element_type.id,
    prims.full.id,
    prims.mul.id,
    prims.neg.id,
    prims.pad.id,
    prims.reshape.id,
    prims.slice_prim.id,
    prims.transpose.id,
    # div and erf are used in GELU and are fused horizontally with RoPE when
    # parallel residual paths are used in the transformer block
    prims.div.id,
    prims.erf.id,
}
torch_compile_cat_ex._implmap = {
    op: ImplInfo(checker=cuda_device_checker) for op in pytorch_ex.implmap if op in supported_ops
}


torch_compile_ex = TorchCompileExecutor(name="torchcompile")
register_executor(torch_compile_ex)
torch_compile_ex._implmap = {op: ImplInfo() for op in pytorch_ex.implmap}
