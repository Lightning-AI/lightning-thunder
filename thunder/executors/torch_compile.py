import time
from collections.abc import Callable, Hashable
from typing import Any

import torch

from thunder.core import prims, utils
from thunder.core.proxies import Proxy, unvariableify, Variable
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance

from thunder.executors.utils import Region
from thunder.extend import FusionExecutor, register_executor


def to_torch_translator(bsym: BoundSymbol) -> Callable:
    """Translates a BoundSymbol to a corresponding traceable by Thunder and
    executable by PyTorch callable.

    Args:
        bsym: The BoundSymbol to translate.

    Returns:
        A callable that can be executed by PyTorch after being traced by Thunder.
    """
    from thunder.executors.torchex import ex as torchex

    def _to_torch(*args, **kwargs) -> Any:
        impl_info = torchex.implmap.get(bsym.sym.id)
        torch_op = None
        if impl_info is not None:
            torch_op = impl_info.symbol
            if impl_info.execution_transform is not None:
                return impl_info.execution_transform(*args, **kwargs)

        if torch_op is None:
            torch_op = torchex.opmap[bsym.sym.name]

        return torch_op(*args, **kwargs)

    return _to_torch


def make_compiled(
    bsyms: list[BoundSymbol], sorted_unique_inputs: list[Proxy], sorted_unique_outputs: list[Proxy]
) -> Callable:
    from thunder import trace
    from thunder.core.transforms import eval_trace

    # Here we construct a trace that will be used to compile the function
    region_trace = TraceCtx(None)
    region_trace.bound_symbols = list(bsyms)
    region_trace.args = sorted_unique_inputs
    region_trace.kwargs = {}
    region_trace.bound_symbols.append(prims.python_return.bind(sorted_unique_outputs, output=()))

    def torch_interpreted_func(*args):
        return eval_trace(region_trace, *args, symbol_mapper=to_torch_translator)

    # Here instead of using thunder.trace we could use torch_trace =
    # passes._transform_for_operator_executor_execution(region_trace, [torchex])
    # but then we would need to handle unpacking of the args explicitly For
    # example with:
    # try:
    #     token = set_tracectx(region_trace)
    #     col = CollectionProxy(region_trace.args, name="args")
    #     _ = prims.unpack_sequence(col, len(region_trace.args))
    # finally:
    #     reset_tracectx(token)
    #     region_trace.bound_symbols.extend(bsyms)
    # But there are some issues with the
    # _transform_for_operator_executor_execution implementation that need to be
    # fixed first. One issue is that it doesn't maintain the ssa form of the
    # trace, which is needed for all the passes to work correctly.
    # TODO: issue "Try using _transform_for_operator_executor_execution for
    # torch.compile executor"
    torch_trace = trace(inline_trace=False)(torch_interpreted_func, *sorted_unique_inputs)

    # Avoid cache limits. For each of `@torch.no_grad(), and `torch.autocast(device_type="cpu"|"cuda")` torch.compile
    # create caches with a guard for the wrapped function. Since the torch.compile caches are per code object, not
    # frame, all the dynamic copies of these context managers share the same code cache.
    # Since Thunder generates many traces, all of them annotated with these context managers, we need to increase
    # the limit here. Alternatively we could pull out the context managers outside of the `torch.compile` region
    torch._dynamo.reset()

    return torch.compile(torch_trace.python_callable(), fullgraph=True)


class TorchCompileExecutor(FusionExecutor):
    def __init__(self):
        super().__init__("torchcompile", version=torch.__version__)

    def fuse(self, region: Region, fusion_counter: int) -> BoundSymbol:
        def keyfn(x: Variable) -> str:
            return x.proxy.name

        sorted_unique_inputs: list[Proxy] = list(unvariableify(x) for x in sorted(region.inputs, key=keyfn))
        sorted_unique_outputs: list[Proxy] = list(unvariableify(x) for x in sorted(region.outputs, key=keyfn))

        compiled: Callable = make_compiled(region.bound_symbols, sorted_unique_inputs, sorted_unique_outputs)

        fusion_name = f"TorchCompile{fusion_counter}"

        ctx = {fusion_name: compiled}

        fusion_sym = Symbol(fusion_name, meta=None, is_fusion=True, executor=self)
        fusion_bsym = BoundSymbol(
            fusion_sym, sorted_unique_inputs, {}, sorted_unique_outputs, region.bound_symbols, _call_ctx=ctx
        )

        return fusion_bsym

    def fusion_pass(self, trace: TraceCtx) -> TraceCtx:
        start_time_ns: int = time.time_ns()

        fusedtrace: TraceCtx = from_trace(trace)

        producers, consumers = utils.producers_and_consumers(trace)
        from thunder.executors.data_dependent_partition import fuse_bound_symbols, Node

        fused_bsyms = []

        # NOTE: Currently the goal is to fuse rotary positional embeddings
        required_ops = {
            "torch.cat",
            prims.cat.id,
            prims.pad.id,
            prims.slice_prim.id,
        }

        def _should_fuse(a: Node, b: Node):
            def _can_fuse_node(n: Node):
                if len(n.group_bsyms) > 1:
                    return True
                bsym: BoundSymbol = n.group_bsyms[0]
                return self.can_fuse(bsym)

            return _can_fuse_node(a) and _can_fuse_node(b)

        bound_symbol_groups = fuse_bound_symbols(trace, _should_fuse)

        # Counts how many fusions (per executor) have been constructed
        fusion_counter: int = 0
        for bsyms in bound_symbol_groups:
            if len(bsyms) == 1:
                bsym: BoundSymbol = bsyms[0]
                if not self.can_fuse(bsym):
                    fused_bsyms.append(bsym)
                    continue

            include_required_ops = any(bsym.sym.id in required_ops for bsym in bsyms)
            if include_required_ops:
                region = Region(producers, consumers, bsyms)
                fusion_bsym: BoundSymbol = self.fuse(region, fusion_counter)
                fusion_counter += 1
                fused_bsyms.append(fusion_bsym)
            else:
                fused_bsyms.extend(bsyms)

        fusedtrace.bound_symbols = fused_bsyms

        end_time_ns: int = time.time_ns()
        elapsed_time_ns: int = end_time_ns - start_time_ns
        elapsed_time_millis: int = elapsed_time_ns // 1000000
        fusedtrace.set_provenance(TraceProvenance(f"Fusion (took {elapsed_time_millis} milliseconds)"))

        return fusedtrace


torch_compile_executor = TorchCompileExecutor()
register_executor(torch_compile_executor)


def register_supported(id: Hashable):
    checker = lambda *args, **kwargs: True
    torch_compile_executor.register_supported(id, checker)


# This is an initial list to support rotary positional embeddings
# TODO: Carefully enable more ops checking that they do improve performance
supported_ops = {
    "torch.split",
    prims.add,
    prims.broadcast_in_dim,
    prims.cat,
    prims.convert_element_type,
    prims.full,
    prims.mul,
    prims.neg,
    prims.pad,
    prims.reshape,
    prims.slice_prim,
    prims.transpose,
}

for op in supported_ops:
    register_supported(op)
