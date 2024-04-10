import time
from collections.abc import Callable, Hashable
from typing import Any

import torch

from thunder.core import prims, utils
from thunder.core.proxies import Proxy, unvariableify, Variable
from thunder.core.rematerialization import rematerialize
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance
from thunder.core.transform_common import dce
from thunder.executors.passes import update_fusion_call_ctx
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
    compiled_func = torch.compile(torch_trace.python_callable())

    def compiled_func_wrapper(*args):
        # PyTorch 2.1 doesn't have this attribute
        if getattr(torch._dynamo.eval_frame, "guarded_backend_cache", None) is None:
            return compiled_func(*args)

        orig = getattr(torch._dynamo.eval_frame.guarded_backend_cache, "skip_backend_check_for_run_only_mode", None)
        try:
            # TODO: Remove this hack
            # Dynamo doesn't recreate a guard for the compiled function called
            # from the backward thread. This is a problem because the guard is
            # created with the forward thread ID, and the guard is not valid
            # for the backward thread.
            # Issue filed: https://github.com/pytorch/pytorch/issues/114674
            # We should be able to remove this hack once we're sure that the
            # above fix has propagated to all supported PyTorch releases.
            torch._dynamo.eval_frame.guarded_backend_cache.skip_backend_check_for_run_only_mode = True
            return compiled_func(*args)
        finally:
            if orig is not None:
                torch._dynamo.eval_frame.guarded_backend_cache.skip_backend_check_for_run_only_mode = orig

    return compiled_func_wrapper


class TorchCompileExecutor(FusionExecutor):
    def __init__(self, name: Hashable = "torchcompile", required_ops: set | None = None):
        super().__init__(name, version=torch.__version__)
        self.required_ops = required_ops

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

        end_time_ns: int = time.time_ns()
        elapsed_time_ns: int = end_time_ns - start_time_ns
        elapsed_time_millis: int = elapsed_time_ns // 1000000
        fusedtrace.set_provenance(TraceProvenance(f"Fusion (took {elapsed_time_millis} milliseconds)"))

        return fusedtrace


from thunder.executors.torchex import ex as pytorch_executor


# NOTE: [torch_compile_executor vs torch_compile_complete_executor]
# The former only relies on `torch.compile` for the operators where it shines the most and is meant to be used
# together with the nvfuser executor. Its current goal is only to fuse RoPE but the set of ops fused will change as each
# of the fusion backends evolve.
# The latter will try to `torch.compile` all the torch operators and is meant to be used without the nvfuser_executor
# since they would be competing over fusion opportunities. The advantage over simply doing `torch.compile` is that you
# still get all of Thunder's advantages, like enabling custom executors (e.g. with custom triton kernels) before it.
required_ops = {
    "torch.cat",
    prims.cat.id,
    prims.pad.id,
    prims.slice_prim.id,
}
torch_compile_executor = TorchCompileExecutor(required_ops=required_ops)
register_executor(torch_compile_executor)
# TODO: Carefully enable more ops checking that they do improve performance
supported_ops = {
    "torch.split",
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
}
assert supported_ops - pytorch_executor.implmap.keys() == set()  # sanity check  # FIXME convert into a test
torch_compile_executor._implmap = {op: info for op, info in pytorch_executor.implmap.items() if op in supported_ops}


torch_compile_complete_executor = TorchCompileExecutor(name="torchcompile_complete")
register_executor(torch_compile_complete_executor)
torch_compile_complete_executor._implmap = dict(pytorch_executor.implmap)
