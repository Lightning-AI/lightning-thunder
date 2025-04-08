import time
from typing import TYPE_CHECKING

from thunder import Transform
from thunder.core import prims
from thunder.core.proxies import Proxy, Variable, unvariableify
from thunder.core.trace import from_trace, tracectx, TraceProvenance, TraceTag
from thunder.core.transforms import (
    _update_forward_with_new_saved_for_backward,
    _update_backward_with_new_saved_for_backward,
)
from thunder.core.transform_common import cse_single_bsym
from thunder.executors.passes import del_last_used


if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface
    from thunder.core.symbol import BoundSymbolRHS, BoundSymbol
    from thunder.core.proxies import TensorProxy

from thunder.executors.functional_teex import _te_fp8_synchronization


class TESynchronizationTransform(Transform):
    def __init__(self):
        self.fp8_recipe = None
        self.swap_map: dict[VariableInterface, TensorProxy] = {}
        self.rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbol] = {}
        self.redundant_map: dict[Variable, Proxy] = {}
        self.new_saved_for_backward = None

    def transform_trace_post_optimization(self, computation_trace, **kwargs):
        if "functional_te" not in map(lambda x: x.name, kwargs["executors_list"]):
            return computation_trace

        start_time_ns = time.perf_counter_ns()

        sync_states = []
        last_fp8_linear_idx = 0
        new_trace = from_trace(computation_trace)
        new_bsyms = []

        for bsym in computation_trace.bound_symbols:
            # Need to understad if it's forward or backward to remove the extra recipies from the saved for backward list.
            if bsym.sym.name == "te_fp8_recipe" and not self.fp8_recipe:
                self.fp8_recipe = bsym

            elif "te_fp8_state" in bsym.sym.name:
                sync_states += [bsym.output]

            elif bsym.sym.id == prims.PrimIDs.DEL:
                continue

            elif bsym.sym.id == prims.PrimIDs.RETURN:
                handling_forward_trace = len(bsym.args) == 2

            if "te_functional_linear" in bsym.sym.name:
                last_fp8_linear_idx = len(new_bsyms)

            if bsym.sym.is_fusion:
                new_bsym = bsym.from_bsym_swap_proxies(self.redundant_map)
            else:
                new_bsym = cse_single_bsym(self.redundant_map, self.rhs_to_bsym_map, bsym)

            if new_bsym:
                new_bsyms.append(new_bsym)

        # couldn't find any TE in the trace
        if not self.fp8_recipe:
            return computation_trace

        half_trace = new_bsyms[:last_fp8_linear_idx]
        new_trace.bound_symbols = half_trace

        with tracectx(new_trace):
            _te_fp8_synchronization(self.fp8_recipe.output, *sync_states, forward=handling_forward_trace)

        new_trace.bound_symbols.extend(new_bsyms[last_fp8_linear_idx:])

        if self.new_saved_for_backward:
            _update_backward_with_new_saved_for_backward(new_trace, self.new_saved_for_backward)

        # If the trace has been generated by thunder autograd then we need to remove extra recipies from the return statement
        if TraceTag.AUGMENTED_FORWARD in computation_trace.tags:
            return_bsym = new_trace.bound_symbols[-1]
            assert return_bsym.sym.id == prims.PrimIDs.RETURN
            _, (saved_for_backward, env) = return_bsym.args
            unique_env = {Variable(x) for x in env}
            self.new_saved_for_backward = (*saved_for_backward, *(unvariableify(x) for x in unique_env))

            _update_forward_with_new_saved_for_backward(new_trace, self.new_saved_for_backward)

        sync_trace = del_last_used(new_trace)

        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000

        sync_trace.set_provenance(
            TraceProvenance(f"TransformerEngine Synchronization transform (took {elapsed_time_millis} milliseconds)")
        )

        return sync_trace
