from thunder import Transform
from thunder.executors.passes import del_last_used
from thunder.core.trace import from_trace, tracectx
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy, variableify

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface

from thunder.executors.functional_teex import _te_fp8_synchronization


class TESynchronizationTransform(Transform):
    def __init__(self):
        self.fp8_recipe = None
        self.handling_forward_trace = None

    def transform_trace_post_optimization(self, computation_trace, **kwargs):
        if "functional_te" not in map(lambda x: x.name, kwargs["executors_list"]):
            return computation_trace

        sync_states = []
        last_fp8_linear_idx = 0
        new_trace = from_trace(computation_trace)
        new_bsyms = []

        swap_map: dict[VariableInterface, TensorProxy] = {}

        for bsym in computation_trace.bound_symbols:
            if bsym.sym.name == "te_fp8_recipe":
                if not self.fp8_recipe:
                    self.fp8_recipe = bsym
                else:
                    swap_map[variableify(bsym.output)] = self.fp8_recipe.output

            elif "te_fp8_state" in bsym.sym.name:
                sync_states += [bsym.output]

            elif bsym.sym.id == PrimIDs.DEL:
                continue

            new_bsym = bsym.from_bsym_swap_proxies(swap_map)
            new_bsyms.append(new_bsym)

            if bsym.sym.name == "te_functional_linear_fwd":
                last_fp8_linear_idx = len(new_bsyms)
                self.handling_forward_trace = True
            elif bsym.sym.name == "te_functional_linear_bwd":
                last_fp8_linear_idx = len(new_bsyms)
                self.handling_forward_trace = False

        # couldn't find any TE in the trace
        if not self.fp8_recipe:
            return computation_trace

        half_trace = new_bsyms[:last_fp8_linear_idx]
        new_trace.bound_symbols = half_trace

        with tracectx(new_trace):
            _te_fp8_synchronization(self.fp8_recipe.output, *sync_states, forward=self.handling_forward_trace)

        new_trace.bound_symbols.extend(new_bsyms[last_fp8_linear_idx:])

        return del_last_used(new_trace)
