import thunder
from thunder.core.trace import from_trace
from thunder.core.proxies import ProxyTag


class ExtractionOnlyPrologueTransform(thunder.Transform):
    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        new_prologue_trace = from_trace(prologue_trace)
        new_bsyms = []

        for bsym in prologue_trace.bound_symbols:
            # NOTE - We assume TensorProxy's tagged with `STATIC_MEMORY_LOCATION` to
            #        be Parameters or Buffer. It should be safe to disable check for
            #        tensors we deem to be static.
            if (
                bsym.sym.id == thunder.prims.PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA
                and ProxyTag.STATIC_MEMORY_LOCATION in bsym.args[0].tags
            ):
                continue

            new_bsyms.append(bsym)

        new_prologue_trace.bound_symbols = new_bsyms

        new_prologue_trace.set_provenance("Extraction only prologue pass")
        return new_prologue_trace, computation_trace, epilogue_trace
