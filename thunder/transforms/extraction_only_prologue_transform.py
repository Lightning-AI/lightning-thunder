from thunder.core.prims import PrimIDs
from thunder.core.proxies import ProxyTag
from thunder.core.trace import from_trace
from thunder.core.transform_common import Transform


__all__ = [
    "ExtractionOnlyPrologueTransform",
]


class ExtractionOnlyPrologueTransform(Transform):
    """Exclude :func:`~thunder.core.prims.check_tensor_shape_and_metadata` from prologue trace of ThunderCompiler.

    This transform is mainly used by :class:`~thunder.dynamo.ThunderCompiler` to remove the check of input tensors
    when either they are :class:`torch.nn.Parameter` or all of them don't have any symbolic shape.

    Args:
        skip_check_on_input_tensors: If :obj:`True`, remove all the check from the prologue trace as TorchDynamo caching would do enough.
            Otherwise, remove the checks of tensor proxies with ``ProxyTag.STATIC_MEMORY_LOCATION``.
    """

    def __init__(self, skip_check_on_input_tensors: bool = False):
        self.skip_check_on_input_tensors = skip_check_on_input_tensors

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        new_prologue_trace = from_trace(prologue_trace)
        new_bsyms = []

        for bsym in prologue_trace.bound_symbols:
            # NOTE - We assume TensorProxy's tagged with `STATIC_MEMORY_LOCATION` to
            #        be Parameters or Buffer. It should be safe to disable check for
            #        tensors we deem to be static.
            if bsym.sym.id == PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA and (
                self.skip_check_on_input_tensors or ProxyTag.STATIC_MEMORY_LOCATION in bsym.args[0].tags
            ):
                continue

            new_bsyms.append(bsym)

        new_prologue_trace.bound_symbols = new_bsyms

        new_prologue_trace.set_provenance("Extraction only prologue pass")
        return new_prologue_trace, computation_trace, epilogue_trace
