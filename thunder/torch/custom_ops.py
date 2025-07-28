from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core.symbol import Symbol
from thunder.core.prims import OpTags

if TYPE_CHECKING:
    from torch._library.custom_ops import CustomOpDef
    from torch._ops import OpOverload
    from torch._ops import OpOverloadPacket


__all__ = [
    "register_custom_op",
]


# TODO: Interpret backward
# TODO: Figure out if we can access `triton_op` of the same name
def register_custom_op(custom_op: CustomOpDef) -> Symbol:
    """Register :func:`~torch.library.custom_op`'ed function to Thunder.

    Args:
        custom_op: :func:`torch.library.custom_op`'ed function. This is not ``torch.ops.{namespace}.{name}``.

    Returns:
        :class:`~thunder.core.symbol.Symbol`: A symbol representing the input ``custom_op``.
    """
    from thunder.core.langctxs import langctx, Languages
    from thunder.executors.torchex import _always_executable
    from thunder.torch import meta_adaptor, _torch_to_thunder_function_map
    from thunder.executors.custom_op_ex import custom_op_ex

    # `custom_op` is `custom_op(name)(my_func)`,
    # `torch.ops.namespace.name` is `OpOverloadPacket.`
    # e.g. `torch.ops.my_lib.foo` is OpOverloadPacket and `torch.ops.my_lib.foo.default` is OpOverload.
    torch_opoverload: OpOverload = custom_op._opoverload
    torch_opoverload_packet: OpOverloadPacket = torch_opoverload._overloadpacket

    meta_fn = langctx(Languages.TORCH)(meta_adaptor(custom_op._abstract_fn))
    fn_name = custom_op._qualname.replace("::", "_")
    op_id = f"torch::ops::{custom_op._qualname}".replace("::", ".")
    symbol = Symbol(
        name=fn_name,
        meta=meta_fn,
        id=op_id,
        is_prim=False,
        # NOTE: Especially when this `custom_op` doesn't have backward, and the caller program
        # involves parameter lifting, somehow the bsym of this custom_op seems to be removed
        # by `thunder/transforms/autodiff.py`'s `AugmentedForwardProcessor` of `grad_transform_on_trace`
        # So this tag marks the bsyms so that the processor does't see them as "constant" for VJP.
        tags=(OpTags.TORCH_COMPILE_COMPLIANT_CUSTOM_OP,),
    )
    op = custom_op_ex.register_operator(fn_name, meta=meta_fn, fn=torch_opoverload)
    custom_op_ex.register_implementation(symbol, op, checker=_always_executable)
    _torch_to_thunder_function_map[torch_opoverload_packet] = symbol
    return symbol
