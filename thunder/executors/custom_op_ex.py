"""Executor for `torch.library.custom_op` operators"""

from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core import baseutils
from thunder.extend import OperatorExecutor
from thunder.extend import register_executor

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
    from thunder.core.symbol import Symbol


__all__ = [
    "custom_op_ex",
    "_override_custom_op_forward",
]


# TODO: Implement a function which allows customizing backward.
def _override_custom_op_forward(
    symbol: Symbol,
    executable_definition: Callable[[Any], Any],
) -> None:
    """Override the forward implementation of ``symbol`` using ``executable_definition``.

    Args:
        symbol: This should be the symbol from :func:`thunder.torch.custom_op._register_custom_op`.
        executable_definition:
            Custom forward definition that has the same signature as the custom op of ``symbol``.
    """
    baseutils.check(
        symbol.id in custom_op_ex._implmap,
        lambda: f"{symbol=} is not found in {custom_op_ex._implmap}",
    )

    custom_def = custom_op_ex.register_operator(
        f"custom_{symbol.name}_forward",
        meta=symbol.meta,
        fn=executable_definition,
    )
    implinfo = custom_op_ex._implmap[symbol.id]
    implinfo.execution_transform = custom_def


custom_op_ex = OperatorExecutor("custom_op")
register_executor(custom_op_ex)
