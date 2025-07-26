"""Executor for `torch.library.custom_op` operators"""

from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.extend import OperatorExecutor

if TYPE_CHECKING:
    pass


__all__ = [
    "custom_op_ex",
]


custom_op_ex = OperatorExecutor("custom_op")
