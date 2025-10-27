"""Executor for `torch.library.custom_op` operators"""

from __future__ import annotations

from thunder.extend import OperatorExecutor
from thunder.extend import register_executor


__all__ = [
    "custom_op_ex",
]


custom_op_ex = OperatorExecutor("custom_op")
register_executor(custom_op_ex)
