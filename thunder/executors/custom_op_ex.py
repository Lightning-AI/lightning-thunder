"""Executor for `torch.library.custom_op` operators"""

from thunder.extend import OperatorExecutor


__all__ = [
    "custom_op_ex",
]


custom_op_ex = OperatorExecutor("custom_op")
