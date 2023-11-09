from typing import Optional, Any, List, Tuple
from collections.abc import Sequence

import thunder.executors.passes as passes

import thunder.extend as extend


# NOTE The executors submodule depends on the extend submodule

__all__ = [
    "passes",
    "get_torch_executor",
    "get_nvfuser_executor",
    "nvfuser_available",
]


def get_nvfuser_executor() -> None | extend.Executor:
    return extend.get_executor("nvfuser")


def get_torch_executor() -> extend.Executor:
    return extend.get_executor("torch")


def nvfuser_available() -> bool:
    return get_nvfuser_executor() is not None
