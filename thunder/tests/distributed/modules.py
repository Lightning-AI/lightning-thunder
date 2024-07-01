from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

import torch.nn as nn

from thunder.core import utils

if TYPE_CHECKING:
    import torch


__all__ = [
    "ParallelMLP",
]


class ParallelMLP(nn.Module):
    """Simplified version of Megatron/NeMo's ParallelMLP.

    Ref: https://github.com/NVIDIA/NeMo/blob/95ca2f4/nemo/collections/nlp/modules/common/megatron/mlp.py#L61
    """

    COLUMN_WISE: ClassVar[tuple[str]] = ("dense_h_to_4h",)
    ROW_WISE: ClassVar[tuple[str]] = ("dense_4h_to_h",)

    SUPPORTED_GELU_APPROX: ClassVar[tuple[str, str]] = ("none", "tanh")

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int | None = None,
        bias: bool = True,
        gelu_approximate: str = "none",
    ) -> None:
        utils.check(
            gelu_approximate in ParallelMLP.SUPPORTED_GELU_APPROX,
            lambda: f"Invalid {gelu_approximate}, supported are {ParallelMLP.SUPPORTED_GELU_APPROX}",
        )
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, ffn_hidden_size, bias=bias)
        self.dense_4h_to_h = nn.Linear(ffn_hidden_size, hidden_size, bias=bias)
        self.gelu = nn.GELU(approximate=gelu_approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        four_h = self.gelu(self.dense_h_to_4h(x))
        h = self.dense_4h_to_h(four_h)
        return h
