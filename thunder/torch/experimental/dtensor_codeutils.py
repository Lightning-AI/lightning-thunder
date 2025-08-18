from typing import Any
import torch
from thunder.torch.experimental.dtensor_utils import run_only_if_distributed_is_available

if torch.distributed.is_available():
    from torch.distributed.tensor._dtensor_spec import DTensorSpec


@run_only_if_distributed_is_available
def is_dtensor_spec(x: Any) -> bool:
    return isinstance(x, DTensorSpec)
