import torch.distributed

from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if torch.distributed.is_available():
    from thunder.distributed.tensor_parallel.column_wise import column_parallel
    from thunder.distributed.tensor_parallel.row_wise import row_parallel
else:
    column_parallel = None
    row_parallel = None


__all__ = [
    "TensorParallelLayerType",
    "column_parallel",
    "row_parallel",
]
