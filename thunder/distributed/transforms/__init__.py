import torch
if torch.distributed.is_available():
    from .ddp import optimize_allreduce_in_ddp_backward
    from .fsdp import FSDPCommBucketing
else:
    optimize_allreduce_in_ddp_backward = None
    FSDPCommBucketing = None

__all__ = [
    "optimize_allreduce_in_ddp_backward",
    "FSDPCommBucketing",
]
