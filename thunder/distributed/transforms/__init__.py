import torch

if torch.distributed.is_available():
    from .ddp import apply_bucketing_to_grad_allreduce
    from .fsdp import FSDPCommBucketing
else:
    apply_bucketing_to_grad_allreduce = None
    FSDPCommBucketing = None

__all__ = [
    "apply_bucketing_to_grad_allreduce",
    "FSDPCommBucketing",
]
