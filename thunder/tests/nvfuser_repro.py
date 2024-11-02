# CUDA devices:
#  0: NVIDIA RTX 6000 Ada Generation
# torch version: 2.6.0a0+git408fe41
# cuda version: 12.6
# nvfuser version: 0.2.11+gitaad7286
import torch
from nvfuser import FusionDefinition, DataType


def nvfuser_fusion_id0(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
    )
    T1 = fd.define_tensor(
        shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
    )
    T2 = fd.ops.add(T0, T1)
    fd.add_output(T2)


with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn(4, dtype=torch.float32, device="cuda:0").as_strided((2, 2), (2, 1)),
    torch.randn(4, dtype=torch.float32, device="cuda:0").as_strided((2, 2), (2, 1)),
]
fd.execute(inputs)
