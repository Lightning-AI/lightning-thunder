# Constructed by Backward pass
import thunder
import thunder.torch as ltorch
import thunder.torch.experimental.dtensor_prims_and_impl
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def backward_fn(saved_for_backward, cotangents):
  # saved_for_backward: "Collection"
  # cotangents: "Collection"
  C0, C1, = saved_for_backward
  # C0: "Collection"
  # C1: "Collection"
  t2, = cotangents
  # t2: "DTensor cuda:0 f32[16, 16]"
  t20, t21, = C0
  # t20: "cuda:0 f32[8, 16]"
  # t21: "cuda:0 f32[8, 16]"
  # C1 (empty sequence)
  t11 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(t2)  # t11: "cuda:0 f32[8, 16]"
  t13 = ltorch.mul(t21, t11)  # t13: "cuda:0 f32[8, 16]"
    # t13 = prims.mul(t21, t11)  # t13: "cuda:0 f32[8, 16]"
  t14 = ltorch.mul(t20, t11)  # t14: "cuda:0 f32[8, 16]"
    # t14 = prims.mul(t20, t11)  # t14: "cuda:0 f32[8, 16]"
  t16 = thunder.torch.experimental.dtensor_prims_and_impl.construct_dtensor(t14, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # t16: "DTensor cuda:0 f32[16, 16]"
  t18 = thunder.torch.experimental.dtensor_prims_and_impl.construct_dtensor(t13, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # t18: "DTensor cuda:0 f32[16, 16]"
  return (t18, None)