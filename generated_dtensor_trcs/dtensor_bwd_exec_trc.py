# Constructed by Delete Last Used (took 0 milliseconds)
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def backward_fn(saved_for_backward, cotangents):
  # saved_for_backward: "Collection"
  # cotangents: "Collection"
  C0, _, = saved_for_backward
  # C0: "Collection"
  # None
  clear_mutable_collection(saved_for_backward)
  del saved_for_backward
  t2, = cotangents
  # t2: "DTensor cuda:0 f32[16, 16]"
  clear_mutable_collection(cotangents)
  del cotangents
  t21, = C0
  # t21: "cuda:0 f32[8, 16]"
  clear_mutable_collection(C0)
  del C0
  bw_t25 = get_dtensor_inner_tensor(t2)  # bw_t25: "cuda:0 f32[8, 16]"
    # bw_t25 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(t2)  # bw_t25: "cuda:0 f32[8, 16]"
  del t2
  [bw_t13] = nvFusion0(t21, bw_t25)
    # bw_t13 = prims.mul(t21, bw_t25)  # bw_t13: "cuda:0 f32[8, 16]"
  del t21, bw_t25
  bw_t27 = construct_dtensor(bw_t13, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # bw_t27: "DTensor cuda:0 f32[16, 16]"
    # bw_t27 = thunder.torch.experimental.dtensor_prims_and_impl.construct_dtensor(bw_t13, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # bw_t27: "DTensor cuda:0 f32[16, 16]"
  del bw_t13
  return (bw_t27, None)