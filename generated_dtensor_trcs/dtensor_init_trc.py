import thunder
import thunder.torch.experimental.dtensor_torch_and_aten_ops
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(l_x_, l_w_):
  # l_x_: "DTensor cuda:0 f32[16, 16]"
  # l_w_: "DTensor cuda:0 f32[16, 16]"

  # <eval_with_key>.10:5: 	    mul = torch.mul(l_x_, l_w_);  l_x_ = l_w_ = None
  mul = thunder.torch.experimental.dtensor_torch_and_aten_ops.dtensor_mul(l_x_, l_w_)  # mul: "DTensor cuda:0 f32[16, 16]"
    # t4 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(l_x_)  # t4: "cuda:0 f32[8, 16]"
    # t5 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(l_w_)  # t5: "cuda:0 f32[8, 16]"
    # t0 = thunder.torch.experimental.dtensor_torch_and_aten_ops.aten_mul(t4, t5)  # t0: "cuda:0 f32[8, 16]"
      # t0 = prims.mul(t4, t5)  # t0: "cuda:0 f32[8, 16]"
    # mul = thunder.torch.experimental.dtensor_prims_and_impl.construct_dtensor(t0, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # mul: "DTensor cuda:0 f32[16, 16]"
  return (mul,)