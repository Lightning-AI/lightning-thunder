# Constructed by Delete Last Used (took 0 milliseconds)
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(l_x_, l_w_):
  # l_x_: "DTensor cuda:0 f32[16, 16]"
  # l_w_: "DTensor cuda:0 f32[16, 16]"

  # <eval_with_key>.10:5: 	    mul = torch.mul(l_x_, l_w_);  l_x_ = l_w_ = None
  t20 = get_dtensor_inner_tensor(l_x_)  # t20: "cuda:0 f32[8, 16]"
    # t20 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(l_x_)  # t20: "cuda:0 f32[8, 16]"
  t21 = get_dtensor_inner_tensor(l_w_)  # t21: "cuda:0 f32[8, 16]"
    # t21 = thunder.torch.experimental.dtensor_prims_and_impl.get_dtensor_inner_tensor(l_w_)  # t21: "cuda:0 f32[8, 16]"
  [t10] = nvFusion0(t20, t21)
    # t10 = prims.mul(t20, t21)  # t10: "cuda:0 f32[8, 16]"
  del t20

  # <eval_with_key>.10:5: 	    mul = torch.mul(l_x_, l_w_);  l_x_ = l_w_ = None
  t23 = construct_dtensor(t10, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # t23: "DTensor cuda:0 f32[16, 16]"
    # t23 = thunder.torch.experimental.dtensor_prims_and_impl.construct_dtensor(t10, DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=(16, 16), stride=(16, 1), dtype=torch.float32)))  # t23: "DTensor cuda:0 f32[16, 16]"
  del t10
  return {'output': (t23,), 'flat_args': [l_x_, l_w_], 'flat_output': (t23,)}, ((t21,), ())