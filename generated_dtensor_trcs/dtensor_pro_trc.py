import thunder
import thunder.core.prims as prims
import thunder.torch.experimental.dtensor_prims_and_impl
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def prologue(*args, **kwargs):
  # args: "Any"
  prims.check_len(args, 2)
  # kwargs: "Any"
  prims.check_len(kwargs, 0)
  l_x_: "DTensor cuda:0 f32[16, 16]" = args[0]
  l_w_: "DTensor cuda:0 f32[16, 16]" = args[1]
  dtensor_spec0: "<class 'NoneType'>" = l_x_._spec
  thunder.torch.experimental.dtensor_prims_and_impl.check_dtensor_spec_repr(dtensor_spec0, "DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=torch.Size([16, 16]), stride=(16, 1), dtype=torch.float32))")
  t1: "cuda:0 f32[8, 16]" = l_x_._local_tensor
  prims.check_tensor_shape_and_metadata(t1, (8, 16), 'cuda:0', torch.float32, True)
  prims.check_tensor_shape_and_metadata(l_x_, (16, 16), 'cuda:0', torch.float32, True)
  dtensor_spec2: "<class 'NoneType'>" = l_w_._spec
  thunder.torch.experimental.dtensor_prims_and_impl.check_dtensor_spec_repr(dtensor_spec2, "DTensorSpec(mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=torch.Size([16, 16]), stride=(16, 1), dtype=torch.float32))")
  t3: "cuda:0 f32[8, 16]" = l_w_._local_tensor
  prims.check_tensor_shape_and_metadata(t3, (8, 16), 'cuda:0', torch.float32, False)
  prims.check_tensor_shape_and_metadata(l_w_, (16, 16), 'cuda:0', torch.float32, False)
  cache_info: "Any" = thunder._get_cache_info()
  cache_info_default_dtype: "<class 'torch.dtype'>" = cache_info['default_dtype']
  prims.check_literal_like(cache_info_default_dtype, torch.float32)
  cache_info_default_device: "<class 'torch.device'>" = cache_info['default_device']
  prims.check_literal_like(cache_info_default_device, torch.device("cpu"))
  cache_info_is_autocast_enabled: "bool False" = cache_info['is_autocast_enabled']
  prims.check_number_type_and_value(cache_info_is_autocast_enabled, False)
  cache_info_alias_tensor_indices: "str" = cache_info['alias_tensor_indices']
  prims.check_string_value(cache_info_alias_tensor_indices, '')
  cache_info_is_grad_enabled: "bool True" = cache_info['is_grad_enabled']
  prims.check_number_type_and_value(cache_info_is_grad_enabled, True)
  cache_info_no_grad_sync: "bool False" = cache_info['no_grad_sync']
  prims.check_number_type_and_value(cache_info_no_grad_sync, False)
  return ((l_x_, l_w_), ())