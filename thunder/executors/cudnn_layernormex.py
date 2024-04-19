from typing import Any

from looseversion import LooseVersion
import torch
import numpy as np

from lightning_utilities.core.imports import package_available

cudnn: None | Any = None
cudnn_backend_version: None | Any = None
if package_available("cudnn"):
    import cudnn

    cudnn_backend_version = cudnn.backend_version()


# WARNING: cudnn layernorm executor is experimental. Tests that use cudnn might fail.
from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Dict

from thunder.torch import TensorLike
import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy

from thunder.executors.cudnnex import CudnnTensorAttributes, torch_to_cudnn_dtype, CudnnexLRUCache, _get_cudnn_handle


from thunder.extend import OperatorExecutor, register_executor

cudnn_layernorm_ex: OperatorExecutor = OperatorExecutor("cudnn_layernorm", version=cudnn_backend_version)
register_executor(cudnn_layernorm_ex)

_cudnn_layernormex_cache = CudnnexLRUCache(maxlen=1024)


def _make_cudnn_layer_norm_graph(a, weight, bias):
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(a.device_index),
    )

    input = graph.tensor(name="input", dim=a.size, stride=a.stride, data_type=torch_to_cudnn_dtype(a.dtype))

    scale = None
    if weight is not None:
        scale = graph.tensor(
            name="scale", dim=weight.size, stride=weight.stride, data_type=torch_to_cudnn_dtype(weight.dtype)
        )

    bias_cudnn = None
    if bias is not None:
        bias_cudnn = graph.tensor(
            name="bias", dim=bias.size, stride=bias.stride, data_type=torch_to_cudnn_dtype(bias.dtype)
        )

    scalar_dim_stride = tuple([1] * len(a.size))
    epsilon = graph.tensor(
        name="epsilon",
        dim=scalar_dim_stride,
        stride=scalar_dim_stride,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    Y, _, _ = graph.layernorm(
        name="LN",
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=input,
        scale=scale,
        bias=bias_cudnn,
        epsilon=epsilon,
    )

    Y.set_output(True).set_data_type(torch_to_cudnn_dtype(a.dtype)).set_stride(a.stride)

    # Validate the graph before querying the cache key
    # Validation makes sure all missing properties are inferred and filled, as they affect cache key.
    graph.validate()
    cache_key = graph.key()

    # If a built graph does not exist in cache already, make one and place it in
    if cache_key not in _cudnn_layernormex_cache:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        _cudnn_layernormex_cache[cache_key] = (input, scale, bias_cudnn, epsilon, Y, graph)
    return _cudnn_layernormex_cache[cache_key]


def _transform_layer_norm_inputs(a, normalized_shape, weight, bias):
    if LooseVersion(cudnn.backend_version_string()) >= "9.1":

        def compute_contiguous_strides(shape):
            strides = [1] * len(shape)
            stride = 1
            for i in reversed(range(len(shape))):
                strides[i] = stride
                stride *= shape[i]
            return tuple(strides)

        a_new = CudnnTensorAttributes(a.shape, compute_contiguous_strides(a.shape), a.dtype, a.device.index)
        weight_new = None
        if weight is not None:
            # Make weight to be of the same dimensionality as other input tensors
            weight_shape = (1,) * (a.ndim - weight.ndim) + weight.shape
            weight_new = CudnnTensorAttributes(
                weight_shape, compute_contiguous_strides(weight_shape), weight.dtype, weight.device.index
            )
        bias_new = None
        if bias is not None:
            # Make bias to be of the same dimensionality as other input tensors
            bias_shape = (1,) * (a.ndim - bias.ndim) + bias.shape
            bias_new = CudnnTensorAttributes(
                bias_shape, compute_contiguous_strides(bias_shape), bias.dtype, bias.device.index
            )

        return a_new, weight_new, bias_new
    else:
        # cudnn only supports following:
        # input tensor shape: N, C, (D), H, W
        # normalized shape:  (C, (D), H, W)
        # convert all tensor shapes to above format
        elements_to_normalize = np.prod(normalized_shape)
        batch_size = np.prod(a.shape[: -len(normalized_shape)], dtype=int)

        # Assume strides to be NCHW contiguous
        assumed_stride = (elements_to_normalize, 1, 1, 1)
        a_4d = CudnnTensorAttributes((batch_size, elements_to_normalize, 1, 1), assumed_stride, a.dtype, a.device.index)
        weight_4d = CudnnTensorAttributes(
            (1, elements_to_normalize, 1, 1), assumed_stride, weight.dtype, weight.device.index
        )
        bias_4d = CudnnTensorAttributes((1, elements_to_normalize, 1, 1), assumed_stride, bias.dtype, bias.device.index)

        return a_4d, weight_4d, bias_4d


def _cudnn_layer_norm_fwd_impl(
    a: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
):
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)
    input, scale, B, epsilon, Y, graph = _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d)

    Y_actual = torch.empty_like(a)
    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")
    workspace = torch.empty(graph.get_workspace_size(), device=a.device, dtype=torch.uint8)

    cudnn_to_torch_tensor = {input: a, scale: weight, B: bias, epsilon: epsilon_cpu, Y: Y_actual}

    # Even though the handle is created on a.device, cudnn still requires to set current device to a.device.
    # This is most probably a bug and is being actively looked into.
    with torch.cuda.device(a.device):
        graph.execute(cudnn_to_torch_tensor, workspace)

    return Y_actual


def _cudnn_layer_norm_checker(
    a: TensorLike,
    normalized_shape: list[int],
    weight: TensorLike | None = None,
    bias: TensorLike | None = None,
    eps: float = 1e-5,
):
    if cudnn is None:
        return False

    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)

    try:
        _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d)
    except cudnn.cudnnGraphNotSupportedError as ex:
        return False
    except Exception as e:
        raise

    return True


import thunder.torch as ltorch

cudnn_layer_norm_fwd = cudnn_layernorm_ex.register_operator(
    "cudnn_layernorm_fwd", like=ltorch.layer_norm, fn=_cudnn_layer_norm_fwd_impl
)
cudnn_layernorm_ex.register_implementation(ltorch.layer_norm, cudnn_layer_norm_fwd, checker=_cudnn_layer_norm_checker)
