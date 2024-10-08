"""
Derivied from
    https://github.com/bitsandbytes-foundation/bitsandbytes

The code for CPU quantization in this file has been adapted from a not-yet-merged
multi-backend-refactor branch
    
MIT License:
    https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import subprocess
from typing import Literal, Optional, Tuple
import warnings
import torch

from bitsandbytes.functional import (
    QuantState,
    get_4bit_type,
)

try:
    # to support Intel CPU/GPU (XPU) backend
    import intel_extension_for_pytorch as ipex

    ipex_cpu = ipex if ipex._C._has_cpu() else None
    ipex_xpu = ipex if ipex._C._has_xpu() else None
except BaseException:
    ipex_cpu = None
    ipex_xpu = None

gxx_available = False
try:
    subprocess.run(["g++", "--version"])
    gxx_available = True
except BaseException:
    warnings.warn("g++ not found, torch.compile disabled for CPU/XPU.")

Tensor = torch.Tensor

def _torch_version_prereq(major, minor):
    ver_major = int(torch.__version__.split(".")[0])
    ver_minor = int(torch.__version__.split(".")[1])
    return ver_major * 32 + ver_minor >= major * 32 + minor

def _maybe_torch_compile(func):
    # torch.compile requires g++ and pytorch >= 2.0
    if gxx_available and _torch_version_prereq(2, 0):
        options = {}
        # fx_graph_cache requires pytorch >= 2.2
        if _torch_version_prereq(2, 2):
            options.update({"fx_graph_cache": True})
        return torch.compile(func, dynamic=True, options=options)
    return func

NF4_QUANT_TABLE = [
    -1.0 - 1e-2,  # 0b0000
    -0.8480964004993439,  # 0b0001
    -0.6106329262256622,  # 0b0010
    -0.4599952697753906,  # 0b0011
    -0.33967943489551544,  # 0b0100
    -0.23460740596055984,  # 0b0101
    -0.13791173323988914,  # 0b0110
    -0.045525018125772476,  # 0b0111
    0.03979014977812767,  # 0b1000
    0.1202552504837513,  # 0b1001
    0.2035212516784668,  # 0b1010
    0.2920137718319893,  # 0b1011
    0.3893125355243683,  # 0b1100
    0.5016634166240692,  # 0b1101
    0.6427869200706482,  # 0b1110
    0.8614784181118011,  # 0b1111
]

FP4_QUANT_TABLE = {
    0 - 1e-2: 0,  # 0b0000
    0.00260417: 1,  # 0b0001
    0.0859375: 6,  # 0b0110
    0.20833333: 7,  # 0b0111
    0.29166667: 4,  # 0b0100
    0.4166667: 5,  # 0b0101
    0.583333: 2,  # 0b0010
    0.8333333: 3,  # 0b0011
}

def assert_on_cpu(tensors):
    on_cpu = True
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        on_cpu &= t.device.type == "cpu"
    if not on_cpu:
        raise TypeError(
            "All input tensors need to be on CPU, but found some tensors to not be on CPU:\n"
            f" {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}"
        )
    return on_cpu

def quantize_4bit_cpu(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=64,
    compress_statistics=False,
    quant_type: Literal["fp4", "nf4"] = "fp4",
    quant_storage=torch.uint8,
) -> Tuple[torch.Tensor, QuantState]:
    if blocksize is None:
        blocksize = 64
    assert_on_cpu([A, absmax, out])
    assert quant_storage == torch.uint8, "CPU backend only supports uint8 quant_storage"
    return quantize_4bit_impl(A, absmax, out, blocksize, compress_statistics, quant_type)

@_maybe_torch_compile
def quantize_4bit_impl(
    A: Tensor,
    absmax: Tensor = None,
    out: Tensor = None,
    blocksize=64,
    compress_statistics=False,
    quant_type="nf4",
) -> Tensor:
    """
    Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized to FP4.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor (8-bit).
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}, only nf4 is supported now

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor with packed 4-bit values.
    tuple(torch.Tensor, torch.Size, torch.dtype, int):
        The quantization state to undo the quantization.
    """
    if quant_type not in ["nf4", "fp4"]:
        raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented for CPU/XPU.")
    if quant_type == "fp4":
        warnings.warn("fp4 quantization is currently slow on CPU/XPU. Please Use nf4 instead for better performance.")
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
    n = A.numel()
    input_shape = A.shape
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0

    if absmax is None:
        absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)

    if out is None:
        out = torch.zeros(((n + 1) // 2), dtype=torch.uint8, device=A.device)

    rem = n % blocksize
    has_rem = rem > 0

    # Scale tensor to [-1, 1]
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)
    # map [-1, 1] to nf4/fp4
    out_uint8 = torch.empty(scaled_A.shape, dtype=torch.uint8)
    if quant_type == "nf4":
        for i in range(len(NF4_QUANT_TABLE)):
            out_uint8[scaled_A > NF4_QUANT_TABLE[i]] = i
    elif quant_type == "fp4":
        sign = scaled_A < 0
        abs_scaled_A = torch.abs(scaled_A)
        for key, val in FP4_QUANT_TABLE.items():
            out_uint8[abs_scaled_A > key] = val
        out_uint8 += sign.to(torch.uint8) * 8
    if out_uint8.size(-1) % 2:
        out_uint8 = torch.nn.functional.pad(out_uint8, (0, 1), value=0)
    out[:] = out_uint8[1::2].bitwise_left_shift(4).bitwise_or_(out_uint8[::2])

    code = get_4bit_type(quant_type, device=A.device)

    if compress_statistics:
        raise NotImplementedError("bnb_4bit_use_double_quant is not supported yet for CPU/XPU")
    else:
        state = QuantState(
            absmax=absmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )

    return out.unsqueeze(0), state
