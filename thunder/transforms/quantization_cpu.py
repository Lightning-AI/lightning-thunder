# NOTE: The code for CPU quantization in this file has been adapted from a not-yet-merged branch of the
# bitsandbytes library (https://github.com/bitsandbytes-foundation/bitsandbytes/tree/multi-backend-refactor).
# Once the changes in that branch are merged into the main bitsandbytes repository, this implementation
# should be replaced with the official, upstream version to ensure better compatibility, performance,
# and future updates.
# Please track the progress of the bitsandbytes library and update this file when necessary.

import warnings

import torch

from bitsandbytes.functional import (
    QuantState,
    get_4bit_type,
)

Tensor = torch.Tensor

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


def get_4bit_type(typename, device=None, blocksize=64):
    if device is None:
        device = "cuda"
    data = None
    if typename == "nf4":
        """Implements the NF4 data type.

        Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
        is normalized into the range [-1, 1].

        For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

        Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
        the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
        """
        data = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
    elif typename == "fp4":
        # 0b000 = 0
        # 0b001 = 0.0625
        # 0b010 = 8
        # 0b011 = 12
        # 0b100 = 4
        # 0b101 = 6
        # 0b110 = 2
        # 0b111 = 3
        # can also be created with bnb.functional.create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    elif typename == "int4":
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    elif typename == "af4":
        # Taken from: NF4 Isn't Information Theoretically Optimal (and that's Good)
        # https://arxiv.org/abs/2306.06965
        if blocksize == 64:
            data = [
                -1.0,
                -0.69441008,
                -0.51243739,
                -0.3736951,
                -0.25607552,
                -0.14982478,
                -0.04934812,
                0.0,
                0.04273164,
                0.12934483,
                0.21961274,
                0.31675666,
                0.42563882,
                0.55496234,
                0.72424863,
                1.0,
            ][::-1]
        else:
            raise NotImplementedError("4-bit AbnormalFloats currently only support blocksize 64.")

    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    data = torch.tensor(data, device=device)
    data.div_(data.abs().max())

    assert data.numel() == 16

    return data


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
