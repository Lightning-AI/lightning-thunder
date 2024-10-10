import collections
from enum import Enum
import functools
import json
import operator
import pathlib
import struct

import numpy
import torch


class GgufMetadataValueType(Enum):  # // gguf_metadata_value_type (uint32_t enum) in llama.cpp
    uint8 = 0
    int8 = 1
    uint16 = 2
    int16 = 3
    uint32 = 4
    int32 = 5
    float32 = 6
    bool = 7
    string = 8
    array = 9
    uint64 = 10
    int64 = 11
    float64 = 12


class GgmlType(Enum):  # uint32
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # 4, 5 are Q4_2/3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = (22,)
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29


GGML_BLOCK_SIZES = {
    GgmlType.F32: 4,
    GgmlType.Q4_0: 2 + 16,
    GgmlType.Q8_0: 2 + 32,
    GgmlType.Q2_K: 256 // 16 + 256 // 4 + 2 + 2,
    GgmlType.Q3_K: 256 // 8 + 256 // 4 + 12 + 2,
    GgmlType.Q4_K: 2 + 2 + 12 + 256 // 2,
    GgmlType.Q5_K: 2 + 2 + 12 + 256 // 8 + 256 // 2,
    GgmlType.Q6_K: 256 // 2 + 256 // 4 + 256 // 16 + 2,
}

GGML_ELEMENTS_PER_BLOCK = {
    GgmlType.F32: 1,
    GgmlType.Q4_0: 32,
    GgmlType.Q8_0: 32,
    GgmlType.Q2_K: 256,
    GgmlType.Q3_K: 256,
    GgmlType.Q4_K: 256,
    GgmlType.Q5_K: 256,
    GgmlType.Q6_K: 256,
}


def compute_number_of_blocks(shape, ggml_type):
    # todo: from the code, it looks like the padding might be per-row
    numel = functools.reduce(operator.mul, shape, 1)
    block_numel = GGML_ELEMENTS_PER_BLOCK[ggml_type]
    num_blocks = (numel + block_numel - 1) // block_numel
    return num_blocks


def read_gguf_string(f):
    (length,) = struct.unpack("L", f.read(8))
    res = f.read(length).decode("utf8")
    return res


def read_metadata_kv(f):
    k = read_gguf_string(f)
    (metadata_value_type,) = struct.unpack("I", f.read(4))
    metadata_value_type = GgufMetadataValueType(metadata_value_type)

    def read_value(f, typ):
        if typ == GgufMetadataValueType.string:
            v = read_gguf_string(f)
        elif typ == GgufMetadataValueType.uint32:
            (v,) = struct.unpack("I", f.read(4))
            v = numpy.uint32(v)
        elif typ == GgufMetadataValueType.int32:
            (v,) = struct.unpack("i", f.read(4))
            v = numpy.int32(v)
        elif typ == GgufMetadataValueType.float32:
            (v,) = struct.unpack("f", f.read(4))
            v = numpy.float32(v)
        elif typ == GgufMetadataValueType.array:
            (element_metadata_value_type,) = struct.unpack("<I", f.read(4))
            element_metadata_value_type = GgufMetadataValueType(element_metadata_value_type)
            (length,) = struct.unpack("L", f.read(8))
            v = [read_value(f, element_metadata_value_type) for _ in range(length)]
        else:
            raise NotImplementedError(f"do not know how to read {k} of type {typ}")
        return v

    v = read_value(f, metadata_value_type)
    return k, v


def read_gguf_header(f):
    f.seek(0)
    header_fixed = f.read(24)
    magic, version, tensor_count, metadata_kv_count = struct.unpack("4sILL", header_fixed)

    assert magic == b"GGUF" and version == 3
    tensor_count, metadata_kv_count

    metadata = {}
    for _ in range(metadata_kv_count):
        key, value = read_metadata_kv(f)
        metadata[key] = value

    alignment = metadata.get("general.alignment", 32)

    tensor_infos = []

    for _ in range(tensor_count):
        name = read_gguf_string(f)
        (dim,) = struct.unpack("I", f.read(4))
        shape = struct.unpack(f"{dim}L", f.read(8 * dim))
        (tensor_type,) = struct.unpack("I", f.read(4))
        tensor_type = GgmlType(tensor_type)
        (offset,) = struct.unpack("L", f.read(8))
        tensor_infos.append((name, shape, tensor_type, offset))

    data_start = f.tell()
    data_start = (data_start + alignment - 1) // alignment * alignment

    processed_tensor_info = [(*i, offset + data_start) for *i, offset in tensor_infos]

    return metadata, processed_tensor_info


def read_gguf_file(fn: str, device="cpu"):
    # does not do any decoding
    f = open(fn, "rb")
    metadata, tensor_infos = read_gguf_header(f)
    state_dict = {}
    tensor_info_dict = {}

    for name, shape, typ, offset in tensor_infos:
        f.seek(offset)
        num_block_bytes = GGML_BLOCK_SIZES[typ]
        num_blocks = compute_number_of_blocks(shape, typ)
        numel = functools.reduce(operator.mul, shape, 1)

        block_data_raw = f.read(num_blocks * num_block_bytes)
        block_data = numpy.frombuffer(block_data_raw, dtype=numpy.uint8).reshape(num_blocks, num_block_bytes)
        if typ == GgmlType.F32:
            # in ggml the first is the fastest moving dimension
            block_data = (
                block_data.view(numpy.float32)
                .reshape(-1)[:numel]
                .reshape(*shape[::-1])
                .transpose(*list(range(len(shape) - 1, -1, -1)))
            )
        elif typ == GgmlType.F16:
            block_data = (
                block_data.view(numpy.float16)
                .reshape(-1)[:numel]
                .reshape(*shape[::-1])
                .transpose(*list(range(len(shape) - 1, -1, -1)))
            )
        state_dict[name] = torch.from_numpy(block_data).to(device)
        tensor_info_dict[name] = (typ, shape)

    return metadata, tensor_info_dict, state_dict


GGML_DEQUANTIZERS = {}


def register_ggml_dequantizer(ggml_type):
    def register_(fn):
        assert ggml_type not in GGML_DEQUANTIZERS
        GGML_DEQUANTIZERS[ggml_type] = fn
        return fn

    return register_


# QK_K = 256
# ypedef struct {
#    uint8_t ql[QK_K/2];      // quants, lower 4 bits
#    uint8_t qh[QK_K/4];      // quants, upper 2 bits
#    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
#    ggml_half d;             // super-block scale
# } block_q6_K;
@register_ggml_dequantizer(GgmlType.Q6_K)
def dequantize_q6_K(block_data, shape, dtype=torch.float32):
    num_blocks, num_block_bytes = block_data.shape
    assert num_block_bytes == GGML_BLOCK_SIZES[GgmlType.Q6_K]
    block_q6_K_ql = block_data[:, :128].reshape(num_blocks, 2, 2, 32)
    block_q6_K_qh = block_data[:, 128:192].reshape(num_blocks, 2, 32)

    block_q6_K_scales = (
        block_data[:, 192:208].view(torch.int8).reshape(num_blocks, 2, 4, 2).repeat_interleave(16, axis=-1)
    )
    block_q6_K_d = block_data[:, 208:210].view(torch.float16).to(torch.float32).reshape(num_blocks, 1, 1)
    y = torch.empty((num_blocks, 2, 4, 32), dtype=dtype, device=block_data.device)
    q1 = ((block_q6_K_ql[:, :, 0, :] & 0xF) | ((block_q6_K_qh[:, :, :] & 0x3) << 4)).view(torch.int8) - 32
    q2 = ((block_q6_K_ql[:, :, 1, :] & 0xF) | (((block_q6_K_qh[:, :, :] >> 2) & 0x3) << 4)).view(torch.int8) - 32
    q3 = ((block_q6_K_ql[:, :, 0, :] >> 4) | (((block_q6_K_qh[:, :, :] >> 4) & 0x3) << 4)).view(torch.int8) - 32
    q4 = ((block_q6_K_ql[:, :, 1, :] >> 4) | (((block_q6_K_qh[:, :, :] >> 6) & 0x3) << 4)).view(torch.int8) - 32
    y[:, :, 0, :] = block_q6_K_d * block_q6_K_scales[:, :, 0, :] * q1
    y[:, :, 1, :] = block_q6_K_d * block_q6_K_scales[:, :, 1, :] * q2
    y[:, :, 2, :] = block_q6_K_d * block_q6_K_scales[:, :, 2, :] * q3
    y[:, :, 3, :] = block_q6_K_d * block_q6_K_scales[:, :, 3, :] * q4
    numel = functools.reduce(operator.mul, shape, 1)
    # in ggml the first is the fastest moving dimension
    y = y.reshape(-1)[:numel].reshape(*shape[::-1]).permute(*list(range(len(shape) - 1, -1, -1)))
    return y


# define QK4_0 32
# typedef struct {
#    ggml_half d;           // delta
#    uint8_t qs[QK4_0 / 2]; // nibbles / quants
# } block_q4_0;
@register_ggml_dequantizer(GgmlType.Q4_0)
def dequantize_q4_0(block_data, shape, dtype=torch.float32):
    num_blocks, num_block_bytes = block_data.shape
    assert num_block_bytes == GGML_BLOCK_SIZES[GgmlType.Q4_0]
    block_q4_0_d = block_data[:, :2].view(torch.float16).to(torch.float32).reshape(num_blocks, 1)
    block_q4_0_qs = block_data[:, 2:]
    num_blocks, _ = block_q4_0_qs.shape
    y = torch.empty((num_blocks, 2, 16), dtype=dtype, device=block_data.device)

    q0 = block_q4_0_qs & 0xF
    q0 = q0.view(torch.int8) - 8
    q1 = (block_q4_0_qs >> 4).view(torch.int8) - 8
    y[:, 0, :] = block_q4_0_d * q0
    y[:, 1, :] = block_q4_0_d * q1
    numel = functools.reduce(operator.mul, shape, 1)
    # in ggml the first is the fastest moving dimension
    y = y.reshape(-1)[:numel].reshape(*shape[::-1]).permute(*list(range(len(shape) - 1, -1, -1)))
    return y


@register_ggml_dequantizer(GgmlType.F32)
@register_ggml_dequantizer(GgmlType.F16)
def dequantize_noop(block_data, shape, dtype=torch.float32):
    assert block_data.shape == shape
    return block_data.to(dtype)


def dequantize(qw, typ, shape, dtype=torch.float32):
    dequantizer = GGML_DEQUANTIZERS.get(typ)
    if typ is None:
        raise NotImplementedError("Cannot decode {typ}")
    return dequantizer(qw, shape, dtype)


def cat_quantized(tensors, infos, dim):
    assert isinstance(tensors, collections.abc.Sequence) and len(tensors) > 0
    assert isinstance(infos, collections.abc.Sequence) and len(infos) == len(tensors)
    typ_0, shape_0 = infos[0]
    assert typ_0 not in (GgmlType.F16, GgmlType.F32), "This only works for quantized"
    total_dim = len(shape_0)
    if dim < 0:
        dim += total_dim
    assert 0 <= dim < total_dim
    concat_dim_size = 0
    reshaped_tensors = []
    for (typ_i, shape_i), tensor_i in zip(infos, tensors):
        assert typ_i == typ_0 and len(shape_i) == total_dim, "concatenated tensors must have same type and dimension"
        assert (
            shape_i[:dim] == shape_0[:dim] and shape_i[dim + 1 :] == shape_0[dim + 1 :]
        ), "shapes must match except in the concat dimension"
        concat_dim_size += shape_i[dim]
        numel_in_back = functools.reduce(operator.mul, shape_i[: dim + 1])
        assert numel_in_back % GGML_ELEMENTS_PER_BLOCK[typ_0] == 0
        blocks_in_back = numel_in_back // GGML_ELEMENTS_PER_BLOCK[typ_0]
        reshaped_tensors.append(tensor_i.reshape(-1, blocks_in_back, GGML_BLOCK_SIZES[typ_0]))
    new_tensor = torch.cat(reshaped_tensors, dim=1).view(-1, GGML_BLOCK_SIZES[typ_0])
    new_info = (typ_0, (*shape_0[:dim], concat_dim_size, *shape_0[dim + 1 :]))
    return new_tensor, new_info


def merge_attention_weights(qq, info_q, qk, info_k, qv, info_v):
    typ_q, shape_q = info_q
    typ_k, shape_k = info_k
    typ_v, shape_v = info_v

    num_blocks = qq.shape[0]
    assert qq.shape[1] == GGML_BLOCK_SIZES[typ_q]
    assert shape_q[0] % GGML_ELEMENTS_PER_BLOCK[typ_q] == 0
    qq_swizzled = (
        qq.view(-1, 64, 2, shape_q[0] // GGML_ELEMENTS_PER_BLOCK[typ_q], qq.shape[1]).transpose(1, 2).reshape(*qq.shape)
    )
    qk_swizzled = (
        qk.view(-1, 64, 2, shape_k[0] // GGML_ELEMENTS_PER_BLOCK[typ_k], qk.shape[1]).transpose(1, 2).reshape(*qk.shape)
    )

    dqq2 = dequantize(qq_swizzled, typ_q, shape_q)
    dqk2 = dequantize(qk_swizzled, typ_k, shape_k)

    assert shape_k[0] % GGML_ELEMENTS_PER_BLOCK[typ_k] == 0
    assert shape_v[0] % GGML_ELEMENTS_PER_BLOCK[typ_v] == 0

    q_all, (typ_all, shape_all) = cat_quantized(
        [qq_swizzled, qk_swizzled, qv],
        [
            (typ_q, (shape_q[0], shape_q[1] // 8, 8)),
            (typ_k, (shape_k[0], shape_k[1] // 8, 8)),
            (typ_v, (shape_v[0], shape_v[1] // 8, 8)),
        ],
        dim=1,
    )
    shape_all = (shape_all[0], shape_all[1] * shape_all[2])
    return q_all, (typ_all, shape_all)


class GgmlDataReader:
    def __init__(self, model_file_name):
        model_file_path = pathlib.Path(model_file_name).expanduser()
        data = json.load(open(model_file_path))
        (model_data,) = [d for d in data["layers"] if d["mediaType"] == "application/vnd.ollama.image.model"]
        model_data_blob = model_data["digest"].replace(":", "-")

        p = model_file_path
        while p.name != ".ollama":
            p = p.parent
        model_blob_path = p / "models" / "blobs" / model_data_blob

        self.f = open(model_blob_path, "rb")
        metadata, tensor_infos = read_gguf_header(self.f)

        self.tensor_infos = {name: (typ, shape, offset) for name, shape, typ, offset in tensor_infos}

    def read_tensor(self, ggml_name):
        typ, shape, offset = self.tensor_infos[ggml_name]

        self.f.seek(offset)
        num_block_bytes = GGML_BLOCK_SIZES[typ]
        num_blocks = compute_number_of_blocks(shape, typ)
        numel = functools.reduce(operator.mul, shape, 1)

        block_data_raw = self.f.read(num_blocks * num_block_bytes)
        block_data = numpy.frombuffer(block_data_raw, dtype=numpy.uint8).reshape(num_blocks, num_block_bytes)
        if typ == GgmlType.F32:
            # in ggml the first is the fastest moving dimension
            block_data = (
                block_data.view(numpy.float32)
                .reshape(-1)[:numel]
                .reshape(*shape[::-1])
                .transpose(*list(range(len(shape) - 1, -1, -1)))
            )
        elif typ == GgmlType.F16:
            block_data = (
                block_data.view(numpy.float16)
                .reshape(-1)[:numel]
                .reshape(*shape[::-1])
                .transpose(*list(range(len(shape) - 1, -1, -1)))
            )
        import warnings

        with warnings.catch_warnings(action="ignore", category=UserWarning):  # ignore read-only warning
            return torch.from_numpy(block_data), (typ, shape)

    def close(self):
        self.f.close()

    def get_parameter(self, name):
        ggml_name = (
            name.replace("transformer.wte.weight", "token_embd.weight")
            .replace("transformer.h.", "blk.")
            .replace(".norm_1.weight", ".attn_norm.weight")
            .replace(".attn.attn.weight", ".attn_q.weight")
            .replace(".attn.proj.weight", ".attn_output.weight")
            .replace(".norm_2.weight", ".ffn_norm.weight")
            .replace(".mlp.fc_1.weight", ".ffn_gate.weight")
            .replace(".mlp.fc_2.weight", ".ffn_up.weight")
            .replace(".mlp.proj.weight", ".ffn_down.weight")
            .replace("transformer.ln_f.weight", "output_norm.weight")
            .replace("lm_head.weight", "output.weight")
        )
        if not name.endswith("attn.attn.weight"):
            q, (typ, shape) = self.read_tensor(ggml_name)
        else:
            ggml_name_key = ggml_name.replace("attn_q", "attn_k")
            ggml_name_value = ggml_name.replace("attn_q", "attn_v")
            q, (typ, shape) = merge_attention_weights(
                *self.read_tensor(ggml_name), *self.read_tensor(ggml_name_key), *self.read_tensor(ggml_name_value)
            )
        return q, (typ, shape)
