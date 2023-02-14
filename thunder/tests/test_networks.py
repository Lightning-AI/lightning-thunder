import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close, make_tensor

import thunder
import thunder.langs.torch as ttorch
from thunder.tests.framework import executors
import thunder.tests.nanogpt_model as nanogpt_model

#
# nanoGPT tests
#


@executors(dtypes=(thunder.float32,))
def test_nanogpt(executor, device, dtype):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=torch.int64, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    idx = make((8, 64), dtype=torch.int64, low=0, high=255)
    torch_result = gpt(idx)

    tom = thunder.make_traced(gpt, executor=executor, _preprocess=True)
    # FIXME: want to make calling convention consistent with PyTorch
    thunder_result = tom(idx)

    assert_close(torch_result, thunder_result)


@executors(dtypes=(thunder.float32,))
def test_nanogpt_csa(executor, device, dtype):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    csa = nanogpt_model.CausalSelfAttention(config).to(device=device, dtype=tdtype)

    inp = make((2, config.block_size, config.n_embd))
    torch_result = csa(inp)

    tom = thunder.make_traced(csa, executor=executor, _preprocess=True)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@executors(dtypes=(thunder.float32,))
def test_nanogpt_block(executor, device, dtype):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    block = nanogpt_model.Block(config).to(device=device, dtype=tdtype)

    inp = make((2, config.block_size, config.n_embd))
    torch_result = block(inp)

    tom = thunder.make_traced(block, executor=executor, _preprocess=True)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp(executor, device, dtype):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    mlp = nanogpt_model.MLP(config).to(device=device, dtype=tdtype)

    inp = make((2, config.n_embd))
    torch_result = mlp(inp)

    tom = thunder.make_traced(mlp, executor=executor, _preprocess=True)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@executors(dtypes=(thunder.float32,))
def test_nanogpt_gelu(executor, device, dtype):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    def new_gelu(a):
        return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))

    inp = make((1024, 1024))
    torch_result = new_gelu(inp)

    tom = thunder.make_traced(new_gelu, executor=executor, _preprocess=True)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)
