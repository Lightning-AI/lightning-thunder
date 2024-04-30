import math
from dataclasses import dataclass
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close, make_tensor

import thunder
import thunder.torch as ttorch
from thunder.tests.framework import instantiate, requiresCUDA
import thunder.tests.nanogpt_model as nanogpt_model
import thunder.tests.hf_bart_self_attn as hf_bart_self_attn

#
# nanoGPT tests
#


@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_complete(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=torch.int64, device=device)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    idx = make((4, 64), dtype=torch.int64, low=0, high=255)
    torch_result = gpt(idx)

    tom = executor.make_callable(gpt, disable_torch_autograd=True)
    thunder_result = tom(idx)

    assert_close(torch_result, thunder_result)


# TODO Investigate grad inconsistency
# TODO: Add float16 and bfloat16 comparison tests here and to all other tests in
# this file.
# See issue "Add half precision dtype tests to test_networks.py"
@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_complete_autograd(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    x = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)
    targets = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)
    torch_result = gpt(x, targets=targets)
    torch_grads = torch.autograd.grad(torch_result[1], gpt.parameters())

    cmodel = executor.make_callable(gpt)
    thunder_result = cmodel(x, targets=targets)
    thunder_grads = torch.autograd.grad(thunder_result[1], gpt.parameters())

    assert_close(torch_result, thunder_result)
    assert_close(torch_grads, thunder_grads, atol=1e-1, rtol=1e-1)


@instantiate(dtypes=(thunder.float32,), devicetypes=(thunder.devices.DeviceType.CUDA,))
@requiresCUDA
def test_nanogpt_complete_cudagraphs(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=torch.int64, device=device)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    idx = make((4, 64), dtype=torch.int64, low=0, high=255)
    torch_result = gpt(idx)

    tom = executor.make_callable(gpt, use_cudagraphs=True, disable_torch_autograd=True)
    thunder_result = tom(idx)

    assert_close(torch_result, thunder_result)


@instantiate(dtypes=(thunder.float32,), devicetypes=(thunder.devices.DeviceType.CUDA,))
@requiresCUDA
def test_nanogpt_complete_cuda_graphs_autograd(executor, device, dtype):
    pytest.skip("https://github.com/Lightning-AI/lightning-thunder/issues/1403")

    tdtype = ttorch.to_torch_dtype(dtype)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    x = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)
    targets = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)
    torch_result = gpt(x, targets=targets)
    torch_grads = torch.autograd.grad(torch_result[1], gpt.parameters())

    cmodel = executor.make_callable(gpt, use_cudagraphs=True)
    thunder_result = cmodel(x, targets=targets)
    thunder_grads = torch.autograd.grad(thunder_result[1], gpt.parameters())

    assert_close(torch_result, thunder_result)
    assert_close(torch_grads, thunder_grads)


@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_csa(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    csa = nanogpt_model.CausalSelfAttention(config).to(device=device, dtype=tdtype)

    inp = make((2, config.block_size, config.n_embd))
    torch_result = csa(inp)

    # TODO: turn disable_torch_autograd=False back on once we have a fix for
    # AssertionError: Tensor-likes are not close!
    # Mismatched elements: 451 / 1572864 (0.0%)
    # Greatest absolute difference: 2.0623207092285156e-05 at index (1, 433, 24) (up to 1e-05 allowed)
    # Greatest relative difference: 0.03444782271981239 at index (0, 484, 119) (up to 1.3e-06 allowed)
    # See: https://github.com/Lightning-AI/lightning-thunder/issues/997
    tom = executor.make_callable(csa, disable_torch_autograd=True)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_block(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    block = nanogpt_model.Block(config).to(device=device, dtype=tdtype)

    inp = make((2, config.block_size, config.n_embd))
    torch_result = block(inp)

    tom = executor.make_callable(block)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_mlp(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    # NOTE: currently setting dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0)
    mlp = nanogpt_model.MLP(config).to(device=device, dtype=tdtype)

    inp = make((2, config.n_embd))
    torch_result = mlp(inp)

    tom = executor.make_callable(mlp)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


@instantiate(dtypes=(thunder.float32,))
def test_nanogpt_gelu(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)

    def new_gelu(a):
        return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))

    inp = make((1024, 1024))
    torch_result = new_gelu(inp)

    tom = executor.make_callable(new_gelu)
    thunder_result = tom(inp)

    assert_close(torch_result, thunder_result)


def test_hf_bart_self_attn():
    model = hf_bart_self_attn.BartAttention(
        1024,
        16,
        dropout=0.0,
    )

    inp = torch.randn(1, 10, 1024)
    torch_result = model(inp, None)
    tom = thunder.jit(model)
    thunder_result = tom(inp, None)
    assert_close(torch_result, thunder_result)


@instantiate(dtypes=(thunder.float32,))
def test_llama_moe(executor, device, dtype):
    # This test is a modified version of the LLaMAMoE from LitGPT:
    # https://github.com/Lightning-AI/litgpt/blob/96836008be96fa2fe5e6909a3fd7a112cc57716e/litgpt/model.py#L325-L349
    class Test(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_expert = 8
            self.n_expert_per_token = 2
            self.C = 2
            self.gate = nn.Linear(self.C, self.n_expert, bias=False)
            self.experts = nn.ModuleList(nn.Linear(2, 2, bias=False) for _ in range(self.n_expert))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
            x = x.view(-1, C)  # (B*T, C)
            router = self.gate(x)  # (B*T, n_expert)
            probs, indices = torch.topk(router, self.n_expert_per_token)  # (B*T, n_expert_per_token)
            probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
            masks = indices.unsqueeze(-1) == torch.arange(self.n_expert, device=x.device)
            masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
            y = torch.zeros_like(x)  # (B*T, C)
            for i in range(self.n_expert):
                # NOTE: zip is not working
                # See https://github.com/Lightning-AI/lightning-thunder/issues/284
                # for (mask, expert) in zip(masks, self.experts):
                mask = masks[i]
                expert = self.experts[i]
                token_idx, expert_idx = torch.where(mask)
                # NOTE: probs[token_idx, expert_idx, None] is not working
                pprobs = probs[token_idx, expert_idx]
                pprobs = pprobs.unsqueeze(-1)
                eexpert = expert(x[token_idx])
                # NOTE: The following line uses += instead of torch.index_add in the original code
                y = torch.index_add(y, 0, token_idx, pprobs * eexpert)
            return y.view(B, T, C)

    model = Test().to(device=device, dtype=ttorch.to_torch_dtype(dtype))
    model = thunder.jit(model, executors=executor.executors_list())

    x = torch.randn(2, 3, 2, device=device, dtype=ttorch.to_torch_dtype(dtype))
    y = model(x)
    assert y.shape == (2, 3, 2)

    y.backward(torch.randn_like(y))
    assert all(p.grad is not None for p in model.parameters())
    print(thunder.last_backward_traces(model)[-1])
    print(thunder.last_traces(model)[-1])
