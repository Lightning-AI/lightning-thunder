import math
from functools import partial

import pytest
import torch
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


def _there_is_cudagraph_sym(trace):
    # So far check for a single CUDAGraph fusion
    bsyms = [bsym for bsym in trace.bound_symbols if bsym.sym.name.startswith("CUDAGraph")]
    return len(bsyms) == 1


@instantiate(dtypes=(thunder.float32,), devicetypes=(thunder.devices.DeviceType.CUDA,))
@requiresCUDA
def test_nanogpt_complete_cudagraphs(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=torch.int64, device=device)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    tom = executor.make_callable(gpt, use_cudagraphs=True, disable_torch_autograd=True)

    # Checking graph cache stats
    from thunder.executors.cudagraphex import build_cuda_graph

    # Cache stats before test runs
    build_graph_stats_old = build_cuda_graph.cache_info()

    for _ in range(2):
        idx = make((4, 64), dtype=torch.int64, low=0, high=255)
        torch_result = gpt(idx)

        thunder_result = tom(idx)
        assert_close(torch_result, thunder_result)

    # Cache stats after test runs
    build_graph_stats_new = build_cuda_graph.cache_info()
    # We ran only a single (forward) graph several times.
    # Test that at most 1 cache miss happened after the runs.
    assert (build_graph_stats_new.misses - build_graph_stats_old.misses) <= 1

    # Check we really run CUDAGraphExecutor {
    assert tom._lc_cd.use_cudagraphs is True
    assert _there_is_cudagraph_sym(thunder.last_traces(tom)[-1])
    # }

    # Let's clear cache if run only in tests
    # TODO: merge with the cache of the thunder.jit callable
    if build_graph_stats_old.misses == 0:
        build_cuda_graph.cache_clear()


@instantiate(dtypes=(thunder.float32,), devicetypes=(thunder.devices.DeviceType.CUDA,))
@requiresCUDA
def test_nanogpt_complete_cuda_graphs_autograd(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)
    cmodel = executor.make_callable(gpt, use_cudagraphs=True)

    # Checking graph cache stats
    from thunder.executors.cudagraphex import build_cuda_graph

    # Cache stats before test runs
    build_graph_stats_old = build_cuda_graph.cache_info()

    # Multiple runs to test whether static buffers are properly updated
    for i in range(3):
        x = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)
        targets = make_tensor((4, 64), dtype=torch.int64, low=0, high=255, device=device)

        torch_result = gpt(x, targets=targets)
        torch_grads = torch.autograd.grad(torch_result[1], gpt.parameters())

        thunder_result = cmodel(x, targets=targets)
        thunder_grads = torch.autograd.grad(thunder_result[1], gpt.parameters())

        assert_close(torch_result, thunder_result)
        assert_close(torch_grads, thunder_grads)

    # Cache stats after test runs
    build_graph_stats_new = build_cuda_graph.cache_info()
    # We ran only at most two (forward and backward) graphs several times.
    # Test that at most 2 cache misses happened after the runs
    # (at most one per each graph)
    assert (build_graph_stats_new.misses - build_graph_stats_old.misses) <= 2

    # Check we really run CUDAGraphExecutor {
    assert cmodel._lc_cd.use_cudagraphs is True
    assert _there_is_cudagraph_sym(thunder.last_traces(cmodel)[-1])
    assert _there_is_cudagraph_sym(thunder.last_backward_traces(cmodel)[-1])
    # }

    # Let's clear cache if run only in tests
    # TODO: merge with the cache of the thunder.jit callable
    if build_graph_stats_old.misses == 0:
        build_cuda_graph.cache_clear()


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


def test_hf_bert():
    import transformers

    @thunder.core.jit_ext.register_general_jit_lookaside(
        transformers.modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask
    )
    @thunder.core.jit_ext.interpreter_needs_wrap
    def dummy(*args):
        pass

    # Transformers 2.41+ adds some more non-essential data-dependent
    # control flow behind a check whether we are compiling
    @thunder.core.jit_ext.register_general_jit_lookaside(torch._dynamo.is_compiling)
    @thunder.core.jit_ext.interpreter_needs_wrap
    def dummy(*args):
        return True

    m = transformers.BertForSequenceClassification(transformers.BertConfig())
    del m.bert.encoder.layer[2:]
    m.eval()
    inp = torch.randint(1, 20, (1, 32))
    jm = thunder.jit(m)
    actual = jm(inp)
    expected = m(inp)

    assert_close(actual, expected)


@requiresCUDA
def test_quantization():
    try:
        import bitsandbytes
    except (ImportError, RuntimeError):
        pytest.skip("bitsandbytes not found")

    from thunder.tests import litgpt_model
    from lightning.fabric.plugins import BitsandbytesPrecision

    config = litgpt_model.Config.from_name("llama2-like")
    with torch.device("cuda"):
        model_fp_reference = litgpt_model.GPT(config).to(torch.bfloat16)

    import lightning as L

    plugins = BitsandbytesPrecision("nf4", torch.bfloat16)
    fabric = L.Fabric(devices=1, precision=None, plugins=plugins)
    with fabric.init_module(empty_init=True):
        model = litgpt_model.GPT(config)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = 20
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()
    model.requires_grad_(False)
    model = fabric.setup_module(model)

    model.load_state_dict(model_fp_reference.state_dict())

    x = torch.randint(1, 255, (1, 10), device="cuda")
    input_pos = torch.arange(10, device="cuda")
    logits_expected = model(x, input_pos)

    from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit, get_bitsandbytes_executor

    bitsandbytes_executor = get_bitsandbytes_executor()

    model_fp_reference.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    model_fp_reference.max_seq_length = 20
    model_fp_reference.requires_grad_(False)
    model_fp_reference.eval()

    jm = thunder.jit(
        model_fp_reference,
        executors=(bitsandbytes_executor,),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )

    logits_thunder = jm(x, input_pos)
    # check_dtype=False due to litgpt returning float32
    # (maybe that also is the numerical discrepancy?)
    assert_close(logits_thunder, logits_expected, atol=2e-2, rtol=1e-3, check_dtype=False)

    sd = {k: v.clone() for k, v in jm.state_dict().items()}
    jm.load_original_state_dict(model_fp_reference.state_dict())
    sd2 = {k: v.clone() for k, v in jm.state_dict().items()}
    assert len(sd) == len(sd2)
    for k, v in sd.items():
        assert_close(v, sd2[k])
