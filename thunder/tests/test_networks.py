import math
from functools import partial
import warnings

import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.torch as ttorch
from thunder.tests.framework import (
    instantiate,
    requiresCUDA,
    DynamoThunderExecutor,
    _all_test_executors,
    version_between,
    BITSANDBYTES_AVAILABLE,
    requiresDeviceMemory,
)
import thunder.tests.nanogpt_model as nanogpt_model
import thunder.tests.hf_bart_self_attn as hf_bart_self_attn

#
# nanoGPT tests
#

# NOTE: DynamoThunderExecutor is not included in the _all_test_executors() list
# because we don't want to run all the tests with the DynamoThunderExecutor by
# default. We only want to run it with the tests that are explicitly marked to
# use the DynamoThunderExecutor. When there's more than one file that uses the
# DynamoThunderExecutor, we should consider adding a separate list of executors
# to the framework.py file.
all_test_executors_and_dynamo = _all_test_executors() + [DynamoThunderExecutor]


# see https://docs.pytest.org/en/stable/how-to/capture-warnings.html#recwarn for the recwarn fixture
@instantiate(dtypes=(thunder.float32,), executors=all_test_executors_and_dynamo)
def test_nanogpt_complete(executor, device, dtype, recwarn):
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

    if recwarn:
        for r in recwarn:
            assert "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed." not in str(r.message)


# TODO Investigate grad inconsistency
# TODO: Add float16 and bfloat16 comparison tests here and to all other tests in
# this file.
# See issue "Add half precision dtype tests to test_networks.py"
@instantiate(dtypes=(thunder.float32,))  # ), executors=all_test_executors_and_dynamo)
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
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=4, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype).requires_grad_(False).eval()

    from thunder.transforms.cudagraph import CUDAGraphTransform

    cgtransform = CUDAGraphTransform()
    tom = executor.make_callable(gpt, transforms=[cgtransform], disable_torch_autograd=True)

    for _ in range(2):
        idx = make((4, 64), dtype=torch.int64, low=0, high=255)
        torch_result = gpt(idx)

        thunder_result = tom(idx)
        assert_close(torch_result, thunder_result)

    # We ran only a single (forward) graph several times.
    # Test that at only 1 cache entry was created during the runs.
    assert len(cgtransform.cuda_graph_runner.cuda_graph_cache) == 1

    # Check we really use CUDA graphs
    assert _there_is_cudagraph_sym(thunder.last_traces(tom)[-1])


@instantiate(
    dtypes=(thunder.float32,),
    devicetypes=(thunder.devices.DeviceType.CUDA,),
)
def test_nanogpt_complete_cudagraphs_autograd(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)

    # Creates a nanoGPT model with a smaller size than any of the default options for testing
    # NOTE Sets dropout to zero for reproducibility
    config = nanogpt_model.GPTConfig(dropout=0, block_size=512, n_layer=6, n_head=6, n_embd=768)
    gpt = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)

    from thunder.transforms.cudagraph import CUDAGraphTransform

    cgtransform = CUDAGraphTransform()
    cmodel = executor.make_callable(gpt, transforms=[cgtransform])

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

    # We ran only at most two (forward and backward) graphs several times.
    # Test that at most 2 cache misses happened after the runs
    # (at most one per each graph)
    assert len(cgtransform.cuda_graph_runner.cuda_graph_cache) == 2

    # Check we have CUDAGraph symbols in forward and backward
    assert _there_is_cudagraph_sym(thunder.last_traces(cmodel)[-1])
    assert _there_is_cudagraph_sym(thunder.last_backward_traces(cmodel)[-1])


@instantiate(dtypes=(thunder.float32,), executors=all_test_executors_and_dynamo)
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


@instantiate(dtypes=(thunder.float32,), executors=all_test_executors_and_dynamo)
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


@instantiate(dtypes=(thunder.float32,), executors=all_test_executors_and_dynamo)
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


@instantiate(
    dtypes=(thunder.float32,),
    executors=all_test_executors_and_dynamo,
)
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

    # transformers accesses the old attrib and causes the future warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch._dynamo.*.is_compiling.*")
        m = transformers.BertForSequenceClassification(transformers.BertConfig())
        del m.bert.encoder.layer[2:]
        m.eval()
        inp = torch.randint(1, 20, (1, 32))
        jm = thunder.jit(m)
        actual = jm(inp)
        expected = m(inp)

    assert_close(actual, expected)


@requiresCUDA
@pytest.mark.skip(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2128")
def test_quantization():
    from thunder.tests import litgpt_model
    from lightning.fabric.plugins import BitsandbytesPrecision

    config = litgpt_model.Config.from_name("llama2-like")
    with torch.device("cuda"):
        model_fp_reference = litgpt_model.GPT(config).to(torch.bfloat16)
        model_fp_reference2 = litgpt_model.GPT(config).to(torch.bfloat16)

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

    model_fp_reference2.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    model_fp_reference2.max_seq_length = 20
    model_fp_reference2.requires_grad_(False)
    model_fp_reference2.eval()

    jm = thunder.jit(
        model_fp_reference,
        executors=(bitsandbytes_executor,),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )

    jm2 = thunder.jit(
        model_fp_reference2,
        executors=(bitsandbytes_executor,),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )
    jm2.load_state_dict(jm.state_dict())

    logits_thunder = jm(x, input_pos)
    logits_thunder2 = jm2(x, input_pos)
    # check_dtype=False due to litgpt returning float32
    # (maybe that also is the numerical discrepancy?)
    assert_close(logits_thunder, logits_expected, atol=2e-2, rtol=1e-3, check_dtype=False)

    assert_close(logits_thunder, logits_thunder2, atol=2e-5, rtol=1e-5)

    sd = {k: v.clone() for k, v in jm.state_dict().items()}
    jm.load_original_state_dict(model_fp_reference.state_dict())
    sd2 = {k: v.clone() for k, v in jm.state_dict().items()}
    assert len(sd) == len(sd2)
    for k, v in sd.items():
        assert_close(v, sd2[k])


@thunder.tests.framework.requiresCUDA
def test_thunderfx_mistral_nemo_small():
    """
    Runs a small version of Mistral-NeMo

    This is largely based on code from Alexandros Koumparoulis.
    """
    import transformers

    model_id = "mistralai/Mistral-Nemo-Base-2407"

    # Setup a "small" version of NeMo-Mistral that does not require downloading
    # weights. This is not a configuration that is worth benchmarking.
    # This was created by using
    #   MistralConfig(num_hidden_layers=1, max_position_embeddings=1024)
    # and then manually diffing that returned object with:
    #   transformers.AutoConfig.from_pretrained(model_id)
    # until they lined up sans the hidden and embeddings changes, above.
    config = transformers.models.mistral.configuration_mistral.MistralConfig(
        num_hidden_layers=1,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=64,
        architectures=["MistralForCausalLM"],
        hidden_size=1024,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=131072,
        head_dim=32,
        _name_or_path=model_id,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    device = torch.device("cuda")
    model.to(device)
    model.train()
    mdl = thunder.dynamo.thunderfx(model)

    batch_size = 1
    iid_size = (batch_size, config.max_position_embeddings)
    input_ids = torch.randint(0, config.vocab_size, iid_size, device=device)
    attention_mask = torch.ones_like(input_ids)

    output = mdl(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    logits = output.logits
    grad_logits = torch.randn_like(logits)
    logits.backward(grad_logits)

    assert mdl._backend.subgraph_infos, "Should have at least 1 subgraph"


def _get_model_config_pairs():
    def phi3():
        from transformers.models.phi3 import Phi3ForCausalLM, Phi3Config

        return Phi3ForCausalLM, Phi3Config

    def qwen2():
        from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Config

        return Qwen2ForCausalLM, Qwen2Config

    return [(phi3), (qwen2)]


@thunder.tests.framework.requiresCUDA
@pytest.mark.parametrize("model_fn", _get_model_config_pairs())
def test_hf_for_nemo(model_fn):
    from thunder.dynamo import thunderfx
    import torch

    model_cls, config_cls = model_fn()

    config = config_cls(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=32,
        max_position_embeddings=16,
        pad_token_id=15,
    )

    with torch.device("cuda"):
        model = model_cls(config).to(torch.bfloat16)

    # thunder.jit doesn't work with Qwen2, so we use torch.compile
    # https://github.com/Lightning-AI/lightning-thunder/issues/1405

    # fullgraph=True used to work with transformers 4.45.2, but it doesn't work
    # with 4.46.2 because of re.findall usage in the loss function
    fullgraph = False
    compiled_model = thunderfx(model, fullgraph=fullgraph)

    input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings), device="cuda")
    ref_output = model(input_ids=input_ids, labels=input_ids)
    ref_loss = ref_output.loss

    compiled_output = compiled_model(input_ids=input_ids, labels=input_ids)
    compiled_loss = compiled_output.loss

    # Less strict tolerance probably due to different type promotion order for bfloat16
    # TODO: Investigate why the loss is different
    # https://github.com/Lightning-AI/lightning-thunder/issues/1407
    torch.testing.assert_close(compiled_loss, ref_loss, rtol=1e-2, atol=1e-2)

    if fullgraph:
        assert len(compiled_model._backend.subgraph_infos) == 1, (
            "Should have exactly 1 subgraph because of fullgraph=True"
        )
    loss_grad = torch.randn_like(compiled_loss)

    grads_ref = torch.autograd.grad(ref_loss, model.parameters(), grad_outputs=loss_grad)
    grads_compiled = torch.autograd.grad(compiled_loss, model.parameters(), grad_outputs=loss_grad)
    torch.testing.assert_close(grads_ref, grads_compiled, rtol=1e-2, atol=1e-2)

    torch._dynamo.reset()


LLAMA_3_2_1B_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
    "_commit_hash": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}


# Both attn implementation have almost same memory requirements
# Default - 697805312
# eager - 698067456
@requiresCUDA
@requiresDeviceMemory(required_memory_bytes=int(0.7 * 1024 * 1024 * 1024))
@pytest.mark.parametrize("attn_implementation", [None, "eager"])
def test_hf_phi3_vision(attn_implementation):
    # This test takes around 697805312 bytes (~0.7GB) of memory.
    # Shapes for data generated with help of the following script
    # https://github.com/microsoft/PhiCookBook/blob/main/code/03.Finetuning/Phi-3-vision-Trainingscript.py
    from transformers import AutoModelForCausalLM, AutoConfig
    from thunder.dynamo import thunderfx

    if attn_implementation is None:
        # Flash Attention is the default implementation.
        # Skip if flash_attn is not installed.
        pytest.importorskip("flash_attn", reason="Flash Attention")

    # trust_remote_code=True is required else you get the following error:
    # ValueError: Loading microsoft/Phi-3-vision-128k-instruct requires you to execute the configuration file in that repo on your local machine.
    cfg = AutoConfig.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)

    # Scale down the model similar to `test_hf_for_nemo`
    cfg.num_hidden_layers = 1
    cfg.vocab_size = 16
    cfg.pad_token_id = 15
    cfg.hidden_size = cfg.num_attention_heads

    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
        )
        input_ids = torch.randint(0, 15, (1, 256))
        pixel_values = torch.randint(0, 254, (1, 256, 256, 3), dtype=torch.uint8)
        labels = input_ids.clone().detach()

        jit_model = thunderfx(model)
        thunder_result = jit_model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        eager_result = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        torch.testing.assert_close(eager_result.loss, thunder_result.loss, atol=1e-2, rtol=1e-2)

        loss_grad = torch.randn_like(eager_result.loss)
        thunder_grads = torch.autograd.grad(
            thunder_result.loss, model.parameters(), grad_outputs=loss_grad, allow_unused=True
        )
        eager_grads = torch.autograd.grad(
            eager_result.loss, model.parameters(), grad_outputs=loss_grad, allow_unused=True
        )
        torch.testing.assert_close(eager_grads, thunder_grads, atol=1e-2, rtol=1e-2)


@requiresCUDA
def test_memory_litgpt_llama3():
    from thunder.tests import litgpt_model

    def forward_backward_peak(m, inp):
        torch.cuda.reset_peak_memory_stats(device=None)
        mem_before = torch.cuda.max_memory_allocated()
        res = m(inp)
        res.sum().backward()
        mem_after = torch.cuda.max_memory_allocated()
        return (mem_after - mem_before) / 2**20

    with torch.device("cuda"):
        m = litgpt_model.GPT.from_name("llama2-like").bfloat16()
        inp = torch.ones((1, 2048), dtype=torch.int64)

    # warmup, allocate grads etc.
    forward_backward_peak(m, inp)
    forward_backward_peak(m, inp)
    jm = thunder.jit(m)
    forward_backward_peak(jm, inp)
    forward_backward_peak(jm, inp)

    mem_thunder = forward_backward_peak(jm, inp)
    mem_eager = forward_backward_peak(m, inp)

    # assert that attention is not automatically recomputed, see
    # https://github.com/Lightning-AI/lightning-thunder/issues/1646
    assert not {
        bsym.sym.name
        for bsym in thunder.last_backward_traces(jm)[-1].bound_symbols
        if ("attention" in bsym.sym.name or "sdpa" in bsym.sym.name)
        and ("forward" in bsym.sym.name or "fwd" in bsym.sym.name)
    }

    assert mem_thunder < mem_eager


@requiresCUDA
def test_checkpointing_thunderfx():
    from thunder.dynamo import thunderfx
    from thunder.tests import litgpt_model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    def forward_backward_peak(m, inp):
        torch.cuda.reset_peak_memory_stats(device=None)
        mem_before = torch.cuda.max_memory_allocated()
        res = m(inp)
        res.sum().backward()
        mem_after = torch.cuda.max_memory_allocated()
        return (mem_after - mem_before) / 2**20

    with torch.device("cuda"):
        m = litgpt_model.GPT.from_name("llama2-like")
        inp = torch.ones((1, 2048), dtype=torch.int64)

    check_fn = lambda submodule: isinstance(submodule, litgpt_model.Block)
    apply_activation_checkpointing(m, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    # warmup, allocate grads etc.
    forward_backward_peak(m, inp)
    forward_backward_peak(m, inp)
    jm = thunderfx(m)
    forward_backward_peak(jm, inp)
    forward_backward_peak(jm, inp)

    mem_thunder = forward_backward_peak(jm, inp)
    mem_eager = forward_backward_peak(m, inp)

    assert mem_thunder < 105  # this ~35% is more than eager, in isolation 100 vs. 74

    ref = m(inp)
    grads_ref = torch.autograd.grad(ref.sum(), [*m.parameters()])

    res = jm(inp)
    grads_res = torch.autograd.grad(res.sum(), [*m.parameters()])

    assert_close(res, ref)
    assert_close(grads_res, grads_ref, atol=1e-3, rtol=1e-3)


@requiresCUDA
def test_hf_kvcache():
    from transformers.models.llama import LlamaForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import logger as llama_logger
    import logging

    # transformers logs a cache deprecation warning
    llama_logger.setLevel(logging.CRITICAL)

    config_args = LLAMA_3_2_1B_CFG.copy()
    config_args["num_hidden_layers"] = 1
    with torch.device("cuda"):
        model = LlamaForCausalLM(LlamaConfig(**config_args)).to(torch.bfloat16).requires_grad_(False).eval()
        model2 = LlamaForCausalLM(LlamaConfig(**config_args)).to(torch.bfloat16).requires_grad_(False).eval()
        model2.load_state_dict(model.state_dict())

    jm = thunder.jit(model)

    j_static_cache = model._get_cache("static", 1, 128, "cuda", config_args)
    ref_static_cache = model2._get_cache("static", 1, 128, "cuda", config_args)
    assert j_static_cache is not ref_static_cache

    args1 = dict(
        cache_position=torch.tensor([0, 1, 2, 3, 4, 5], device="cuda:0"),
        input_ids=torch.tensor([[128000, 791, 1401, 311, 2324, 374]], device="cuda:0"),
        inputs_embeds=None,
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1]], device="cuda:0"),
        use_cache=True,
    )

    args1["past_key_values"] = j_static_cache
    res = jm(**args1)
    args1["past_key_values"] = ref_static_cache
    expected = model2(**args1)

    assert res.past_key_values is j_static_cache
    assert expected.past_key_values is ref_static_cache

    # we cannot compare the StaticCache instances in assert_close
    res["past_key_values"] = None
    expected["past_key_values"] = None

    assert_close(res, expected, rtol=1e-1, atol=1e-1)

    assert_close(j_static_cache.key_cache, ref_static_cache.key_cache, rtol=1e-1, atol=1e-1)
    assert_close(j_static_cache.value_cache, ref_static_cache.value_cache, rtol=1e-1, atol=1e-1)

    res["past_key_values"] = j_static_cache
    expected["past_key_values"] = ref_static_cache

    args2 = dict(
        cache_position=torch.tensor([6], device="cuda:0"),
        input_ids=torch.tensor([[311]], device="cuda:0"),
        inputs_embeds=None,
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1]], device="cuda:0"),
        use_cache=True,
    )

    res2 = jm(past_key_values=j_static_cache, **args2)
    expected2 = model2(past_key_values=ref_static_cache, **args2)

    assert res2.past_key_values is j_static_cache
    assert expected2.past_key_values is ref_static_cache
    res2["past_key_values"] = None
    expected2["past_key_values"] = None
    assert_close(res2, expected2, rtol=1e-1, atol=1e-1)

    assert_close(j_static_cache.key_cache, ref_static_cache.key_cache, rtol=1e-1, atol=1e-1)
    assert_close(j_static_cache.value_cache, ref_static_cache.value_cache, rtol=1e-1, atol=1e-1)
