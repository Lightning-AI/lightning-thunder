import torch
from torch.testing import assert_close

import thunder
from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform, nvtx_push, nvtx_pop
from thunder.tests.framework import requiresCUDA
from thunder.tests import litgpt_model


@requiresCUDA
def test_nvtx_transform():
    DIM = 2

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(DIM, DIM)
            self.fc2 = torch.nn.Linear(DIM, DIM)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc2(x)
            return x

    model = Model()
    x = torch.randn(4, DIM)

    # Transform for profiling with NVTX markers.
    # This transform adds NVTX markers to all the computation symbols in the execution trace.

    nvtx_profile_transform = NvtxProfileTransform()

    jmodel = thunder.jit(
        model,
        transforms=[
            nvtx_profile_transform,
        ],
    )
    o = jmodel(x)
    o.sum().backward()

    fwd_trc = thunder.last_traces(jmodel)[-1]
    bwd_trc = thunder.last_backward_traces(jmodel)[-1]

    def _test_equal_nvtx_push_and_pop(trc):
        # This test verifies that we see equal number of `nvtx_push` and `nvtx_pop` symbols
        # and also order is `nvtx_push` followed by `nvtx_pop`

        # First NVTX symbol will always be push.
        EXPECT_PUSH = True
        push_cnt = 0
        pop_cnt = 0
        for bsym in trc.bound_symbols:
            if bsym.sym.id in (nvtx_push.id, nvtx_pop.id):
                if EXPECT_PUSH:
                    assert bsym.sym.id == nvtx_push.id
                    push_cnt += 1
                    # After `nvtx_push` we expect next the `nvtx` symbol to be `nvtx_pop`
                    EXPECT_PUSH = False
                else:
                    assert bsym.sym.id == nvtx_pop.id
                    pop_cnt += 1
                    # After `nvtx_pop` we expect next the `nvtx` symbol to be `nvtx_push`
                    EXPECT_PUSH = True

        assert push_cnt == pop_cnt

    _test_equal_nvtx_push_and_pop(fwd_trc)
    _test_equal_nvtx_push_and_pop(bwd_trc)


@requiresCUDA
def test_materialization():
    from thunder.transforms import MaterializationTransform

    config = litgpt_model.Config.from_name("llama2-like")
    with torch.device("cuda"):
        ref_m = litgpt_model.GPT(config).to(torch.bfloat16).eval().requires_grad_(False)
    with torch.device("meta"):
        m = litgpt_model.GPT(config).to(torch.bfloat16).eval().requires_grad_(False)

    ref_m.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    ref_m.max_seq_length = 20

    # the kvcache is not in the state dict, so it must be cuda to start
    m.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    m.max_seq_length = 20
    m.cos, m.sin = ref_m.cos.clone(), ref_m.sin.clone()

    for p in m.parameters():
        p.__thunder_device = torch.device("cuda")

    init_from_sd = MaterializationTransform.init_from_original_state_dict(ref_m.state_dict())
    jm = thunder.jit(m, transforms=[MaterializationTransform("cuda", init=init_from_sd)], executors=())

    x = torch.randint(1, 255, (1, 10), device="cuda")
    input_pos = torch.arange(10, device="cuda")

    expected = ref_m(x, input_pos)
    actual = jm(x, input_pos)

    for n, p in ref_m.named_parameters():
        p2 = jm.get_parameter(n)
        assert_close(p, p2)
    for n, b in ref_m.named_buffers():
        b2 = jm.get_buffer(n)
        assert_close(b2, b)

    assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@requiresCUDA
def test_quantization_on_meta():
    from thunder.transforms import MaterializationTransform
    from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit, get_bitsandbytes_executor

    bitsandbytes_executor = get_bitsandbytes_executor()

    config = litgpt_model.Config.from_name("llama2-like")
    with torch.device("cuda"):
        ref_m = litgpt_model.GPT(config).to(torch.bfloat16).eval().requires_grad_(False)
    with torch.device("meta"):
        m = litgpt_model.GPT(config).to(torch.bfloat16).eval().requires_grad_(False)

    ref_m.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    ref_m.max_seq_length = 20

    # the kvcache is not in the state dict, so it must be cuda to start
    m.set_kv_cache(1, device="cuda", dtype=torch.bfloat16)
    m.max_seq_length = 20
    m.cos, m.sin = ref_m.cos.clone(), ref_m.sin.clone()

    for p in m.parameters():
        p.__thunder_device = torch.device("cuda")

    init_from_sd = MaterializationTransform.init_from_original_state_dict(ref_m.state_dict())

    jm_ref = thunder.jit(
        ref_m,
        executors=(bitsandbytes_executor,),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )
    jm = thunder.jit(
        m,
        executors=(bitsandbytes_executor,),
        transforms=[
            BitsAndBytesLinearQuant4bit(),
            MaterializationTransform("cuda", init=init_from_sd),
        ],
    )

    x = torch.randint(1, 255, (1, 10), device="cuda")
    input_pos = torch.arange(10, device="cuda")

    expected = jm_ref(x, input_pos)
    actual = jm(x, input_pos)

    for n, p in jm_ref.named_parameters():
        p2 = jm.get_parameter(n)
        assert_close(p, p2)
    for n, b in jm_ref.named_buffers():
        b2 = jm.get_buffer(n)
        assert_close(b2, b)

    assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_nvfuser_cse():
    with torch.device("cuda"):
        mlp = (
            torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 512),
            )
            .eval()
            .requires_grad_(False)
        )
        inp = torch.randn(1, 512)

    from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit, get_bitsandbytes_executor
    from thunder.executors.nvfuserex import nvfuserex

    bitsandbytes_executor = get_bitsandbytes_executor()

    jm = thunder.jit(
        mlp,
        executors=(bitsandbytes_executor, nvfuserex),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )

    actual = jm(inp)
    expected = mlp(inp)

    assert_close(actual, expected, atol=1e-1, rtol=1e-1)
