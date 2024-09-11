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

    for p in m.parameters():
        p._thunder_device = torch.device("cuda")

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
        p._thunder_device = torch.device("cuda")

    jm_ref = thunder.jit(
        ref_m,
        executors=(bitsandbytes_executor,),
        transforms=[BitsAndBytesLinearQuant4bit()],
    )
    init_from_orig_sd = MaterializationTransform.init_from_original_state_dict(ref_m.state_dict())
    jm = thunder.jit(
        m,
        executors=(bitsandbytes_executor,),
        transforms=[
            BitsAndBytesLinearQuant4bit(),
            MaterializationTransform("cuda", init=init_from_orig_sd),
        ],
    )
    init_from_transformed_sd = MaterializationTransform.init_from_transformed_state_dict(jm_ref.state_dict())
    jm2 = thunder.jit(
        m,
        executors=(bitsandbytes_executor,),
        transforms=[
            BitsAndBytesLinearQuant4bit(),
            MaterializationTransform("cuda", init=init_from_transformed_sd),
        ],
    )

    x = torch.randint(1, 255, (1, 10), device="cuda")
    input_pos = torch.arange(10, device="cuda")

    expected = jm_ref(x, input_pos)
    actual = jm(x, input_pos)
    actual2 = jm2(x, input_pos)

    for n, p in jm_ref.named_parameters():
        p2 = jm.get_parameter(n)
        assert_close(p, p2)
        p2 = jm2.get_parameter(n)
        assert_close(p, p2)
    for n, b in jm_ref.named_buffers():
        b2 = jm.get_buffer(n)
        assert_close(b2, b)
        b2 = jm2.get_buffer(n)
        assert_close(b2, b)

    assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert_close(actual, actual2)


@requiresCUDA
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

    assert_close(actual, expected, atol=2e-1, rtol=2e-1)

    cache_info, comp_inps, _ = thunder.compile_data(jm).get_computation_and_inputs(inp)
    for t, comp_proxy, prologue_proxy in zip(
        comp_inps, cache_info.computation_traces[-1].args, cache_info.prologue_traces[-1].bound_symbols[-1].args[0][0]
    ):
        # this needs relaxing for dynamic shapes
        assert comp_proxy.shape == t.shape, f"{comp_proxy} does not match {t.shape=}"
        assert prologue_proxy.shape == t.shape
        assert comp_proxy.device == thunder.core.devices.to_device(t.device)
        assert prologue_proxy.device == thunder.core.devices.to_device(t.device)
        assert comp_proxy.dtype == thunder.core.dtypes.to_dtype(t.dtype)
        assert prologue_proxy.dtype == thunder.core.dtypes.to_dtype(t.dtype)


def test_debug_transform():
    from thunder.dev_utils.debug_transform import debug_execution_trace

    # Only use the primitive operations in `fn` so that
    # we can count them easily.
    N_PRIMITIVE_OPS = 3

    def fn(x, y):
        return (x + y * y) / x

    def pre_callback(bsym, *args, **kwargs):
        return f"Pre - {bsym.sym.name}"

    def post_callback(bsym, *args, **kwargs):
        return f"Post - {bsym.sym.name}"

    jfn = debug_execution_trace(thunder.jit(fn), pre_callback=pre_callback, post_callback=post_callback)
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    jfn(x, y)

    fwd_exec_trace = thunder.last_traces(jfn)[-1]

    debug_syms = set()
    for bsym in fwd_exec_trace.bound_symbols:
        if bsym.sym.name.startswith("debug"):
            debug_syms.add(bsym)

    n_expected_debug_syms = 2 * N_PRIMITIVE_OPS  # Multiply by 2 as we have both pre and post callbacks
    assert len(debug_syms) == n_expected_debug_syms

    # As `debug_syms` have name of the form `debug_{pre|post}_{sym_name}_{debug_count}`,
    # we expect to see debug_sym with `N_PRIMITIVE_OPS` at `{debug_count}` part of the name.
    assert any(map(lambda bsym: bsym.sym.name.endswith(f"{str(N_PRIMITIVE_OPS)}"), debug_syms))

    # Verify that we have correctly set the header for all debug_syms.
    debug_headers = {sym.header for sym in debug_syms}
    expected_headers = {"Pre - true_divide", "Pre - add", "Pre - mul", "Post - true_divide", "Post - add", "Post - mul"}
    assert debug_headers == expected_headers


@requiresCUDA
def test_cudagraph_warmup_runs_with_correct_buffers():
    """
    Tests whether newly-created buffers are being properly initialized.
    Otherwise we should expect failures because of incorrect values.
    """

    from thunder.transforms.cudagraph import CUDAGraphTransform

    weights = torch.tensor([0, 10, 3, 0], device="cuda", dtype=torch.float)

    def f(x):
        return torch.multinomial(x, num_samples=3, replacement=True)

    jf = thunder.jit(f, transforms=[CUDAGraphTransform()])
    jf(weights)
    jf(weights)


@requiresCUDA
def test_materialization_init():
    from thunder.transforms import MaterializationTransform
    from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit, get_bitsandbytes_executor

    bitsandbytes_executor = get_bitsandbytes_executor()

    def get_model():
        m0 = torch.nn.Linear(2, 2)

        # to not change the rng state
        with torch.device("meta"):
            m4 = torch.nn.Linear(2, 2)

        m4.weight = m0.weight
        m4.bias = m0.bias

        return (
            torch.nn.Sequential(
                m0,
                torch.nn.GELU(),
                torch.nn.Linear(2, 2),
                torch.nn.GELU(),
                m4,
            )
            .eval()
            .requires_grad_(False)
        )

    torch.manual_seed(1234)
    with torch.device("cuda"):
        m_ref = get_model()
        inp = torch.randn(3, 2)

    jm_ref = thunder.jit(m_ref, transforms=[BitsAndBytesLinearQuant4bit()], executors=(bitsandbytes_executor,))

    torch.manual_seed(1234)
    init_from_module_init = MaterializationTransform.init_from_original_module_init()
    with torch.device("meta"):
        m = get_model()

    jm = thunder.jit(
        m,
        transforms=[BitsAndBytesLinearQuant4bit(), MaterializationTransform("cuda", init=init_from_module_init)],
        executors=(bitsandbytes_executor,),
    )

    assert_close(jm(inp), jm_ref(inp))

    assert jm_ref._get_shared_names()["0.weight"] == {"0.weight", "4.weight"}
    assert jm._get_shared_names()["0.weight"] == {"0.weight", "4.weight"}
