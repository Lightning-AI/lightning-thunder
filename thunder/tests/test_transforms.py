import torch
from torch.testing import assert_close
from lightning_utilities.core.imports import package_available
import pytest

import thunder
from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform, nvtx_push, nvtx_pop
from thunder.tests.framework import requiresCUDA, version_between, BITSANDBYTES_AVAILABLE


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
    from thunder.tests.litgpt_model import Config

    from litgpt.model import GPT

    config = Config.from_name("llama2-like")
    with torch.device("cuda"):
        ref_m = GPT(config).to(torch.bfloat16).eval().requires_grad_(False)
    with torch.device("meta"):
        m = GPT(config).to(torch.bfloat16).eval().requires_grad_(False)

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


@pytest.mark.skipif(
    version_between(torch.__version__, min_ver="2.6.0dev0", max_ver="2.6.0a99"),
    reason="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1413",
)
@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="`bitsandbytes` is not available")
@requiresCUDA
def test_quantization_on_meta():
    from thunder.transforms import MaterializationTransform
    from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit, get_bitsandbytes_executor
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT

    bitsandbytes_executor = get_bitsandbytes_executor()

    config = Config.from_name("llama2-like")
    with torch.device("cuda"):
        ref_m = GPT(config).to(torch.bfloat16).eval().requires_grad_(False)
    with torch.device("meta"):
        m = GPT(config).to(torch.bfloat16).eval().requires_grad_(False)

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


@pytest.mark.skipif(
    version_between(torch.__version__, min_ver="2.6.0dev0", max_ver="2.6.0a99"),
    reason="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1413",
)
@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="`bitsandbytes` is not available")
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


@pytest.mark.skipif(
    version_between(torch.__version__, min_ver="2.6.0dev0", max_ver="2.6.0a99"),
    reason="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1413",
)
@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="`bitsandbytes` is not available")
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


def test_saved_for_backward_recomputation():
    import torch.nn as nn
    from thunder.core.compile_data import compile_data_and_stats
    from thunder.core.proxies import variableify, TensorProxy, unvariableify
    from thunder.core.transforms import recompute_saved_for_backward
    from thunder.core.vjp_utils import get_saved_for_backward_tensors

    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(10, 15), nn.ReLU(), nn.Linear(15, 15), nn.ReLU(), nn.Linear(15, 10))

        def forward(self, x):
            return self.layers(x)

    a = torch.randn((10, 10), requires_grad=True)
    model = MyModel()

    # Do not recompute anything
    jmodel = thunder.jit(model)
    jmodel(a)

    fwd_trace = None

    for trace in thunder.last_traces(jmodel):
        if str(trace.get_provenance()) == "# Constructed by Augmented forward pass":
            fwd_trace = trace
            break

    assert fwd_trace is not None, "Unable to find augmented forward pass trace"

    bwd_trace = thunder.last_backward_traces(jmodel)[0]
    cd = thunder.compile_data(jmodel)
    cs = thunder.compile_stats(jmodel)

    # Do not recompute any
    cd.compile_options["recomputation_policy"] = lambda x: set()
    with compile_data_and_stats(cd, cs):
        new_fwd, new_bwd = recompute_saved_for_backward(fwd_trace, bwd_trace)

    # Check that the return in the fwd is the same
    new_out = variableify(new_fwd.output[0]["output"])
    old_out = variableify(fwd_trace.output[0]["output"])
    assert new_out == old_out, f"New fwd output differs from the old, expected {old_out}, found {new_out}."

    new_flat_args = new_fwd.output[0]["flat_args"]
    old_flat_args = fwd_trace.output[0]["flat_args"]
    for new, old in zip(new_flat_args, old_flat_args):
        new = variableify(new)
        old = variableify(old)
        assert new == old, f"Return arguments expected to be the same, expected {old} but found {new}"

    # Check that the unpack in the bwd is also the same, making a set as the order might differ
    new_unpack_outs = set(map(lambda x: variableify(x), new_bwd.bound_symbols[4].flat_outs))
    old_unpack_outs = set(map(lambda x: variableify(x), bwd_trace.bound_symbols[4].flat_outs))
    assert (
        len(new_unpack_outs - old_unpack_outs) == 0
    ), f"Unpack arguments expected to be the same, expected {old_unpack_outs} but found {new_unpack_outs}"

    # Recompute all tensors
    saved_for_bw = get_saved_for_backward_tensors(fwd_trace)
    fwd_trace_args = {variableify(j) for j in fwd_trace.args}
    old_saved_for_bwd = {variableify(j) for j in saved_for_bw}

    all_rematerializable = old_saved_for_bwd - fwd_trace_args

    cd.compile_options["recomputation_policy"] = lambda x: x
    with compile_data_and_stats(cd, cs):
        _, new_bwd = recompute_saved_for_backward(fwd_trace, bwd_trace)

    # List the outputs after the unpacks
    bwd_bsym_out = set(
        map(
            lambda x: variableify(x.output),
            filter(lambda x: isinstance(x.output, TensorProxy), new_bwd.bound_symbols[6:]),
        )
    )
    # check that all the fwd are recomputed
    for rematerializable in all_rematerializable:
        assert rematerializable in bwd_bsym_out

    # Recompute only one tensor
    cd.compile_options["recomputation_policy"] = lambda x: set(filter(lambda i: unvariableify(i).name == "t7", x))
    t7 = set(filter(lambda x: unvariableify(x).name == "t7", all_rematerializable))
    with compile_data_and_stats(cd, cs):
        _, new_bwd = recompute_saved_for_backward(fwd_trace, bwd_trace)

    bwd_bsym_out = set(
        map(
            lambda x: variableify(x.output),
            filter(lambda x: isinstance(x.output, TensorProxy), new_bwd.bound_symbols[6:]),
        )
    )
    assert t7 not in bwd_bsym_out, "Unexpected tensor rematerialized in the backward."


def test_lora_transform_linear():
    from thunder.transforms import LORATransform

    DIM = 512
    rank = 16
    alpha = 32

    seed = torch.manual_seed(0)

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

    class Original(torch.nn.Module):
        def __init__(self, rank, alpha) -> None:
            from litgpt.lora import LoRALinear

            super().__init__()
            self.fc1 = LoRALinear(DIM, DIM, r=rank, lora_alpha=alpha)
            self.fc2 = LoRALinear(DIM, DIM, r=rank, lora_alpha=alpha)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc2(x)
            return x

    model = Model()
    x = torch.randn(DIM, DIM)

    loratransform = LORATransform(r=rank, lora_alpha=alpha)

    jmodel = thunder.jit(
        model,
        transforms=[
            loratransform,
        ],
    )
    actual = jmodel(x)
    original_jmodel = thunder.jit(model)
    expected = original_jmodel(x)
    assert_close(actual, expected, atol=2e-1, rtol=2e-1)

    # rename tensor names
    rename_state_dict = {}
    for k, v in model.state_dict().items():
        name = k.split(".")[0]
        weight_type = k.split(".")[1]
        rename_state_dict[f"{name}.linear.{weight_type}"] = v

    original_model = Original(rank, alpha)
    original_model.load_state_dict(rename_state_dict, strict=False)
    litgpt_lora_output = original_model(x)
    assert_close(actual, litgpt_lora_output, atol=2e-1, rtol=2e-1)


def test_constant_folding():

    # Helper to verify we see the expected constant tensors
    # in exec_trace.
    def assert_in_trace(exec_trace, sym, arg_vals):
        for bsym in exec_trace.bound_symbols:
            if bsym.sym.id == sym and bsym.args == arg_vals:
                return

        err = f"Expected to find symbol {sym} with arguments {arg_vals} in execution trace but didn't find any."
        raise RuntimeError(err)

    from thunder.transforms.constant_folding import ConstantFolding

    def forward():
        const_t = torch.tensor([2])
        getitem = (const_t * 2)[0]
        return (getitem, const_t)  # (4, [2])

    jforward = thunder.jit(forward, transforms=[ConstantFolding()])
    actual = jforward()
    expected = forward()
    torch.testing.assert_close(actual, expected)
    exec_trace = thunder.last_traces(jforward)[-1]
    assert_in_trace(exec_trace, "tensor", ([2],))
    assert_in_trace(exec_trace, "full", ((), 4))

    def forward(x):
        const_t = torch.tensor([2])
        getitem = const_t[0]  # 2
        getitem_2 = (
            torch.zeros(
                2,
            )
            + 1
        )[
            0
        ]  # 1
        return x + getitem + getitem_2

    jforward = thunder.jit(forward, transforms=[ConstantFolding()])
    x = torch.randn(3, 3)
    actual = jforward(x)
    expected = forward(x)
    torch.testing.assert_close(actual, expected)
    exec_trace = thunder.last_traces(jforward)[-1]
    assert_in_trace(exec_trace, "full", ((), 2))
    assert_in_trace(exec_trace, "full", ((), 1.0))

    def forward(x):
        const_t = torch.tensor([2], dtype=torch.float16)
        ones_t = torch.ones(1, dtype=torch.float32)
        s1 = const_t * 2  # 4
        s2 = const_t / 1  # 2
        s3 = s1 * s2 + 10  # 18
        ones_mul_10 = ones_t * 10  # 10
        return x[0, 0] + s3 + ones_mul_10

    jforward = thunder.jit(forward, transforms=[ConstantFolding()])
    x = torch.randn(3, 3)
    actual = jforward(x)
    expected = forward(x)
    torch.testing.assert_close(actual, expected)
    exec_trace = thunder.last_traces(jforward)[-1]
    assert_in_trace(exec_trace, "tensor", ([18.0],))
    assert_in_trace(exec_trace, "tensor", ([10.0],))

    # Constant folding of Python constants.
    def forward(x):
        t = torch.tensor(2.0)
        return x + (t.item() + (t + 1).item())

    jforward = thunder.jit(forward, transforms=[ConstantFolding()])
    x = torch.randn(3, 3)
    actual = jforward(x)
    expected = forward(x)
    torch.testing.assert_close(actual, expected)
    exec_trace = thunder.last_traces(jforward)[-1]
    # exec_trace will look something like this
    # def computation(x):
    #     # x: "cpu f32[3]"
    #     t5 = torch.add(x, 5.0, alpha=1)  # t5: "cpu f32[3]"
    #         # t5 = ltorch.add(x, 5.0, alpha=1)  # t5: "cpu f32[3]"
    #         # t5 = prims.add(x, 5.0)  # t5: "cpu f32[3]"
    #     return t5

    # So we check that torch.add has 5.0 in it's arguments.
    for bsym in exec_trace.bound_symbols:
        if bsym.sym.id == "add":
            assert bsym.args[1] == 5.0
            break
    else:
        raise RuntimeError("Failed to find `add` symbol in trace")


@requiresCUDA
def test_cudagraph_empty_inputs():
    def fn():
        a = torch.ones(5, 5, device="cuda")
        b = a * 2
        return b

    from thunder.transforms.cudagraph import CUDAGraphTransform

    jfn = thunder.jit(fn, transforms=(CUDAGraphTransform(),), executors=())
    assert_close(jfn(), fn())

    assert any(("CUDAGraph" in bsym.sym.name) for bsym in thunder.last_traces(jfn)[-1].bound_symbols)


def test_disable_params_and_buffer_check():
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT
    from thunder.transforms.extraction_only_prologue_transform import ExtractionOnlyPrologueTransform

    model = GPT(Config.from_name("llama1-like", n_layer=1))
    x = torch.randint(model.max_seq_length, (2, 5))
    cmodel = thunder.jit(model, transforms=[ExtractionOnlyPrologueTransform()])
    _ = cmodel(x)
    prologue_trc = thunder.last_prologue_traces(cmodel)[-1]

    check_bsyms = tuple(
        filter(
            lambda bsym: bsym.sym.id == thunder.executors.pythonex.check_tensor_shape_and_metadata.id,
            prologue_trc.bound_symbols,
        )
    )

    assert len(check_bsyms) == 1  # We only have the check for input.
