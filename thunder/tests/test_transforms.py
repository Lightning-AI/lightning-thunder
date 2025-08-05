import torch
from torch.testing import assert_close
import pytest

import thunder
from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform, nvtx_push, nvtx_pop
from thunder.tests.framework import requiresCUDA, BITSANDBYTES_AVAILABLE


class MiniModel(torch.nn.Module):
    def __init__(self, DIM) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(DIM, DIM)
        self.fc2 = torch.nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


@requiresCUDA
def test_nvtx_transform():
    DIM = 2

    model = MiniModel(DIM)
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

    def post_callback(bsym, output, *args, **kwargs):
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

    tr = thunder.last_traces(jmodel)[-1]
    flat_arg_names = [a.name for a in tr.bound_symbols[-1].args[0]["flat_args"]]
    arg_names = [a.name for a in tr.args]
    assert flat_arg_names == arg_names


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
        )[0]  # 1
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


@requiresCUDA
def test_cudagraph_fw_bw():
    import torch
    import thunder
    import litgpt
    from thunder.tests.litgpt_model import Config
    from torch.testing import make_tensor
    from functools import partial
    from thunder.transforms.cudagraph import CUDAGraphTransform

    device = torch.device("cuda")

    cfg = Config.from_name("llama2-like")
    with device:
        make = partial(make_tensor, low=0, high=255, device=device, dtype=torch.long, requires_grad=False)
        shape = (1,) + (cfg.block_size,)

        x = make(shape)
        m = litgpt.GPT(cfg)

    cg_transform = CUDAGraphTransform(share_mem_pool=True)
    m = thunder.jit(m, transforms=[cg_transform])

    before_snapshot = torch.cuda.memory_snapshot()

    o = m(x)
    o.sum().backward()

    after_snapshot = torch.cuda.memory_snapshot()

    # Ensure all saved for backwards tensors are marked as static inputs
    # the grad_out and idx will not be in a static mem location, all others should be
    assert all(cg_transform.cuda_graph_runner.python_callables["CUDAGraph2"][1][1:-2])

    # Ensure that all newly allocated segments are allocated in the shared memeory pool or the global pool
    for segment in after_snapshot:
        if segment in before_snapshot:
            continue
        else:
            assert (
                segment["segment_pool_id"] == cg_transform.cuda_graph_runner.mem_pool[0]
                or str(segment["segment_pool_id"]) == "(0, 0)"
            )


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


def test_disable_params_check_thunderfx():
    from thunder.dynamo import thunderfx

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(16, 16)
            self.register_buffer(
                "buffer",
                torch.randn(
                    16,
                ),
            )
            self.fc2 = torch.nn.Linear(16, 16)

        def forward(self, x):
            return self.fc1(x) + self.buffer + self.fc2(x)

    model = Model()
    x = torch.randn(16, 16)
    # NOTE: The `ExtractionOnlyPrologueTransform` transform is applied by default on `thunderfx` path.
    cmodel = thunderfx(model)
    _ = cmodel(x)
    tfn = cmodel._backend.subgraph_infos[0].thunder_compiled_fns[0]
    prologue_trc = thunder.last_prologue_traces(tfn)[-1]

    check_bsyms = tuple(
        filter(
            lambda bsym: bsym.sym.id == thunder.executors.pythonex.check_tensor_shape_and_metadata.id,
            prologue_trc.bound_symbols,
        )
    )

    # Currently we don't detect buffers on thunderfx path and hence don't remove
    # the corresponding checks from prologue.
    # This will fails when we detect buffers and remove their checks from prologue.
    assert len(check_bsyms) == 2  # 1 check for input and 1 for buffer (and 0 for parameters)


def test_buffer_dtype_casting():
    import torch.nn as nn
    import itertools

    class CastBuffers(thunder.core.transform_common.Transform):
        def __init__(self):
            self.cast_states = {}

        def transform_module(self, model: thunder.ThunderModule):
            self.thunder_module = model
            for n, b in model._model.named_buffers():
                qb = b.to(torch.bfloat16)
                self.cast_states[n] = {
                    "dtype": b.dtype,
                    "shape": tuple(b.shape),
                    "qb.dtype": qb.dtype,
                    "qb.shape": tuple(qb.shape),
                }
                model._overrides_buffers[n] = qb

        def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
            tm = self.thunder_module

            checks = thunder.transforms.utils.get_checks(prologue_trace)

            prologue_proxy_map = {
                get_param_bsym.output.name: dict(
                    shape=self.cast_states[model_weight_name]["qb.shape"],
                    dtype=thunder.dtypes.to_dtype(self.cast_states[model_weight_name]["qb.dtype"]),
                )
                for model_weight_name, (check_bsym, get_param_bsym) in checks.items()
                if model_weight_name in self.cast_states
            }

            # here we switch the prologue_trace to a copy with new metadata
            prologue_trace = thunder.transforms.utils.trace_with_replaced_proxy_metadata(
                prologue_trace, prologue_proxy_map
            )

            checks = thunder.transforms.utils.get_checks(prologue_trace)
            for n, qs in self.cast_states.items():
                check, get_param = checks[n]
                # check has args: tensor, shape, device, dtype, requires_grad
                proxy, _, device, _, requires_grad = check.args
                check.args = (
                    proxy,
                    qs["qb.shape"],
                    device,
                    qs["qb.dtype"],
                    False,
                )

            computation_proxy_map = {
                csym.name: dict(
                    shape=psym.shape,
                    dtype=psym.dtype,
                )
                for psym, csym in zip(prologue_trace.bound_symbols[-1].args[0][0], computation_trace.args)
                if psym.shape != csym.shape or psym.dtype != csym.dtype
            }

            new_computation_trace = thunder.transforms.utils.trace_with_replaced_proxy_metadata(
                computation_trace, computation_proxy_map
            )

            producers, consumers = thunder.core.utils.producers_and_consumers(new_computation_trace)

            bound_symbols = new_computation_trace.bound_symbols
            new_computation_trace.bound_symbols = []

            new_computation_trace._siginfo.args = [(a.name, None) for a in new_computation_trace.args]

            computation_proxy_map = {}
            new_bound_symbols = []
            for bsym in bound_symbols:
                if bsym.sym == thunder.torch.to and producers[bsym.args[0]].sym == thunder.core.prims.unpack_trivial:
                    inp = bsym.args[0]
                    args = (inp, inp.dtype, *bsym.args[2:])
                    computation_proxy_map[bsym.output.name] = dict(shape=inp.shape, dtype=inp.dtype)
                    assert (
                        len(bsym.subsymbols) == 1 and bsym.subsymbols[0].sym == thunder.core.prims.convert_element_type
                    )
                    subsymbols = [bsym.subsymbols[0].from_bsym(args=(inp, inp.dtype))]
                    new_bound_symbols.append(bsym.from_bsym(args=args, subsymbols=subsymbols))
                else:
                    new_bound_symbols.append(bsym.from_bsym())

            new_computation_trace.bound_symbols = new_bound_symbols

            new_computation_trace = thunder.transforms.utils.trace_with_replaced_proxy_metadata(
                new_computation_trace, computation_proxy_map
            )

            new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("Dtype Convert"))
            return prologue_trace, new_computation_trace, epilogue_trace

    class cast(nn.Module):
        def __init__(
            self,
            k_shape: tuple[int, int, int, int],
            v_shape: tuple[int, int, int, int],
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
            self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

        def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # move the buffer to the activation dtype for when AMP is used
            self.k = self.k.to(k.dtype)
            self.v = self.v.to(v.dtype)
            # update the cache
            return self.k, self.v

    with torch.device("cpu"):
        k_shape = (2, 3, 4, 5)
        v_shape = (2, 3, 4, 5)
        device = torch.device("cpu")
        dtype = torch.float32
        model = cast(k_shape, v_shape, device=device, dtype=dtype).eval().requires_grad_(False)

    k = torch.randn(2, 3, 4, 5, device=device, dtype=torch.half)
    v = torch.randn(2, 3, 4, 5, device=device, dtype=torch.half)
    cast_jit = thunder.jit(
        model,
        transforms=[
            CastBuffers(),
        ],
    )
    output_k, output_v = cast_jit(k, v)

    def check_dtypes(bsym):
        for a in itertools.chain(bsym.flat_args, bsym.flat_outs):
            if isinstance(a, thunder.TensorProxy):
                assert a.dtype == thunder.dtypes.bfloat16
        for sbsym in bsym.subsymbols:
            check_dtypes(sbsym)

    for tr in thunder.last_traces(cast_jit):
        if str(tr.get_provenance()) == "# Constructed by Dtype Convert":
            for bsym in tr.bound_symbols:
                check_dtypes(bsym)


def test_prune_prologue_checks():
    DIM = 2
    m = MiniModel(DIM)
    inp = torch.randn(4, DIM)

    def count_tensor_checks(tr):
        return len([bsym for bsym in tr.bound_symbols if "check_tensor" in bsym.sym.name])

    # without pruning: checks for 1 input and each parameter
    jm = thunder.jit(m)
    jm(inp)
    assert count_tensor_checks(thunder.last_prologue_traces(jm)[-1]) == 1 + len(list(jm.parameters()))

    # with default of pruning module checks: check for 1 input
    jm = thunder.jit(m, transforms=(thunder.transforms.PrunePrologueChecks(),))
    jm(inp)
    assert count_tensor_checks(thunder.last_prologue_traces(jm)[-1]) == 1

    # with pruning all checks: none left
    jm = thunder.jit(m, transforms=(thunder.transforms.PrunePrologueChecks(prune_all_checks=True),))
    jm(inp)
    assert count_tensor_checks(thunder.last_prologue_traces(jm)[-1]) == 0


def test_dce_duplicate_number_proxies():
    from thunder.core.prims import PrimIDs

    def fn(x):
        shape_0 = x.shape
        shape_1 = x.clone().shape  # duplicate shape query
        return sum(shape_0)

    # symbolic values is necessary to have the shape query in trace
    jfn = thunder.jit(fn, cache="symbolic values")

    a = torch.randn(2, 3, 4, 5)
    out = jfn(a)

    def _count_shape_query(trace):
        count = 0
        for bsym in trace.bound_symbols:
            if bsym.sym.id == PrimIDs.SHAPE:
                count += 1
        return count

    # original two shape queries should both exist in the original trace
    trace = thunder.last_traces(jfn)[0]
    assert _count_shape_query(trace) == 2

    # dce should remove duplicate shape queries
    trace = thunder.core.transforms.dce(trace)
    assert _count_shape_query(trace) == 1


def test_cache_symbolic_values_grad_matmul():
    def foo(a, w):
        return torch.nn.functional.linear(a, w)

    jfoo = thunder.jit(foo, cache="symbolic values")

    a = torch.randn(2, 8, 6)
    b = torch.randn(4, 6)
    a_ref = a.clone()
    b_ref = b.clone()
    for x in (a, b, a_ref, b_ref):
        x.requires_grad_()
    actual = jfoo(a, b)
    expected = foo(a_ref, b_ref)
    actual.sum().backward()
    expected.sum().backward()

    assert_close(actual, expected)
    assert_close(a.grad, a_ref.grad)
    assert_close(b.grad, b_ref.grad)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    a = torch.randn(4, 4, 2)
    b = torch.randn(8, 2)
    a_ref = a.clone()
    b_ref = b.clone()
    for x in (a, b, a_ref, b_ref):
        x.requires_grad_()
    actual = jfoo(a, b)
    expected = foo(a_ref, b_ref)
    actual.sum().backward()
    expected.sum().backward()

    assert_close(actual, expected)
    assert_close(a.grad, a_ref.grad)
    assert_close(b.grad, b_ref.grad)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1


def test_cache_symbolic_values_grad_unsqueeze():
    def foo(x):
        cache = torch.arange(0, 128, 1)
        cache_unsqueezed = cache.unsqueeze(0)
        return x + cache_unsqueezed

    jfoo = thunder.jit(foo, cache="symbolic values")

    a = torch.randn(2, 8, 128)
    a_ref = a.clone()
    for x in (a, a_ref):
        x.requires_grad_()
    actual = jfoo(a)
    expected = foo(a_ref)
    actual.sum().backward()
    expected.sum().backward()
    assert_close(actual, expected)
    assert_close(a.grad, a_ref.grad)
