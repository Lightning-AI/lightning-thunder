import pytest
from functools import partial

import torch

import thunder
import thunder.examine as examine
from thunder.examine import get_fusions
from thunder.executors.nvfuserex import nvfuserex
import thunder.torch as ltorch
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
import thunder.core.prims as prims
from thunder.core.pytree import tree_map
from thunder.core.rematerialization import (
    apply_rematerialization_for_consumer,
    apply_rematerialization_for_producer,
    find_cut,
    find_external_producer_outputs,
    find_filtered_producer_consumer_pairs,
    find_nvfuser_producer_consumer_pairs,
)
from thunder.core import utils
from thunder.core.transforms import value_and_grad

from thunder.tests.framework import (
    instantiate,
    TestExecutor,
    NOTHING,
    ops,
    run_snippet,
    assert_closer,
    nvFuserExecutor,
    TorchExecutor,
)
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.opinfos import opinfos, push_away_from_singularities, tensor_creation_ops, get_opinfo


@instantiate(
    dtypes=NOTHING,
)
def test_rematerialization_with_forward_and_backward_from_trace(executor: TestExecutor, device: str, _) -> None:
    from thunder import trace
    from thunder.clang import cos, sin
    import thunder.torch as ltorch
    from thunder.core.transforms import forward_and_backward_from_trace, value_and_grad
    from thunder.common import transform_for_execution
    from thunder.core.rematerialization import rematerialize_forward_and_backward

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

    expected_vjp_func = executor.make_callable_legacy(value_and_grad(func))

    a = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    b = make_tensor((2, 3), device=device, dtype=torch.float64, requires_grad=True)
    c = make_tensor(
        (
            2,
            3,
        ),
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    trace = trace(inline_trace=False)(func, a, b, c=c)
    fw_trace, bw_trace = forward_and_backward_from_trace(trace)

    fw_extraces = transform_for_execution(
        fw_trace, executors_list=executor.executors_list(), use_rematerialization=False
    )
    bw_extraces = transform_for_execution(
        bw_trace, executors_list=executor.executors_list(), use_rematerialization=False
    )
    fw_extrace, bw_extrace = rematerialize_forward_and_backward(fw_extraces[-1], bw_extraces[-1])

    fw = fw_extrace.python_callable()
    bw = bw_extrace.python_callable()

    fw_out, saved_for_backward = fw(a, b, c=c)
    expected_fw_out, expected_grads = expected_vjp_func(a, b, c=c)
    torch.testing.assert_close(fw_out, expected_fw_out)

    output_grads = tree_map(lambda x: torch.ones_like(x), fw_out)
    bw_out = bw(saved_for_backward, output_grads)
    torch.testing.assert_close(bw_out, expected_grads)


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_redundant_cast_basic(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a):
        b = a.to(torch.float16)
        c = b.to(torch.float64)
        return c

    cfoo = thunder.jit(foo)
    cfoo(a)

    traces = thunder.last_traces(cfoo)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)

    # Verifies that there is a single fusion with only one operation
    assert len(fusions) == 1
    fusion = fusions[0]
    assert len(fusion.subsymbols) == 1

    # Tests a longer chain of operations
    def bar(a):
        b = a.to(torch.float16)
        c = b.to(torch.float64)
        d = c.to(torch.float32)
        e = d.to(torch.float16)
        return e

    cbar = thunder.jit(bar)
    cbar(a)

    traces = thunder.last_traces(cbar)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)

    # Verifies that there is a single fusion with only one operation
    assert len(fusions) == 1
    fusion = fusions[0]
    assert len(fusion.subsymbols) == 1


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_redundant_intermediate_consumers(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a):
        b = a.to(torch.float64)
        c = b + 5
        d = b.to(torch.float16)
        return c, d

    cfoo = thunder.functional.jit(foo)
    cfoo(a)

    traces = thunder.last_traces(cfoo)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)

    # Verifies that there is a single fusion with three each operation
    assert len(fusions) == 1
    fusion = fusions[0]
    assert len(fusion.subsymbols) == 3

    # Verifies that the second conversion consumes the output of the first conversion
    #   (because the first conversion's output is used in an intermediate operation)
    assert fusion.subsymbols[-1].args[0].name == "a"


# NOTE the test relies on matmul not being executable by nvFuser
# Test that rrc pass can handle subsymbols in nvFusion, and the inputs of successors are handled properly
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_redundant_cast_nvfusion(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    x = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, x):
        b = a + 5
        c = b.to(torch.float16)
        d = c.to(torch.float32)
        e = d @ x
        f = d + 3
        g = e + x
        g1 = g.to(torch.float64)
        g2 = g1 + d
        g3 = g1.to(torch.half)
        h = f.to(torch.float16)
        i = h.to(torch.float32)
        y = i.to(torch.float64)
        return d, g, y, i, g2, g3

    cfoo = thunder.functional.jit(foo)
    cfoo(a, x)
    traces = thunder.last_traces(cfoo)

    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)
    assert len(fusions) == 2

    # Verifies that the nvFusion inputs and outputs are updated properly
    t0 = fusions[0].output
    assert fusions[1].args[0].name == "t0"
    assert t0[0].name == "t0"
    assert extrace.output[0].name == "t0"
    assert len(fusions[0].subsymbols) == 3

    # Verifies the intermediate consumer
    assert fusions[1].subsymbols[-1].args[0].name == "t5"


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_redundant_no_op(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a):
        return a.to(torch.float32)

    cfoo = thunder.jit(foo)
    cfoo(a)

    traces = thunder.last_traces(cfoo)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)

    # Verifies that no operations are performed
    assert len(fusions) == 0

    def bar(a):
        b = a.to(torch.float32)
        c = b.to(torch.float64)
        d = c.to(torch.float16)
        e = c.to(torch.float16)
        f = b.to(torch.float32)
        g = d.to(torch.float32)
        return d, e, f, g

    cbar = thunder.jit(bar)
    cbar(a)

    traces = thunder.last_traces(cbar)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)

    # Verifies a single fusion of two operations
    assert len(fusions) == 1
    fusion = fusions[0]
    assert len(fusion.subsymbols) == 1

    # Verifies that the trace outputs are updated properly
    d, e, f, g = extrace.output
    assert d.name == "d"
    assert e.name == "d"
    assert f.name == "a"
    assert g.name == "a"


@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CUDA,), executors=(nvFuserExecutor,))
def test_cse_subsymbol_removal(executor, device, _):
    from thunder.core.pytree import tree_flatten

    def func(x):
        t0 = x.relu()
        t1 = t0 + 5
        t2 = t0 + 5
        t3 = t0 @ t0
        t4 = torch.where(t3 > t1, t1, t2)
        return t4

    x = make_tensor(5, 5, dtype=torch.float16, device=device)
    compiled_func = thunder.functional.jit(func, executors=executor.executors_list())
    compiled_func(x)

    fw_trace = thunder.last_traces(compiled_func)[-1]
    fusion_bsyms = tuple(filter(lambda a: a.sym.is_fusion, fw_trace.bound_symbols))

    # There are two nvfuser fusion groups separated by the matmul operation.
    assert len(fusion_bsyms) == 2
    nvf_0, nvf_1 = fusion_bsyms

    # CSE removes the redundant (t0 + 5) operation
    assert len(nvf_0.subsymbols) == 5
    # Return t0 and t1 from the first fusion
    assert [t.name for t in tree_flatten(nvf_0.output)[0]] == ["t1", "t4"]

    # CSE does not change the second fusion
    assert len(nvf_1.subsymbols) == 2
    assert [t.name for t in tree_flatten(nvf_1.output)[0]] == ["t10"]


@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CUDA,), executors=(nvFuserExecutor,))
def test_cse_subsymbol_redundant_args(executor, device, _):
    from thunder.core.pytree import tree_flatten

    def func(w, x, y, z):
        t0 = x @ y
        t1 = t0 + z
        t2 = x @ y
        t3 = t2 + w
        t4 = t1 + t3
        return t4

    w = make_tensor(5, 5, dtype=torch.float16, device=device)
    x = make_tensor(5, 5, dtype=torch.float16, device=device)
    y = make_tensor(5, 5, dtype=torch.float16, device=device)
    z = make_tensor(5, 5, dtype=torch.float16, device=device)
    compiled_func = thunder.functional.jit(func, executors=executor.executors_list())
    compiled_func(w, x, y, z)

    fw_trace = thunder.last_traces(compiled_func)[-1]
    fusion_bsyms = tuple(filter(lambda a: a.sym.is_fusion, fw_trace.bound_symbols))

    # There is a single nvfuser fusion group.
    assert len(fusion_bsyms) == 1
    nvf_0 = fusion_bsyms[0]

    assert [t.name for t in tree_flatten(nvf_0.args)[0]] == ["t0", "w", "z"]
    assert len(nvf_0.subsymbols) == 7
    assert [t.name for t in tree_flatten(nvf_0.output)[0]] == ["t13"]


@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CUDA,), executors=(nvFuserExecutor,))
def test_cse_rematerialization(executor, device, _):
    # Unit test for "llama2.c example failed with bookend disabled."
    from thunder.tests.llama2_model import Transformer, ModelArgs
    from thunder.core.pytree import tree_flatten

    batch_size = 2
    max_seq_len = 32
    vocab_size = 32
    model_args = dict(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    model.to(device)

    x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    compiled_func = thunder.jit(
        model.eval(),
        disable_torch_autograd=True,
        executors=executor.executors_list(),
        nv_enable_bookend=False,
    )
    compiled_func(x, y)

    # Rematerialization can replace saved intermediates between fusions with extra computation.
    # In any downstream fusions, an input argument is replaced with duplicate computation.
    # This test case can only occur if rematerialization is active.
    assert nvfuserex._use_rematerialization

    fw_trace = thunder.last_traces(compiled_func)[-1]
    fusion_bsyms = tuple(filter(lambda a: a.sym.is_fusion, fw_trace.bound_symbols))
    assert len(fusion_bsyms) == 11
    # fusion groups 1 and 6 correspond with the apply_rotary_emb function
    # Nvfuser with recomputation should use precomputed cos and sin values.
    assert len(fusion_bsyms[1].args) == len(fusion_bsyms[6].args)
    assert fusion_bsyms[1].subsymbols[0].output.name == "freqs_cos"
    assert fusion_bsyms[1].subsymbols[1].output.name == "freqs_sin"
    assert fusion_bsyms[6].subsymbols[0].output.name == "freqs_cos"
    assert fusion_bsyms[6].subsymbols[1].output.name == "freqs_sin"


# Tests that two separated nvFuser regions can be merged when they don't depend
#   on an intermediate PyTorch region
# TODO Create a testing operator that can only be executed by PyTorch so that
#   these tests don't rely on matmul not being executable by nvFuser
# TODO Explicitly use the nvFuserExecutor in these tests
#   (by creating executor.make_callable_with_info?)
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_basic(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - b

        return c, d, e

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged when they have no
#   dependencies
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_independent(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - b
        f = b @ a
        g = a * b

        return c, d, e, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged when the middle region
#   depends on the first region
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent0(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - c
        f = b @ a
        g = a * b

        return c, d, e, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged when the middle
#   and final regions depend on the first one
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent1(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - c
        f = b @ a
        g = c * b

        return c, d, e, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged when each region
#   depends on the other
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent2(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - c
        f = b @ a
        g = c * e

        return c, d, e, f, g

    cfoo = thunder.jit(foo)

    result = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged when the first region
#   is entirely consumed by later regions
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent3(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - c
        f = b @ a
        g = c * e

        return d, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can be merged even if a PyTorch region has to be reordered BEFORE them
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent4(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = a @ b
        e = a - c
        f = b @ a
        g = d * e

        return d, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1


# Tests that three separated nvFuser regions can only be partially merged
#   if there's a PyTorch data dependency between them
@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_toposort_dependent5(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a, b):
        c = a + b
        d = c @ b
        e = a - c
        f = b @ a
        g = d * e

        return d, f, g

    cfoo = thunder.jit(foo)

    _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)

    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 2


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_cse_issue1789(executor, device, _):
    def func(x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = x + x
        v1 = a.view([6])
        v2 = a.view([6])
        s = s + s
        s1 = s.squeeze([0])
        s2 = s.squeeze([0])
        t1 = s.transpose(0, 1)
        t2 = s.transpose(0, 1)
        return v1 + v2, s1 + s2 + t1 + t2

    x = make_tensor(2, 3, device=device, dtype=torch.float32)
    s = make_tensor(1, 3, device=device, dtype=torch.float32)
    compiled_func = thunder.jit(func)
    compiled_func(x, s)

    traces = thunder.last_traces(compiled_func)
    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)
    assert len(fusions) == 1
    assert [subsymbol.sym.id for subsymbol in fusions[0].subsymbols].count(prims.PrimIDs.RESHAPE) == 1
    assert [subsymbol.sym.id for subsymbol in fusions[0].subsymbols].count(prims.PrimIDs.SQUEEZE) == 1
    assert [subsymbol.sym.id for subsymbol in fusions[0].subsymbols].count(prims.PrimIDs.TRANSPOSE) == 1


@instantiate(
    dtypes=NOTHING,
    executors=(
        nvFuserExecutor,
        # NOTE We might want to do transpose bookend optimization for other executors than nvFuser.
    ),
)
def test_bookend_meta_optimization(executor, device, _):
    a = torch.ones(2, 3, 5, device=device, dtype=torch.float32)

    def subtest(fn, n):
        # Enable bookending so it gets tested.
        cfn = thunder.jit(fn, nv_enable_bookend=True)

        _ = cfn(a)
        traces = thunder.last_traces(cfn)
        execution_trace = traces[-1]

        transposes_in_fusions = 0
        for bsym in execution_trace.bound_symbols:
            sym = bsym.sym
            if sym.is_fusion:
                for sbsym in bsym.subsymbols:
                    ssym = sbsym.sym
                    if ssym.id is prims.PrimIDs.TRANSPOSE:
                        transposes_in_fusions += 1

        assert (
            transposes_in_fusions == n
        ), f"Expected {n} prims.transpose operations in fusions, but found {transposes_in_fusions} transpose in fusions in the trace {traces[-1]}"

    # one transpose at the beginning
    # should be moved out of fusion
    def func_0(t):
        t0 = t.transpose(0, 1)
        t1 = t0.tanh()
        t2 = t1.sin()
        return t2

    subtest(func_0, 0)

    # one transpose at the end
    # should be moved out of fusion
    def func_1(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)
        return t2

    subtest(func_1, 0)

    # one transpose at the beginning and another at the end
    # both should be moved out of fusion
    def func_2(t):
        t0 = t.transpose(0, 1)
        t1 = t0.tanh()
        t2 = t1.sin()
        t3 = t2.transpose(0, 2)
        return t3

    subtest(func_2, 0)

    # a couple independent transposes at the beginning
    # both should be moved out of fusion
    def func_3(t):
        t0 = t.transpose(0, 1)
        t1 = t0.tanh()
        t2 = t1.sin()

        t3 = t.transpose(0, 2)
        t4 = t3.sin()
        t5 = t4.tanh()
        return t2, t5

    subtest(func_3, 0)

    # a couple independent transposes at the end
    # both should be moved out of fusion
    def func_4(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)

        t3 = t.sin()
        t4 = t3.tanh()
        t5 = t4.transpose(0, 2)
        return t2, t5

    subtest(func_4, 0)

    # a couple chained transposes at the beginning
    # both should be moved out of fusion
    def func_5(t):
        t0 = t.transpose(0, 1)
        t1 = t0.transpose(0, 2)
        t2 = t1.tanh()
        t3 = t2.sin()
        return t3

    subtest(func_5, 0)

    # a couple chained transposes at the end
    # both should be moved out of fusion
    def func_6(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)
        t3 = t2.transpose(0, 2)
        return t3

    subtest(func_6, 0)

    # a couple chained transposes at the beginning and end
    # both should be moved out of fusion
    def func_7(t):
        t0 = t.transpose(0, 1)
        t1 = t0.transpose(0, 2)
        t2 = t1.tanh()
        t3 = t2.sin()
        t4 = t3.transpose(0, 1)
        t5 = t4.transpose(0, 2)
        return t5

    subtest(func_7, 0)

    # complicated case, where two non-meta ops are each sandwiched by transpose
    # the two transposes on the edge should be moved out of fusion
    def func_8(t):
        t0 = t.transpose(0, 1)
        t1 = t0.tanh()
        # transpose in the middle should stay
        t2 = t1.transpose(0, 1)
        t3 = t2.sin()
        t4 = t3.transpose(0, 2)
        return t4

    subtest(func_8, 1)

    # NOTE func_9 and func_10 are symmetrical, this is designed to double check our toposort based approach can break
    # ties

    # complicated case, where two branches have transpose ops towards the end
    # the two transposes on the edge should be moved out of fusion
    def func_9(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)
        t3 = t2.transpose(2, 1)

        t4 = t.sin()
        t5 = t4.tanh()
        t6 = t5.transpose(0, 2)
        t7 = t6.sin()
        return t3, t7

    subtest(func_9, 1)

    # complicated case, where two branches have transpose ops towards the end
    # the two transposes on the edge should be moved out of fusion
    def func_10(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)
        t3 = t2.sin()

        t4 = t.sin()
        t5 = t4.tanh()
        t6 = t5.transpose(0, 2)
        t7 = t6.transpose(2, 1)
        return t3, t7

    subtest(func_10, 1)

    # complicated case, where a chain of transposed operations is both an output and consumed as an intermediate
    # no transposes should be removed
    def func_11(t):
        t0 = t.tanh()
        t1 = t0.sin()
        t2 = t1.transpose(0, 1)
        t3 = t2.transpose(0, 2)

        t4 = t3.sin()
        return t3, t4

    subtest(func_11, 2)

    # complicated case
    def func_12(t):
        t0 = t.transpose(0, 1)
        t1 = t0.transpose(0, 2)
        t2 = t1.tanh()
        t3 = t2 + 1.0
        t4 = t3.transpose(2, 1)
        t4 = t4.transpose(0, 1)

        t5 = t * 0.5
        # this is the only transpose that should stay in fusion, because it is surrounded by non-meta ops
        t6 = t5.transpose(0, 2)
        t7 = t6.tanh()

        t8 = t1.transpose(1, 2)

        t9 = t.transpose(2, 1)
        t10 = t9.tanh()

        t11 = t.transpose(1, 2)
        t12 = t11.transpose(0, 2)
        t13 = t12.transpose(0, 1)

        return t4, t6, t7, t8, t10, t13

    subtest(func_12, 1)


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_optimization_fuel(executor, device, _):
    def fn(x):
        return x.tanh()

    def get_num_fusions(cfn):
        traces = thunder.last_traces(cfn)
        fusions = examine.get_fusions(traces[-1])
        return len(fusions)

    nvfuserex.set_fuel(1)

    # Only the first compilation is fueled.
    x = torch.ones(2, 3, device=device, dtype=torch.float32)
    cfn_with_fusion = thunder.jit(fn)
    cfn_with_fusion(x)
    assert get_num_fusions(cfn_with_fusion) == 1

    cfn_without_fusion = thunder.jit(fn)
    cfn_without_fusion(x)
    assert get_num_fusions(cfn_without_fusion) == 0

    nvfuserex.set_fuel(thunder.extend.FUEL_LEVEL.UNLIMITED)
