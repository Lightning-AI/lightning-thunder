import pytest
from functools import partial

import torch

import thunder
import thunder.examine as examine
from thunder.examine import get_fusions
from thunder.executors.nvfuserex import nvfuser_version, nvfuserex
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
from thunder.tests.opinfos import (
    opinfos,
    tensor_creation_ops,
    get_opinfo,
    linear_opinfo,
    matmul_opinfo,
)
from looseversion import LooseVersion


@instantiate(
    dtypes=NOTHING,
)
def test_rematerialization_with_forward_and_backward_from_trace(executor: TestExecutor, device: str, _) -> None:
    from thunder import trace
    from thunder.clang import cos, sin
    import thunder.torch as ltorch
    from thunder.core.transforms import forward_and_backward_from_trace, value_and_grad
    from thunder.core.transform_common import wrap_return_value_together_with_arguments
    from thunder.common import transform_for_execution
    from thunder.core.rematerialization import rematerialize_forward_and_backward

    def func(a, b, *, c):
        d = a + b + c
        e = d * a + d * b + d * c
        return sin(e) + cos(e), e, ltorch.sin(e) + ltorch.cos(e)

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
    trace = wrap_return_value_together_with_arguments(trace)
    fw_trace, bw_trace = forward_and_backward_from_trace(trace)

    fw_extraces = transform_for_execution(fw_trace, executors_list=executor.executors_list())
    bw_extraces = transform_for_execution(bw_trace, executors_list=executor.executors_list())
    fw_extrace, bw_extrace = rematerialize_forward_and_backward(fw_extraces[-1], bw_extraces[-1])

    fw = fw_extrace.python_callable()
    bw = bw_extrace.python_callable()

    fw_out, saved_for_backward = fw(a, b, c=c)

    initial_trace = thunder.trace()(value_and_grad(func), a, b, c=c)
    expected_vjp_func = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
    expected_fw_out, expected_grads = expected_vjp_func(a, b, c=c)
    torch.testing.assert_close(fw_out["output"], expected_fw_out)

    output_grads = tree_map(lambda x: torch.ones_like(x), fw_out["output"])
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

    cfoo = thunder.jit(foo)
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

    cfoo = thunder.jit(foo)
    cfoo(a, x)
    traces = thunder.last_traces(cfoo)

    extrace = traces[-1]
    fusions = examine.get_fusion_symbols(extrace)
    assert len(fusions) == 2

    # Verifies that the nvFusion inputs and outputs are updated properly
    t0 = fusions[0].output[0]
    assert fusions[1].args[2].name == "b"
    assert t0.name == "b"
    assert extrace.output[0].name == "b"
    assert len(fusions[0].subsymbols) == 3

    # Verifies the intermediate consumer
    assert fusions[1].subsymbols[-1].args[0].name == "g"


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
    compiled_func = thunder.jit(func, executors=executor.executors_list())
    compiled_func(x)

    fw_trace = thunder.last_traces(compiled_func)[-1]
    fusion_bsyms = tuple(filter(lambda a: a.sym.is_fusion, fw_trace.bound_symbols))

    # There are two nvfuser fusion groups separated by the matmul operation.
    assert len(fusion_bsyms) == 2

    # CSE removes the redundant (t0 + 5) operation
    nvf_0, nvf_1 = fusion_bsyms
    assert len(nvf_0.subsymbols) + len(nvf_1.subsymbols) == 7

    outside_fusion_syms = ["unpack_trivial", "matmul", "python_return", "python_del"]
    assert {el.sym.name for el in fw_trace.bound_symbols if not el.sym.is_fusion} == set(outside_fusion_syms)


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
    compiled_func = thunder.jit(func, executors=executor.executors_list())
    compiled_func(w, x, y, z)

    fw_trace = thunder.last_traces(compiled_func)[-1]
    fusion_bsyms = tuple(filter(lambda a: a.sym.is_fusion, fw_trace.bound_symbols))

    # There is a single nvfuser fusion group.
    assert len(fusion_bsyms) == 1
    nvf_0 = fusion_bsyms[0]

    assert [t.name for t in tree_flatten(nvf_0.args)[0]] == ["t0", "z", "w"]
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
    assert len(fusion_bsyms) == 9
    # fusion groups 1 and 6 correspond with the apply_rotary_emb function
    # Nvfuser with recomputation should use precomputed cos and sin values.
    assert len(fusion_bsyms[1].args) == len(fusion_bsyms[5].args)

    # Below, we check that freqs_sin and freqs_cos are used
    # in the same operation in both fusions.
    (fusion1_freqs_sin_arg,) = (a for a in fusion_bsyms[1].args if a.name == "freqs_sin")
    (fusion1_freqs_cos_arg,) = (a for a in fusion_bsyms[1].args if a.name == "freqs_cos")
    (fusion5_freqs_sin_arg,) = (a for a in fusion_bsyms[5].args if a.name == "freqs_sin")
    (fusion5_freqs_cos_arg,) = (a for a in fusion_bsyms[5].args if a.name == "freqs_cos")

    (fusion1_freqs_sin_user,) = (s for s in fusion_bsyms[1].subsymbols if s.args[0] is fusion1_freqs_sin_arg)
    (fusion6_freqs_sin_user,) = (s for s in fusion_bsyms[5].subsymbols if s.args[0] is fusion5_freqs_sin_arg)

    assert fusion1_freqs_sin_user.sym is fusion6_freqs_sin_user.sym
    assert fusion1_freqs_sin_user.args[1:] == fusion6_freqs_sin_user.args[1:]
    (fusion1_freqs_cos_user,) = (s for s in fusion_bsyms[1].subsymbols if s.args[0] is fusion1_freqs_cos_arg)
    (fusion5_freqs_cos_user,) = (s for s in fusion_bsyms[5].subsymbols if s.args[0] is fusion5_freqs_cos_arg)

    assert fusion1_freqs_cos_user.sym is fusion5_freqs_cos_user.sym
    assert fusion1_freqs_cos_user.args[1:] == fusion5_freqs_cos_user.args[1:]


# Tests that two separated nvFuser regions can be merged when they don't depend
#   on an intermediate PyTorch region
# TODO Create a testing operator that can only be executed by PyTorch so that
#   these tests don't rely on matmul not being executable by nvFuser
# TODO Explicitly use the nvFuserExecutor in these tests
#   (by creating executor.make_callable?)
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


@instantiate(
    dtypes=(thunder.float16, thunder.bfloat16),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
    decorators=(
        pytest.mark.skipif(
            nvfuser_version() is None or nvfuser_version() < LooseVersion("0.2.3"),
            reason="Requires nvFuser version 0.2.3 or later",
        ),
        pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"]),
    ),
)
def test_linear(executor, device: str, dtype: dtypes.dtype, has_bias: bool):

    def fn(a, b, bias=None):
        return torch.nn.functional.linear(a, b, bias)

    for sample in linear_opinfo.sample_inputs(device, dtype):
        if nvfuser_version() < LooseVersion("0.2.5") and sample.args[0].ndim != 2:
            # Only 2D inputs are supported for version < 0.2.5.
            continue

    compiled_func = thunder.jit(fn, executors_list=executor.executors_list(), nv_enable_linear=True)

    out = compiled_func(*sample.args)
    traces = thunder.last_traces(compiled_func)
    fusions = examine.get_fusions(traces[-1])

    assert len(fusions) == 1
    torch.testing.assert_close(out, torch.nn.functional.linear(*sample.args))


@instantiate(
    dtypes=(thunder.float16, thunder.bfloat16),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
    decorators=(
        pytest.mark.skipif(
            nvfuser_version() is None or nvfuser_version() < LooseVersion("0.2.2"),
            reason="Requires nvFuser version 0.2.2 or later",
        ),
    ),
)
def test_matmul(executor, device: str, dtype: dtypes.dtype):

    def fn(a, b):
        return torch.matmul(a, b)

    for sample in matmul_opinfo.sample_inputs(device, dtype):
        if nvfuser_version() < LooseVersion("0.2.4") and (sample.args[0].ndim != 2 or sample.args[1].ndim != 2):
            # Only 2D inputs are supported for version < 0.2.4.
            continue

        compiled_func = thunder.jit(fn, executors_list=executor.executors_list(), nv_enable_matmul=True)

        out = compiled_func(*sample.args)
        traces = thunder.last_traces(compiled_func)
        fusions = examine.get_fusions(traces[-1])

        assert len(fusions) == 1
        torch.testing.assert_close(out, torch.matmul(*sample.args))


@instantiate(
    dtypes=NOTHING,
    executors=(nvFuserExecutor,),
)
def test_rm_unused_inputs_of_nvfusion(executor, device, _):
    import operator

    def foo(t, ab):
        return operator.getitem(t, ab)

    t = make_tensor(5, 3, device=device, dtype=torch.float32)
    ab = (slice(3, 1), slice(1, 2))
    # enable bookend would remove the error and let you look at the trace without fusion.
    jfoo = thunder.jit(
        foo,
        cache="no caching",
        disable_torch_autograd=True,
        nv_enable_bookend=False,
    )
    out = jfoo(t, ab)
    out_ref = foo(t, ab)

    assert out.equal(out_ref)


# TODO: we should improve our consistency testing
# to also include checks for the result of meta functions.
@instantiate(
    dtypes=(thunder.int64, thunder.int32),
    executors=(nvFuserExecutor,),
)
def test_div_truediv_integer_tensors_consistency_nvfuser(executor, device, thunder_dtype):
    dtype = ltorch.to_torch_dtype(thunder_dtype)

    def div(a, b):
        return thunder.prims.div(a, b)

    def truediv(a, b):
        return a // b

    def make_integer_tensor():
        half_len = 5
        t = torch.tensor([*range(-half_len, 0), *range(1, half_len + 1)], device=device, dtype=dtype)
        perm = torch.randperm(2 * half_len)
        return t[perm]

    x = make_integer_tensor()
    y = make_integer_tensor()

    for f in (thunder.jit(div), thunder.jit(truediv)):
        rout = f(x.cpu(), y.cpu()).to(device)
        jout = f(x, y)
        assert rout.equal(jout)


@instantiate(
    dtypes=(thunder.float16, thunder.bfloat16),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
    decorators=(
        pytest.mark.skipif(
            nvfuser_version() is None or nvfuser_version() < LooseVersion("0.2.10"),
            reason="Requires nvFuser version 0.2.10 or later",
        ),
        pytest.mark.skipif(
            torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9,
            reason="Requires CUDA compute capability >= 9.0",
        ),
        pytest.mark.parametrize("dropout_p", [0.0, 0.2]),
        pytest.mark.parametrize("is_causal", [False, True]),
        pytest.mark.parametrize("scale", [None, 1e-3]),
    ),
)
def test_sdpa(
    executor,
    device: str,
    thunder_dtype: dtypes.dtype,
    dropout_p: None | float,
    is_causal: None | bool,
    scale: None | float,
):

    def sdpa_fn(q, k, v, dropout_p, is_causal, scale):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    torch.manual_seed(0)
    dtype = ltorch.to_torch_dtype(thunder_dtype)

    N, H, L, S, E = 4, 8, 16, 16, 8
    q = make_tensor((N, H, L, E), device=device, dtype=dtype, requires_grad=True)
    k = make_tensor((N, H, S, E), device=device, dtype=dtype, requires_grad=True)
    v = make_tensor((N, H, S, E), device=device, dtype=dtype, requires_grad=True)
    grad_out = make_tensor((N, H, L, E), device=device, dtype=dtype)

    tensor_inputs = [q, k, v]
    scalar_inputs = [dropout_p, is_causal, scale]

    compiled_func = thunder.jit(sdpa_fn, executors_list=executor.executors_list(), nv_enable_sdpa=True)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        attn_out = compiled_func(*tensor_inputs, *scalar_inputs)
    attn_out.backward(grad_out)
    fwd_trace = thunder.last_traces(compiled_func)[-1]
    bwd_trace = thunder.last_backward_traces(compiled_func)[-1]
    fwd_fusion = examine.get_fusions(fwd_trace)
    bwd_fusion = examine.get_fusions(bwd_trace)

    assert len(fwd_fusion) == 1
    assert len(bwd_fusion) == 1
    assert "nv_sdpfa_fwd" in fwd_fusion[-1][-1].name

    # Check nv_sdpfa_fwd is not in bwd_fusion -> that would indicate rematerialization
    assert "nv_sdpfa_bwd" in bwd_fusion[-1][-1].name and "nv_sdpfa_fwd" not in bwd_fusion[-1][-1].name

    nvf_fd = bwd_fusion[-1][-1].last_used
    repro_script = None
    # Legacy repro script API
    if nvfuser_version() < LooseVersion("0.2.14"):
        repro_script = nvf_fd.getReproString()
    else:
        repro_script = nvf_fd.repro_script_for()
    assert (
        repro_script.count("is_cpu=True") == 2
    ), "Expected philox_seed and philox_offset inputs to be CPU scalar tensors."

    # Torch reference computation
    # Clone the inputs to verify gradients with torch reference
    ref_tensor_inputs = []
    for inp in tensor_inputs:
        ref_inp = inp.clone().detach()
        ref_inp.requires_grad = True
        ref_tensor_inputs.append(ref_inp)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]) and sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref_attn_out = sdpa_fn(*ref_tensor_inputs, *scalar_inputs)
    ref_attn_out.backward(grad_out)

    nv_outputs = (attn_out, q.grad, k.grad, v.grad)
    ref_outputs = (ref_attn_out, *(inp.grad for inp in ref_tensor_inputs))
    for nv_out, ref_out in zip(nv_outputs, ref_outputs):
        torch.testing.assert_close(nv_out, ref_out)


@instantiate(
    dtypes=(thunder.float32,),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
    decorators=(
        pytest.mark.skipif(
            nvfuser_version() is None or nvfuser_version() < LooseVersion("0.2.23"),
            reason="Requires nvFuser version 0.2.23 or later",
        ),
    ),
)
def test_enable_disable_options(executor, device: str, thunder_dtype: dtypes.dtype):

    def fn(a, b):
        return torch.matmul(a, b)

    m, n, k = 24, 16, 16

    dtype = ltorch.to_torch_dtype(thunder_dtype)
    inps = [
        torch.randn(m, k, device="cuda", dtype=dtype),
        torch.randn(k, n, device="cuda", dtype=dtype),
    ]

    compiled_func = thunder.jit(
        fn,
        executors_list=executor.executors_list(),
        nv_enable_matmul=True,
        nv_enable_options=["fuse_matmul"],
        nv_disable_options=["matmul_expr_eval", "kernel_reuse"],
    )
    # The above combination of options enables matmul codegen and disables expr evaluation for matmul.
    # Since matmul scheduler does not support float32 inputs, the execution should raise an error.
    # By default, without using these options, the given fusion will run through expr eval scheduler correctly.
    # NOTE: This test relies on `float32` being unsupported by nvFuser matmul scheduler.
    # If this support is added, the test will need to be updated since it will no longer
    # verify the functionality of the above flags.
    with pytest.raises(RuntimeError, match="Can not find a scheduler to schedule fusion segment"):
        out = compiled_func(*inps)


@instantiate(
    dtypes=(thunder.float32,),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
)
def test_no_shape_only_fusion_region(executor, device: str, thunder_dtype: dtypes.dtype):
    x = make_tensor(2, 2, 2, device=device, dtype=ltorch.to_torch_dtype(thunder_dtype))

    def fn(x):
        return x.view(4, -1).transpose(0, 1)

    jfn = thunder.jit(fn)

    expected = fn(x)
    actual = jfn(x)

    torch.testing.assert_close(actual, expected)

    fwd_trace = thunder.last_traces(jfn)[-1]

    # Make sure there are no fusion symbols.
    assert all(not bsym.sym.is_fusion for bsym in fwd_trace.bound_symbols)

    # Verify that we create fusion even if we have a single compute op.
    def fn(x):
        # There is a `sin` which is not a shape op.
        return x.view(4, -1).transpose(0, 1).sin().transpose(0, 1).view(2, 2, 2)

    jfn = thunder.jit(fn)
    expected = fn(x)
    actual = jfn(x)

    torch.testing.assert_close(actual, expected)

    fwd_trace = thunder.last_traces(jfn)[-1]

    # Make sure there is a fusion symbol.
    assert any(bsym.sym.is_fusion for bsym in fwd_trace.bound_symbols)
