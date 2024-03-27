from numbers import Number

import numpy as np
import torch
from torch.testing import assert_close

import thunder
import thunder.core.devices as devices
from thunder.core.langctxs import langctx
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import grad, get_grad, put_grads
from thunder.extend import OperatorExecutor, register_executor, deregister_executor, get_all_executors
from lightning_utilities.core.imports import package_available


def test_extend_core():
    myex = OperatorExecutor("myex", version="0.1")
    register_executor(myex)

    def multimul_impl(
        a: Number | TensorProxy,
        b: Number | TensorProxy,
        c: Number | TensorProxy,
        d: Number | TensorProxy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return np.multiply(a, b), np.multiply(c, d)

    @langctx("torch")
    def multimul_like(
        a: Number | TensorProxy,
        b: Number | TensorProxy,
        c: Number | TensorProxy,
        d: Number | TensorProxy,
    ):
        return a * b, c * d

    multimul = myex.register_operator("multimul", like=multimul_like, fn=multimul_impl)

    def foo(a, b):
        return multimul(a, b, 0, 0)

    cfoo = thunder.jit(foo, executors=[myex, thunder.pytorch_executor])

    a = torch.randn((4, 4), device="cpu", dtype=torch.float32, requires_grad=False)
    b = torch.randn((4, 4), device="cpu", dtype=torch.float32, requires_grad=False)

    thunder_result = cfoo(a, b)

    assert_close(thunder_result[0], a * b)

    cfoo_grad = grad(cfoo)
    cfoo_grad(a, b)

    def mul_to_multimul(a: Number | TensorProxy, b: Number | TensorProxy) -> TensorProxy:
        result, _ = multimul(a, b, 0, 0)
        return result

    def mul_to_multimul_checker(a: Number | TensorProxy, b: Number | TensorProxy) -> bool:
        def is_cpu(x: Number | TensorProxy) -> bool:
            if isinstance(a, TensorProxy):
                return a.device.devicetype == devices.DeviceType.CPU
            return True

        return all(is_cpu(x) for x in (a, b))

    @langctx("torch")
    def mymul_grad(a: TensorProxy, b: TensorProxy) -> TensorProxy:
        fwd = a * b

        g = get_grad(fwd)
        a_grad, b_grad = multimul(b, g, a, g)
        put_grads((a, b), (a_grad, b_grad))

        return fwd

    myex.register_implementation(
        thunder.torch.mul,
        checker=mul_to_multimul_checker,
        execution_transform=mul_to_multimul,
        grad_transform=mymul_grad,
    )

    def bar(a, b):
        return a * b

    a = torch.randn((4, 4), device="cpu", dtype=torch.float32, requires_grad=False)
    b = torch.randn((4, 4), device="cpu", dtype=torch.float32, requires_grad=False)

    cbar = thunder.jit(bar, executors=[myex, thunder.pytorch_executor])
    cbar(a, b)
    traces = thunder.last_traces(cbar)

    found_multimul: bool = False
    for bsym in traces[-1].bound_symbols:
        if bsym.sym.id == "multimul":
            found_multimul = True
            break

    assert found_multimul, "Failed to find multimul"

    a.requires_grad_(True)
    b.requires_grad_(True)

    cbar_grad = grad(cbar)
    _ = cbar_grad(a, b)
    traces = thunder.last_traces(cbar_grad)

    multimul_count: int = 0
    for bsym in traces[-1].bound_symbols:
        if bsym.sym.id == "multimul":
            multimul_count += 1

    assert multimul_count == 1

    deregister_executor(myex)


def test_get_all_executors_includes_all_native_executors():
    executors = get_all_executors()
    actual = {e.name for e in executors}
    expected = {
        "apex",
        "cudnn",
        "torch",
        "cudnn_layernorm",
        "sdpa",
        "torchcompile",
        "python",
        "transformer_engine",
    }
    if package_available("triton"):
        # `triton` maybe installed on a system without GPU.
        expected.update({"triton"})
    if torch.cuda.is_available():
        expected.update({"nvfuser"})
    assert actual == expected


def test_register_implementation_custom_op():
    myex = OperatorExecutor("myex", version="0.1")
    register_executor(myex)

    def official_add(a, b):
        return a + b

    def _myadd(a, b):
        return a + b

    myadd1 = myex.register_operator("myadd1", like=_myadd, fn=_myadd, replaces=official_add)
    myadd2 = myex.register_operator("myadd2", like=_myadd, fn=_myadd)

    def fn(a, b):
        return official_add(a, b)

    cfn = thunder.jit(fn, executors=[myex])

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    res = cfn(a, b)

    assert "myadd1" in str(thunder.last_traces(cfn)[-1])

    def myadd_trafo(a, b):
        return myadd2(a, b)

    def myadd_grad_trafo(a, b):
        res = myadd2(a, b)
        grad_res = get_grad(res)
        put_grads((a, b), (grad_res, grad_res))
        return res

    myex.register_implementation(myadd1, execution_transform=myadd_trafo, grad_transform=myadd_grad_trafo)

    cfn = thunder.jit(fn, executors=[myex])
    res = cfn(a, b)

    s = str(thunder.last_traces(cfn)[-1])
    assert "myadd2" in s and "myadd1" not in s

    a.requires_grad_()

    res = cfn(a, b)

    s = str(thunder.last_traces(cfn)[-1])
    assert "myadd2" in s and "myadd1" not in s

    a.requires_grad_()

    # without the executor, we just (should and do) jit through official_add
    cfn = thunder.jit(fn)
    res = cfn(a, b)

    s = str(thunder.last_traces(cfn)[-1])
    assert "myadd2" not in s and "myadd1" not in s

    deregister_executor(myex)
