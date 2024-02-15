from numbers import Number

import numpy as np
import torch
from torch.testing import make_tensor, assert_close

import thunder
from thunder.core.langctxs import langctx
from thunder.extend import OperatorExecutor, register_executor, get_default_executors, add_default_executor
from thunder.core.proxies import TensorProxy
from thunder.core.utils import check
from thunder.core.transforms import grad, get_grad, put_grad, put_grads
import thunder.core.devices as devices


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

    # TODO Restore this once nonlocals are supported
    # def foo(a, b):
    #     return multimul(a, b, 0, 0)

    # cfoo = thunder.compile(foo, executors_list=[myex, thunder.pytorch_executor])

    # a = torch.randn((4, 4), device='cpu', dtype=torch.float32, requires_grad=False)
    # b = torch.randn((4, 4), device='cpu', dtype=torch.float32, requires_grad=False)

    # thunder_result = cfoo(a, b)

    # assert_close(thunder_result, a * b)

    # cfoo_grad = grad(cfoo)
    # cfoo_grad(a, b)

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

    cbar = thunder.compile(bar, executors_list=[myex, thunder.pytorch_executor])
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
    result = cbar_grad(a, b)
    traces = thunder.last_traces(cbar_grad)

    multimul_count: int = 0
    for bsym in traces[-1].bound_symbols:
        if bsym.sym.id == "multimul":
            multimul_count += 1

    assert multimul_count == 1
