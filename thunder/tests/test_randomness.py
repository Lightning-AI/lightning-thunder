import random

from torch.testing import assert_close

import thunder
import thunder.torch as ltorch
from thunder.core import devices, dtypes
from thunder.tests.framework import TorchExecutor, nvFuserExecutor, instantiate, NOTHING


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_uniform_philox(executor, device: str, dtype: dtypes.dtype):
    shape = (10, 30)
    rng_seed = random.randint(0, 123902390)
    rng_offset = random.randint(0, 123902390) * (4 if executor == TorchExecutor else 1)

    def func(shape, dtype, device, rng_seed, rng_offset):
        return ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=rng_seed, offset=rng_offset)

    cf = thunder.jit(func, executors=executor.executors_list())

    outputs = [cf(shape, dtype, device, rng_seed, rng_offset) for _ in range(3)]
    for o in outputs:
        assert_close(o, outputs[0])


@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CUDA,), executors=(nvFuserExecutor,))
def test_rng_state_prims(executor, device: str, _):
    import thunder.core.prims as prims
    import torch

    def func(device):
        s0, o0 = prims.get_and_update_rng_state(None, None, device=device)
        s1, o1 = prims.get_and_update_rng_state(s0, o0, device)
        return s0, o0, s1, o1

    dev = devices.to_device(device)
    cuda_generator = torch.cuda.default_generators[dev.index]
    jfunc = thunder.jit(func, executors=executor.executors_list())
    with torch.random.fork_rng(devices=(device,)):
        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)
        s0, o0, s1, o1 = jfunc(dev)
        assert_close(s0, 2)
        assert_close(s1, 2)
        assert_close(o0, 8 // 4)
        assert_close(o1, 12 // 4)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
)
def test_uniform_philox_with_rng_state_prims(executor, device: str, dtype: dtypes.dtype):
    import thunder.core.prims as prims
    import torch

    def func1(shape, dtype, device):
        seed0, offset0 = prims.get_and_update_rng_state(None, None, device=device)
        out1 = ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=seed0, offset=offset0)

        seed1, offset1 = prims.get_and_update_rng_state(seed0, offset0, device)
        out2 = ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=seed1, offset=offset1)

        return out1, out2

    def func2(shape, dtype, device):
        out1 = ltorch.uniform(shape, device=device, dtype=dtype)
        out2 = ltorch.uniform(shape, device=device, dtype=dtype)
        return out1, out2

    dev = devices.to_device(device)

    cuda_generator = torch.cuda.default_generators[dev.index]
    jfunc1 = thunder.jit(func1, executors=executor.executors_list())
    jfunc2 = thunder.jit(func2, executors=executor.executors_list())
    shape = (2, 3, 4, 5)
    with torch.random.fork_rng(devices=(device,)):
        cuda_generator.manual_seed(2)
        uniform_philox_o1, uniform_philox_o2 = jfunc1(shape, dtype, dev)
        state1 = cuda_generator.get_state()

        cuda_generator.manual_seed(2)
        uniform_o1, uniform_o2 = jfunc2(shape, dtype, dev)
        state2 = cuda_generator.get_state()

        assert_close(state1, state2)
        assert_close(uniform_o1, uniform_philox_o1)
        assert_close(uniform_o2, uniform_philox_o2)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
)
def test_rng_state_uniform_philox_reproducibility(executor, device: str, dtype: dtypes.dtype):
    import torch

    def func(a):
        b = ltorch.uniform_like(a, device=a.device, dtype=a.dtype)
        d = torch.nn.functional.dropout(a, p=0.5)
        c = ltorch.uniform_like(a, device=a.device, dtype=a.dtype)
        return c * b * a * d

    dev = devices.to_torch_device(device)
    a = torch.randn(2, 2, device=dev, dtype=dtypes.to_torch_dtype(dtype), requires_grad=True)
    a1 = a.detach().clone()
    a1.requires_grad_()

    jfunc = thunder.jit(func, executors_list=executor.executors_list())

    with torch.random.fork_rng(devices=(dev,)):
        torch.cuda.manual_seed(20)
        expects = []
        for _ in range(4):
            out = jfunc(a)
            out.sum().backward()
            expects.append(out)
            expects.append(a.grad)

        results = []
        torch.cuda.manual_seed(20)
        for _ in range(4):
            out = jfunc(a1)
            out.sum().backward()
            results.append(out)
            results.append(a1.grad)

    for expected, result in zip(expects, results):
        assert_close(expected, result)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
)
def test_uniform_philox_vs_uniform(executor, device: str, dtype: dtypes.dtype):
    import torch

    dev = devices.to_torch_device(device)
    cuda_generator = torch.cuda.default_generators[dev.index]

    def func(a):
        b = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        e = a * b
        c = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        f = e + c
        d = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        return f + d

    a = torch.randn(2, 2, device=dev, dtype=dtypes.to_torch_dtype(dtype), requires_grad=True)
    a1 = a.detach().clone().requires_grad_()

    jfunc = thunder.jit(func, executors_list=executor.executors_list())

    # TODO: Check the backward results when #231 is fixed
    with torch.random.fork_rng(devices=(dev,)):
        cuda_generator.manual_seed(20)
        expects = []
        # get the results of uniform_philox with RNG state updates
        for _ in range(4):
            out = jfunc(a)
            out.sum().backward()
            expects.append(out)
        assert cuda_generator.get_offset() == 12 * 4
        rng_syms = (
            "unpack_rng_state_prim_impl",
            "get_rng_state_prim_impl",
            "update_rng_state_prim_impl",
            "set_rng_state_prim_impl",
        )
        # check the transform has inserted the rng state operators
        assert any(t.sym.id in rng_syms for t in thunder.last_traces(jfunc)[-1].bound_symbols)

        # get the results of uniform
        results = []
        cuda_generator.manual_seed(20)
        from unittest.mock import patch, MagicMock

        # mock the replace_uniform transform to return the input trace
        replace_uniform_mock = MagicMock(side_effect=lambda trc: trc)

        with patch("thunder.core.rematerialization.replace_uniform", new=replace_uniform_mock):
            jfunc = thunder.jit(func, executors_list=executor.executors_list())
            for _ in range(4):
                out = jfunc(a1)
                out.sum().backward()
                results.append(out)
            assert cuda_generator.get_offset() == 12 * 4

    for expected, result in zip(expects, results):
        assert_close(expected, result)
