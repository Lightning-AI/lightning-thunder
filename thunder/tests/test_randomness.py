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
        b = prims.get_rng_state(None, device=device)
        c = prims.get_rng_state(b, device=device)
        seed, offset = prims.unpack_rng_state(b)
        new_state1 = prims.update_rng_state(seed, offset)

        new_state1 = prims.set_rng_state(new_state1, device)
        new_state1_1 = prims.get_rng_state(new_state1, device)
        state1_seed, state1_offset = prims.unpack_rng_state(new_state1_1)
        return b, c, seed, offset, new_state1_1, state1_seed, state1_offset

    dev = devices.to_device(device)
    cuda_generator = torch.cuda.default_generators[dev.index]
    jfunc = thunder.jit(func, executors=executor.executors_list())
    with torch.random.fork_rng(devices=(device,)):
        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)
        ori_state, ori_state_1, ori_seed, ori_offset, state1, s1_seed, s1_offset = jfunc(dev)

        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)

        assert_close(cuda_generator.get_state(), ori_state)
        assert_close(cuda_generator.get_state(), ori_state_1)
        assert_close(ori_seed, cuda_generator.initial_seed())
        assert_close(cuda_generator.get_offset() // 4, ori_offset)

        cuda_generator.set_offset(cuda_generator.get_offset() + 4)
        assert_close(cuda_generator.get_state(), state1)
        assert_close(cuda_generator.initial_seed(), s1_seed)
        assert_close(cuda_generator.get_offset() // 4, s1_offset)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(nvFuserExecutor,),
)
def test_uniform_philox_with_rng_state_prims(executor, device: str, dtype: dtypes.dtype):
    import thunder.core.prims as prims
    import torch

    def func1(shape, dtype, device):
        b = prims.get_rng_state(None, device=device)
        seed, offset = prims.unpack_rng_state(b)
        out1 = ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=seed, offset=offset)
        new_state1 = prims.update_rng_state(seed, offset)
        new_state1 = prims.set_rng_state(new_state1, dev)

        new_state1_1 = prims.get_rng_state(new_state1, device)
        state1_seed, state1_offset = prims.unpack_rng_state(new_state1_1)
        out2 = ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=state1_seed, offset=state1_offset)
        new_state2 = prims.update_rng_state(state1_seed, state1_offset)
        new_state2_2 = prims.set_rng_state(new_state2, device)

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
