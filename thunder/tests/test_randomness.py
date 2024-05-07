import random

from torch.testing import assert_close

import thunder
import thunder.torch as ltorch
from thunder import compile as lc_compile
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

    cf = lc_compile(func, disable_preprocessing=True, executors_list=executor.executors_list())

    outputs = [cf(shape, dtype, device, rng_seed, rng_offset) for _ in range(3)]
    for o in outputs:
        assert_close(o, outputs[0])


@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CUDA,), executors=(nvFuserExecutor,))
def test_rng_state_prims(executor, device: str, _):
    import thunder.core.prims as prims
    import torch

    dev = devices.to_device(device)

    def func():
        b = prims.get_rng_state(None, device=dev)
        c = prims.get_rng_state(b, device=dev)
        seed, offset = prims.unpack_rng_state(b)
        new_state1 = prims.update_rng_state(seed, offset)

        new_state1 = prims.set_rng_state(new_state1, dev)
        new_state1_1 = prims.get_rng_state(new_state1, dev)
        state1_seed, state1_offset = prims.unpack_rng_state(new_state1_1)
        return b, c, seed, offset, new_state1_1, state1_seed, state1_offset

    cuda_generator = torch.cuda.default_generators[dev.index]
    jfunc = thunder.jit(func, executors_list=executor.executors_list())
    torch_device = thunder.core.devices.to_torch_device(dev)
    with torch.random.fork_rng(devices=(torch_device,)):
        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)
        ori_state, ori_state_1, ori_seed, ori_offset, state1, s1_seed, s1_offset = jfunc()

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
