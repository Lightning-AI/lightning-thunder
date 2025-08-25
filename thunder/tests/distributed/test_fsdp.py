import os
import unittest
import weakref
from itertools import product

import pytest
import torch
import torch.distributed as tdist

if not tdist.is_available():
    pytest.skip(allow_module_level=True)
import torch.nn as nn
from torch.distributed import distributed_c10d as c10d
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.testing import assert_close, make_tensor

import thunder
import thunder.executors
import thunder.torch as ltorch
from thunder.core import devices
from thunder.distributed import FSDPBucketingStrategy, FSDPType
from thunder.distributed import fsdp
from thunder.tests.framework import instantiate, TorchExecutor

from thunder.executors.transformer_engineex import (
    transformer_engine_ex,
    TE_AVAILABLE,
)

from thunder.executors.transformer_engine_v2ex import transformer_engine_v2_ex, TransformerEngineTransformV2


is_fp8_supported: bool = False
# This will be correctly updated below when TE Engine is installed
# and if the current environment doesn't support FP8.
fp8_support_reason: str = ""
if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch.fp8 import (
        check_fp8_support,
        FP8GlobalStateManager,
        get_default_fp8_recipe,
    )
    from transformer_engine.common.recipe import MXFP8BlockScaling
    import transformer_engine

    is_fp8_supported, fp8_support_reason = check_fp8_support()

from thunder.tests.distributed.helper import (
    ToyModel,
    DistributedParallelTestCase,
    new_gelu,
    executors_map,
    distributed_wrapper,
    create_per_process_dataloader,
    SmallModel,
    run_test_no_sync_grad_accumulation,
    init_per_process_distributed,
)
from torch.testing._internal import common_utils


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "FSDP test requires CUDA and NCCL `torch.distributed` backend",
)
class FSDPTest(DistributedParallelTestCase):
    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_sort_waits(self, executor):
        from thunder.distributed.utils import sort_waits

        _executor = executors_map[executor]

        def func(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
        ):
            d = ltorch.all_reduce(a, group=process_group, async_op=True).wait()
            c = a + b
            e = c @ b + a
            return e, d

        cfunc = thunder.jit(func, executors=_executor.executors_list())
        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()
        _ = cfunc(a, b, process_group)
        traces = thunder.last_traces(cfunc)

        del_last_used_indices = [
            i
            for i, trace in enumerate(traces)
            if (tp := trace.get_provenance()) is not None and "Delete Last Used" in tp.pss
        ]
        assert del_last_used_indices and del_last_used_indices[0] >= 1
        trace_before_del_last_used = traces[del_last_used_indices[0] - 1]

        # sort_waits is supposed to be called just before del_last_used
        sorted_trace = sort_waits(trace_before_del_last_used)

        # assert that there is at least one node between the all_reduce and wait
        all_reduce_idx = sorted_trace.bound_symbols.index(
            next(filter(lambda n: n.sym.name == "torch_all_reduce_prim_impl", sorted_trace.bound_symbols))
        )
        wait_idx = sorted_trace.bound_symbols.index(
            next(filter(lambda n: n.sym.name == "torch_wait_prim_impl", sorted_trace.bound_symbols))
        )
        self.assertGreater(wait_idx - all_reduce_idx, 1)
        self.assertEqual(wait_idx, len(sorted_trace.bound_symbols) - 2)

    @pytest.mark.xfail(strict=True, reason="This is not updated yet for joint forward-backward trace")
    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype",
        product(
            tuple(executors_map.keys()),
            (FSDPBucketingStrategy.BLOCK,),
            (FSDPType.ZERO2, FSDPType.ZERO3),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_fsdp_with_no_sync_grad_accumulation(
        self,
        executor: str,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.common import CACHE_OPTIONS
        from thunder.distributed import fsdp

        def get_model_and_optimizer(device):
            m = ToyModel().to(device)
            jitted_m = thunder.jit(
                m,
                cache_mode=CACHE_OPTIONS.CONSTANT_VALUES,
                executors=executors_map[executor].executors_list(),
            )
            jitted_fsdp_m = fsdp(jitted_m, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype)
            optimizer = torch.optim.SGD(jitted_fsdp_m.parameters(), lr=1e-3)
            return jitted_fsdp_m, optimizer

        def is_comm(k: str) -> bool:
            return "reducescatter" in k or "reduce_scatter" in k

        run_test_no_sync_grad_accumulation(self, get_model_and_optimizer, is_comm, dataset_size=2)

    # TODO(crcrpar): Add torch compile to executors_list
    @pytest.mark.xfail(strict=True, reason="This is not updated yet for joint forward-backward trace")
    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype",
        product(
            tuple(executors_map.keys()),
            (
                FSDPBucketingStrategy.LAYER,
                FSDPBucketingStrategy.BLOCK,
            ),
            (FSDPType.ZERO2, FSDPType.ZERO3),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_fsdp_grad_parity_with_without_bucketing(
        self,
        executor,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.distributed import fsdp

        device = torch.device("cuda", self.rank)
        initial_model_state = ToyModel().state_dict()

        for strategy in (FSDPBucketingStrategy.NONE, bucketing_strategy):
            m = ToyModel()
            m.load_state_dict(initial_model_state)
            cm = fsdp(
                thunder.jit(m.to(device), executors=executors_map[executor].executors_list()),
                device=device,
                bucketing_strategy=bucketing_strategy,
                sharding_strategy=fsdptype,
            )
            x = torch.ones((2, 12), device=device)
            loss = cm(x).mean()
            loss.backward()

            if strategy == FSDPBucketingStrategy.NONE:
                gradients = tuple(p.grad for p in cm.parameters() if p.grad is not None)
                orig_loss = loss.detach()
            else:
                self.assertEqual(loss, orig_loss)
                self.assertEqual(tuple(p.grad for p in cm.parameters() if p.grad is not None), gradients)

                # Make sure that at least one of "pack" takes multiple tensors.
                from thunder.executors.torchex import pack_for_fsdp_prim_impl
                from thunder.distributed.prims import PrimIDs as DistPrimIDs

                for ex_trace in (thunder.last_traces(cm)[-1], thunder.last_backward_traces(cm)[-1]):
                    pack_bsyms = list(
                        filter(
                            lambda bsym: bsym.sym.id in {DistPrimIDs.PACK_FOR_FSDP, pack_for_fsdp_prim_impl.id},
                            ex_trace.bound_symbols,
                        )
                    )
                    self.assertGreater(len(pack_bsyms), 0)
                    has_pack_multiple_tensors = False
                    for bsym in pack_bsyms:
                        first_arg = bsym.args[0]
                        self.assertIsInstance(first_arg, list)
                        has_pack_multiple_tensors |= len(first_arg) > 1
                    # note(crcrpar): The way creating a bucket name from an FQN could be better for models with simple structure
                    # see https://github.com/Lightning-AI/lightning-thunder/blob/b24e5b23/thunder/distributed/__init__.py#L278-L301
                    if bucketing_strategy == FSDPBucketingStrategy.LAYER:
                        self.assertTrue(has_pack_multiple_tensors, msg=f"{[bsym.args[0] for bsym in pack_bsyms]=}")

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    @common_utils.parametrize(
        "bucketing_strategy,fsdptype",
        product(
            (
                FSDPBucketingStrategy.NONE,
                FSDPBucketingStrategy.BLOCK,
            ),
            (FSDPType.ZERO2, FSDPType.ZERO3),
        ),
        name_fn=lambda bucketing_strategy, fsdptype: (
            f"bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_fsdp_with_padding(
        self,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.core.prims import PrimIDs
        from thunder.core.transforms import unwrap_one_level_of_subsymbols
        from thunder.executors.torchex import pad_prim_impl
        from thunder.executors.torchex import slice_prim_impl

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 13)
                self.l2 = nn.Linear(13, 1)

            def forward(self, x):
                return self.l2(new_gelu(self.l1(x)))

        device = torch.device(f"cuda:{self.rank}")
        m = M().to(device)
        jitted = fsdp(thunder.jit(m), bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype)

        x = torch.randn(4, 4, device=device)
        y = jitted(x)
        y.mean().backward()

        fw_extrace = thunder.last_traces(jitted)[-1]
        # `slice` and `pad` may appear in nvFusion subsymbols.
        fw_extrace = unwrap_one_level_of_subsymbols(fw_extrace)
        fw_symids = [bsym.sym.id for bsym in fw_extrace.bound_symbols]
        self.assertTrue(any(sym_id in {PrimIDs.SLICE, slice_prim_impl.id} for sym_id in fw_symids))

        bw_trace = thunder.last_backward_traces(jitted)[0]
        bw_trace = unwrap_one_level_of_subsymbols(bw_trace)
        bw_symids = [bsym.sym.id for bsym in bw_trace.bound_symbols]
        self.assertTrue(any(sym_id in {PrimIDs.PAD, pad_prim_impl.id} for sym_id in bw_symids))

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdp_shard_unshard(self):
        from thunder.distributed import _shard_params, _unshard_params

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        model = torch.nn.Linear(3, 5, bias=False, device="meta")
        with pytest.raises(RuntimeError, match=r"parameter 'weight' \(5\) to be divisible by the world size \(2\)"):
            _shard_params(model, pg, device, None)
        _shard_params(model, pg, device, None, allow_padding_for_fsdp=True)

        model = torch.nn.Linear(3, 4, bias=False, device="meta")
        weight = torch.arange(3 * 4, device="cpu", dtype=torch.float).view(4, 3)
        model.load_state_dict({"weight": weight}, assign=True)

        # each shard got its corresponding piece of the weight
        _shard_params(model, pg, device, None)
        expected = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] if self.rank == 0 else [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        # the weight was moved to device
        assert torch.equal(model.weight, torch.tensor(expected, device=device))

        # unsharding reconstructs the original weight (and cpu offloads)
        _unshard_params(model, pg, cpu_offload=True)
        assert torch.equal(model.weight, weight)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdp_broadcast_from(self):
        from thunder.distributed import _shard_params

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        model = torch.nn.Linear(3, 4, bias=False, device="meta")
        model.register_buffer("foo", torch.tensor([123.0]), persistent=False)
        weight = torch.arange(3 * 4, device="cpu", dtype=torch.float).view(4, 3)
        if self.rank == 0:
            weight *= -1.0
            model.foo *= -1.0
        model.load_state_dict({"weight": weight}, assign=True)

        _shard_params(model, pg, device, 0)
        # since rank 0's params are negative and rank 1's are positive, we know broadcasting worked if all params are negative
        expected = (
            [[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]] if self.rank == 0 else [[-6.0, -7.0, -8.0], [-9.0, -10.0, -11.0]]
        )
        # the weight was moved to device
        assert torch.equal(model.weight, torch.tensor(expected, device=device))
        # same check for the buffer
        assert torch.equal(model.foo, torch.tensor([-123.0], device=device))

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_materialize_meta_tensors(self):
        from thunder.distributed import _shard_params

        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(4, 8)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.tensor(0))
                self.l = torch.nn.Linear(2, 4)
                self.inner = Submodule()

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        with torch.device("meta"):
            model = MyModel()
        with pytest.raises(TypeError, match="MyModel.reset_parameters` method is implemented"):
            _shard_params(model, pg, device, None)

        class MyModel2(MyModel):
            def reset_parameters(self):
                self.buf = torch.empty_like(self.buf)

        with torch.device("meta"):
            model = MyModel2()

        _shard_params(model, pg, device, None)
        # all parameters were moved
        assert len(list(model.parameters())) == 4
        assert all(p.device.type == "cuda" for p in model.parameters())
        # buffers were moved too
        assert model.buf.device.type == "cuda"

    # This is not updated yet for joint forward-backward trace
    @common_utils.decorateIf(
        unittest.expectedFailure,
        lambda params: params["bucketing_strategy"] in (FSDPBucketingStrategy.LAYER, FSDPBucketingStrategy.BLOCK),
    )
    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype",
        product(
            tuple(executors_map.keys()),
            (FSDPBucketingStrategy.NONE, FSDPBucketingStrategy.LAYER, FSDPBucketingStrategy.BLOCK),
            (FSDPType.ZERO3,),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_limit_in_flight_allgathers(
        self,
        executor,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.distributed import fsdp
        from thunder.tests.nanogpt_model import Block, GPTConfig

        def check_inflight_allgather_number(trc, n: int, is_bucket: bool):
            from thunder.core.utils import producers
            from thunder.executors.torchex import all_gather_prim_impl, pack_for_fsdp_prim_impl, wait_prim_impl

            producers = producers(trc)
            cnt = 0
            for idx, bsym in enumerate(trc.bound_symbols):
                if bsym.sym.id == all_gather_prim_impl.id:
                    cnt += 1
                    if is_bucket:
                        self.assertEqual(trc.bound_symbols[idx - 1].sym.id, pack_for_fsdp_prim_impl.id)
                self.assertLessEqual(cnt, n)
                if bsym.sym.id == wait_prim_impl.id:
                    if producers[bsym.flat_proxy_args[0]].sym.id == all_gather_prim_impl.id:
                        cnt -= 1

        device = torch.device("cuda", self.rank)
        config = GPTConfig(dropout=0)
        m = Block(config).to(device=device)
        cm = thunder.jit(
            m,
            executors=executors_map[executor].executors_list(),
        )
        cm = fsdp(
            cm, device=device, broadcast_from=0, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype
        )
        x = torch.ones((2, config.block_size, config.n_embd), device=device)
        loss = cm(x).mean()
        loss.backward()

        # get the trace before sorting
        fwd_trc = thunder.last_traces(cm)[-2]
        bwd_trc = thunder.last_backward_traces(cm)[-1]

        from thunder.distributed.utils import limit_in_flight_allgathers

        is_bucketing = bucketing_strategy != FSDPBucketingStrategy.NONE
        for i in range(1, 12):
            aft_trc = limit_in_flight_allgathers(fwd_trc, i, is_bucketing)
            check_inflight_allgather_number(aft_trc, i, is_bucketing)
        for i in range(1, 6):
            aft_trc = limit_in_flight_allgathers(bwd_trc, i, is_bucketing)
            check_inflight_allgather_number(aft_trc, i, is_bucketing)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdp_weight_sharing(self):
        # This test is to verify that weight sharing works with fsdp.
        # NOTE: Currently we end up creating 2 copies of shared weight during execution.
        #       This should be fixed and we should update this test to check for that.
        device = torch.device("cuda", self.rank)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 16, bias=False)
                self.fc2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return self.fc1(x) + self.fc2(x)

        def _test_model_output_and_gradients(model, x, duplicate_all_gather):
            output = model(x)
            with device:
                grad_output = torch.ones_like(output)
            output.backward(grad_output)
            expected_shape = (4, 16)

            assert output.shape == expected_shape, f"{output.shape=} - {expected_shape=}"

            # Verify that both params point to same grad tensor.
            assert id(model.get_parameter("fc1.weight").grad) == id(model.get_parameter("fc2.weight").grad)

            # Verify that we accumulate the gradients for the shared parameter.
            gathered_grad_shape = (model.get_parameter("fc1.weight").shape[0] * self.world_size,) + model.get_parameter(
                "fc1.weight"
            ).shape[1:]
            with device:
                actual_grad_gathered = torch.empty(gathered_grad_shape)

            tdist.all_gather_into_tensor(actual_grad_gathered, model.get_parameter("fc1.weight").grad)

            # Based on the forward, grad for both params is `(grad_output.T @ x)`. Multiplying by 2 as the grad will be accumulated.
            expected_grad = 2 * (grad_output.T @ x)
            torch.testing.assert_close(actual_grad_gathered, expected_grad)

            forward_exec_trace = thunder.last_traces(model)[-1]
            gathered_params = set()
            for bsym in forward_exec_trace.bound_symbols:
                if bsym.sym.id in (
                    thunder.distributed.prims.PrimIDs.ALL_GATHER,
                    thunder.executors.torchex.all_gather_prim_impl.id,
                ):
                    gathered_params.add(bsym.args[0].name)

            # Check trace to see we don't have duplicate AllGather for shared parameters.
            if duplicate_all_gather:
                # Both params are gathered.
                assert "t_fc1_weight" in gathered_params and "t_fc2_weight" in gathered_params
            else:
                # Either of the param was gathered but not both.
                assert ("t_fc1_weight" in gathered_params) ^ ("t_fc2_weight" in gathered_params)

        with device:
            fsdp_jit_model = Model()
            x = torch.ones(4, 16)

        # Check `fsdp(jit(model))` works
        fsdp_jit_model.fc1.weight = fsdp_jit_model.fc2.weight

        fsdp_jit_model = thunder.distributed.fsdp(thunder.jit(fsdp_jit_model, executors=["torch"]))

        _test_model_output_and_gradients(fsdp_jit_model, x, duplicate_all_gather=False)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_load_original_state_dict(self):
        device = torch.device("cuda", self.rank)
        with device:
            x = torch.randn((2, ToyModel.N_IN))
        with torch.device("cuda"):
            model1 = ToyModel()
            model2 = ToyModel()

        sd = {k: v.clone() for k, v in model1.state_dict().items()}

        jm1 = fsdp(thunder.jit(model1), device=device)
        jm2 = fsdp(thunder.jit(model2), device=device)
        jm2.load_original_state_dict(sd)

        y_1 = jm1(x)
        y_2 = jm2(x)

        torch.testing.assert_close(y_1, y_2)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_original_state_dict(self):
        device = torch.device("cuda", self.rank)

        for move_state_dict_to_cpu in (False, True):
            with torch.device("cuda"):
                model = ToyModel()

            init_state_dict = model.state_dict()
            jitted = fsdp(thunder.jit(model), device=device, move_state_dict_to_cpu=move_state_dict_to_cpu)

            sharded_state_dict = jitted.state_dict()
            original_state_dict = jitted.original_state_dict()
            for key, unsharded in original_state_dict.items():
                self.assertTrue(key in init_state_dict and key in sharded_state_dict)
                self.assertEqual(len(init_state_dict[key]), len(unsharded))
                self.assertGreater(len(unsharded), len(sharded_state_dict[key]))
                if move_state_dict_to_cpu:
                    self.assertEqual(unsharded.device, torch.device("cpu"))
                else:
                    self.assertEqual(unsharded.device, device)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdpv2_with_1layer_llama_meta_init(self):
        from thunder.tests.litgpt_model import Config, GPT

        device = torch.device("cuda", self.rank)
        config = Config("Llama-2-7b-hf")
        config.n_layer = 1
        with torch.device("meta"):
            model = GPT(config)
        jitted = fsdp(thunder.jit(model), device=device)
        with device:
            model_ref = GPT(config)
        jitted_ref = fsdp(thunder.jit(model_ref), device=device)

        jitted_ref.load_state_dict(jitted.state_dict())

        t = config.block_size
        data = torch.randint(
            0,
            100,
            (
                1,
                t + 1,
            ),
            dtype=torch.int64,
        )
        x = data[:, :t]
        x = x.to(device=device)
        result = jitted(x)
        expected = jitted_ref(x)
        assert_close(result, expected)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdpv2_no_grad(self):
        from thunder.tests.litgpt_model import Config, GPT

        device = torch.device("cuda", self.rank)
        config = Config("Llama-2-7b-hf")
        config.n_layer = 1
        with torch.device("meta"):
            model = GPT(config)
        with device:
            model_ref = GPT(config)
        jitted = fsdp(thunder.jit(model), device=device)
        jitted.load_original_state_dict(model_ref.state_dict())

        t = config.block_size
        data = torch.randint(
            0,
            100,
            (
                1,
                t + 1,
            ),
            dtype=torch.int64,
        )
        x = data[:, :t]
        x = x.to(device=device)
        with torch.no_grad():
            result = jitted(x)
            expected = model_ref(x)
        assert_close(result, expected)


common_utils.instantiate_parametrized_tests(FSDPTest)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "FSDP test requires CUDA and NCCL `torch.distributed` backend",
)
class FSDPDDPHybridTest(DistributedParallelTestCase):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires 4 devices")
    def test_fsdp_ddp_hybrid(self):
        import torch
        import thunder
        import torch.distributed
        from torch.testing import assert_close
        from thunder.distributed.transforms.fsdp_v2 import FSDPTransform
        from thunder.distributed.transforms.ddp_v2 import DDPTransform

        torch.manual_seed(1337)

        mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (2, 2), mesh_dim_names=("ddp", "fsdp"))
        global_rank = mesh.get_rank()
        fsdp_rank = mesh.get_local_rank("fsdp")

        with torch.device("cuda"):
            m = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256))
            inp = torch.randn(4, 256)

        jm = thunder.jit(
            m,
            transforms=[
                FSDPTransform(process_group=mesh["fsdp"].get_group()),
                DDPTransform(mesh["ddp"].get_group(), broadcast_from=0, bucket_size_in_mb=25.0),
            ],
        )

        res = jm(inp)
        go = torch.randn_like(res)
        grads = torch.autograd.grad(res, jm.parameters(), go)
        ref = m(inp)
        ref_grads = torch.autograd.grad(ref, m.parameters(), go)
        assert_close(res, ref)
        for g, rg in zip(grads, ref_grads):
            slice_size = rg.size(0) // 2
            assert_close(g, rg[slice_size * fsdp_rank : slice_size * (fsdp_rank + 1)])

    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires 4 devices")
    @pytest.mark.xfail(RuntimeError, reason="requires fix ...")  # todo
    def test_fsdp_ddp_plugin(self):
        import thunder
        import torch.distributed
        from thunder.plugins import FSDP
        from torch.testing import assert_close

        torch.manual_seed(1337)

        mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (2, 2), mesh_dim_names=("ddp", "fsdp"))

        with torch.device("cuda"):
            m = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256))
            inp = torch.randn(4, 256)

        plugin = FSDP(process_group=mesh)

        jm = thunder.compile(m, plugins=[plugin])
        res = jm(inp)
        grads = torch.autograd.grad(res, jm.parameters(), go)
        ref = m(inp)
        ref_grads = torch.autograd.grad(ref, m.parameters(), go)
        assert_close(res, ref)
        for g, rg in zip(grads, ref_grads):
            slice_size = rg.size(0) // 2
            assert_close(g, rg[slice_size * fsdp_rank : slice_size * (fsdp_rank + 1)])


common_utils.instantiate_parametrized_tests(FSDPDDPHybridTest)


def _test_native_fsdp_helper(input_data):
    init_method, world_size, rank, executor, device, dtype, kwargs = input_data
    bucketing_strategy = kwargs["fsdp_bucketing_strategy"]

    num_samples = 2
    tensor_shape = (2, 2)
    sample_seed = 3456
    num_epochs = 1
    devicetype = devices.device_from_string(device).devicetype
    torch_dtype = ltorch.to_torch_dtype(dtype)

    pg = init_per_process_distributed(init_method, devicetype, world_size, rank)
    tdist.barrier(pg)

    def finalize_pg(pg):
        # NOTE This function is undocumented; its definition is here:
        # https://github.com/pytorch/pytorch/blob/416bf4e/torch/distributed/distributed_c10d.py#L1359
        tdist.barrier(pg)
        tdist.destroy_process_group(pg)

    weakref.finalize(pg, finalize_pg, pg)

    dataloader = create_per_process_dataloader(
        rank,
        num_samples=num_samples,
        tensor_shape=tensor_shape,
        tensor_dtype=torch_dtype,
        sample_seed=sample_seed,
        devicetype=devicetype,
    )

    # Creates, compiles, and FSDPs the model
    model = SmallModel(device, torch_dtype)

    original_weight_net1_shape = model.net1.weight.shape

    cmodel0 = thunder.jit(
        model,
        executors=executor.executors_list(),
    )
    cmodel = fsdp(cmodel0, bucketing_strategy=bucketing_strategy, device=device)

    # Check that the model is sharded
    sharded_weight_net1 = cmodel.get_parameter("net1.weight")
    assert sharded_weight_net1.shape != original_weight_net1_shape
    assert sharded_weight_net1.shape == (1, 2)

    comparison_exceptions = []
    for _ in range(num_epochs):
        for step, data in enumerate(dataloader):
            (inp,) = data
            pred = cmodel(inp)

            # Validates that each process got the same result by gathering all the tensors
            #   on rank 0 and comparing them
            # NOTE Exceptions thrown during the comparison process are recorded and returned
            #   to the spawning process for analysis
            gather_list = None
            if rank == 0:
                gather_list = []
                for _ in range(world_size):
                    gather_list.append(torch.empty_like(pred))

            tdist.gather(pred, gather_list, dst=0, group=pg, async_op=False)

            if rank == 0:
                for other in gather_list:
                    try:
                        assert_close(pred, other)
                    except Exception as e:
                        comparison_exceptions.append(e)

            pred.mean().backward()

            for param_with_grad in filter(lambda p: p.grad is not None, cmodel.parameters()):
                sharded_grad = param_with_grad.grad
                assert sharded_grad.shape == param_with_grad.shape

    if rank == 0:
        return comparison_exceptions

    return None


def _test_fsdp_transformer_engine(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value and
    # fp8 meta state is same after `n_iter`.
    init_method, world_size, rank, executor, device, _unused_dtype, kwargs = input_data
    thunder_fsdp_strategy, intermediate_activation_sharding = kwargs["thunder_fsdp_strategy_and_intermediate_sharding"]
    devicetype = devices.device_from_string(device).devicetype

    # Setting LOCAL_RANK is necessary for thunder.distributed.fsdp
    with unittest.mock.patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
        init_per_process_distributed(init_method, devicetype, world_size, rank)
        torch.cuda.set_device(rank)

        dim = 256
        # Running more iterations leads to `nan` for both eager and thunder
        # with BlockScaling.
        n_iter = 5

        class ThunderModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim, bias=False)
                self.fc2 = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        # Weights
        fc1_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")
        fc2_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")

        # Inputs (different input on different rank).
        if rank == 0:
            x = torch.arange(dim * dim, dtype=torch.float, device="cuda").view(dim, dim)
        if rank == 1:
            x = torch.randn(dim, dim, device="cuda") * 100

        with torch.device("cuda"):
            thunder_model = ThunderModel()
        thunder_model.fc1.weight.data = fc1_weight.clone()
        thunder_model.fc2.weight.data = fc2_weight.clone()

        jit_model = thunder.distributed.fsdp(
            thunder.jit(
                thunder_model,
                executors=[
                    transformer_engine_ex,
                ]
                + executor.executors_list(),
                fp8_shard_intermediate_activation=intermediate_activation_sharding,
            ),
            sharding_strategy=thunder_fsdp_strategy,
        )

        optim = torch.optim.SGD(jit_model.parameters())

        for _ in range(n_iter):
            o = jit_model(x).sum()
            o.backward()
            optim.step()
            optim.zero_grad()

        # See https://github.com/NVIDIA/TransformerEngine/issues/814
        FP8GlobalStateManager.reset()

        class TEModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = TELinear(dim, dim, bias=False)
                self.fc2 = TELinear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        with torch.device("cuda"):
            te_model = TEModel()
        te_model.fc1.weight.data = fc1_weight.clone()
        te_model.fc2.weight.data = fc2_weight.clone()

        fsdp_model = FullyShardedDataParallel(te_model, auto_wrap_policy=always_wrap_policy)
        if intermediate_activation_sharding:
            transformer_engine.pytorch.distributed.prepare_te_modules_for_fsdp(fsdp_model)
        optim = torch.optim.SGD(te_model.parameters())

        for _ in range(n_iter):
            with fp8_autocast():
                o = fsdp_model(x).sum()

            o.backward()
            optim.step()
            optim.zero_grad()

        thunder_to_te_layer_map = {"te_linear_0": te_model.fc1, "te_linear_1": te_model.fc2}

        fwd_traces = thunder.last_traces(jit_model)

        def is_same_across_ranks(t):
            t_clone = t.clone()
            torch.distributed.all_reduce(t_clone, op=torch.distributed.ReduceOp.AVG)
            assert_close(t, t_clone)

        # Compare the state of the two models.
        comparison_exceptions = []
        if not isinstance(
            get_default_fp8_recipe(), MXFP8BlockScaling
        ):  # BlockScaling recipe doesn't have state like scale, amax_history.
            for bound_symbol in fwd_traces[-1].bound_symbols:
                if "te_linear" in bound_symbol.sym.name:
                    thunder_fp8_meta = bound_symbol._call_ctx[bound_symbol.sym.name].func.fp8_meta
                    te_fp8_meta = thunder_to_te_layer_map[bound_symbol.sym.name].fp8_meta
                    try:
                        # fwd tensor history
                        assert_close(thunder_fp8_meta["scaling_fwd"].scale, te_fp8_meta["scaling_fwd"].scale)
                        assert_close(
                            thunder_fp8_meta["scaling_fwd"].amax_history, te_fp8_meta["scaling_fwd"].amax_history
                        )
                        # bwd tensor history
                        assert_close(thunder_fp8_meta["scaling_bwd"].scale, te_fp8_meta["scaling_bwd"].scale)
                        assert_close(
                            thunder_fp8_meta["scaling_bwd"].amax_history, te_fp8_meta["scaling_bwd"].amax_history
                        )

                        # This has to be on all ranks so that the computation is not blocked
                        is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale)
                        # See NOTE: TE forward tensor meta-data sync
                        is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history[1:])
                        is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                        is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
                    except Exception as e:
                        # Return exceptions only for rank==0
                        if rank == 0:
                            comparison_exceptions.append(e)

        # Compare weights after `n_iters`
        shard_size = int(dim / world_size)
        fsdp_te_params = tuple(te_model.parameters())
        try:
            assert_close(jit_model.get_parameter("fc1.weight"), fsdp_te_params[0].view(shard_size, dim))
            assert_close(jit_model.get_parameter("fc2.weight"), fsdp_te_params[1].view(shard_size, dim))
        except Exception as e:
            # Return exceptions only for rank==0
            if rank == 0:
                comparison_exceptions.append(e)

        return comparison_exceptions


def _test_fsdp_transformer_engine_bucketing(input_data):
    # Test Description: Test is to that TE works with bucketing.
    from thunder.tests.llama2_model import Transformer, ModelArgs

    init_method, world_size, rank, executor, device, _unused_dtype, kwargs = input_data
    thunder_fsdp_strategy, bucketing = kwargs["thunder_fsdp_strategy_and_bucketing"]
    devicetype = devices.device_from_string(device).devicetype

    # Setting LOCAL_RANK is necessary for thunder.distributed.fsdp
    with unittest.mock.patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
        init_per_process_distributed(init_method, devicetype, world_size, rank)
        torch.cuda.set_device(rank)

        # data
        batch_size = 64
        max_seq_len = 64
        vocab_size = 64

        model_args = dict(
            dim=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            vocab_size=vocab_size,
            multiple_of=32,
            max_seq_len=max_seq_len,
            dropout=0.0,
            hidden_dim=64,
        )
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        model.to(device)
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
        jit_model = thunder.distributed.fsdp(
            thunder.jit(model, executors=(transformer_engine_ex,) + thunder.get_default_executors()),
            sharding_strategy=thunder_fsdp_strategy,
            bucketing_strategy=bucketing,
        )

        sanity_exceptions = []
        try:
            for _ in range(5):
                out = jit_model(x, y).sum()
                out.backward()

            # Verifies te_linear was called
            forward_trace = thunder.last_traces(jit_model)
            backward_trace = thunder.last_backward_traces(jit_model)
            assert any(bsym.sym.name.startswith("te_linear") for bsym in forward_trace[-1].bound_symbols)
            assert any(
                bsym.sym.name.startswith("te_functional_linear_backward") for bsym in backward_trace[-1].bound_symbols
            )
        except Exception as e:
            sanity_exceptions.append(e)

        if rank == 0:
            return sanity_exceptions
        return None


# NOTE CPU is skipped because of
# RuntimeError: no support for _allgather_base in Gloo process group
@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    decorators=(
        pytest.mark.parametrize(
            "fsdp_bucketing_strategy",
            (
                FSDPBucketingStrategy.NONE,
                FSDPBucketingStrategy.LAYER,
                FSDPBucketingStrategy.BLOCK,
            ),
        ),
    ),
)
@distributed_wrapper("test_native_fsdp", _test_native_fsdp_helper)
def test_native_fsdp(executor, devices, dtype, fsdp_bucketing_strategy):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        # NOTE: ddp_wrapper
        pytest.mark.parametrize(
            "thunder_fsdp_strategy_and_intermediate_sharding",
            (
                (FSDPType.ZERO2, False),
                (FSDPType.ZERO3, False),
                # Intermediate sharding is only availabe TE v1.8 onwards
                pytest.param(
                    (FSDPType.ZERO3, True),
                    marks=pytest.mark.skip("Intermediate sharding is errors in TE 2.0 (also with eager)."),
                ),
            ),
        ),
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_fsdp_transformer_engine", _test_fsdp_transformer_engine)
def test_fsdp_transformer_engine(executor, devices, dtype, thunder_fsdp_strategy_and_intermediate_sharding):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        # NOTE: ddp_wrapper
        pytest.mark.parametrize(
            "thunder_fsdp_strategy_and_bucketing",
            (
                (FSDPType.ZERO3, FSDPBucketingStrategy.LAYER),
                (FSDPType.ZERO3, FSDPBucketingStrategy.BLOCK),
            ),
        ),
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_fsdp_transformer_engine_bucketing", _test_fsdp_transformer_engine_bucketing)
def test_fsdp_transformer_engine_bucketing(executor, devices, dtype, thunder_fsdp_strategy_and_bucketing):
    pass


def _test_fsdp_transformer_engine_v2(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value after `n_iter`.

    init_method, world_size, rank, executor, device, _unused_dtype, kwargs = input_data
    thunder_fsdp_strategy, intermediate_activation_sharding = kwargs["thunder_fsdp_strategy_and_intermediate_sharding"]
    devicetype = devices.device_from_string(device).devicetype

    fp8_recipe = get_default_fp8_recipe()

    # Setting LOCAL_RANK is necessary for thunder.distributed.fsdp
    with unittest.mock.patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
        init_per_process_distributed(init_method, devicetype, world_size, rank)
        torch.cuda.set_device(rank)

        dim = 256
        # Running more iterations leads to `nan` for both eager and thunder
        # with BlockScaling.
        n_iter = 5

        class ThunderModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim, bias=False)
                self.fc2 = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        # Weights
        fc1_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")
        fc2_weight = torch.randn(dim, dim, requires_grad=True, device="cuda")

        # Inputs (different input on different rank).
        if rank == 0:
            x = torch.arange(dim * dim, dtype=torch.float, device="cuda").view(dim, dim)
        if rank == 1:
            x = torch.randn(dim, dim, device="cuda") * 100

        with torch.device("cuda"):
            thunder_model = ThunderModel()
        thunder_model.fc1.weight.data = fc1_weight.clone()
        thunder_model.fc2.weight.data = fc2_weight.clone()

        jit_model = thunder.distributed.fsdp(
            thunder.jit(
                thunder_model,
                executors=[
                    transformer_engine_v2_ex,
                ]
                + executor.executors_list(),
                fp8_shard_intermediate_activation=intermediate_activation_sharding,
                transforms=[TransformerEngineTransformV2()],
            ),
            sharding_strategy=thunder_fsdp_strategy,
        )

        optim = torch.optim.SGD(jit_model.parameters())

        for _ in range(n_iter):
            with fp8_autocast(fp8_recipe=fp8_recipe):
                o = jit_model(x).sum()
            o.backward()
            optim.step()
            optim.zero_grad()

        FP8GlobalStateManager.reset()

        class TEModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = TELinear(dim, dim, bias=False)
                self.fc2 = TELinear(dim, dim, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.relu(self.fc1(x)))

        with torch.device("cuda"):
            te_model = TEModel()
        te_model.fc1.weight.data = fc1_weight.clone()
        te_model.fc2.weight.data = fc2_weight.clone()

        fsdp_model = FullyShardedDataParallel(te_model, auto_wrap_policy=always_wrap_policy)
        if intermediate_activation_sharding:
            transformer_engine.pytorch.distributed.prepare_te_modules_for_fsdp(fsdp_model)
        optim = torch.optim.SGD(te_model.parameters())

        for _ in range(n_iter):
            with fp8_autocast(fp8_recipe=fp8_recipe):
                o = fsdp_model(x).sum()

            o.backward()
            optim.step()
            optim.zero_grad()

        # Compare weights after `n_iters`
        comparison_exceptions = []
        shard_size = int(dim / world_size)
        fsdp_te_params = tuple(te_model.parameters())
        try:
            assert_close(jit_model.get_parameter("fc1.weight"), fsdp_te_params[0].view(shard_size, dim))
            assert_close(jit_model.get_parameter("fc2.weight"), fsdp_te_params[1].view(shard_size, dim))
        except Exception as e:
            # Return exceptions only for rank==0
            if rank == 0:
                comparison_exceptions.append(e)

        return comparison_exceptions


def _test_fsdp_transformer_engine_v2_bucketing(input_data):
    # Test Description: Test is to that TEv2 works with bucketing.
    from thunder.tests.llama2_model import Transformer, ModelArgs

    init_method, world_size, rank, executor, device, _unused_dtype, kwargs = input_data
    thunder_fsdp_strategy, bucketing = kwargs["thunder_fsdp_strategy_and_bucketing"]
    devicetype = devices.device_from_string(device).devicetype

    fp8_recipe = get_default_fp8_recipe()

    # Setting LOCAL_RANK is necessary for thunder.distributed.fsdp
    with unittest.mock.patch.dict(os.environ, {"LOCAL_RANK": str(rank)}):
        init_per_process_distributed(init_method, devicetype, world_size, rank)
        torch.cuda.set_device(rank)

        # data
        batch_size = 64
        max_seq_len = 64
        vocab_size = 64

        model_args = dict(
            dim=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            vocab_size=vocab_size,
            multiple_of=32,
            max_seq_len=max_seq_len,
            dropout=0.0,
            hidden_dim=64,
        )
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        model.to(device)
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
        jit_model = thunder.distributed.fsdp(
            thunder.jit(
                model,
                executors=(transformer_engine_v2_ex,) + thunder.get_default_executors(),
                transforms=[TransformerEngineTransformV2()],
            ),
            sharding_strategy=thunder_fsdp_strategy,
            bucketing_strategy=bucketing,
        )

        sanity_exceptions = []
        try:
            for _ in range(5):
                with fp8_autocast(fp8_recipe=fp8_recipe):
                    out = jit_model(x, y).sum()
                out.backward()

            # Verifies te_linear was called
            forward_trace = thunder.last_traces(jit_model)
            backward_trace = thunder.last_backward_traces(jit_model)
            assert any(bsym.sym.name.startswith("te_functional_linear_fwd") for bsym in forward_trace[-1].bound_symbols)
            assert any(
                bsym.sym.name.startswith("te_functional_linear_bwd") for bsym in backward_trace[-1].bound_symbols
            )
        except Exception as e:
            sanity_exceptions.append(e)

        if rank == 0:
            return sanity_exceptions
        return None


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        # NOTE: ddp_wrapper
        pytest.mark.parametrize(
            "thunder_fsdp_strategy_and_intermediate_sharding",
            (
                (FSDPType.ZERO2, False),
                (FSDPType.ZERO3, False),
                # Intermediate sharding is only availabe TE v1.8 onwards
                pytest.param(
                    (FSDPType.ZERO3, True),
                    marks=pytest.mark.skip("Intermediate sharding is errors in TE 2.0 (also with eager)."),
                ),
            ),
        ),
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_fsdp_transformer_engine_v2", _test_fsdp_transformer_engine_v2)
def test_fsdp_transformer_engine_v2(executor, devices, dtype, thunder_fsdp_strategy_and_intermediate_sharding):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        # NOTE: ddp_wrapper
        pytest.mark.parametrize(
            "thunder_fsdp_strategy_and_bucketing",
            (
                (FSDPType.ZERO3, FSDPBucketingStrategy.LAYER),
                (FSDPType.ZERO3, FSDPBucketingStrategy.BLOCK),
            ),
        ),
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_fsdp_transformer_engine_v2_bucketing", _test_fsdp_transformer_engine_v2_bucketing)
def test_fsdp_transformer_engine_v2_bucketing(executor, devices, dtype, thunder_fsdp_strategy_and_bucketing):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
