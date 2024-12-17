import os
import unittest
import weakref
from itertools import product
from collections.abc import Callable

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


is_fp8_supported: bool = False
# This will be correctly updated below when TE Engine is installed
# and if the current environment doesn't support FP8.
fp8_support_reason: str = ""
if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch.fp8 import check_fp8_support, FP8GlobalStateManager
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

    def test_rematerialize_all_gather(self):
        device = torch.device("cuda", self.rank)
        m = ToyModel().to(device)
        cm = thunder.jit(
            fsdp(m, device=device, broadcast_from=0),
        )
        x = torch.ones((2, 12), device=device)
        cm(x).mean().backward()

        fwd_trc = [
            t for t in thunder.last_traces(cm) if getattr(t.get_provenance(), "pss", "") == "Augmented forward pass"
        ][0]
        bwd_trc = thunder.last_backward_traces(cm)[0]
        from thunder.core.rematerialization import rematerialize_all_gather

        result_fwd_trc, result_bwd_trc = rematerialize_all_gather(fwd_trc, bwd_trc)

        # check the return statement in forward trace is updated
        # TODO: this is not stable w.r.t. details of the processing, the sharded correspond to ("t_net1_weight", "t_net2_weight")
        #       in the original trace and are inputs to all_gather, the unshard are the outputs fo the corresponding wait
        #       If you fix this to be dynamically discerned, you'll be my hero.
        sharded_param_names = ("t_net1_weight", "t_net2_weight")
        # t5 and t20 are all-gather'ed t_net1_weight and t_net2_weight, respectively.
        unshard_param_names = ("t5", "t20")
        result_saved_for_bwd = [x.name for x in fwd_trc.bound_symbols[-1].args[1][0]]
        self.assertTrue(all(t not in sharded_param_names for t in result_saved_for_bwd))
        self.assertTrue(all(t in result_saved_for_bwd for t in unshard_param_names))

        result_saved_for_bwd = [x.name for x in result_fwd_trc.bound_symbols[-1].args[1][0]]
        self.assertTrue(all(t in result_saved_for_bwd for t in sharded_param_names))
        self.assertTrue(all(t not in unshard_param_names for t in result_saved_for_bwd))

        # check allgather is inserted in backward trace
        from thunder.distributed.prims import PrimIDs

        self.assertTrue(all(bsym.sym.id != PrimIDs.ALL_GATHER for bsym in bwd_trc.bound_symbols))
        self.assertTrue(any(bsym.sym.id == PrimIDs.ALL_GATHER for bsym in result_bwd_trc.bound_symbols))

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
            fsdp_m = fsdp(m, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype)
            jitted_ddp_m = thunder.jit(
                fsdp_m,
                cache_mode=CACHE_OPTIONS.CONSTANT_VALUES,
                executors=executors_map[executor].executors_list(),
            )
            optimizer = torch.optim.SGD(jitted_ddp_m.parameters(), lr=1e-3)
            return jitted_ddp_m, optimizer

        def is_comm(k: str) -> bool:
            return "reducescatter" in k or "reduce_scatter" in k

        run_test_no_sync_grad_accumulation(self, get_model_and_optimizer, is_comm, dataset_size=2)

    def _test_no_sync_grad_accumulation(
        self,
        get_model_and_optimizer: Callable[[torch.device], tuple[torch.nn.Module, torch.optim.Optimizer]],
        is_comm: Callable[[str], bool],
        dataset_size,
    ):
        from collections import defaultdict
        from contextlib import nullcontext
        from thunder.distributed import get_skip_data_parallel_grad_sync

        device = torch.device("cuda", self.rank)
        batch_size = 128
        num_micro_batch = 4
        micro_batch_size = batch_size // num_micro_batch
        with torch.no_grad():
            dataloader = [
                (torch.randn(batch_size, 12, device=device), torch.randn(batch_size, 8, device=device))
                for _ in range(dataset_size)
            ]

        # TODO(crcrpar): Use `last_traces` to check if allreduce was called, instead of `torch.profiler.profile`
        # See: https://github.com/Lightning-AI/lightning-thunder/pull/1881#issuecomment-1910455732
        def run_fwd_bwd(iter_count, model, x, y, num_grad_accum_steps: int | None = None):
            with torch.profiler.profile() as prof:
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                if num_grad_accum_steps is not None:
                    loss /= num_grad_accum_steps
                loss.backward()

            keys = tuple([e.key for e in prof.key_averages()])
            has_comms = any(is_comm(k) for k in keys)
            msg = f"{keys=}"
            if get_skip_data_parallel_grad_sync():
                self.assertFalse(has_comms, msg=msg)
            else:
                self.assertTrue(has_comms, msg=msg)

            return loss

        def get_ground_truth_loss_grads(device, dataloader):
            compiled_ddp_m, optimizer = get_model_and_optimizer(device)
            initial_state_dict = compiled_ddp_m.state_dict()

            losses, grads = [], []

            for iter_count, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                losses.append(run_fwd_bwd(iter_count, compiled_ddp_m, x, y, num_grad_accum_steps=None))
                grads.append([p.grad for p in compiled_ddp_m.parameters() if p.grad is not None])
                optimizer.step()

            return initial_state_dict, losses, grads

        device = torch.device("cuda", self.rank)
        initial_state_dict, ground_truth_losses, ground_truth_grads = get_ground_truth_loss_grads(device, dataloader)

        gradients = defaultdict(list)
        for use_no_sync in (True, False):
            jitted_model, optimizer = get_model_and_optimizer(device)
            jitted_model.load_state_dict(initial_state_dict)

            for iter_count, (x, y) in enumerate(dataloader):
                loss = torch.zeros((), device=device)
                with jitted_model.no_sync() if use_no_sync else nullcontext():
                    for i in range(num_micro_batch - 1):
                        cur_loss = run_fwd_bwd(
                            iter_count,
                            jitted_model,
                            x[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            y[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            num_micro_batch,
                        )
                        with torch.no_grad():
                            loss += cur_loss
                        if use_no_sync and i == 0 and iter_count == 0:
                            # make sure the backward trace under `no_sync` has actual math computations.
                            no_sync_bwd_trc = thunder.last_backward_traces(jitted_model)[-1]
                            self.assertGreater(len(no_sync_bwd_trc.bound_symbols), 1)
                cur_loss = run_fwd_bwd(
                    iter_count, jitted_model, x[-micro_batch_size:, :], y[-micro_batch_size:, :], num_micro_batch
                )
                with torch.no_grad():
                    loss += cur_loss
                optimizer.step()
                gradients[use_no_sync].append([p.grad for p in jitted_model.parameters() if p.grad is not None])
                optimizer.zero_grad(set_to_none=True)

                num_expected_caches: int
                if use_no_sync:
                    num_expected_caches = 2
                else:
                    num_expected_caches = 1
                self.assertEqual(len(jitted_model._lc_cs.interpreter_cache), num_expected_caches)

                torch.testing.assert_close(loss, ground_truth_losses[iter_count], atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(
                    actual=gradients[use_no_sync][iter_count],
                    expected=ground_truth_grads[iter_count],
                    atol=5e-5,
                    rtol=5e-3,
                )
                if not use_no_sync:
                    torch.testing.assert_close(
                        actual=gradients[True][iter_count],
                        expected=gradients[False][iter_count],
                    )

    # TODO(crcrpar): Add torch compile to executors_list
    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype,apply_fsdp_first",
        product(
            tuple(executors_map.keys()),
            (
                FSDPBucketingStrategy.LAYER,
                FSDPBucketingStrategy.BLOCK,
            ),
            (FSDPType.ZERO2, FSDPType.ZERO3),
            (True, False),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype, apply_fsdp_first: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}_{'jit_fsdp' if apply_fsdp_first else 'fsdp_jit'}"
        ),
    )
    def test_fsdp_grad_parity_with_without_bucketing(
        self,
        executor,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
        apply_fsdp_first: bool,
    ):
        from thunder.distributed import fsdp

        device = torch.device("cuda", self.rank)
        initial_model_state = ToyModel().state_dict()

        for strategy in (FSDPBucketingStrategy.NONE, bucketing_strategy):
            m = ToyModel()
            m.load_state_dict(initial_model_state)
            if apply_fsdp_first:
                cm = thunder.jit(
                    fsdp(m, device=device, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype),
                    executors=executors_map[executor].executors_list(),
                )
            else:
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
        "bucketing_strategy,fsdptype,apply_fsdp_first",
        product(
            (
                FSDPBucketingStrategy.NONE,
                FSDPBucketingStrategy.BLOCK,
            ),
            (FSDPType.ZERO2, FSDPType.ZERO3),
            (True, False),
        ),
        name_fn=lambda bucketing_strategy, fsdptype, apply_fsdp_first: (
            f"bucketing_{str(bucketing_strategy).split('.')[1].lower()}_"
            f"{(str(fsdptype).lower().split('.')[1])}_{'jit_fsdp' if apply_fsdp_first else 'fsdp_jit'}"
        ),
    )
    def test_fsdp_with_padding(
        self,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
        apply_fsdp_first: bool,
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
        if apply_fsdp_first:
            jitted = thunder.jit(fsdp(m, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype))
        else:
            jitted = fsdp(thunder.jit(m), bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype)

        x = torch.randn(4, 4, device=device)
        y = jitted(x)
        y.mean().backward()

        fw_extrace = thunder.last_traces(jitted)[-1]
        # When bookend is turned off, `slice` and `pad` may appear in nvFusion subsymbols.
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
            fsdp(m, device=device, broadcast_from=0, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype),
            executors=executors_map[executor].executors_list(),
        )
        x = torch.ones((2, config.block_size, config.n_embd), device=device)
        loss = cm(x).mean()
        loss.backward()

        # get the trace before sorting
        fwd_trc = thunder.last_traces(cm)[-2]
        bwd_trc = thunder.last_backward_traces(cm)[-2]

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
            jit_fsdp_model = Model()
            fsdp_jit_model = Model()
            x = torch.ones(4, 16)

        # Check `jit(fsdp(model))` works
        jit_fsdp_model.fc1.weight = jit_fsdp_model.fc2.weight

        jit_fsdp_model = thunder.jit(thunder.distributed.fsdp(jit_fsdp_model), executors=["torch"])

        _test_model_output_and_gradients(jit_fsdp_model, x, duplicate_all_gather=True)

        # Check `fsdp(jit(model))` works
        fsdp_jit_model.fc1.weight = fsdp_jit_model.fc2.weight

        fsdp_jit_model = thunder.distributed.fsdp(thunder.jit(fsdp_jit_model, executors=["torch"]))

        _test_model_output_and_gradients(fsdp_jit_model, x, duplicate_all_gather=False)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    @common_utils.parametrize("model_device", ["cuda", "meta"])
    def test_memory_consumption(self, model_device):
        import gc

        device = torch.device("cuda", self.rank)
        with device:
            x_1 = torch.randn((2, ToyModel.N_IN))
        with torch.device(model_device):
            model = ToyModel()
        jit_fsdp_model = thunder.jit(fsdp(model, device=device))
        y_1 = jit_fsdp_model(x_1)
        active_mem_jit_fsdp = torch.cuda.memory_stats()["active_bytes.all.current"]

        del x_1, y_1, jit_fsdp_model, model
        gc.collect()
        torch.cuda.empty_cache()

        with device:
            x_2 = torch.randn((2, ToyModel.N_IN))
        with torch.device(model_device):
            model = ToyModel()
        fsdp_jit_model = fsdp(thunder.jit(model), device=device)
        y_2 = fsdp_jit_model(x_2)
        active_mem_fsdp_jit = torch.cuda.memory_stats()["active_bytes.all.current"]
        self.assertAlmostEqual(active_mem_fsdp_jit, active_mem_jit_fsdp)

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
        import re
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


common_utils.instantiate_parametrized_tests(FSDPTest)


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
        n_iter = 10

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

        jit_model = thunder.jit(
            thunder.distributed.fsdp(thunder_model, sharding_strategy=thunder_fsdp_strategy),
            executors=[
                transformer_engine_ex,
            ]
            + executor.executors_list(),
            fp8_shard_intermediate_activation=intermediate_activation_sharding,
        )

        optim = torch.optim.SGD(thunder_model.parameters())

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
        for bound_symbol in fwd_traces[-1].bound_symbols:
            if "te_linear" in bound_symbol.sym.name:
                thunder_fp8_meta = bound_symbol._call_ctx[bound_symbol.sym.name].func.fp8_meta
                te_fp8_meta = thunder_to_te_layer_map[bound_symbol.sym.name].fp8_meta
                try:
                    # fwd tensor history
                    assert_close(thunder_fp8_meta["scaling_fwd"].scale, te_fp8_meta["scaling_fwd"].scale)
                    assert_close(thunder_fp8_meta["scaling_fwd"].scale_inv, te_fp8_meta["scaling_fwd"].scale_inv)
                    assert_close(thunder_fp8_meta["scaling_fwd"].amax_history, te_fp8_meta["scaling_fwd"].amax_history)
                    # bwd tensor history
                    assert_close(thunder_fp8_meta["scaling_bwd"].scale, te_fp8_meta["scaling_bwd"].scale)
                    assert_close(thunder_fp8_meta["scaling_bwd"].scale_inv, te_fp8_meta["scaling_bwd"].scale_inv)
                    assert_close(thunder_fp8_meta["scaling_bwd"].amax_history, te_fp8_meta["scaling_bwd"].amax_history)

                    # This has to be on all ranks so that the computation is not blocked
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale)
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale_inv)
                    # See NOTE: TE forward tensor meta-data sync
                    is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history[1:])
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale_inv)
                    is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
                except Exception as e:
                    # Return exceptions only for rank==0
                    if rank == 0:
                        comparison_exceptions.append(e)

        # Compare weights after `n_iters`
        shard_size = int(dim / world_size)
        fsdp_te_params = tuple(te_model.parameters())
        try:
            assert_close(thunder_model.fc1.weight, fsdp_te_params[0].view(shard_size, dim))
            assert_close(thunder_model.fc2.weight, fsdp_te_params[1].view(shard_size, dim))
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
        jit_model = thunder.jit(
            thunder.distributed.fsdp(model, sharding_strategy=thunder_fsdp_strategy, bucketing_strategy=bucketing),
            executors=(transformer_engine_ex,) + thunder.get_default_executors(),
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
                (FSDPType.ZERO3, True),
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


if __name__ == "__main__":
    common_utils.run_tests()
