import os
import unittest
from itertools import product

import pytest
import torch
import torch.distributed as tdist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import thunder
import thunder.executors
import thunder.torch as ltorch
from thunder.core import devices
from thunder.distributed import ddp
from thunder.tests.framework import instantiate, TorchExecutor

from thunder.executors.transformer_engineex import (
    transformer_engine_ex,
    TE_AVAILABLE,
    te_sync_fp8_meta_bwd,
)


is_fp8_supported: bool = False
# This will be correctly updated below when TE Engine is installed
# and if the current environment doesn't support FP8.
fp8_support_reason: str = ""
if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch.fp8 import check_fp8_support, FP8GlobalStateManager

    is_fp8_supported, fp8_support_reason = check_fp8_support()

from thunder.tests.distributed.helper import (
    ToyModel,
    DistributedParallelTestCase,
    executors_map,
    SmallModel,
    create_per_process_dataloader,
    run_test_no_sync_grad_accumulation,
    distributed_wrapper,
    init_per_process_distributed,
)
from torch.testing._internal import common_utils


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DDP test requires CUDA and NCCL `torch.distributed` backend",
)
class DDPTest(DistributedParallelTestCase):
    # Reference issue "Add an example of DDP(compile(model)) to tests"
    def test_ddp_compile_module(self):
        model = ToyModel().to(self.rank)
        ddp_model = DDP(thunder.jit(model, device_ids=[self.rank]))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

        x, labels = torch.randn(20, 12).to(self.rank), torch.randn(20, 8).to(self.rank)

        init_loss, last_loss = None, None
        for i in range(3):
            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == 0:
                init_loss = loss.detach().item()
            if i == 2:
                last_loss = loss.detach().item()
        assert init_loss > last_loss

    # Reference issue "[tracker] Support DistributedDataParallel"
    def test_compile_ddp_module(self):
        model = ToyModel().to(self.rank)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"DistributedDataParallel.*not supported",
        ):
            cm = thunder.jit(DDP(model, device_ids=[self.rank]))
            x = torch.randn(20, 12).to(self.rank)
            cm(x)

    @common_utils.parametrize("executor,bucket_size_in_mb", product(tuple(executors_map.keys()), (0, 1000)))
    def test_ddp_grad_bucketing(self, executor, bucket_size_in_mb: int):
        from thunder.distributed import ddp
        from thunder.executors.torchex import (
            pack_prim_impl,
            unpack_prim_impl,
            update_bucket_view_prim_impl,
            all_reduce_prim_impl,
        )

        device = torch.device("cuda", self.rank)
        m = ToyModel().to(device)
        cm = thunder.jit(
            ddp(m, bucket_size_in_mb=bucket_size_in_mb),
            executors=executors_map[executor].executors_list(),
        )
        x = torch.ones((2, 12)).to(device)
        cm(x).mean().backward()

        bwd_extrace = thunder.last_backward_traces(cm)[-1]
        bsym_sym_id_list = [bsym.sym.id for bsym in bwd_extrace.bound_symbols]
        pack_syms = tuple(filter(lambda a: a == pack_prim_impl.id, bsym_sym_id_list))
        unpack_syms = tuple(filter(lambda a: a == unpack_prim_impl.id, bsym_sym_id_list))
        update_bucket_view_syms = tuple(filter(lambda a: a == update_bucket_view_prim_impl.id, bsym_sym_id_list))
        if bucket_size_in_mb == 0:
            self.assertEqual(len(pack_syms), 0)
            self.assertEqual(len(unpack_syms), 0)
            self.assertEqual(len(update_bucket_view_syms), 0)
            for bsym in bwd_extrace.bound_symbols:
                if bsym.sym.id == all_reduce_prim_impl.id:
                    # oh, everything is put into `bsym.args`?
                    msg = f"{bsym.args=}, {bsym.kwargs=}"
                    self.assertTrue(bsym.args[-1], msg=msg)
        else:
            self.assertEqual(len(pack_syms), 1, msg=f"{pack_syms}")
            self.assertEqual(len(unpack_syms), 1, msg=f"{unpack_syms}")
            self.assertEqual(len(update_bucket_view_syms), 4, msg=f"{update_bucket_view_prim_impl}")

    @unittest.mock.patch.dict(os.environ, {"KINETO_LOG_LEVEL": "5"})  # silence torch.profiler logs
    @common_utils.parametrize(
        "executor,bucket_size_in_mb,dataset_size",
        product(tuple(executors_map.keys()), (0, 25), (1, 2)),
    )
    def test_ddp_with_no_sync_grad_accumulation(self, executor: str, bucket_size_in_mb: float, dataset_size: int):
        # This case tries to guarantee the parity between `thunder.distributed.ddp` with and without `no_sync`
        # from the perspectives of trace and numeric.
        # At trace level, in `no_sync`, the backward trace should NOT have AllReduce while outside of `no_sync`,
        # the trace should have.
        # For numerical parity, we compare the accumulated gradients with and without `no_sync` and even against gradients without accumulation.
        # If they are different, it'd be impossible to keep replicas identical.
        from thunder.common import CACHE_OPTIONS
        from thunder.distributed import ddp

        def get_model_and_optimizer(device):
            m = ToyModel().to(device)
            ddp_m = ddp(m, bucket_size_in_mb=bucket_size_in_mb)
            jitted_ddp_m = thunder.jit(
                ddp_m,
                cache_mode=CACHE_OPTIONS.CONSTANT_VALUES,
                executors=executors_map[executor].executors_list(),
            )
            optimizer = torch.optim.SGD(jitted_ddp_m.parameters(), lr=1e-3)
            return jitted_ddp_m, optimizer

        def is_comm(k: str) -> bool:
            return "allreduce_" in k or "all_reduce" in k

        run_test_no_sync_grad_accumulation(self, get_model_and_optimizer, is_comm, dataset_size)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_ddp_grad_parity_with_without_bucketing(self, executor):
        from thunder.distributed import ddp

        device = torch.device("cuda", self.rank)
        initial_model_state = ToyModel().to(device).state_dict()

        for bucket_size_in_mb in (0, 100):
            m = ToyModel().to(device)
            m.load_state_dict(initial_model_state)
            cm = thunder.jit(
                ddp(m, bucket_size_in_mb=bucket_size_in_mb),
                executors=executors_map[executor].executors_list(),
            )
            x = torch.ones((2, 12)).to(device)
            cm(x).mean().backward()

            if bucket_size_in_mb == 0:
                gradients = tuple(p.grad for p in cm.parameters() if p.grad is not None)
            else:
                self.assertEqual(tuple(p.grad for p in cm.parameters() if p.grad is not None), gradients)

    def test_ddp_model_as_argument(self):
        # Sanity test to make sure passing model as argument to
        # thunder.jit with `ddp` compiles.
        device = torch.device("cuda", self.rank)
        model = torch.nn.Linear(5, 10, bias=False, device=device)
        x = torch.randn(2, 5, device=device)

        def fwd_loss(m, x):
            return m(x).sum()

        model = thunder.distributed.ddp(model)
        fwd_loss = thunder.jit(fwd_loss)
        fwd_loss(model, x)

        # notice how we cannot do `model.no_sync()` because it's not a ThunderModule
        with thunder.ThunderModule.no_sync(model):
            fwd_loss(model, x)


common_utils.instantiate_parametrized_tests(DDPTest)


# NOTE This assumes that one process will have rank=0 -- could generalize that to root
# TODO Test training, this test just currently tests forward
def _test_native_ddp_helper(input_data):
    init_method, world_size, rank, executor, device, dtype, kwargs = input_data
    bucket_size_in_mb = kwargs.get("bucket_size_in_mb", 0)

    num_samples = 2
    tensor_shape = (2, 2)
    sample_seed = 3456
    num_epochs = 1
    devicetype = devices.device_from_string(device).devicetype
    torch_dtype = ltorch.to_torch_dtype(dtype)

    pg = init_per_process_distributed(init_method, devicetype, world_size, rank)

    tdist.barrier(pg)

    dataloader = create_per_process_dataloader(
        rank,
        num_samples=num_samples,
        tensor_shape=tensor_shape,
        tensor_dtype=torch_dtype,
        sample_seed=sample_seed,
        devicetype=devicetype,
    )

    # Creates, compiles, and DDPs the model
    model = SmallModel(device, torch_dtype)
    ddp_model = ddp(model)
    cmodel = thunder.jit(
        ddp_model,
        executors=executor.executors_list(),
    )

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

            grad_gather_list = None
            for param_with_grad in filter(lambda p: p.grad is not None, cmodel.parameters()):
                if rank == 0:
                    grad_gather_list = []
                    for _ in range(world_size):
                        grad_gather_list.append(torch.empty_like(param_with_grad))

                grad = param_with_grad.grad

                tdist.gather(grad, grad_gather_list, dst=0, group=pg, async_op=False)

                if rank == 0:
                    for other in grad_gather_list:
                        try:
                            assert_close(grad, other)
                        except Exception as e:
                            comparison_exceptions.append(e)

    # NOTE This function is undocumented; its definition is here:
    #   https://github.com/pytorch/pytorch/blob/416bf4e/torch/distributed/distributed_c10d.py#L1359
    tdist.barrier(pg)
    tdist.destroy_process_group(pg)

    if rank == 0:
        bwd_extrace_sym_ids = [bsym.sym.id for bsym in thunder.last_backward_traces(cmodel)[-1].bound_symbols]
        pack_unpack_update_bucket_view_found = (
            "torch_pack_prim_impl" in bwd_extrace_sym_ids
            and "torch_unpack_prim_impl" in bwd_extrace_sym_ids
            and "torch_update_bucket_view_prim_impl" in bwd_extrace_sym_ids
        )
        return comparison_exceptions and (pack_unpack_update_bucket_view_found or bucket_size_in_mb == 0)

    return None


def _test_ddp_transformer_engine(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value and
    # fp8 meta state is same after `n_iter`.
    init_method, world_size, rank, executor, device, dtype, _unused_kwargs = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
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
    fc1_weight = torch.randn(dim, dim, requires_grad=True).cuda()
    fc2_weight = torch.randn(dim, dim, requires_grad=True).cuda()

    # Inputs (different input on different rank).
    if rank == 0:
        x = torch.arange(dim * dim, dtype=torch.float).view(dim, dim).cuda()
    if rank == 1:
        x = torch.randn(dim, dim).cuda() * 100

    thunder_model = ThunderModel().cuda()
    thunder_model.fc1.weight.data = fc1_weight.clone()
    thunder_model.fc2.weight.data = fc2_weight.clone()

    jit_model = thunder.jit(
        thunder.distributed.ddp(thunder_model),
        executors=[
            transformer_engine_ex,
        ]
        + executor.executors_list(),
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

    te_model = TEModel().cuda()
    te_model.fc1.weight.data = fc1_weight.clone()
    te_model.fc2.weight.data = fc2_weight.clone()

    ddp_model = DDP(te_model)

    optim = torch.optim.SGD(te_model.parameters())

    for _ in range(n_iter):
        with fp8_autocast():
            o = ddp_model(x).sum()

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
                # NOTE: TE forward tensor meta-data sync
                # Syncing of FP8 meta-data happens in two step in the forward pass.
                # 1. When we enter the fp8_autocast(), all the forward fp8 meta-data
                # in global buffer is synced.
                # See: https://github.com/NVIDIA/TransformerEngine/blob/6a9edc38bf9b941b7d369af5103fa8fe0b121d61/transformer_engine/pytorch/fp8.py#L409-L412
                # 2. Post this, in the forward pass of the module in `prepare_forward`,
                # we read from the global-buffer the synced meta-data.
                # See: https://github.com/NVIDIA/TransformerEngine/blob/6a9edc38bf9b941b7d369af5103fa8fe0b121d61/transformer_engine/pytorch/module/base.py#L539-L545
                # However, at the end of this forward pass, we have seen new inputs and outputs. Their amax are recorded on
                # 0th row of `amax_history` (which will be synced only in the next forward pass).
                # So, here we check that every row except for `0` is same.
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history[1:])
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale_inv)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
            except Exception as e:
                # Return exceptions only for rank==0
                if rank == 0:
                    comparison_exceptions.append(e)

    # Compare weights after `n_iters`
    try:
        assert_close(thunder_model.fc1.weight, te_model.fc1.weight)
        assert_close(thunder_model.fc2.weight, te_model.fc2.weight)
    except Exception as e:
        # Return exceptions only for rank==0
        if rank == 0:
            comparison_exceptions.append(e)

    return comparison_exceptions


def _test_ddp_transformer_engine_llama_sanity(input_data):
    # Test Description: We run a dummy training loop for a Transformer Model
    # We run a few iterations to see that TransformerEngine doesn't throw internal assertion
    # due to reordering of forward and backward operators.
    # (This test will fail without `_rearrange_transformer_engine_linear` in `torch_autograd.py`)
    # For more details, see docstring for `_rearrange_transformer_engine_linear` in transformer_engine_ex.py.
    from thunder.tests.llama2_model import Transformer, ModelArgs

    init_method, world_size, rank, executor, device, dtype, _unused_kwargs = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
    init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.cuda.set_device(rank)
    # data
    batch_size = 2
    max_seq_len = 32
    vocab_size = 32

    model_args = dict(
        dim=32,
        n_layers=1,
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
        thunder.distributed.ddp(model), executors=(transformer_engine_ex,) + thunder.get_default_executors()
    )

    sanity_exceptions = []
    try:
        for _ in range(5):
            out = jit_model(x, y).sum()
            out.backward()

        bwd_exec_trace = thunder.last_backward_traces(jit_model)[-1]

        # Last symbol of the trace should be `return`
        return_sym_idx = len(bwd_exec_trace.bound_symbols) - 1
        assert thunder.core.prims.PrimIDs.RETURN == bwd_exec_trace.bound_symbols[return_sym_idx].sym.id

        # Verify that the symbol to sync backward
        # fp8 metadata is present in backward trace.
        for idx, bsym in enumerate(bwd_exec_trace.bound_symbols):
            if bsym.sym.id == te_sync_fp8_meta_bwd.id:
                # Verify that `te_sync_fp8_meta_bwd` is before the last symbol of the trace
                # which is `return`
                assert idx < return_sym_idx
                break
        else:
            raise RuntimeError("Backward sync symbol not found.")
    except Exception as e:
        sanity_exceptions.append(e)

    if rank == 0:
        return sanity_exceptions
    return None


# NOTE This is just a stub, see the NOTE for ddp_wrapper
@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    # CPU broke around PyTorch 2.3.1, see PR #545
    devicetypes=(devices.DeviceType.CUDA,),
    decorators=(pytest.mark.parametrize("bucket_size_in_mb", (0, 25)),),
)
@distributed_wrapper("test_native_ddp", _test_native_ddp_helper)
def test_native_ddp(executor, devices, dtype, bucket_size_in_mb):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # NOTE: Setting `NVTE_TORCH_COMPILE`
        # It is important to set this flag so that TE doesn't use
        # `torch.compile` to fuse a few operations. This is because
        # `torch.compile` creates a new process and that leads to
        # the error : daemonic processes are not allowed to have children
        # when running the tests.
        # With the setting below, we use `torch.jit` for this test suite
        # See: https://github.com/NVIDIA/TransformerEngine/blob/a38b291b0d1b04847e8ab1df8550df642a03a27d/transformer_engine/pytorch/jit.py#L11-L19
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_ddp_transformer_engine", _test_ddp_transformer_engine)
def test_ddp_transformer_engine(executor, devices, dtype):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        # NOTE: We don't pass `clear=True` to `unittest.mock.patch.dict` as that may clear paths
        # from environment leading to picking up of incorrect dependencies in the spawned process.
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}),
    ),
)
@distributed_wrapper("test_ddp_transformer_engine_llama_sanity", _test_ddp_transformer_engine_llama_sanity)
def test_ddp_transformer_engine_llama_sanity(executor, devices, dtype):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
