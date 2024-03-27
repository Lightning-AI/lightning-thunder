import os
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch.distributed import distributed_c10d as c10d
from torch.distributed._tensor import DTensor
from torch.testing._internal import common_utils

import thunder
from thunder.distributed import _unshard_params
from thunder.distributed.checkpoint import (
    _split_state_dict,
    has_fsdp_modules,
    load_model_state_dict,
    StateDictOptions,
    save,
    get_model_state_dict,
    _TORCH_GREATER_EQUAL_2_3,
)
from thunder.tests.distributed.test_ddp import DataParallelTestCase


class Submodule(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.l = torch.nn.Linear(4, h * 2, bias=False)

    def forward(self, x):
        # defined just because preprocessing fails otherwise
        ...


class MyModel(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.register_buffer("buf", torch.tensor(0))
        self.l = torch.nn.Linear(2, h)
        self.inner = Submodule(h)

    def forward(self):
        # defined just because preprocessing fails otherwise
        ...


def test_split_state_dict():
    model = torch.nn.Linear(1, 1, bias=False)
    params, rest = _split_state_dict(model)
    assert set(params) == {"weight"}
    assert set(rest) == set()

    model = MyModel(4)
    model.inner = MyModel(2)
    params, rest = _split_state_dict(model)
    assert set(params) == {"inner.inner.l.weight", "l.weight", "l.bias", "inner.l.weight", "inner.l.bias"}
    assert set(rest) == {"inner.buf", "buf"}


def distributed_ckpt_to_regular(path):
    """From ``torch.distributed.checkpoint.format_utils.dcp_to_torch_save``."""
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    from torch.distributed.checkpoint import FileSystemReader

    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    else:
        from torch.distributed.checkpoint._traverse import set_element
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def set_up_planner(self, state_dict, metadata, is_coordinator):
                assert not state_dict
                # rebuild the state dict from the metadata
                for k, v in metadata.state_dict_metadata.items():
                    if isinstance(v, TensorStorageMetadata):
                        v = torch.empty(v.size, dtype=v.properties.dtype)
                    if k in metadata.planner_data:
                        set_element(state_dict, metadata.planner_data[k], v)
                    else:
                        state_dict[k] = v
                super().set_up_planner(state_dict, metadata, is_coordinator)

    state_dict = {}
    storage_reader = FileSystemReader(path)
    _load_state_dict(state_dict, storage_reader=storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    return state_dict


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "Distributed checkpoint tests require CUDA and NCCL `torch.distributed` backend",
)
class DistributedCheckpointTest(DataParallelTestCase):
    @property
    def tmp_path(self) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
        broadcast_list = [tmp_path]
        torch.distributed.broadcast_object_list(broadcast_list)
        tmp_path = broadcast_list[0]
        return tmp_path

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_get_model_state_dict(self):
        device = torch.device("cuda", self.rank)

        with device:
            model = MyModel(4)
        expected = model.state_dict()
        assert not has_fsdp_modules(model)

        # No sharding - full state dict
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        state_dict = get_model_state_dict(model, options, self.rank)
        torch.testing.assert_close(state_dict, expected)

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options, self.rank)
        expected_cpu = {k: v.to(device="cpu") for k, v in expected.items()}
        torch.testing.assert_close(state_dict, expected_cpu)

        options = StateDictOptions(full_state_dict=True, rank0_only=True)
        state_dict = get_model_state_dict(model, options, self.rank)
        torch.testing.assert_close(state_dict, expected if self.rank == 0 else {})

        # No sharding - sharded state dict
        options = StateDictOptions(full_state_dict=False)
        with pytest.raises(ValueError, match="cannot be used"):
            get_model_state_dict(model, options, self.rank)

        sharded_model = thunder.distributed.fsdp(model)
        assert has_fsdp_modules(model)

        # Sharding - full state dict
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        state_dict = get_model_state_dict(sharded_model, options, self.rank)
        torch.testing.assert_close(state_dict, expected)

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(sharded_model, options, self.rank)
        torch.testing.assert_close(state_dict, expected_cpu)

        options = StateDictOptions(full_state_dict=True, rank0_only=True)
        state_dict = get_model_state_dict(sharded_model, options, self.rank)
        torch.testing.assert_close(state_dict, expected if self.rank == 0 else {})

        tmp_path = self.tmp_path

        # Sharding - sharded state dict
        for value in (False, True):
            options = StateDictOptions(full_state_dict=False, cpu_offload=value)
            state_dict = get_model_state_dict(sharded_model, options, self.rank)

            assert isinstance(state_dict["buf"], torch.Tensor)
            assert isinstance(state_dict["l.weight"], DTensor)
            assert state_dict["buf"].device.type == ("cpu" if value else "cuda")
            assert state_dict["l.weight"].to_local().device.type == "cuda"

            save(state_dict, tmp_path)
            torch.distributed.barrier()
            assert set(os.listdir(tmp_path)) == {"__1_0.distcp", "__0_0.distcp", ".metadata"}
            checkpoint = distributed_ckpt_to_regular(tmp_path)
            if self.rank == 0:
                torch.testing.assert_close(checkpoint, expected_cpu)

        options = StateDictOptions(full_state_dict=False, rank0_only=True)
        with pytest.raises(ValueError, match="cannot be used"):
            get_model_state_dict(sharded_model, options, self.rank)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_load_model_state_dict(self):
        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        state_dict = MyModel(4).to(device=device).state_dict()

        # No sharding - full state dict
        model = MyModel(4).to(device=device)
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        load_model_state_dict(state_dict, model, options, self.rank)
        torch.testing.assert_close(model.state_dict(), state_dict)

        # cpu_offload=True is not relevant for this case

        model = MyModel(4).to(device=device)
        options = StateDictOptions(full_state_dict=True, rank0_only=True)
        load_model_state_dict(state_dict, model, options, self.rank)
        if self.rank == 0:
            torch.testing.assert_close(model.state_dict(), state_dict)
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(model.state_dict(), state_dict)

        # No sharding - sharded state dict
        options = StateDictOptions(full_state_dict=False)
        with pytest.raises(ValueError, match="cannot be used"):
            load_model_state_dict(state_dict, model, options, self.rank)

        # Sharding - full state dict
        for kwargs in ({"cpu_offload": True}, {"cpu_offload": False}, {"rank0_only": True}):
            model = MyModel(4).to(device=device)
            sharded_model = thunder.distributed.fsdp(model)
            options = StateDictOptions(full_state_dict=True, **kwargs)
            load_model_state_dict(state_dict, sharded_model, options, self.rank)
            _unshard_params(sharded_model, pg)
            torch.testing.assert_close(model.state_dict(), state_dict)

        # Create a sharded state_dict that can be loaded
        model = MyModel(4).to(device=device)
        sharded_model_expected = thunder.distributed.fsdp(model)
        options = StateDictOptions(full_state_dict=False)
        sharded_state_dict = get_model_state_dict(sharded_model_expected, options, self.rank)

        # Sharding - sharded state dict
        for kwargs in ({"cpu_offload": True}, {"cpu_offload": False}, {"rank0_only": True}):
            model = MyModel(4).to(device=device)
            sharded_model = thunder.distributed.fsdp(model)
            options = StateDictOptions(full_state_dict=False, **kwargs)
            load_model_state_dict(sharded_state_dict, sharded_model, options, self.rank)
            torch.testing.assert_close(sharded_model.state_dict(), sharded_model_expected.state_dict())


common_utils.instantiate_parametrized_tests(DistributedCheckpointTest)
