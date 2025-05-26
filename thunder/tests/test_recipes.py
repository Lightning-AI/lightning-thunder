from unittest.mock import patch

import pytest
import thunder
import transformers
import torch

from thunder.extend import deregister_executor
from torch.testing import assert_close, make_tensor
from thunder.recipes import HFTransformers
from thunder.executors import nvfuser_available
from thunder.tests.framework import version_between, IS_WINDOWS


@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
@pytest.mark.skipif(IS_WINDOWS, reason="slow on Windows")
def test_default_recipe_basic_bert():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    thunder_bert = thunder.compile(bert)

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)


@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
@pytest.mark.skipif(IS_WINDOWS, reason="slow on Windows")
def test_recipe_basic_bert():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    expected = bert(inp)

    thunder_bert = thunder.compile(bert, recipe="hf-transformers")

    actual = thunder_bert(inp)

    assert_close(actual, expected)

    from thunder.recipes import HFTransformers

    thunder_bert = thunder.compile(bert, recipe=HFTransformers())

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)

    # cleanup after test
    deregister_executor("inplace_index_copy_ex")


@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
def test_recipe_basic_bert_fx():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    from thunder.recipes import HFTransformers

    thunder_bert = thunder.compile(bert, recipe=HFTransformers(interpreter="thunder.fx"))

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)

    # cleanup after test
    deregister_executor("inplace_index_copy_ex")


@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
def test_recipe_mlp():
    model = torch.nn.Sequential(torch.nn.Linear(2048, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 64))

    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    print(thunder_model)

    print(thunder.last_traces(thunder_model)[-1])


def test_recipe_errors():
    class BrokenRecipe(HFTransformers):
        def __init__(self):
            super().__init__()
            self.executor_names = ["cudnn", "nonexistent_executor"]

    recipe = BrokenRecipe()

    with pytest.raises(
        ValueError,
        match="Executor 'nonexistent_executor' was specified in the recipe but is not available in the current environment.",
    ):
        recipe.setup_executors()

    # cleanup after test
    deregister_executor("inplace_index_copy_ex")


@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
def test_plugins_basics():
    model = torch.nn.Sequential(torch.nn.Linear(2048, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 64))

    from thunder import compile_data as get_compile_data
    from thunder.recipes.base import BaseRecipe

    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    _ = thunder_model(x)
    cd = get_compile_data(thunder_model)
    assert cd is not None
    for ex in thunder.get_default_executors():
        assert ex.name in [el.name for el in cd.executors_list]


# test skipped if nvfuser isn't available because providing plugins calls BaseRecipe
@pytest.mark.skipif(not nvfuser_available(), reason="nvFuser is not available")
@pytest.mark.skipif(IS_WINDOWS, reason="libuv error with PT build on windows")
def test_plugins_composition(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2048, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 64))

    monkeypatch.setenv("LOCAL_RANK", "0")

    with patch("thunder.jit") as mock_jit:
        _ = thunder.compile(model, plugins="fp8")
        call_args = mock_jit.call_args
        assert "transformer_engine" in [el.name for el in call_args.kwargs["executors"]]
        for ex in thunder.get_default_executors():
            assert ex.name in [el.name for el in call_args.kwargs["executors"]]

        _ = thunder.compile(model, plugins=["fp8"])
        call_args = mock_jit.call_args
        assert "transformer_engine" in [el.name for el in call_args.kwargs["executors"]]
        for ex in thunder.get_default_executors():
            assert ex.name in [el.name for el in call_args.kwargs["executors"]]

        from thunder.plugins import FP8

        _ = thunder.compile(model, plugins=[FP8()])
        call_args = mock_jit.call_args
        assert "transformer_engine" in [el.name for el in call_args.kwargs["executors"]]
        for ex in thunder.get_default_executors():
            assert ex.name in [el.name for el in call_args.kwargs["executors"]]

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:12345")

    with patch("thunder.jit") as mock_jit:
        _ = thunder.compile(model, plugins=["fsdp"])
        call_args = mock_jit.call_args
        expected_transforms = [
            thunder.distributed.transforms.fsdp_v2.FSDPTransform,
            thunder.transforms.materialization.MaterializationTransform,
        ]
        transforms = call_args.kwargs["transforms"]
        for expected in expected_transforms:
            assert any(isinstance(el, expected) for el in transforms)

    with patch("thunder.jit") as mock_jit:
        _ = thunder.compile(model, plugins=["fsdp", "fp8"])
        call_args = mock_jit.call_args
        expected_transforms = [
            thunder.distributed.transforms.fsdp_v2.FSDPTransform,
            thunder.transforms.materialization.MaterializationTransform,
        ]
        transforms = call_args.kwargs["transforms"]
        for expected in expected_transforms:
            assert any(isinstance(el, expected) for el in transforms)
        assert "transformer_engine" in [el.name for el in call_args.kwargs["executors"]]


@pytest.mark.skipif(IS_WINDOWS, reason="libuv error with PT build on windows")
def test_plugins_hybrid_ddpfsdp(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2048, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 64))

    monkeypatch.setenv("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:1234")
    from thunder.plugins import FSDP
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))  # single-dim mesh

    plugin = FSDP(process_group=mesh)

    with patch("thunder.jit") as mock_jit:
        _ = thunder.compile(model, plugins=[plugin])
        call_args = mock_jit.call_args
        transforms = call_args.kwargs["transforms"]
        expected = thunder.distributed.transforms.fsdp_v2.FSDPTransform
        assert any(isinstance(el, expected) for el in transforms)

    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("ddp", "fsdp"))
    plugin = FSDP(process_group=mesh)

    with patch("thunder.jit") as mock_jit:
        _ = thunder.compile(model, plugins=[plugin])
        call_args = mock_jit.call_args
        transforms = call_args.kwargs["transforms"]
        expected = [
            thunder.distributed.transforms.fsdp_v2.FSDPTransform,
            thunder.distributed.transforms.ddp_v2.DDPTransform,
        ]
        for e in expected:
            assert any(isinstance(el, e) for el in transforms)
