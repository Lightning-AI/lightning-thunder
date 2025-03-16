import thunder
import transformers
import torch
import pytest

from torch.testing import assert_close, make_tensor
from thunder.tests.framework import version_between, IS_WINDOWS


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


def test_recipe_mlp():
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 64)
    )

    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    print(thunder_model)

    print(thunder.last_traces(thunder_model)[-1])
