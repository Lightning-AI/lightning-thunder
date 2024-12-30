import thunder
import transformers
import torch
import pytest

from torch.testing import assert_close, make_tensor
from thunder.tests.framework import version_between


def test_recipe_basic_bert():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    from thunder.recipes.hf_bert import HFBertBasic

    thunder_bert = thunder.compile(bert, recipe=HFBertBasic())

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)


@pytest.mark.skipif(
    version_between(torch.__version__, min_ver="2.6.0dev0", max_ver="2.6.0a99"),
    reason="https://github.com/Lightning-AI/lightning-thunder/issues/1471",
)
def test_recipe_basic_bert_dynamo():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    from thunder.core.recipe import DynamoRecipe

    thunder_bert = thunder.compile(bert, recipe=DynamoRecipe())

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)
