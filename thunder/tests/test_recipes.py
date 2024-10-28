import thunder
import transformers
import torch

from torch.testing import assert_close, make_tensor


class HFBertBasicFX(thunder.Recipe):
    compiler = "torch.compile"


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


def test_recipe_basic_bert_fx():
    bert = transformers.BertForSequenceClassification(transformers.BertConfig())
    del bert.bert.encoder.layer[1:]
    bert.eval()

    inp = torch.randint(1, 20, (1, 32))

    thunder_bert = thunder.compile(bert, recipe=HFBertBasicFX())

    actual = thunder_bert(inp)
    expected = bert(inp)

    assert_close(actual, expected)
