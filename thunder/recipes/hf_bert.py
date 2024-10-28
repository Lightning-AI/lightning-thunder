import thunder
import transformers
import torch


class HFBertBasic(thunder.Recipe):
    def validate(self, model):
        if not isinstance(model, transformers.BertForSequenceClassification):
            raise ValueError("The model must be a BertForSequenceClassification")

    def setup_lookasides(self):
        warn_lookaside = thunder.Lookaside(
            fn=transformers.modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask,
            replace_with=lambda *args: None,
        )

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            is_compiling = torch.compiler.is_compiling
        else:
            is_compiling = torch._dynamo.is_compiling

        is_compiling_lookaside = thunder.Lookaside(fn=is_compiling, replace_with=lambda *_: True)

        return [warn_lookaside, is_compiling_lookaside]
