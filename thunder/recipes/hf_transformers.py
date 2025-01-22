from distutils.version import LooseVersion
from functools import wraps
import warnings

import transformers

import thunder
from thunder.transforms.cudagraph import CUDAGraphTransform
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks
from thunder.extend import get_default_executors
from thunder import executors


def pretty_warnings(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning)
            result = func(*args, **kwargs)
            for warning in caught_warnings:
                if issubclass(warning.category, UserWarning):
                    print(f"{warning.category.__name__}: {warning.message}")
            return result
    return wrapped


class HFTransformers(thunder.Recipe):
    def __init__(self, reduce_overhead=True, fuser="nvfuser"):
        super().__init__()
        self.reduce_overhead = reduce_overhead
        self.fuser = fuser

    @pretty_warnings
    def validate(self, model):
        version = LooseVersion(transformers.__version__)
        min_version = LooseVersion("4.46.0")

        if version < min_version:
            warnings.warn(f"`transformers` version {version} detected. The HFTransformers recipe supports >= {min_version}.")

        supported = [
            transformers.BertPreTrainedModel,
            transformers.LlamaPreTrainedModel,
            transformers.Phi3PreTrainedModel,
            transformers.Qwen2PreTrainedModel,
        ]
        supported_str = "\n".join(f"* {el.__name__}" for el in supported)

        if not any(cls for cls in supported if isinstance(model, cls)):
            warnings.warn(f"instance of {type(model).__name__} found. The HFTransformers recipe supports:\n{supported_str}")

        if not isinstance(model, transformers.PreTrainedModel):
            raise ValueError(f"The model must be an instance of PreTrainedModel, found {type(model)}")

    def setup_lookasides(self):
        warn_lookaside = thunder.Lookaside(
            fn=transformers.modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask,
            replace_with=lambda *args: None,
        )

        return [warn_lookaside]

    def setup_transforms(self):
        transforms = [PrunePrologueChecks()]

        if self.reduce_overhead:
            return transforms + [CUDAGraphTransform()]

        return transforms

    def setup_executors(self):
        ex = get_default_executors()

        if self.fuser == "nvfuser":
            return ex
        elif self.fuser == "torch.compile":
            ex = [el for el in ex if el.name not in ["torchcompile_cat", "nvfuser"]]
            ex.append(executors.torch_compile.torch_compile_ex)
            return ex

        raise ValueError(f"Invalid fuser {self.fuser}. Allowed fusers: 'nvfuser', 'torch.compile'.")
