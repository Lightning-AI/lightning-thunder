from distutils.version import LooseVersion
from functools import partial, wraps
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
    def __init__(self, reduce_overhead=True, fuser="nvfuser", show_progress=False):
        super().__init__()
        self.reduce_overhead = reduce_overhead
        self.fuser = fuser
        self.show_progress = show_progress

    @pretty_warnings
    def validate(self, model):
        version = LooseVersion(transformers.__version__)
        min_version = LooseVersion("4.46.0")
        max_version = LooseVersion("4.46.3")

        if version < min_version or version > max_version:
            warnings.warn(f"`transformers` version {version} detected. The HFTransformers recipe supports {min_version} to {max_version}.")

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

    def setup_config(self):
        if not self.show_progress:
            return {}
        return dict(debug_options=thunder.DebugOptions(show_interpreter_progress=True))

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

    def apply(self, model):
        fix_generate = not isinstance(model, transformers.BertPreTrainedModel)
        thunder_model = super().apply(model)

        if fix_generate:
            thunder_model.generate = partial(thunder_model.generate.__func__, thunder_model)
            thunder_model._sample = partial(thunder_model._sample.__func__, thunder_model)

        return thunder_model
