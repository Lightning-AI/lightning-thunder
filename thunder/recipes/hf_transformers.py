from distutils.version import LooseVersion
from functools import partial
import warnings

import thunder
from thunder.recipes import BaseRecipe
from thunder import Recipe


class InplaceIndexCopyTransform(thunder.Transform):
    def __init__(self):
        super().__init__()
        self.executor = thunder.extend.OperatorExecutor("inplace_index_copy_ex")
        thunder.extend.register_executor(self.executor)  # needed if you want to pickle traces

        def inplace_index_copy_meta(buffer, dim, idx, val):
            return thunder.TensorProxy(like=buffer)

        def inplace_index_copy_impl(buffer, dim, idx, val):
            return buffer.index_copy_(dim, idx, val)

        self.inplace_index_copy = self.executor.register_operator(
            "inplace_index_copy", fn=inplace_index_copy_impl, meta=inplace_index_copy_meta
        )

    def get_executor(self):
        return self.executor

    def transform_traces_pre_prologue(self, pro, comp, epi, **kwargs):
        comp_new = thunder.core.trace.from_trace(comp)
        for bsym in comp.bound_symbols:
            if bsym.sym == thunder.torch.index_copy:
                bsym.args[0].tags.add(thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION)
                bsym = bsym.from_bsym(sym=self.inplace_index_copy)
            else:
                bsym = bsym.from_bsym()
            comp_new.bound_symbols.append(bsym)
        return pro, comp_new, epi


@Recipe.register("transformers")
class HFTransformers(BaseRecipe):
    """
    Recipe tuned for Hugging Face ``transformers`` models.

    Args:
        show_progress (bool, optional): Forwarded to :class:`BaseRecipe`.
        interpreter (str, optional): Thunder interpreter to use.
        plugins (Iterable | None, optional): Extra Thunder plugins.
    """

    def __init__(
        self,
        show_progress=False,
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(show_progress=show_progress, interpreter=interpreter, plugins=plugins)

        # for kv-cache inplace ops
        self.inplace_index_copy_transform = InplaceIndexCopyTransform()
        self.executor_names.append(self.inplace_index_copy_transform.executor.name)

    @classmethod
    def validate(cls, model):
        """
        Emit warnings (or errors) if *model* falls outside the supported
        transformer versions or base classes.

        Args:
            model (transformers.PreTrainedModel): Model instance to vet.

        Raises:
            ValueError: If *model* is not a ``PreTrainedModel``.
        """
        import transformers

        version = LooseVersion(transformers.__version__)
        min_version = LooseVersion("4.46.0")
        max_version = LooseVersion("5.0.0")

        if version < min_version or version > max_version:
            warnings.warn(
                f"`transformers` version {version} detected. The HFTransformers recipe supports {min_version} to {max_version}."
            )

        supported = [
            transformers.BertPreTrainedModel,
            transformers.LlamaPreTrainedModel,
            transformers.Phi3PreTrainedModel,
            transformers.Qwen2PreTrainedModel,
        ]
        supported_str = "\n".join(f"* {el.__name__}" for el in supported)

        if not any(cls for cls in supported if isinstance(model, cls)):
            warnings.warn(
                f"instance of {type(model).__name__} found. The HFTransformers recipe supports:\n{supported_str}"
            )

        if not isinstance(model, transformers.PreTrainedModel):
            raise ValueError(f"The model must be an instance of PreTrainedModel, found {type(model)}")

    def setup_config(self):
        """
        Enable NV-kernelised linear, matmul and SDPA ops on top of the
        base recipe’s debug configuration.

        Returns:
            dict[str, Any]: Thunder config dictionary augmented with
            ``nv_enable_*`` flags.
        """
        config = super().setup_config()
        config.update(nv_enable_linear=True, nv_enable_matmul=True, nv_enable_sdpa=True)
        return config

    def setup_lookasides(self):
        """
        Swap out the warning helper when running under
        the non Thunder-FX interpreter.

        Returns:
            list[thunder.core.recipe.Lookaside] | None
        """
        if self.interpreter == thunder.core.recipe.Interpreter.THUNDER_FX:
            return None

        import transformers

        warn_lookaside = thunder.core.recipe.Lookaside(
            fn=transformers.modeling_utils.PreTrainedModel.warn_if_padding_and_no_attention_mask,
            replace_with=lambda *args: None,
        )

        # HF transformers (4.52.4) wraps something in autocast(device, enabled=False)
        # but PyTorch (2.7) does not like this call when device is meta.
        # So to trace on the meta device and because don't care about much about the autocast,
        # we replace it with the nullcontext.
        # We might allow more cases (call autocast if iot is nt with meta or not enabled=False?)
        def autocast_lookaside(*args, enabled=True, **kwargs):
            from contextlib import nullcontext

            if enabled:
                raise RuntimeError("don't do autocast")
            return nullcontext()

        import torch

        autocast_lookaside = thunder.core.recipe.Lookaside(fn=torch.autocast, replace_with=autocast_lookaside)
        return [warn_lookaside, autocast_lookaside]

    def setup_transforms(self):
        """
        Prepend the ``InplaceIndexCopyTransform`` to the default
        transform list.

        Returns:
            list[thunder.Transform]: transform list.
        """
        transforms = super().setup_transforms()
        return [self.inplace_index_copy_transform] + transforms

    def apply(self, model):
        """
        Apply the recipe (compile the model) and patch ``generate`` / ``_sample``
        so they work after tracing.

        Args:
            model (transformers.PreTrainedModel): The model to compile.

        Returns:
            transformers.PreTrainedModel: Thunder-compiled model ready
            for inference.
        """
        thunder_model = super().apply(model)

        if getattr(thunder_model, "generate", None):
            thunder_model.generate = partial(thunder_model.generate.__func__, thunder_model)

        if getattr(thunder_model, "_sample", None):
            thunder_model._sample = partial(thunder_model._sample.__func__, thunder_model)

        return thunder_model
