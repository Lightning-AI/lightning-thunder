from distutils.version import LooseVersion
from functools import partial
import warnings
from collections.abc import Callable

import torch

import thunder
from thunder.recipes import BaseRecipe
from thunder import Recipe


# for materializing models, we need reset_parameters, which is part of the unwritten
# spec for idiomatic PyTorch, but not implemented everywhere
def RotaryEmbedding_reset_parameters(self):
    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device)
    with torch.no_grad():
        self.inv_freq.copy_(inv_freq)


def RMSNorm_reset_parameters(self):
    with torch.no_grad():
        self.weight.fill_(1)


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
        import transformers

        cache_classes = (transformers.cache_utils.StaticCache,)
        if hasattr(transformers.cache_utils, "StaticLayer"):
            cache_classes += (transformers.cache_utils.StaticLayer,)
        import thunder.core.utils as utils

        # Identify the indices of cache-related proxies in the prologue trace.
        # In the prologue, the cache related proxies are represented for example as:
        # any0: "<class 'transformers.cache_utils.StaticCache'>" = kwargs['past_key_values']
        # obj1: "Any" = any0.layers
        # subscr2: "Any" = obj1[0]
        # key_tensors: "cuda:0 bf16[1, 8, 108, 64]" = subscr2.keys
        # We start from the AnyProxy whose underlying objects are of type `StaticCache` or `StaticLayer` and recursively collect the outputs of the consumers(getattr or getitem) as the target proxies
        bsyms = pro.bound_symbols
        consumers = utils.consumers(bsyms, _map_to_numbers=False)
        target_proxies = []

        def recursively_collect_consumers_outputs(proxy, consumers_dict, visited, collected):
            if proxy.name in visited:
                return
            visited.add(proxy.name)
            static_consumers = consumers_dict.get(proxy, None)
            if not static_consumers:
                return
            for consumer in static_consumers:
                if consumer.sym.name in ("unpack_attr", "unpack_getitem"):
                    collected.extend(consumer.flat_proxy_outs)
                    for out_proxy in consumer.flat_proxy_outs:
                        recursively_collect_consumers_outputs(out_proxy, consumers_dict, visited, collected)

        for bsym in bsyms:
            for out in bsym.flat_proxy_outs:
                if isinstance(out, thunder.core.proxies.AnyProxy) and isinstance(out._o, cache_classes):
                    visited = set()
                    recursively_collect_consumers_outputs(out, consumers, visited, target_proxies)
        return_bsym = bsyms[-1]
        assert return_bsym.sym.name == "python_return", f"Expected return symbol, got {return_bsym.sym.name}"
        # the values passed to the compute trace
        passed_to_compute_trace = return_bsym.args[0][0]
        target_proxies_names = [proxy.name for proxy in target_proxies]
        static_proxy_idx = []
        for idx, passed_value in enumerate(passed_to_compute_trace):
            if passed_value.name in target_proxies_names:
                static_proxy_idx.append(idx)

        for idx, arg in enumerate(comp.args):
            if idx in static_proxy_idx:
                arg.tags.add(thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION)

        return pro, comp, epi


class SDPAMaskTransform(thunder.Transform):
    def __init__(self):
        self._MASK_FUNCTIONS = {}
        self.executor = thunder.extend.OperatorExecutor("sdpa_mask_transform_ex")
        thunder.extend.register_executor(self.executor)  # needed if you want to pickle traces
        import transformers

        if LooseVersion(transformers.__version__) < LooseVersion("4.55"):
            return
        import transformers.masking_utils

        def transformers_masking_utils_sdpa_mask_recent_torch_meta(
            batch_size: int,
            cache_position: thunder.TensorProxy,
            kv_length: int,
            kv_offset: int = 0,
            attention_mask: thunder.TensorProxy | None = None,
            local_size: int | None = None,
            allow_is_causal_skip: bool = True,
            mask_function: str | None = None,
        ):
            # batch, num_attention_heads, seq_len, kvcache_len
            return thunder.TensorProxy(
                shape=(batch_size, 1, cache_position.shape[0], kv_length),
                dtype=thunder.dtypes.bool8,
                device=cache_position.device,
            )

        def transformers_masking_utils_sdpa_mask_recent_torch_impl(
            batch_size: int,
            cache_position: torch.Tensor,
            kv_length: int,
            kv_offset: int = 0,
            attention_mask: torch.Tensor | None = None,
            local_size: int | None = None,
            allow_is_causal_skip: bool = True,
            mask_function: str | None = None,
        ):
            import transformers

            return transformers.masking_utils.sdpa_mask_recent_torch(
                batch_size,
                cache_position,
                kv_length,
                kv_offset=kv_offset,
                attention_mask=attention_mask,
                local_size=local_size,
                allow_is_causal_skip=allow_is_causal_skip,
                mask_function=self._MASK_FUNCTIONS[mask_function],
            )

        self.transformers_masking_utils_sdpa_mask_recent_torch_sym = self.executor.register_operator(
            "transformers_masking_utils_sdpa_mask_recent_torch",
            meta=transformers_masking_utils_sdpa_mask_recent_torch_meta,
            fn=transformers_masking_utils_sdpa_mask_recent_torch_impl,
        )

        def transformers_masking_utils_sdpa_mask_recent_torch_lookaside(
            batch_size: int,
            cache_position: torch.Tensor | thunder.TensorProxy,
            kv_length: int,
            kv_offset: int = 0,
            mask_function: Callable = transformers.masking_utils.causal_mask_function,
            attention_mask: torch.Tensor | thunder.TensorProxy | None = None,
            local_size: int | None = None,
            allow_is_causal_skip: bool = True,
            **kwargs,
        ):
            assert set(kwargs.keys()) == {"config", "dtype"}  # ignore
            mask_name = mask_function.__name__
            self._MASK_FUNCTIONS[mask_name] = mask_function
            return self.transformers_masking_utils_sdpa_mask_recent_torch_sym(
                batch_size,
                cache_position,
                kv_length,
                kv_offset=kv_offset,
                attention_mask=attention_mask,
                local_size=local_size,
                allow_is_causal_skip=allow_is_causal_skip,
                mask_function=mask_name,
            )

        self.executor._lookasides[transformers.masking_utils.sdpa_mask_recent_torch] = (
            transformers_masking_utils_sdpa_mask_recent_torch_lookaside
        )

    def get_executor(self):
        return self.executor


@Recipe.register("hf-transformers")
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
        self.sdpa_mask_transform = SDPAMaskTransform()
        self.executor_names.append(self.inplace_index_copy_transform.executor.name)
        self.executor_names.append(self.sdpa_mask_transform.executor.name)

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
        max_version = LooseVersion("4.55.4")

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

        # We need reset_parameters for initialization of buffers in materialization.
        # This seems to work for transformers 4.5x with Llama, Llama4 and Qwen2 at least
        for submodule in model.modules():
            cls = submodule.__class__
            if cls.__name__.endswith("RotaryEmbedding") and not hasattr(cls, "reset_parameters"):
                cls.reset_parameters = RotaryEmbedding_reset_parameters
            elif cls.__name__.endswith("RMSNorm") and not hasattr(cls, "reset_parameters"):
                cls.reset_parameters = RMSNorm_reset_parameters

        thunder_model = super().apply(model)

        if getattr(thunder_model, "generate", None):
            thunder_model.generate = partial(thunder_model.generate.__func__, thunder_model)

        if getattr(thunder_model, "_sample", None):
            thunder_model._sample = partial(thunder_model._sample.__func__, thunder_model)

        if getattr(thunder_model, "_valid_auto_compile_criteria", None):
            thunder_model._valid_auto_compile_criteria = lambda *args, **kwargs: False

        return thunder_model
