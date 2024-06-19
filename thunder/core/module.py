from contextlib import contextmanager
import itertools
from typing import Any

import torch as pytorch

from thunder.core.compile_data import get_compile_data


class ThunderModule(pytorch.nn.Module):
    """A wrapper nn.Module subclass.

    This wrapper is returned by ``thunder.jit``, you would typically not
    instantiate it manually.

    """

    def __init__(self, model, compiled_model_call):
        """"""
        super().__init__()
        self._model = model

        # We delete self.training in order for training to be used from
        # the model itself through `__getattr__`.
        del self.training

        self._forward_fn = compiled_model_call

        # overrides for parameters and buffers (see get_buffer/get_parameter)
        # we populate these here for performance reasons (sam as module cache),
        # a single dict lookup is cheaper than traversin the module
        # hierarchy, see https://github.com/Lightning-AI/lightning-thunder/issues/396#issuecomment-2113231498
        self._overrides_parameters = dict(self._model.named_parameters())
        self._overrides_buffers = dict(self._model.named_buffers())
        self._module_cache = {k: v for k, v in self._model.named_modules()}
        self._null = object()

    def get_buffer(self, name):
        p = self._overrides_buffers.get(name, self._null)
        if p is not self._null:
            return p
        return self._model.get_buffer(name)

    def set_buffer(self, name, value):
        p = self._overrides_buffers[name] = value

    def get_parameter(self, name):
        p = self._overrides_parameters.get(name, self._null)
        if p is not self._null:
            return p
        return self._model.get_parameter(name)

    def get_submodule(self, name):
        p = self._module_cache.get(name, self._null)
        if p is not self._null:
            return p
        return self._model.get_submodule(name)

    def forward(self, *args, **kwargs):
        res = self._forward_fn(*args, **kwargs)
        return res

    def _named_parameters_or_buffers(self, overrides, orig_iter, prefix="", recurse=True, remove_duplicate=True):
        seen_ids = set()
        seen_names = set()
        for k, v in itertools.chain(overrides.items(), orig_iter(remove_duplicate=remove_duplicate)):
            if remove_duplicate:
                id_v = id(v)
                if id_v in seen_ids:
                    continue
                seen_ids.add(id_v)

            mod, _, base_param = k.rpartition(".")
            if recurse or not mod:
                if k not in seen_names:
                    seen_names.add(k)
                    if prefix:
                        yield (f"{prefix}.{k}", v)
                    else:
                        yield (k, v)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_parameters_or_buffers(
            self._overrides_parameters,
            self._model.named_parameters,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_parameters_or_buffers(
            self._overrides_buffers,
            self._model.named_buffers,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def load_original_state_dict(state_dict):
        # this loads the state dict incrementally to not exhaust memory
        module_names = {n for n, _ in self.named_modules()}
        sd_per_module = {}
        for k, v in state_dict.items():
            prefix, sep, _ = k.rpartition(".")
            # not great but should not happen too often / deep
            while prefix not in module_names:
                prefix, sep, _ = prefix.rpartition(".")
            sd_per_module[prefix][k[len(prefix) + len(sep) :]] = v

        for submodule_name, sd_part in sd_per_module.items():
            prefix = submodule_name + ("." if submodule_name else "")
            for transform in fn_._lc_early_transforms:
                sd_part = transform.transform_state_dict_for_submodule(self, submodule_name.sd_part)
            for k, v in sd_part:
                full_k = prefix + k
                if k in model._overrides_parameters:
                    model._overrides_parameters[full_k] = v
                elif k in model._overrides_buffers:
                    model._overrides_buffers[full_k] = v
                else:
                    raise NotImplementedError(f"don't know how to handle {full_k}")

    @contextmanager
    def no_sync(self):
        r"""Context manager to disable gradient synchronization in data parallel mode.

        This context manager is intended to be used in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel` to disable gradient
        synchronization in the backward pass. It will not have any effect when
        used with other modules.

        .. note::

            This could lead to different accumulated gradients with ``torch.nn.parallel.distributed.DistributedDataParallel.no_sync``.
            PyTorch's gradient synchronization is implemented by applying all-reduce to gradient buckets of ``torch.nn.Parameter.grad``.
            Thus the ``no_sync`` context leads to :math:`\text{AllReduce} \left( \sum_{i = 0}^{\text{ga_steps}} g_i \right)` where :math:`\text{ga_steps}` means the number of gradient accumulation steps.
            In contrast, this synchronizes accumulated gradients when exiting, leading to
            :math:`\text{AllReduce} \left( \sum_{i = 0}^{\text{ga_steps - 1}} g_i \right) + \text{AllReduce}(g_{\text{ga_steps}})`.

        .. warning::

            You must reuse this context manager in each group of gradient accumulation iterations since gradients will get synchronized
            on context manager exit.

            .. code-block:: python

                with model.no_sync():
                    for _ in range(len(gradient_accumulation_iters)):
                        loss(model(x)).backward()  # uses no-sync-backward trace
                loss(model(x)).backward()  # uses the regular backward trace
                optimizer.step()

        """
        from thunder.distributed import (
            set_skip_data_parallel_grad_sync,
            reset_skip_data_parallel_grad_sync,
            _sync_grads,
        )

        token = set_skip_data_parallel_grad_sync(True)
        try:
            yield
        finally:
            reset_skip_data_parallel_grad_sync(token)
            _sync_grads(self)

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            return self._modules["_model"]
        return getattr(self._model, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.load_state_dict(*args, **kwargs)


def get_thunder_module(model):
    cd = get_compile_data()

    # to not hold a reference to model in the _thunder_module_map dict, we index by id.
    # But this means that we need to check if it is actually the right model, which we do with the if below.
    tm = cd._thunder_module_map.get(id(model))
    if tm and tm._model is not model:
        tm = None

    if tm is None:
        # TODO: we would like to raise an error here, but this would require
        #       us wrapping models that are passed in closures etc.
        # raise RuntimeError("could not find ThunderModule")
        return model
    return tm
