from __future__ import annotations
from contextlib import contextmanager
import itertools
from typing import TYPE_CHECKING
import collections

import torch as pytorch
from torch.utils.weak import WeakTensorKeyDictionary

import thunder
from thunder.core.compile_data import get_compile_data

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any
    from thunder.core.transform_common import Transform


def _convert_state_dict_to_per_module(state_dict: dict[str, Any], module_names: set[str]) -> dict[str, dict[str, Any]]:
    state_dict_per_module = collections.defaultdict(dict)
    for k, v in state_dict.items():
        prefix, sep, _ = k.rpartition(".")
        # not great but should not happen too often / deep
        while prefix not in module_names:
            prefix, sep, _ = prefix.rpartition(".")
        state_dict_per_module[prefix][k[len(prefix) + len(sep) :]] = v
    return state_dict_per_module


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
        self._overrides_buffers[name] = value

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
        for k, v in itertools.chain(overrides.items(), orig_iter):
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
            self._model.named_parameters(remove_duplicate=remove_duplicate),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True, *, persistent=None):
        if persistent is not None:
            orig_buffers = self._model.named_buffers(remove_duplicate=remove_duplicate, persistent=persistent)
        else:
            orig_buffers = self._model.named_buffers(remove_duplicate=remove_duplicate)

        yield from self._named_parameters_or_buffers(
            self._overrides_buffers,
            orig_buffers,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def _get_shared_names(self):
        parameters_to_names = WeakTensorKeyDictionary()
        for name, v in itertools.chain(
            self.named_parameters(remove_duplicate=False), self.named_buffers(remove_duplicate=False)
        ):
            parameters_to_names.setdefault(v, set()).add(name)
        shared_names: dict[str, set[str]] = {}
        for s in parameters_to_names.values():
            for n in s:
                shared_names[n] = s
        return shared_names

    def load_original_state_dict(self, state_dict):
        # this loads the state dict incrementally to not exhaust memory
        sd_per_module = _convert_state_dict_to_per_module(state_dict, {n for n, _ in self._model.named_modules()})

        shared_names = self._get_shared_names()
        processed_names = set()

        for submodule_name, sd_part in sd_per_module.items():
            self._transform_and_load_for_submodule(submodule_name, sd_part, shared_names, processed_names)

    def _transform_and_load_for_submodule(self, submodule_name, sd_part, shared_names, processed_names):
        prefix = submodule_name + ("." if submodule_name else "")
        for transform in self._lc_transforms:
            sd_part = transform.transform_state_dict_for_submodule(self, submodule_name, sd_part)

        for k, v in sd_part.items():
            full_k = prefix + k

            # cater for shared parameters
            processed_copies = shared_names[full_k] & processed_names
            if processed_copies:
                copy_name = next(iter(processed_copies))
                if full_k in self._overrides_parameters:
                    self._overrides_parameters[full_k] = self._overrides_parameters[copy_name]
                elif full_k in self._overrides_buffers:
                    self._overrides_buffers[full_k] = self._overrides_buffers[copy_name]
                else:
                    raise NotImplementedError(f"don't know how to handle {full_k}")
                processed_names.add(full_k)
                continue

            if full_k in self._overrides_parameters:
                p = self._overrides_parameters[full_k]
                if p.dtype == v.dtype and p.shape == v.shape:
                    with pytorch.no_grad():
                        p.copy_(v)
                else:
                    with pytorch.no_grad():
                        self._overrides_parameters[full_k] = pytorch.nn.Parameter(
                            v.to(p.device), requires_grad=p.requires_grad
                        )
            elif full_k in self._overrides_buffers:
                b = self._overrides_buffers[full_k]
                if b.dtype == v.dtype and b.shape == v.shape:
                    with pytorch.no_grad():
                        b.copy_(v)
                else:
                    with pytorch.no_grad():
                        self._overrides_parameters[full_k] = v.to(b.device).requires_grad_(b.requires_grad)
            else:
                raise NotImplementedError(f"don't know how to handle {full_k}")
            processed_names.add(full_k)

    def state_dict(self, *, destination=None, prefix="", keep_vars=False):
        """
        Returns the state dict of the (transformed) Thunder module.

        Args:
            destination: if given, use this mutable mapping as the dict container.
            prefix: a prefix for the keys.
            keep_vars: do not detach

        Note that this is similar but rather more rudimentary than the original state_dict (e.g. no hook suport yet).
        """
        if destination is None:
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        for name, param in self.named_parameters(prefix=prefix):
            if param is not None:
                destination[name] = param if keep_vars else param.detach()

        non_persistent_buffers_set = set()
        for name, submodule in self._model.named_modules(prefix=prefix):
            subprefix = f"{name}." if name else name
            non_persistent_buffers_set.update(f"{subprefix}{bname}" for bname in submodule._non_persistent_buffers_set)

        for name, buf in self.named_buffers(prefix=prefix):
            if buf is not None and name not in non_persistent_buffers_set:
                destination[name] = buf if keep_vars else buf.detach()

        for name, submodule in self._model.named_modules(prefix=prefix):
            subprefix = f"{name}." if name else name
            extra_state_key = subprefix + pytorch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX
            if (
                getattr(self.__class__, "get_extra_state", pytorch.nn.Module.get_extra_state)
                is not pytorch.nn.Module.get_extra_state
            ):
                destination[extra_state_key] = self.get_extra_state()
        return destination

    def original_state_dict(
        self,
        *,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Returns the state dict of the transformed :class:`ThunderModule` with reverse transform applied.

        For example, :func:`ThunderModule.state_dict` returns a state dict of sharded tensors if
        a model is :func:`thunder.distributed.fsdp` applied while :func:`ThunderModule.original_state_dict`
        returns a state dict of unsharded tensors.

        Args:
            destination: if given, use this mutable mapping as the dict container.
            prefix: a prefix for the keys.
            keep_vars: do not detach

        """
        module_names = {name for name, _ in self._model.named_modules()}
        state_dict_per_submodule = _convert_state_dict_to_per_module(self.state_dict(), module_names)

        if destination is None:
            destination = collections.OrderedDict()
            destination._metadata = collections.OrderedDict()

        transform: Transform
        for submodule_name, submodule_state_dict in state_dict_per_submodule.items():
            for transform in reversed(self._lc_transforms):
                submodule_state_dict = transform.reverse_transform_state_dict_for_submodule(
                    self,
                    submodule_name,
                    submodule_state_dict,
                )
            destination.update({f"{prefix}{submodule_name}.{k}": v for k, v in submodule_state_dict.items()})
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Loads the state dict to a transformed module.

        Args:
            state_dict: the state dict to load.
            strict: error on missing / unused state dict members
            assign: assign the state dict tensors instead of copying the data

        This is similar much more simple than the original load_state_dict.
        (Regarding hooks, customization etc.)
        """
        # non-persistent buffer overrides?
        non_persistent_buffers_set = set()
        for name, submodule in self._model.named_modules():
            subprefix = f"{name}." if name else name
            non_persistent_buffers_set.update(f"{subprefix}{bname}" for bname in submodule._non_persistent_buffers_set)

        keys_unused = thunder.core.utils.OrderedSet(state_dict.keys())
        keys_missing = thunder.core.utils.OrderedSet()
        errors = []
        for name, v in itertools.chain(self.named_parameters(), self.named_buffers()):
            if name in non_persistent_buffers_set:
                continue
            if name not in state_dict:
                keys_missing.add(name)
                continue
            keys_unused.remove(name)
            sd_v = state_dict[name]
            if not pytorch.overrides.is_tensor_like(sd_v):
                errors.append(
                    f'While copying the parameter named "{name}", '
                    "expected torch.Tensor or Tensor-like object from checkpoint but "
                    f"received {type(sd_v)}"
                )
                continue
            if sd_v.shape != v.shape:
                errors.append(
                    f"size mismatch for {name}: copying a param with shape {sd_v.shape} from checkpoint, "
                    f"the shape in current model is {v.shape}."
                )
                continue

            # We don't check dtype because PyTorch also does dtype conversion on load.
            with pytorch.no_grad():
                if assign:
                    if isinstance(v, pytorch.nn.Parameter):
                        if not isinstance(sd_v, pytorch.nn.Parameter):
                            sd_v = pytorch.nn.Parameter(sd_v, requires_grad=v.requires_grad)
                        else:
                            sd_v.requires_grad_(v.requires_grad)
                        self._overrides_parameters[name] = sd_v
                    else:
                        self._overrides_buffers[name] = sd_v

                else:
                    v.copy_(sd_v)
        if strict:
            if keys_unused:
                errors.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in keys_unused)),
                )
            if keys_missing:
                errors.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(", ".join(f'"{k}"' for k in keys_missing)),
                )

        if errors:
            raise RuntimeError("Error(s) in loading state_dict:\n\t{}".format("\n\t".join(errors)))

        return pytorch.nn.modules.module._IncompatibleKeys(keys_missing, keys_unused)

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
