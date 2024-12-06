from __future__ import annotations
from collections.abc import Callable
import copy
from itertools import chain
from typing import TYPE_CHECKING

import torch

from thunder.core.transform_common import Transform

if TYPE_CHECKING:
    from thunder.core.module import ThunderModule


def module_init_from_original_module_init(
    transform: MaterializationTransform, tm: ThunderModule, *, non_persistent_only: bool = False
):
    if non_persistent_only:
        non_persistent_buffers_set = set()
        for name, submodule in tm._model.named_modules():
            prefix = f"{name}." if name else name
            non_persistent_buffers_set.update(f"{prefix}{bname}" for bname in submodule._non_persistent_buffers_set)

        def should_skip_materialization(n):
            return n not in non_persistent_buffers_set

    else:

        def should_skip_materialization(n):
            return False

    shared_names = tm._get_shared_names()
    processed_names = set()

    # Shared parameters in PyTorch eager are parameters of module which have different name but share the underlying tensor.
    # For shared parameter, we replace all occurence shared parameter with its corresponding `base` parameter.
    # In our implementation `base` parameter is the parameter and corresponding name which we see the first time while
    # iterating our parameters (see below). We track subsequent parameter which share the underlying Tensor with this `base` parameter
    # in `shared_params_name` dictionary.

    for module_name, _ in tm._model.named_modules():
        prefix = module_name if not module_name else f"{module_name}."
        submodule = tm.get_submodule(module_name)

        # Materialize meta-parameters on-device if necessary.
        # This is done before sharding in case the materialization logic depends on the tensor shape.
        # The tradeoff is that all of a module's direct parameters need to fit in device.
        # Each module only initializes its own parameters and not those of its children (recurse=False)
        if any(
            chain(
                (
                    (n in transform.have_materialized) and not should_skip_materialization(n)
                    for n, _ in submodule.named_parameters(recurse=False, prefix=module_name)
                ),
                (
                    (n in transform.have_materialized) and not should_skip_materialization(n)
                    for n, _ in submodule.named_buffers(recurse=False, prefix=module_name)
                ),
            )
        ):
            # we use a copy to let the user's module alone
            module_copy = copy.copy(submodule)
            module_copy._parameters = module_copy._parameters.copy()
            module_copy._buffers = module_copy._buffers.copy()
            module_copy._modules = module_copy._modules.__class__()

            # we need to initialize the module unless all parameters are duplicatess
            need_init = not all(
                (shared_names[n] & processed_names) or should_skip_materialization(n)
                for n, _ in chain(
                    module_copy.named_parameters(prefix=module_name, recurse=False),
                    module_copy.named_buffers(prefix=module_name, recurse=False),
                )
            )
            if need_init:
                # TODO: we could also support calling a "param_init_fn" argument like PyTorch
                module_copy.to_empty(device=transform.device, recurse=False)
                if not hasattr(module_copy, "reset_parameters"):
                    raise TypeError(
                        f"Materialization requires that the `{type(submodule).__name__}.reset_parameters` method is implemented."
                        " This method is used to initialize any children parameters or buffers in this module."
                    )
                module_copy.reset_parameters()

            sd = {
                n: p
                for n, p in chain(module_copy.named_parameters(recurse=False), module_copy.named_buffers(recurse=False))
            }
            tm._transform_and_load_for_submodule(module_name, sd, shared_names, processed_names)


class MaterializationTransform(Transform):
    """Materialize a model that can fit on a device only after transforms applied.

    Args:
        device: Device to host :class:`~thunder.core.module.ThunderModule` after materialization.
                The transform will annotate any unannotated parameters on the meta device as to be initialized on this device.

    Keyword Args:
        init: Post-processing callable applied to :class:`~thunder.core.module.ThunderModule` after materialization.
              possible values are obtained from

              - `MaterializationTransform.init_from_original_state_dict(state_dict)`
                populate weights from a `state_dict` from the untransformed module,
              - `MaterializationTransform.init_from_transformed_state_dict(state_dict)`
                populate weights from a `state_dict` from the transformed module,
              - `MaterializationTransform.init_from_original_module_init()`
                initialize using the weight initialization of the original module (`reset_parameters`)
    """

    def __init__(
        self,
        device: str | torch.device,
        *,
        init: Callable[[MaterializationTransform, ThunderModule], None],
    ) -> None:
        self.device = torch.device(device)
        self.init = init

    def transform_module(self, model: ThunderModule):
        for p in chain(model._model.parameters(), model._model.buffers()):
            if p.device.type == "meta" and not hasattr(p, "_thunder_device"):
                p._thunder_device = self.device

        shared_names = model._get_shared_names()
        self.have_materialized: set[str] = set()

        # note: the iterations below are without duplicates
        for n, p in list(model.named_parameters()):
            if p.device.type == "meta":
                p = torch.nn.Parameter(
                    torch.empty_like(p, device=getattr(p, "_thunder_device", self.device)),
                    requires_grad=p.requires_grad,
                )
                for n2 in shared_names[n]:
                    model._overrides_parameters[n2] = p
                    self.have_materialized.add(n2)

        for n, b in list(model.named_buffers()):
            if b.device.type == "meta":
                b = torch.empty_like(
                    b, device=getattr(b, "_thunder_device", self.device), requires_grad=b.requires_grad
                )
                for n2 in shared_names[n]:
                    model._overrides_buffers[n2] = b
                    self.have_materialized.add(n2)

        self.init(self, model)

    @staticmethod
    def init_from_original_state_dict(state_dict):
        def module_init_from_original_state_dict(transform: MaterializationTransform, model: ThunderModule):
            # initialize things not persisted in the state dict
            module_init_from_original_module_init(transform, model, non_persistent_only=True)
            # load (and implicitly transform) the original state dict
            model.load_original_state_dict(state_dict)

        return module_init_from_original_state_dict

    @staticmethod
    def init_from_transformed_state_dict(state_dict):
        def module_init_from_transformed_state_dict(transform: MaterializationTransform, model: ThunderModule):
            # initialize things not persisted in the state dict
            module_init_from_original_module_init(transform, model, non_persistent_only=True)
            # load the state dict
            model.load_state_dict(state_dict)

        return module_init_from_transformed_state_dict

    @staticmethod
    def init_from_original_module_init():
        return module_init_from_original_module_init
