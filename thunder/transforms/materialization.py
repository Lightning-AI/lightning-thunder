from __future__ import annotations
from collections.abc import Callable
import copy
from itertools import chain
from typing import TYPE_CHECKING

import torch

from thunder.core.transform_common import Transform

if TYPE_CHECKING:
    from thunder.core.module import ThunderModule


class MaterializationTransform(Transform):
    """Materialize a model that can fit on a device only after transforms applied.

    Args:
        device: Device to host :class:`~thunder.core.module.ThunderModule` after materialization.

    Keyword Args:
        init: Post-processing callable applied to :class:`~thunder.core.module.ThunderModule` after materialization.
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

        for n, p in model.named_parameters():
            if p.device.type == "meta":
                model._overrides_parameters[n] = torch.nn.Parameter(
                    torch.empty_like(p, device=self.device), requires_grad=p.requires_grad
                )
        for n, b in model.named_buffers():
            if b.device.type == "meta":
                model._overrides_buffers[n] = torch.empty_like(b, device=self.device, requires_grad=b.requires_grad)
        self.init(self, model)

    @staticmethod
    def init_from_original_state_dict(state_dict):
        def module_init_from_original_state_dict(transform: MaterializationTransform, model: ThunderModule):
            # transform is unused
            model.load_original_state_dict(state_dict)

        return module_init_from_original_state_dict

    @staticmethod
    def init_from_transformed_state_dict(state_dict):
        def module_init_from_transformed_state_dict(transform: MaterializationTransform, model: ThunderModule):
            # transform is unused
            model.load_state_dict(state_dict)

        return module_init_from_transformed_state_dict

    @staticmethod
    def init_from_original_module_init():
        def module_init_from_original_module_init(transform: MaterializationTransform, tm: thunder.ThunderModule):
            for module_name, _ in tm._model.named_modules():
                prefix = module_name if not module_name else f"{module_name}."
                submodule = tm.get_submodule(module_name)

                # we use a copy to let the user's module alone
                module_copy = copy.copy(submodule)

                # Materialize meta-parameters on-device if necessary.
                # This is done before sharding in case the materialization logic depends on the tensor shape.
                # The tradeoff is that all of a module's direct parameters need to fit in device.
                # Each module only initializes its own parameters and not those of its children (recurse=False)
                if any(
                    t.is_meta for t in chain(module_copy.parameters(recurse=False), module_copy.buffers(recurse=False))
                ):
                    # TODO: we could also support calling a "param_init_fn" argument like PyTorch
                    module_copy.to_empty(device=transform.device, recurse=False)
                    if not hasattr(module_copy, "reset_parameters"):
                        raise TypeError(
                            f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
                            " This method is used to initialize any children parameters or buffers in this module."
                        )
                    module_copy.reset_parameters()

                    # TODO: non-persistent buffers?
                    sd = {
                        n: p
                        for n, p in chain(
                            module_copy.named_parameters(recurse=False), module_copy.named_buffers(recurse=False)
                        )
                    }
                    tm.transform_and_load_for_submodule(module_name, sd)

        return module_init_from_original_module_init
