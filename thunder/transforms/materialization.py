from __future__ import annotations
from collections.abc import Callable
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
