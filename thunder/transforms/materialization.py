import torch
import thunder
from thunder.core.transform_common import Transform


class MaterializationTransform(Transform):
    def __init__(self, device, *, init):
        self.device = torch.device(device)
        self.init = init

    def transform_module(self, model: thunder.ThunderModule):
        for n, p in model.named_parameters():
            if p.device.type == 'meta':
                model._overrides_parameters[n] = torch.nn.Parameter(torch.empty_like(p, device=self.device), requires_grad=p.requires_grad)
        for n, b in model.named_buffers():
            if b.device.type == 'meta':
                model._overrides_buffers[n] = torch.empty_like(b, device=self.device, requires_grad=b.requires_grad)
        self.init(self, model)

    @staticmethod
    def init_from_original_state_dict(state_dict):
        def module_init_from_original_state_dict(transform: MaterializationTransform, model: thunder.ThunderModule):
            # transform is unused
            model.load_original_state_dict(state_dict)
        return module_init_from_original_state_dict
