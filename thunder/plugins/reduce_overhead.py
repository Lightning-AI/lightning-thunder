from thunder import Plugin
from thunder.core.recipe import Plugin, PluginPolicy
from thunder.transforms.cudagraph import CUDAGraphTransform


class ReduceOverhead(Plugin):
    policy = PluginPolicy.POST

    def setup_transforms(self):
        return [CUDAGraphTransform()]
