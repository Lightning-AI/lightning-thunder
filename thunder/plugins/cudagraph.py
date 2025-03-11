from thunder import Plugin
from thunder.core.recipe import Plugin, PluginPolicy
from thunder.transforms.cudagraph import CUDAGraphTransform


class CUDAGraph(Plugin):
    policy = PluginPolicy.POST

    def setup_transforms(self):
        return [CUDAGraphTransform()]
