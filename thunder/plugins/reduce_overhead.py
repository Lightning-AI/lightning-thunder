from thunder.core.recipe import Plugin, PluginPolicy
from thunder.transforms.cudagraph import CUDAGraphTransform


class ReduceOverhead(Plugin):
    """
    Plugin to enable CUDA Graphs and reduce CPU overhead.
    """

    policy = PluginPolicy.POST

    def setup_transforms(self):
        """
        Fetches the CUDAGraph transform.

        Returns:
            list[Transform]: A list containing the CUDAGraph transform.
        """
        return [CUDAGraphTransform()]
