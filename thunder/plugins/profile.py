from thunder.core.recipe import Plugin, PluginPolicy
from thunder.dev_utils.profile_transform import ProfileTransform


class Profile(Plugin):
    policy = PluginPolicy.POST

    def __init__(self, input_match, from_match_idx=0, to_match_idx=1):
        self.profile_transform = ProfileTransform(
            input_match=input_match, start_idx=from_match_idx, end_idx=to_match_idx
        )

    def setup_transforms(self):
        return [self.profile_transform]
