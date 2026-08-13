import operator

from lightning_utilities.core.imports import compare_version

# PyTorch version constants
# `use_base_version` so that dev builds like "2.13.0a0+gitabc123" compare as "2.13.0"
_TORCH_GREATER_EQUAL_2_13 = compare_version("torch", operator.ge, "2.13.0", use_base_version=True)
