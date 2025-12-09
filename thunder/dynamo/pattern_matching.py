import operator
from pathlib import Path

import textwrap

import torch.fx
import torch._inductor.pattern_matcher
from torch._inductor.pattern_matcher import (
    _TargetExpr,
    PatternExpr,
    PatternPrettyPrinter,
    fx_to_pattern,
)

from thunder.core.utils import OrderedSet

SERIALIZED_PATTERN_PATH = Path(__file__).parent
SERIALIZED_PATTERN_FILENAME = "serialized_patterns.py"

# TODO programmatically load all the patterns from a "serialized_patterns" folder
#     m = importlib.import_module(
#     f"thunder.dynamo.serialized_patterns.{pattern_name}"
# )

_serialized_patterns: OrderedSet[str] = OrderedSet()


# Monkey patch the repr fn to include the module.
# This is required to preserve module information for operations because pattern matching in Thunder
# occurs before decomposition.
# Without this patch, we might lose the distinction between operations from different modules.
def fns_repr(self) -> str:
    first_repr = self.fns[0]
    if not isinstance(first_repr, str):
        first_repr = first_repr.__name__

    if len(self.fns) > 1:
        return f"[{first_repr}, ...]"
    elif self.fns[0] is getattr(torch, first_repr, None):
        return f"torch.{first_repr}"
    elif self.fns[0] is getattr(operator, first_repr, None):
        return f"operator.{first_repr}"
    # sequence from thunder/torch/__init__.py#202-219
    elif self.fns[0] is getattr(torch.nn.functional, first_repr, None):
        return f"torch.nn.functional.{first_repr}"
    elif self.fns[0] is getattr(torch.Tensor, first_repr, None):
        return f"torch.Tensor.{first_repr}"
    elif self.fns[0] is getattr(torch.special, first_repr, None):
        return f"torch.special.{first_repr}"
    elif isinstance(self.fns[0], torch._ops.OpOverload):
        return str(self.fns[0])
    else:
        return first_repr


# Monkey patch here!
_TargetExpr.fns_repr = fns_repr


def serialize_pattern(unique_name: str, graph_module: torch.fx.GraphModule) -> PatternExpr:
    """
    Serialize a GraphModule into a pattern expression and save it to a file.

    Args:
        unique_name: A unique identifier for the pattern. This name will be used to import the pattern later.
        graph_module: The FX GraphModule to be serialized.

    Returns:
        The generated PatternExpr.
    """

    def get_file_template() -> str:
        auto_generated_msg = textwrap.dedent(
            """\
            # This is auto-generated. Please do not modify it by hand.
            """
        )

        file_template = textwrap.dedent(
            """\

            {msg}
            import torch
            import operator

            """
        ).format(msg=auto_generated_msg)

        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            try:
                if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
                    # pyrefly: ignore [bad-argument-type]
                    pattern_matcher_imports.append(name)
            except TypeError:
                pass

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}")

    # TODO one file contains multiple patterns each with unique name as specified in the args but the file name is TBD here
    pattern_name = SERIALIZED_PATTERN_FILENAME

    from torch._functorch import config as functorch_config

    with functorch_config.patch(functionalize_rng_ops=False):
        pattern = fx_to_pattern(
            graph_module,
            # ignore_types=(int, float, list, torch.device, torch.dtype)
        )

    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    file_template = get_file_template()

    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    return pattern
