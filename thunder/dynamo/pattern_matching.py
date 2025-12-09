import operator
from pathlib import Path

import textwrap

import torch.fx
import torch._inductor.pattern_matcher
from torch._inductor.pattern_matcher import (
    _TargetExpr,
    Arg,
    CallFunction,
    Ignored,
    KeywordArg,
    MultiOutputPattern,
    PatternExpr,
    PatternPrettyPrinter,
    fx_to_pattern,
)

from thunder.core.utils import OrderedSet

SERIALIZED_PATTERN_PATH = Path(__file__).parent
SERIALIZED_PATTERN_FILENAME = "serialized_patterns"

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

def pattern_to_graphmodule(pattern: PatternExpr) -> torch.fx.GraphModule:
    """
    Convert a PatternExpr back into an FX GraphModule.

    This is essentially the inverse of `torch._inductor.pattern_matcher.fx_to_pattern`.
    It reconstructs an executable graph from a pattern expression by traversing
    the pattern tree and creating corresponding FX nodes.

    Args:
        pattern: The PatternExpr to convert. Can be a simple pattern or a
            MultiOutputPattern for patterns with multiple outputs.

    Returns:
        A GraphModule representing the pattern's computation graph. The output
        is always wrapped in a tuple for consistency with Inductor's conventions.

    Note:
        - KeywordArg patterns become named placeholders
        - Arg/Ignored patterns become wildcard placeholders
        - Literal values are passed through unchanged
        - The function deduplicates nodes when the same PatternExpr instance
          appears multiple times in the pattern tree
    """
    graph = torch.fx.Graph()
    placeholders = {}
    # Cache to deduplicate nodes: PatternExpr instance -> fx.Node
    node_map = {}

    def visit(expr):
        # If we've already visited this specific PatternExpr object, return the existing node
        if id(expr) in node_map:
            return node_map[id(expr)]

        node = None
        if isinstance(expr, MultiOutputPattern):
            # This is usually the top-level container.
            # We recursively visit all outputs.
            outputs = [visit(out) for out in expr.outputs if out is not None]
            # The MultiOutputPattern itself doesn't correspond to a single node
            # in the graph body, but rather the Output node's args.
            # We return the list/tuple of outputs.
            return tuple(outputs)

        elif isinstance(expr, CallFunction):
            # Extract args from flat_args_kwargs (this part depends on internal structure)
            # We assume simple structure for the sketch
            # In reality, you'd need to parse expr.flat_args_kwargs or expr.args/kwargs
            # Note: PatternExpr stores args/kwargs in a specific way

            # Helper to process arguments
            def process_arg(arg):
                if isinstance(arg, PatternExpr):
                    return visit(arg)
                elif isinstance(arg, (list, tuple)):
                    return type(arg)(process_arg(x) for x in arg)
                return arg

            args = [process_arg(a) for a in expr.args]
            kwargs = {k: process_arg(v) for k, v in expr.kwargs.items()}

            node = graph.call_function(expr.fns[0], args=tuple(args), kwargs=kwargs)

        elif isinstance(expr, KeywordArg):
            if expr.name not in placeholders:
                placeholders[expr.name] = graph.placeholder(expr.name)
            node = placeholders[expr.name]

        elif isinstance(expr, (Arg, Ignored)):
            # Create a unique wildcard placeholder
            name = f"wildcard_{len(placeholders)}"
            placeholders[name] = graph.placeholder(name)
            node = placeholders[name]
        else:
            # Literal value
            return expr

        # Cache the created node
        if node is not None:
            node_map[id(expr)] = node
        return node

    # Build the graph
    output_value = visit(pattern)

    # If the result is a tuple (from MultiOutputPattern), output it as such
    # If it's a single node (from standard PatternExpr), output that
    if isinstance(output_value, tuple):
        graph.output(output_value)
    else:
        graph.output((output_value,))  # Inductor often expects tuple output

    return torch.fx.GraphModule(torch.nn.Module(), graph)
