from typing import Any

from thunder.core import prims
from thunder.core.pytree import tree_map
from thunder.core.trace import VariableInterface
from thunder.core.utils import safe_map_flat, sequencify


# TODO: Currently we use trace.args and trace.kwargs to get the arguments
# Maybe we should use these instead
transform_skip_list = (
    prims.PrimIDs.UNPACK_EMPTY_DICT,
    prims.PrimIDs.UNPACK_KEY,
    prims.PrimIDs.UNPACK_SEQUENCE,
    prims.PrimIDs.UNPACK_TRIVIAL,
    prims.PrimIDs.RETURN,
)


def eval_trace(trace, *args, symbol_mapper=None, with_env=False, **kwargs):
    """Evaluate a trace.

    Args:
        trace: trace to evaluate
        *args: arguments to evaluate the trace with
        symbol_mapper: function that maps a symbol to a function that evaluates it
        **kwargs: keyword arguments to evaluate the trace with

    Returns:
        result of evaluating the trace
    """
    env = {}

    def read(x: VariableInterface | Any) -> Any:
        if isinstance(x, VariableInterface):
            return env[x.name]
        else:
            return x

    def write(v: VariableInterface | Any, val: Any, allow_duplicates=False) -> None:
        if not isinstance(v, VariableInterface):
            return
        # Duplicates are allowed and overwritten
        if v.name in env:
            if allow_duplicates:
                return
            raise ValueError(f"Variable {v.name} is being overwritten this is not allowed")
        env[v.name] = val

    safe_map_flat(write, list(trace.args), list(args))
    safe_map_flat(write, list(trace.kwargs.values()), list(kwargs.values()))

    for symbol in trace.bound_symbols:
        if symbol.sym.id in transform_skip_list:
            continue
        args = tree_map(read, symbol.args)
        kwargs = tree_map(read, symbol.kwargs)
        symbol_mapper = symbol_mapper if symbol_mapper is not None else symbol.sym
        prim_func = symbol_mapper(symbol)
        if prim_func is None:
            continue
        result = prim_func(*args, **kwargs)
        try:
            safe_map_flat(write, list(sequencify(symbol.output)), list(sequencify(result)))
        except AssertionError as e:
            raise RuntimeError(
                f"Error while assigning the result of dispatched function {prim_func} to the output of the original symbol {symbol}."
                " This is likely due to a mismatch in the number of outputs."
                f" The original symbol has {len(symbol.output)} outputs and the dispatched function has {len(sequencify(result))} outputs."
            ) from e

    if with_env:
        return tree_map(read, trace.output), env

    return tree_map(read, trace.output)
