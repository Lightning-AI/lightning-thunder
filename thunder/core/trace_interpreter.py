from functools import partial
from typing import Any

from thunder.core import prims
from thunder.core.pytree import tree_map
from thunder.core.trace import VariableInterface, from_trace, tracectx
from thunder.core.baseutils import ProxyInterface, TensorProxyInterface
from thunder.core.utils import safe_map_flat, sequencify
from thunder.core.proxies import variableify, ProxyTag
from thunder.core.transform_common import VJPDual
from thunder.core.symbol import has_tags
from thunder.core.compile_data import get_compile_option


# TODO: Currently we use trace.args and trace.kwargs to get the arguments
# Maybe we should use these instead
trace_interpreter_skip_list = (
    prims.PrimIDs.UNPACK_EMPTY_DICT,
    prims.PrimIDs.UNPACK_KEY,
    prims.PrimIDs.UNPACK_SEQUENCE,
    prims.PrimIDs.UNPACK_TRIVIAL,
    prims.PrimIDs.RETURN,
)


def interpret_trace(trace, *args, symbol_mapper=None, with_env=False, **kwargs):
    """Interpret a trace.

    Args:
        trace: trace to interpret
        *args: arguments to interpret the trace with
        symbol_mapper: function that redirects the evaluation of a BoundSymbol to a different function
        with_env: whether to return the environment after interpreting the trace. Environment is a dictionary
            that maps VariableInterface objects to their values.
        **kwargs: keyword arguments to interpret the trace with

    Returns:
        result of interpreting the trace, optionally with the environment that saves all intermediate values
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
        if v.name in env:
            if allow_duplicates:
                # Duplicates are allowed and not overwritten
                return
            raise ValueError(f"Variable {v.name} is being overwritten this is not allowed")
        env[v.name] = val

    safe_map_flat(write, list(trace.args), list(args))
    safe_map_flat(write, list(trace.kwargs.values()), list(kwargs.values()))

    for symbol in trace.bound_symbols:
        if symbol.sym.id in trace_interpreter_skip_list:
            continue
        args = tree_map(read, symbol.args)
        kwargs = tree_map(read, symbol.kwargs)
        prim_func = symbol_mapper(symbol) if symbol_mapper is not None else symbol.sym
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


# TODO: refactor to use TraceSubstitutionProcessor to split out the gradient-specific part
def interpret_trace_to_trace(trace, *args, symbol_mapper=None, with_env=False, **kwargs):
    """Interpret a trace.

    Args:
        trace: trace to interpret
        *args: arguments to interpret the trace with
        symbol_mapper: function that redirects the evaluation of a BoundSymbol to a different function
        with_env: whether to return the environment after interpreting the trace. Environment is a dictionary
            that maps VariableInterface objects to their values.
        **kwargs: keyword arguments to interpret the trace with

    Returns:
        result of interpreting the trace, optionally with the environment that saves all intermediate values
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
        if v.name in env:
            if allow_duplicates:
                # Duplicates are allowed and not overwritten
                return
            raise ValueError(f"Variable {v.name} is being overwritten this is not allowed")
        env[v.name] = val

    def add_to_swap_map(old, new):
        if isinstance(old, ProxyInterface):
            if isinstance(new, ProxyInterface) and variableify(new) in env:
                # the new isn't new, but something returned the input
                # this means we need to map the old to the new
                old, new = new, old
            elif isinstance(old, TensorProxyInterface):
                # should we have a fix shapes pass? the sharding
                # (FSDP, tensor parallel) transforms do "break" shape metadata
                new_trace.names.remove(old.name)  # taken by the .replace proxy
                if isinstance(new, VJPDual):
                    old = old.replace(shape=new.primal._shape)
                else:
                    old = old.replace(shape=new._shape)

            if isinstance(new, VJPDual):
                swap_map[variableify(new.primal)] = old
                new.primal = old
            else:
                assert isinstance(new, ProxyInterface), (old, new)
                swap_map[variableify(new)] = old

    def do_swap(v):
        if isinstance(v, VJPDual):
            v.primal = tree_map(do_swap, v.primal)
            v.residuals = tree_map(do_swap, v.residuals)
            return v
        if not isinstance(v, ProxyInterface):
            return v
        return swap_map.get(variableify(v), v)

    def unwrap_vjpdual(v):
        if isinstance(v, VJPDual):
            return v.primal
        return v

    new_trace = from_trace(trace)
    with tracectx(new_trace):
        swap_map = {}

        safe_map_flat(add_to_swap_map, list(trace.args), list(args))
        safe_map_flat(add_to_swap_map, list(trace.kwargs.values()), list(kwargs.values()))
        args, kwargs = tree_map(do_swap, (args, kwargs))

        safe_map_flat(write, list(trace.args), list(args))
        safe_map_flat(write, list(trace.kwargs.values()), list(kwargs.values()))

        for bsym in trace.bound_symbols:
            if bsym.sym == prims.python_return:
                # For return we need to read the args and swap because it may
                # include inputs that need replacing.
                # However, we need to unwrap VJP dual for recording in the symbols.
                # This is in particular needed for the flat_inputs the autograd transform.
                args = tree_map(lambda x: unwrap_vjpdual(read(x)), bsym.args)
                new_trace.bound_symbols.append(bsym.from_bsym(args=args).from_bsym_swap_proxies(swap_map))
                continue

            if bsym.sym.id in trace_interpreter_skip_list:
                # swap proxies, also for UNPACK symbols if inputs are replaced
                new_trace.bound_symbols.append(bsym.from_bsym_swap_proxies(swap_map))
                continue
            args = tree_map(read, bsym.args)
            kwargs = tree_map(read, bsym.kwargs)

            prim_func = symbol_mapper(bsym) if symbol_mapper is not None else bsym.sym
            if prim_func is None:
                continue

            new_trace.push_scope([])
            result = prim_func(*args, **kwargs)
            new_bsyms = new_trace.pop_scope()

            swap_map = {}

            # TODO: if inputs are returned, the old outputs should be mapped on the new ones (= the inputs) instead of the other way round
            if not new_bsyms:
                # empty result means we want to swap references to the old
                # result to the new result (which will be one of the args)
                safe_map_flat(add_to_swap_map, list(sequencify(result)), list(sequencify(bsym.output)))
            else:
                safe_map_flat(add_to_swap_map, list(sequencify(bsym.output)), list(sequencify(result)))

            ### replace bsyms

            for new_bsym in new_bsyms:
                # TODO: what to do with bsym header? Maybe have a combined from_bsym_swap_proxies and from_bsym?
                auto_recompute_intermediates: None | bool = get_compile_option(
                    "auto_recompute_intermediates",
                    "Whether to mark intermediates as up for recomputation. This experimental flag reduces the number tensors saved for backward, at the expense of more compute",
                )

                if auto_recompute_intermediates and not has_tags(
                    bsym,
                    {
                        prims.OpTags.RANDOM_OP,
                        prims.OpTags.DONT_RECOMPUTE_IN_BACKWARD,
                        prims.OpTags.DONT_AUTO_RECOMPUTE_IN_BACKWARD,
                    },
                ):
                    # when we decompose to compute the forward/backward, we mark intermediates as to be recomputed in the backward.
                    # Typically our decompositions are for things that will then be fused together.
                    # We tag "expensive" operations (sdpa, matmul) as DONT_AUTO_RECOMPUTE_IN_BACKWARD to block this.
                    for o in new_bsym.flat_proxy_outs:
                        if variableify(o) not in swap_map:
                            o.tags.add(ProxyTag.RECOMPUTE_IN_BACKWARD)
                new_trace.bound_symbols.append(
                    new_bsym.from_bsym_swap_proxies(swap_map).from_bsym(
                        source_filename=bsym.source_filename, source_positions=bsym.source_positions
                    )
                )

            result = tree_map(do_swap, result)

            try:
                safe_map_flat(write, list(sequencify(bsym.output)), list(sequencify(result)))
            except AssertionError as e:
                raise RuntimeError(
                    f"Error while assigning the result of dispatched function {prim_func} to the output of the original symbol {bsym}."
                    " This is likely due to a mismatch in the number of outputs."
                    f" The original symbol has {len(bsym.output)} outputs and the dispatched function has {len(sequencify(result))} outputs."
                ) from e

    if with_env:
        return new_trace, tree_map(read, trace.output), env

    return new_trace, tree_map(read, trace.output)


class TraceSubstitutionProcessor:
    """This processes a trace in an interpretation-style way by looping over the bound symbols.
    This processing aims to preserve as much information on the proxies as possible.

    Args:
        trace: trace to process
        *args: arguments to process the trace with
        **kwargs: keyword arguments to process the trace with

    The user is expected to subclass the trace and implement process_bsym with the help of add_unprocessed_bsyms (useful eg for using subsymbols to compute a symbol), add_processed_bsyms, and add_bsyms_from_function.

    Calling the instantiated object initiates the processing and returns
    the new trace and a mapping of the outputs.

    See the OpExProcessor in thunder.executors.passes._transform_for_operator_executor_execution for an example of subclassing.
    """

    NULL = object()

    def __init__(self, trace, *args, **kwargs):
        self.env = {}
        self.trace = trace
        self.new_trace = from_trace(self.trace)
        self.have_processed_args = False

    def read(self, x: VariableInterface | Any) -> Any:
        if isinstance(x, VariableInterface):
            return self.env[x.name]
        else:
            return x

    def write(self, v: VariableInterface | Any, val: Any, allow_duplicates=True) -> None:
        if not isinstance(v, VariableInterface):
            return
        if v.name in self.env:
            if allow_duplicates:
                # Duplicates are allowed and not overwritten
                return
            raise ValueError(f"Variable {v.name} is being overwritten this is not allowed")
        # inherit tags
        val.tags.update(v.tags)
        self.env[v.name] = val

    def add_to_swap_map(self, old, new):
        if old is new:
            return
        if isinstance(old, ProxyInterface):
            if isinstance(new, ProxyInterface) and variableify(new) in self.env:
                # the new isn't new, but something returned the input
                # this means we need to map the old to the new
                old, new = new, old
            elif isinstance(old, TensorProxyInterface):
                # should we have a fix shapes pass? the sharding
                # (FSDP, tensor parallel) transforms do "break" shape metadata
                self.new_trace.names.remove(old.name)  # taken by the .replace proxy
                if isinstance(new, VJPDual):
                    old = old.replace(shape=new.primal._shape)
                else:
                    old = old.replace(shape=new._shape)

            if isinstance(new, VJPDual):
                self.swap_map[variableify(old.primal)] = new
                new.primal = old
            else:
                assert isinstance(new, ProxyInterface), (old, new)
                self.swap_map[variableify(old)] = new

    def do_swap(self, v):
        if isinstance(v, VJPDual):
            v.primal = tree_map(self.do_swap, v.primal)
            v.residuals = tree_map(self.do_swap, v.residuals)
            return v
        if not isinstance(v, ProxyInterface):
            return v
        return self.swap_map.get(variableify(v), v)

    def add_unprocessed_bsyms(self, bsyms):
        self.unprocessed_bsyms[:0] = bsyms

    def add_bsyms_from_function(self, fn, /, *args, tags=None, **kwargs):
        self.new_trace.push_scope([])
        result = fn(*args, **kwargs)
        new_bsyms = self.new_trace.pop_scope()
        if tags is not None:
            for bsym in new_bsyms:
                bsym.tags.update(tags)
        self.new_bsyms += new_bsyms

        self.set_result(result)
        return result

    def add_processed_bsyms(self, bsyms):
        self.new_bsyms += bsyms

    def set_result(self, result):
        self.replacement_result = result

    def process_bsym(self, bsym):
        raise NotImplementedError("This needs to be implemented in subclasses")

    def process_args(self, *args, **kwargs):
        self.have_processed_args = True
        with tracectx(self.new_trace):
            self.swap_map = {}

            safe_map_flat(self.add_to_swap_map, list(self.trace.args), list(args))
            safe_map_flat(self.add_to_swap_map, list(self.trace.kwargs.values()), list(kwargs.values()))
            args, kwargs = tree_map(self.do_swap, (args, kwargs))

            safe_map_flat(self.write, list(self.trace.args), list(args))
            safe_map_flat(self.write, list(self.trace.kwargs.values()), list(kwargs.values()))

    def __call__(self):
        with tracectx(self.new_trace):
            self.unprocessed_bsyms = self.trace.bound_symbols[:]
            self.swap_map = {}

            while self.unprocessed_bsyms:
                bsym = self.unprocessed_bsyms.pop(0)

                if self.have_processed_args and bsym.sym.id in trace_interpreter_skip_list:
                    self.new_trace.bound_symbols.append(bsym.from_bsym())
                    continue

                # this should be prettier
                self.replacement_result = self.NULL
                self.new_bsyms = []

                self.process_bsym(bsym)

                if self.new_bsyms:
                    assert self.replacement_result is not self.NULL, "Need to call set_result if producing new bsyms"

                if self.replacement_result is not self.NULL:
                    # TODO: if inputs are returned, the old outputs should be mapped on the new ones (= the inputs) instead of the other way round
                    if not self.new_bsyms:
                        # empty result means we want to swap references to the old
                        # result to the new result (which will be one of the args)
                        safe_map_flat(
                            self.add_to_swap_map,
                            list(sequencify(self.replacement_result)),
                            list(sequencify(bsym.output)),
                        )
                    else:
                        safe_map_flat(
                            self.add_to_swap_map,
                            list(sequencify(bsym.output)),
                            list(sequencify(self.replacement_result)),
                        )

                    ### replace bsyms

                    for new_bsym in self.new_bsyms:
                        # TODO: what to do with bsym header? Maybe have a combined from_bsym_swap_proxies and from_bsym?
                        self.new_trace.bound_symbols.append(
                            new_bsym.from_bsym_swap_proxies(self.swap_map).from_bsym(
                                source_filename=bsym.source_filename, source_positions=bsym.source_positions
                            )
                        )

                    result = tree_map(self.do_swap, self.replacement_result)

                    # we need to allow duplicates here because the re-interpretation is not necessairly DCEed when subsymbols symbols are flattened into the trace after re-execution.
                    try:
                        safe_map_flat(
                            partial(self.write, allow_duplicates=True),
                            list(sequencify(bsym.output)),
                            list(sequencify(result)),
                        )
                    except AssertionError as e:
                        raise RuntimeError(
                            f"Error while assigning the result of dispatched function {prim_func} to the output of the original symbol {bsym}."
                            " This is likely due to a mismatch in the number of outputs."
                            f" The original symbol has {len(bsym.output)} outputs and the dispatched function has {len(sequencify(result))} outputs."
                        ) from e

        return self.new_trace, tree_map(self.read, self.trace.output)


def rerun_trace(trace):  # rerun trace to fix metadata
    class RerunProcessor(TraceSubstitutionProcessor):
        def process_bsym(self, bsym):
            if bsym.sym == prims.unpack_trivial:
                self.add_processed_bsyms([bsym.from_bsym()])
                self.set_result(bsym.output)
                return
            bsym = bsym.from_bsym_swap_proxies(self.swap_map)
            self.add_bsyms_from_function(bsym.sym, *bsym.args, **bsym.kwargs)

    new_trace, _ = RerunProcessor(trace)()
    return new_trace
