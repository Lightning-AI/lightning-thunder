from __future__ import annotations
import time
from typing import TYPE_CHECKING
from abc import ABC
from collections.abc import Sequence
import dataclasses
from itertools import filterfalse
from functools import partial

import thunder
import thunder.core.prims as prims
from thunder.core.baseutils import BoundSymbolInterface, NumberProxyInterface
from thunder.core.proxies import Proxy, variableify, Variable, TensorProxy, unvariableify
from thunder.core.pytree import tree_flatten, tree_iter, tree_map, tree_unflatten
from thunder.core.symbol import BoundSymbol, BoundSymbolRHS, has_tags
from thunder.core.trace import from_trace, TraceProvenance, TraceCtx as Trace, tracectx
from thunder.core.utils import ProxyDict, producers, check

if TYPE_CHECKING:
    from numbers import Number
    from typing import Any
    from thunder.core.module import ThunderModule


#
# Common optimization and transform passes
#
# NOTE This file avoids transforms depending on passes, since passes depend on transforms
@dataclasses.dataclass  # (frozen=True)
class VJPDual:
    """A pair of primal and saved information for backward (residuals).

    Args:
        primal (Union[Proxy, Number]): Primal value, i.e., the value being differentiated.
        residuals (Tuple[Proxy, ...]): Residuals, i.e., the values that are
            saved for the backward.

    Yields:
        Tuple[Variable, Tuple[Variable, ...], Callable]: Primal and residuals
    """

    primal: Proxy | Number
    residuals: tuple[Proxy, ...]

    def __iter__(self):
        yield self.primal
        yield self.residuals


# Modifies an existing BoundSymbol, removing its "no-op" subsymbols (recursively) which perform no operations
def _remove_noop_subsymbols(bsym: BoundSymbol) -> None:
    nsbsyms: list[BoundSymbol] = []
    sbsym: BoundSymbol
    for sbsym in bsym.subsymbols:
        if len(sbsym.subsymbols) == 0 and not sbsym.sym.is_prim:
            continue
        # if all outputs are constants, we elmininate the subsymbol
        if not has_tags(bsym, {prims.OpTags.DONT_DCE}) and not any(
            o is not None for o in sbsym.flat_proxy_outs
        ):  # is not None to avoid cast to bool
            continue
        _remove_noop_subsymbols(sbsym)
        nsbsyms.append(sbsym)

    bsym.subsymbols = nsbsyms


def _inplace_copy_sanity_check(extrace: Trace):
    """The sanity check is based on the sharp edge of nvfuser's `add_ouput(output, input)` interface,
    it makes sure that the `copy_to` argument of `prims.copy_` is not used as input for any of its subsequent operators in a nvFusion fused operator

    Anti-pattern:

    .. code-block:: python

        [t2] = nvFusion0(x, y)
            # result = prims.mul(x, y)
            # a = prims.copy_(result, x)
            # t2 = prims.add(a, y) or t2 = prims.add(x, y)

    Do not use the `copy_to` variable `x` or `a` after it has been updated, use the `copy_from` variable `result` instead to reflect the dependency:

    .. code-block:: python

        [t2] = nvFusion0(x, y)
            # result = prims.mul(x, y)
            # a = prims.copy_(result, x)
            # t2 = prims.add(result, y)
    """

    from thunder.core.utils import consumers

    nvfuser_symbols = (bsym for bsym in extrace.bound_symbols if bsym.sym.name.startswith("nvFusion"))
    for bsym in nvfuser_symbols:
        consumer_dict = consumers(list(bsym.subsymbols), _map_to_numbers=True)
        inplace_copy_idx = ((idx, sym) for idx, sym in enumerate(bsym.subsymbols) if sym.sym.id == prims.PrimIDs.COPY_)
        for idx, subbsym in inplace_copy_idx:
            copy_to_arg = subbsym.flat_args[1]
            copy_to_out = subbsym.output

            def check(inp, log_str):
                if inp is not None and inp in consumer_dict:
                    last_used_idx = max(consumer_dict[inp])
                    if last_used_idx > idx:
                        raise NotImplementedError(
                            f"{bsym.subsymbols[last_used_idx]} trying to use {inp} (the {log_str} of 'prims.copy_') as input, which is not safe."
                            f" There is a risk of accessing the wrong memory. If you are sure you don't want to use this check, it can be disabled by setting `disable_inplace_copy_check=True` in `thunder.jit`."
                        )

            check(copy_to_arg, "'copy_to' argument")
            check(copy_to_out, "output")


def remove_duplicate_number_proxies(bsyms: Sequence[BoundSymbol]) -> list[BoundSymbol]:
    """This removes duplicate number proxies when they are returned multiple times.
    The remaining DCE pass does not see them (because they often are in a tuple?).
    In particular, proxies may be extracted multiple times when using the thunder.jit's
    symbolic constraints mode.
    """
    seen = set()

    def keep_or_swap(p):
        if not isinstance(p, NumberProxyInterface):
            return p
        if p.name in seen:
            return p.value  # don't make it a duplicate
        seen.add(p.name)
        return p

    new_bsyms = []
    for bsym in bsyms:
        output = tree_map(keep_or_swap, bsym.output)
        new_bsyms.append(bsym.from_bsym(output=output))
    return new_bsyms


# TODO This calls variableify(), but we could directly construct Variable objects instead, which might slightly
#   improve performance
# Runs a Dead Code Elimination (DCE) pass
# NOTE Today we are only interested in computations that produce proxies, so this will eliminate operations
#   that only produce non-proxy objects
# NOTE needed_proxies is an in/out argument, it takes an initial set of Variables you want to keep, and return
#   all the needed proxies of the input trace
def dce(trace: Trace, needed_proxies: None | set[Variable] = None) -> Trace:
    start_time_ns = time.perf_counter_ns()

    producer_map: ProxyDict = producers(trace)

    flat_trace_outputs, _ = tree_flatten(trace.output)
    if needed_proxies is None:
        needed_proxies: set[Variable] = set(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    else:
        needed_proxies.update(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    dced = []

    bsym: BoundSymbol
    for bsym in reversed(trace.bound_symbols):
        # Preserves symbols that should never be collected
        if has_tags(bsym, {prims.OpTags.DONT_DCE}):
            needed = True
        else:
            needed = False

        # NOTE This block is run even if we know we're preserving the operation, because it
        #   may mark some of the operation's outputs as unused
        some_unused = False
        for out in bsym.flat_proxy_outs:
            if variableify(out) in needed_proxies and producer_map[out] == bsym:
                needed = True
            else:
                some_unused = True

        if needed:
            nbsym: BoundSymbol = bsym

            # Replaces unused Proxy outputs with None
            if some_unused:

                def _helper(x):
                    if isinstance(x, Proxy) and (variableify(x) not in needed_proxies or producer_map[x] != bsym):
                        return None
                    return x

                nbsym_output = tree_map(_helper, bsym.output)
                nbsym = bsym.from_bsym(output=nbsym_output)

            # Eliminates no-op subsymbols
            # NOTE In general editing subsymbols doesn't do anything, but no-op subsymbols are a pain
            #   for transforms to deal with. Transforms typically look for a "flattened" version of an
            #   operator for which they can apply their rules, and no-op subsymbols have no
            #   flattening, requiring each transform handle them explicitly or DCE them themselves
            #   while flattening.
            _remove_noop_subsymbols(nbsym)

            dced.append(nbsym)
            for x in nbsym.flat_proxy_args:
                needed_proxies.add(variableify(x))

    dcetrace = from_trace(trace)
    dced_bound_symbols = list(reversed(dced))
    # duplicate number proxies happen with the symbolic shapes and are
    # not covered by the above (due to being in tuples?).
    dced_bound_symbols = remove_duplicate_number_proxies(dced_bound_symbols)
    dcetrace.bound_symbols = dced_bound_symbols

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    dcetrace.set_provenance(TraceProvenance(f"Dead Code Elimination (took {elapsed_time_millis} milliseconds)"))

    return dcetrace


#
# Functions related to the Common Subexpression Elimination (CSE) pass
#
def replace_redundant_inputs(
    redundant_map: dict[Variable, Proxy], bsyms: Sequence[BoundSymbol]
) -> Sequence[BoundSymbol]:
    new_bsyms = []
    for bsym in bsyms:
        # Checks if the bound symbol has redundant inputs (that need to be replaced)
        has_redundant_inputs: bool = False
        for x in bsym.flat_proxy_args:
            if Variable(x) in redundant_map:
                has_redundant_inputs = True
                break

        # Bound symbols without redundant inputs need no modification
        if not has_redundant_inputs:
            new_bsyms.append(bsym)
            continue

        # Bound symbols with redundant inputs have to remap those inputs, and the
        #   inputs of their subsymbols, to the original computations
        new_bsym = bsym.from_bsym_swap_proxies(
            redundant_map,
            skip_inputs=False,
            skip_output=False,
            skip_subsymbols=False,
        )
        new_bsyms.append(new_bsym)

    return new_bsyms


# These are ops that are not referentially transparent. We need to treat such
# ops specially when optimizing; for example, CSE cannot coalesce two calls
# into one for ops in this set.
NON_FUNCTIONAL_OPS: set[prims.PrimIDs | str] = {
    prims.PrimIDs.UNIFORM,
    "torch.uniform",  # this doesn't exist as of the PR
    "torch.uniform_like",  # this doesn't exist as of the PR
    # thunder.core.prims doesn't support. See https://pytorch.org/docs/stable/generated/torch.rand.html.
    # "torch.rand",
    # "torch.rand_like",
    "torch.nn.functional.dropout",
    "torch.nn.functional.scaled_dot_product_attention",
}


# This helper function applies cse transformation to bound symbol that is not a fusion.
def cse_single_bsym(
    redundant_map: dict[Variable, Proxy],
    rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbolInterface],
    bsym: BoundSymbolInterface,
) -> BoundSymbolInterface:
    check(
        bsym.sym.is_fusion != True,
        lambda: f"Expected bound symbol not to be a fusion in _cse_single_bsym",
        exception_type=AssertionError,
    )

    # `NON_FUNCTIONAL_OPS` are a op that's not deterministic, for example, `torch.nn.functiona.dropout`
    # and `torch.nn.functional.scaled_dot_product_attention` depending on PRNG.
    if bsym.sym.id in NON_FUNCTIONAL_OPS:
        return bsym

    # We can replace any redundant `bsym.args` and `bsym.kwargs` if it is available in the current context.
    new_bsym = bsym.from_bsym_swap_proxies(
        swap_map=redundant_map,
        skip_inputs=False,
        skip_output=True,
    )

    # Skip appending this bsym to the new bound symbols due to its rhs being a common subexpression.
    rhs = new_bsym.rhs
    if (prior_bsym := rhs_to_bsym_map.get(rhs)) is not None and bsym._executor is prior_bsym._executor:
        for src, dst in zip(bsym.flat_outs, prior_bsym.flat_outs):
            # Detects (and avoids) aliasing
            vsrc, vdst = variableify(src), variableify(dst)
            if vsrc == vdst:
                continue
            redundant_map[vsrc] = dst
        return None
    else:
        rhs_to_bsym_map[rhs] = bsym
        return new_bsym


# TODO Update the replacement of redundant proxies to use a visitor pattern
#   when that architecture is added in the future
def cse(trace: Trace) -> Trace:
    """Remove bound symbols whose right hand side is common expression.

    This does two things:
        1. Removes bound symbols if their right hand side expression is already seen.
        2. Replaces variables of arguments and keyword arguments of the bound symbols to use and
           their subsymbols with the preceding ones using the map of a variable to another with
           the same right hand side expression.

    Say we have `foo` defined in the below code snippet.

    .. code::

        def foo(x, y):
            a = x + y
            b = y - x
            c = x + y  # Expected to be removed in favor of `a`.
            d = y - x  # Expected to be removed in favor of `b`.
            z = a + b  # Expected to be intact.
            w = c + d  # Expected to be converted to `w = a + b` and then removed in favor of `z`.
            m = w + 1  # Expected to be updated to `m = z + 1`.
            return z, w, m  # Expected to be (z, z, z + 1)

        # CPU tensors
        @torch.no_grad()
        def thunder_140374621612752(x, y):
          # x: "cpu f32[2, 2]" =  x  (trivial unpack)
          # y: "cpu f32[2, 2]" =  y  (trivial unpack)
          t0 = torch.add(x, y)  # t0: "cpu f32[2, 2]"
          t1 = torch.sub(y, x)  # t1: "cpu f32[2, 2]"
          del [y, x]
          t4 = torch.add(t0, t1)  # t4: "cpu f32[2, 2]"
          del [t0, t1]
          t6 = torch.add(t4, 1)  # t6: "cpu f32[2, 2]"
          return (t4, t4, t6)

        # CUDA tensors & nvFuser
        @torch.no_grad()
        def thunder_140410131706304(x, y):
          # x: "cuda:0 f32[2, 2]" =  x  (trivial unpack)
          # y: "cuda:0 f32[2, 2]" =  y  (trivial unpack)
          (t4, t6) = nvFusion0(x, y)
            # t0 = prims.add(x, y)  # t0: "cuda:0 f32[2, 2]"
            # t1 = prims.sub(y, x)  # t1: "cuda:0 f32[2, 2]"
            # t4 = prims.add(t0, t1)  # t4: "cuda:0 f32[2, 2]"
            # t6 = prims.add(t4, 1.0)  # t6: "cuda:0 f32[2, 2]"
          del [x, y]
          return (t4, t4, t6)

    Args:
        trace:

    Returns:
        :class:`TraceCtx` with common subexpression eliminated.
    """
    start_time_ns = time.perf_counter_ns()

    cse_trace = from_trace(trace)

    cse_trace_bound_symbols = []
    rhs_to_bsym_map = {}
    redundant_map = {}

    # Identifies redundant operations and maps redundant proxies to originally
    #   computed proxies
    cse_bound_symbols = map(partial(cse_single_bsym, redundant_map, rhs_to_bsym_map), trace.bound_symbols)
    cse_trace_bound_symbols = tuple(filterfalse(lambda a: a is None, cse_bound_symbols))

    # Updates the bound symbols in the trace
    # NOTE This uses the cse_trace_bound_symbols list, which has filtered
    #   redundant symbols
    new_bsyms = replace_redundant_inputs(redundant_map, cse_trace_bound_symbols)
    cse_trace.bound_symbols = new_bsyms

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    cse_trace.set_provenance(
        TraceProvenance(f"Common Subexpression Elimination (took {elapsed_time_millis} milliseconds)")
    )
    return cse_trace


# The transform class, at different points in the processing, the methods are executed. Typically, a subset of them is implemented.
class Transform(ABC):
    def transform_traces_pre_prologue(
        self, prologue_trace: Trace, computation_trace: Trace, epilogue_trace: Trace | None, **kwargs
    ):
        """
        transform_traces_pre_prologue enables transforming prologue, computation and epilogue trace.
        Note that the computation trace here is before the autograd transform, so any update to
        the computation trace will also update backward trace.
        """
        # default to noop
        return prologue_trace, computation_trace, epilogue_trace

    def transform_module(self, model: ThunderModule) -> None:
        """Transforms the ThunderModule. This is executed once on application of the transform"""
        pass

    def transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Implement this to transform the state dict (mostly parameters and buffers) of a module, e.g. when loading
        from a state dict of the original model.

        Expected to return a state dict (for chaining or populating overrides).

        Note that state dict keys do not include the submodule name as prefix.
        """
        return state_dict

    def transform_trace_post_optimization(self, computation_trace: Trace, **kwargs):
        """
        transform_trace_post_optimization enables transforming computation trace after optimization pass.
        Note that this transform will also be applied to the backward trace if the the autograd transform was enabled.
        """
        return computation_trace

    def reverse_transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        return state_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}()"


def order_proxies(bsyms: Sequence[BoundSymbol]) -> dict[str, int]:
    """computes a canonical ordering of proxies in the bound symbols based on the order of appearance
    note that it would not cover unused inputs when applied to traces.bound_symbols
    """
    counter = 0
    proxy_order: dict[str, int] = {}  # names to order

    def process_bound_symbols(bound_symbols):
        nonlocal counter
        for bsym in bound_symbols:
            if len(bsym.subsymbols) > 0:
                process_bound_symbols(bsym.subsymbols)
            for p in tree_iter((bsym.args, bsym.kwargs, bsym.output)):  # should kwargs be sorted by name?
                if isinstance(p, thunder.Proxy) and p.name not in proxy_order:
                    counter += 1
                    proxy_order[p.name] = counter

    process_bound_symbols(bsyms)

    return proxy_order


def canonicalize_proxies(bsyms: Sequence[BoundSymbol]) -> Sequence[BoundSymbol]:
    output = []
    counter = 0

    proxymap: dict[str, thunder.Proxy] = {}

    def map_proxy(p):
        nonlocal counter
        if isinstance(p, thunder.Proxy):
            if p.name in proxymap:
                return proxymap[p.name]
            np = p.replace(name=f"p{counter}")
            counter += 1
            proxymap[p.name] = np
            return np
        return p

    def process_bound_symbols(src_bound_symbols, target_bound_symbols):
        for bsym in src_bound_symbols:
            new_subsymbols = []
            if len(bsym.subsymbols) > 0:
                process_bound_symbols(bsym.subsymbols, new_subsymbols)
            new_args = tree_map(map_proxy, bsym.args)
            new_kwargs = tree_map(map_proxy, bsym.kwargs)  # should this be sorted by key word?
            new_output = tree_map(map_proxy, bsym.output)
            new_bsym = bsym.from_bsym(output=new_output, args=new_args, kwargs=new_kwargs, subsymbols=new_subsymbols)
            target_bound_symbols.append(new_bsym)

    with thunder.core.trace.tracectx(thunder.TraceCtx()):
        process_bound_symbols(bsyms, output)

    return output


def wrap_return_value_together_with_arguments(trace: Trace) -> Trace:
    last = trace.bound_symbols[-1]
    assert last.sym.id == prims.PrimIDs.RETURN
    flat_args, _ = tree_flatten((trace.args, trace.kwargs))
    new_return_value = {"output": last.args[0], "flat_args": flat_args}
    new_return_bsym = last.from_bsym(args=(new_return_value,))

    new_trace = from_trace(trace)
    new_trace.bound_symbols = trace.bound_symbols[:-1] + [new_return_bsym]
    new_trace.set_provenance(TraceProvenance("Return arguments to track copies onto them"))
    return new_trace


def unwrap_return_value(trace: Trace) -> Trace:
    last = trace.bound_symbols[-1]
    assert last.sym.id == prims.PrimIDs.RETURN
    new_return_bsym = last.from_bsym(args=(last.args[0]["output"],))

    new_trace = from_trace(trace)
    new_trace.bound_symbols = trace.bound_symbols[:-1] + [new_return_bsym]
    new_trace.set_provenance(TraceProvenance("Unwrap the actual return value"))
    return new_trace


def remove_context_manager_prims_from_trace(trace: Trace) -> Trace:
    def is_context_manager_prim(bsym):
        # context manager prims would/should be explicitly tagged.
        if bsym.sym.tags is None:
            return False
        return prims.OpTags.CTX_MANAGER_ENTER_EXIT_OP in bsym.sym.tags

    filtered_bsyms = list(filterfalse(is_context_manager_prim, trace.bound_symbols))
    new_trace = from_trace(trace)
    new_trace.bound_symbols = filtered_bsyms
    new_trace.set_provenance(TraceProvenance("Remove context manager prims"))
    return new_trace
