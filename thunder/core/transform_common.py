import time
from typing import Any, Dict
from collections.abc import Sequence
from itertools import filterfalse
from functools import partial

import thunder.core.prims as prims
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.proxies import Proxy, variableify, Variable
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.symbol import BoundSymbol, BoundSymbolRHS, has_tags
from thunder.core.trace import from_trace, TraceProvenance, TraceCtx as Trace
from thunder.core.utils import ProxyDict, producers, check


#
# Common optimization and transform passes
#
# NOTE This file avoids transforms depending on passes, since passes depend on transforms


# Modifies an existing BoundSymbol, removing its "no-op" subsymbols (recursively) which perform no operations
def _remove_noop_subsymbols(bsym: BoundSymbol) -> None:
    nsbsyms: list[BoundSymbol] = []
    sbsym: BoundSymbol
    for sbsym in bsym.subsymbols:
        if len(sbsym.subsymbols) == 0 and not sbsym.sym.is_prim:
            continue

        _remove_noop_subsymbols(sbsym)
        nsbsyms.append(sbsym)

    bsym.subsymbols = nsbsyms


# TODO This calls variableify(), but we could directly construct Variable objects instead, which might slightly
#   improve performance
# Runs a Dead Code Elimination (DCE) pass
# NOTE Today we are only interested in computations that produce proxies, so this will eliminate operations
#   that only produce non-proxy objects
def dce(trace: Trace) -> Trace:
    start_time_ns = time.time_ns()

    producer_map: ProxyDict = producers(trace)

    flat_trace_outputs, _ = tree_flatten(trace.output)
    needed_proxies: set[Variable] = set(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
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
    dcetrace.bound_symbols = list(reversed(dced))

    end_time_ns = time.time_ns()
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
        skip_subsymbols=True,
    )

    # Skip appending this bsym to the new bound symbols due to its rhs being a common subexpression.
    rhs = new_bsym.rhs()
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
    start_time_ns = time.time_ns()

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

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    cse_trace.set_provenance(
        TraceProvenance(f"Common Subexpression Elimination (took {elapsed_time_millis} milliseconds)")
    )
    return cse_trace
