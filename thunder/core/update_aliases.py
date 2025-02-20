from collections import defaultdict
from itertools import chain
from functools import reduce, partial

import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy, variableify, unvariableify
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, tracectx, TraceCtx as Trace, TraceProvenance


def _get_update_bsym(group, swap_map, new_aliases):
    aliases = tuple(unvariableify(alias) for alias in group)
    update_aliases_bsym = prims.update_aliases.bind(aliases, output=new_aliases)
    update_aliases_bsym = update_aliases_bsym.from_bsym_swap_proxies(swap_map)
    for new_alias, old_alias in zip(new_aliases, group):
        swap_map[old_alias] = new_alias
    return update_aliases_bsym, swap_map


def _get_new_alias(alias, trace):
    with tracectx(trace):
        # !! better to use the meta function?
        new_alias = TensorProxy(like=alias.proxy, requires_grad=alias.proxy.requires_grad)
    return new_alias


def _get_new_aliases(aliases, trace):
    return tuple(map(partial(_get_new_alias, trace=trace), aliases))


def _is_inplace_op(bsym):
    return bsym.sym.tags and prims.OpTags.IN_PLACE in bsym.sym.tags


def _is_view_creation_op(bsym):
    import thunder.torch as ltorch

    # !!! check on _syms_that_may_return_views?
    return bsym.sym in ltorch._syms_returning_views


def _involves_viewed_args(bsym, viewed):
    if bsym.sym.id == prims.PrimIDs.RETURN:
        return False
    return any(isinstance(p, TensorProxy) and variableify(p) in viewed for p in bsym.flat_proxy_args)


def insert_alias_updates(computation_trace: Trace) -> Trace:
    swap_map = dict()
    bsyms = []

    # First pass: identify views, their originals, and operands involved in inplace ops
    view_groups = []
    inplace_inputs = set()
    for bsym in computation_trace.bound_symbols:
        if _is_inplace_op(bsym) or _is_view_creation_op(bsym):
            # only interested in the input which is modified by the inplace op
            in_tensor = variableify(bsym.flat_proxy_args[0])
            out_tensors = set(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_outs)))
            if _is_inplace_op(bsym):
                inplace_inputs.add(in_tensor)
            for group in view_groups:
                if in_tensor in group:
                    group.update(out_tensors)
                    group.add(in_tensor)
                    break
            else:
                view_groups.append(out_tensors.union({in_tensor}))

    # filter out view groups that don't have any tensors involved in inplace ops
    view_groups = [group for group in view_groups if len(group.intersection(inplace_inputs)) != 0]
    viewed = set(reduce(set.union, view_groups, set()))
    encountered = set()
    for bsym in computation_trace.bound_symbols:
        if _is_inplace_op(bsym) or _is_view_creation_op(bsym) or _involves_viewed_args(bsym, viewed):
            in_tensors = map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_args))
            if _is_inplace_op(bsym):
                in_tensors = {list(in_tensors)[0]}
            else:
                in_tensors = set(in_tensors)
            out_tensors = set(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_outs)))
            encountered.update(in_tensors)
            group = next((g for g in view_groups if any(g.intersection(in_tensors))), None)
            if group is None:
                # this is a view creation with operands that are not involved in any inplace ops
                bsyms.append(bsym.from_bsym_swap_proxies(swap_map, skip_output=True))
                continue
            views_encountered = group.intersection(encountered)
            new_aliases = _get_new_aliases(views_encountered, computation_trace)

            update_bsym, swap_map = _get_update_bsym(views_encountered, swap_map, new_aliases)
            bsyms.append(update_bsym)
            encountered.update(out_tensors)
            new_bsym = bsym.from_bsym_swap_proxies(swap_map)
            if _is_inplace_op(bsym):
                #  This relies on these being one element sets
                swap_map[in_tensors.pop()] = unvariableify(out_tensors.pop())
            bsyms.append(new_bsym)

        else:
            bsyms.append(bsym.from_bsym_swap_proxies(swap_map))

    alias_updated_trace = from_trace(computation_trace)
    alias_updated_trace.set_provenance(TraceProvenance("Update aliases for in-place ops"))
    alias_updated_trace.bound_symbols = bsyms
    return alias_updated_trace
