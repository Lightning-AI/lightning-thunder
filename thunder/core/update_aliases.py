from functools import reduce, partial

import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy, variableify, unvariableify
from thunder.core.pytree import tree_flatten
from thunder.core.symbol import BoundSymbol, BoundSymbolTag, has_tags
from thunder.core.trace import from_trace, tracectx, TraceCtx as Trace, TraceProvenance, VariableInterface
from thunder.torch import setitem_


def _update_swap_map(swap_map, old_alias, new_alias):
    visited = set()
    while old_alias in swap_map:
        if old_alias in visited:
            raise ValueError(f"Cycle detected in swap map for aliases updates: {old_alias}")
        visited.add(old_alias)
        old_alias = variableify(swap_map[old_alias])
    swap_map[old_alias] = new_alias
    return swap_map


def _get_update_bsym(group, swap_map, new_aliases):
    aliases = tuple(unvariableify(alias) for alias in group)
    update_aliases_bsym = prims.update_aliases.bind(aliases, output=new_aliases)
    update_aliases_bsym = update_aliases_bsym.from_bsym_swap_proxies(swap_map)
    for new_alias, old_alias in zip(new_aliases, group):
        swap_map = _update_swap_map(swap_map, old_alias, new_alias)
    return update_aliases_bsym, swap_map


def _get_new_alias(alias, trace):
    with tracectx(trace):
        new_alias = TensorProxy(like=alias.proxy, requires_grad=alias.proxy.requires_grad)
    return new_alias


def _get_new_aliases(aliases, trace):
    return tuple(map(partial(_get_new_alias, trace=trace), aliases))


def _is_inplace_op(bsym):
    return (bsym.sym.tags and prims.OpTags.IN_PLACE in bsym.sym.tags) or (
        bsym.subsymbols and bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_
    )


def _is_view_creation_op(bsym):
    import thunder.torch as ltorch

    return bsym.sym in ltorch._syms_returning_views or bsym.sym in ltorch._syms_that_may_return_views


def _involves_viewed_args(bsym, viewed):
    if bsym.sym.id == prims.PrimIDs.RETURN:
        return False
    return any(isinstance(p, TensorProxy) and variableify(p) in viewed for p in bsym.flat_proxy_args)


def replace_args_with_alias_map(
    computation_trace: Trace,
    alias_tensor_indices: list[list[int]],
) -> tuple[Trace, dict[VariableInterface, TensorProxy]]:
    if not alias_tensor_indices:
        return computation_trace, {}
    bsyms: list[BoundSymbol] = []
    flat_args, _ = tree_flatten((computation_trace.args, computation_trace.kwargs))
    swap_map_for_aliases: dict[VariableInterface, TensorProxy] = {}
    arg_to_optional_bsyms: dict[VariableInterface, BoundSymbol] = {}
    for indices in alias_tensor_indices:
        arg = flat_args[indices[0]]
        for idx in filter(lambda idx: idx < len(flat_args), indices[1:]):
            arg_to_replace = flat_args[idx]
            reshaped_arg = arg
            if arg_to_replace.shape != arg.shape:
                with tracectx(computation_trace):
                    reshaped_arg = prims.reshape.meta(arg, arg_to_replace.shape)
                    arg_to_optional_bsyms[variableify(arg_to_replace)] = prims.reshape.bind(
                        arg,
                        arg_to_replace.shape,
                        output=reshaped_arg,
                    )
            swap_map_for_aliases[variableify(arg_to_replace)] = reshaped_arg
    appended_bsyms = {}
    for bsym in computation_trace.bound_symbols:
        for arg in filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args):
            reshape_bsym = arg_to_optional_bsyms.get(variableify(arg))
            if reshape_bsym is not None:
                if reshape_bsym not in appended_bsyms:
                    bsyms.append(reshape_bsym)
                    appended_bsyms[reshape_bsym] = arg
        if replaced_args_map := {
            x.name: swap_map_for_aliases[variableify(x)].name
            for x in filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args)
            if variableify(x) in swap_map_for_aliases
        }:
            bsyms.append(bsym.from_bsym_swap_proxies(swap_map_for_aliases, skip_output=True))
            if len(replaced_args_map) == 1:
                bsyms[
                    -1
                ].header = f"[alias tensor args] `{list(replaced_args_map.keys())[0]}` is replaced by `{list(replaced_args_map.values())[0]}`"
            else:
                bsyms[
                    -1
                ].header = f"[alias tensor args] {list(replaced_args_map.keys())} are replaced by {list(replaced_args_map.values())}, respectively"
        else:
            bsyms.append(bsym)
    no_implicit_alias_trace = from_trace(computation_trace)
    no_implicit_alias_trace.bound_symbols = bsyms
    str_map = {unvariableify(k).name: v.name for k, v in swap_map_for_aliases.items()}
    no_implicit_alias_trace.set_provenance(TraceProvenance(f"Duplicate alias args using {str_map}"))
    return no_implicit_alias_trace, swap_map_for_aliases


def insert_alias_updates(computation_trace: Trace, alias_tensor_indices: list[list[int]]) -> Trace:
    if not any(_is_inplace_op(bsym) for bsym in computation_trace.bound_symbols):
        return computation_trace

    swap_map = dict()
    bsyms = []

    # First pass: identify inputs which are views of each other and swap them out with a default,
    # reshaping if necessary.
    computation_trace, _ = replace_args_with_alias_map(computation_trace, alias_tensor_indices)

    # Second pass: identify views, their originals, and operands involved in inplace ops
    view_groups = []
    inplace_inputs = set()
    for bsym in computation_trace.bound_symbols:
        if _is_inplace_op(bsym) or _is_view_creation_op(bsym):
            # only interested in the input which is modified by the inplace op
            in_tensor = variableify(bsym.flat_proxy_args[0])
            out_tensors = set(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_outs)))
            if _is_inplace_op(bsym):
                inplace_inputs.add(in_tensor)
                out_tensors = set()
            for group in view_groups:
                if in_tensor in group:
                    group.update(out_tensors)
                    break
            else:
                view_groups.append(out_tensors.union({in_tensor}))

    # filter out view groups that don't have any tensors involved in inplace ops
    view_groups = [group for group in view_groups if len(group.intersection(inplace_inputs)) != 0]
    viewed = set(reduce(set.union, view_groups, set()))
    encountered = set()

    # Third pass: insert alias updates
    for bsym in computation_trace.bound_symbols:
        if _is_inplace_op(bsym) or _is_view_creation_op(bsym) or _involves_viewed_args(bsym, viewed):
            in_tensors = list(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_args)))
            if _is_inplace_op(bsym) and in_tensors:
                in_tensors = {in_tensors[0]}
            else:
                in_tensors = set(in_tensors)
            out_tensors = set(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_outs)))
            encountered.update(in_tensors)
            group = set(reduce(set.union, filter(lambda g: any(g.intersection(in_tensors)), view_groups), set()))
            if not group or not (views_encountered := group.intersection(encountered)):
                # If group is empty, this is a view creation with operands that are not involved in any inplace ops.
                bsyms.append(bsym.from_bsym_swap_proxies(swap_map, skip_output=True))
                continue

            new_aliases = _get_new_aliases(views_encountered, computation_trace)

            update_bsym, swap_map = _get_update_bsym(views_encountered, swap_map, new_aliases)
            new_bsym = bsym.from_bsym_swap_proxies(swap_map)
            if has_tags(bsym, {BoundSymbolTag.BACKWARD}):
                update_bsym.tags.add(BoundSymbolTag.BACKWARD)
            bsyms.append(update_bsym)
            encountered.update(out_tensors)
            bsyms.append(new_bsym)
            if _is_inplace_op(bsym) and len(out_tensors) == 1 and len(in_tensors) == 1:
                #  This relies on these being one element sets (ltorch.setitem_ yields no outs).
                swap_map = _update_swap_map(swap_map, in_tensors.pop(), unvariableify(out_tensors.pop()))

        else:
            bsyms.append(bsym.from_bsym_swap_proxies(swap_map))

    alias_updated_trace = from_trace(computation_trace)
    alias_updated_trace.set_provenance(TraceProvenance("Update aliases for in-place ops"))
    alias_updated_trace.bound_symbols = bsyms
    return alias_updated_trace
