from collections import defaultdict
from itertools import chain
from functools import reduce, partial

import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy, variableify, unvariableify
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, tracectx, TraceCtx as Trace, TraceProvenance

# from thunder.core.utils import ProxyDict

# # this doesn't work because if an inplace operation is performed on a view, the
# # result won't be reflected in the base.
# def _create_functional_bsym_from(inplace_bsym: BoundSymbol) -> BoundSymbol:
#     from thunder.torch import _inplace_to_out_of_place, setitem_, setitem

#     functional_sym, optional_inplace_arg_index = _inplace_to_out_of_place[inplace_bsym.sym]
#     args, kwargs = inplace_bsym.args, inplace_bsym.kwargs
#     if optional_inplace_arg_index > -1:
#         # Update `inplace` from `True` to `False`. e.g. `relu(x, inplace=True)` -> `relu(x, inplace=False)`
#         flat_args, flat_args_spec = tree_flatten((args, kwargs))
#         flat_args[optional_inplace_arg_index] = False
#         args, kwargs = tree_unflatten(flat_args, flat_args_spec)
#     functional_output = inplace_bsym.output
#     if inplace_bsym.sym is setitem_:
#         # setitem does not return a value, take the output of the setitem subsymbol
#         assert inplace_bsym.subsymbols[0].sym is setitem
#         functional_output = inplace_bsym.subsymbols[0].output
#     functional_bsym = functional_sym.bind(
#         *args,
#         **kwargs,
#         output=functional_output,
#         subsymbols=inplace_bsym.subsymbols,
#         _call_ctx=inplace_bsym._call_ctx,
#     )
#     if functional_bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_:
#         last_ssym_out = functional_bsym.subsymbols[-1].flat_proxy_outs[0]
#         functional_bsym.subsymbols = functional_bsym.subsymbols[:-1]
#         penultimate_ssym = functional_bsym.subsymbols[-1]
#         new_last_ssym = penultimate_ssym.from_bsym_swap_proxies({variableify(penultimate_ssym.flat_proxy_outs[0]): last_ssym_out})
#         functional_bsym.subsymbols[-1] = new_last_ssym
#     if len(functional_bsym.subsymbols) == 1 and functional_bsym.rhs == functional_bsym.subsymbols[0].rhs:
#         functional_bsym.subsymbols = functional_bsym.subsymbols[0].subsymbols
#     return functional_bsym

# def _maybe_create_functional_bsym_from(bsym):
#     if _is_inplace_op(bsym) and bsym.sym.id != 'setitem_':
#         return _create_functional_bsym_from(bsym)
#     else:
#         return bsym


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
    return (bsym.sym.tags and prims.OpTags.IN_PLACE in bsym.sym.tags) or (
        bsym.subsymbols and bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_
    )


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
                out_tensors = set()
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
    # trace_args_set = ProxyDict()
    # for a in filter(
    #     lambda a: isinstance(a, TensorProxy), tree_flatten((computation_trace.args, computation_trace.kwargs))[0]
    # ):
    #     trace_args_set[a] = a

    for bsym in computation_trace.bound_symbols:
        if _is_inplace_op(bsym) or _is_view_creation_op(bsym) or _involves_viewed_args(bsym, viewed):
            in_tensors = map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_args))
            if _is_inplace_op(bsym):
                in_tensor = list(in_tensors)[0]
                in_tensors = {in_tensor}
                # if unvariableify(in_tensor) in trace_args_set:
                #     # this is an inplace op on an input to the computation, and we don't want to update the aliases
                #     # for this input, so we skip it to keep nvfuser happy.
                #     encountered.add(in_tensor)
                #     bsyms.append(bsym.from_bsym_swap_proxies(swap_map, skip_output=True))
                #     continue
            else:
                in_tensors = set(in_tensors)
            out_tensors = set(map(variableify, filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_outs)))
            encountered.update(in_tensors)
            group = set(reduce(set.union, filter(lambda g: any(g.intersection(in_tensors)), view_groups), set()))
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
            # new_bsym = _maybe_create_functional_bsym_from(new_bsym)
            bsyms.append(new_bsym)
            if _is_inplace_op(bsym) and len(out_tensors) == 1:
                #  This relies on these being one element sets (ltorch.setitem_ yields no outs).
                swap_map[in_tensors.pop()] = unvariableify(out_tensors.pop())

        else:
            bsyms.append(bsym.from_bsym_swap_proxies(swap_map))

    alias_updated_trace = from_trace(computation_trace)
    alias_updated_trace.set_provenance(TraceProvenance("Update aliases for in-place ops"))
    alias_updated_trace.bound_symbols = bsyms
    return alias_updated_trace
