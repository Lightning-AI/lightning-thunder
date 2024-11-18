from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

import thunder.core.prims as prims
from thunder.core.proxies import variableify, TensorProxy, unvariableify, ProxyInterface
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceProvenance, TraceCtx as Trace, tracectx
from thunder.core.utils import ProxyDict, producers, check, consumers

if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface


__all__ = [
    "check_inplace_to_views",
    "functionalize_inplace_ops",
]


# For `ltorch.to`, we can AOT tell whether or not the return value of the op is a new tensor or self.
def bsym_of_to_return_self(bsym: BoundSymbol):
    import thunder.torch as ltorch

    a, tensor_dtype_or_device, optional_positional_dtype = bsym.args
    device = bsym.kwargs.get("device")
    dtype = bsym.kwargs.get("dtype")
    copy = bsym.kwargs.get("copy")
    memory_format = bsym.kwargs.get("memory_format")
    device, dtype = ltorch._parse_to_device_and_dtype(
        tensor_dtype_or_device,
        optional_positional_dtype,
        device=device,
        dtype=dtype,
    )
    input_device, input_dtype = a.device, a.dtype
    result_is_self = ltorch._will_to_return_self(input_device, input_dtype, device, dtype, memory_format, copy)
    return result_is_self


def check_inplace_to_views(computation_trace: Trace) -> dict[VariableInterface, TensorProxy]:
    """Error out if in-place op that outputs of different number of elements from the input and the input has other consumers."""
    import thunder.torch as ltorch

    producer_bsyms = producers(computation_trace)
    trace_args_set = ProxyDict()
    for a in filter(
        lambda a: isinstance(a, TensorProxy), tree_flatten((computation_trace.args, computation_trace.kwargs))[0]
    ):
        trace_args_set[a] = a

    # note(crcrpar): Why not using :func:`~thunder.core.symbol.has_tags`?
    # Because it looks into `.sym.tags` of the input bsym and its subsymbols,
    # thus even `ltorch.batch_norm` is regarded as `prims.OpTags.IN_PLACE`.
    def has_tag(bsym: BoundSymbol, tag: prims.OpTags) -> bool:
        return bsym.sym.tags and tag in bsym.sym.tags

    swap_map: dict[VariableInterface, TensorProxy] = {}
    consumer_map = consumers(computation_trace)
    bsym: BoundSymbol
    for bsym in filter(lambda b: has_tag(b, prims.OpTags.IN_PLACE), computation_trace.bound_symbols):
        in_tensor: TensorProxy = list(filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_args))[0]

        if in_tensor in trace_args_set:
            continue
        prod_bsym: BoundSymbol = producer_bsyms[in_tensor]

        flat_tensor_proxy_args = tuple(filter(lambda p: isinstance(p, TensorProxy), prod_bsym.flat_args))
        if not flat_tensor_proxy_args:
            # assuming `prod_bsym` is a tensor factory method such as `torch.empty`, `torch.zeros`, and `torch.ones`
            continue
        orig_tensor = flat_tensor_proxy_args[0]
        consumer_of_orig_tensor = consumer_map[orig_tensor]
        # When the orig tensor is not used by consumers other than `prod_bsym`, it'd be safe.
        # Otherwise, we'd need to replace the use of ``orig_tensor`` with a view, unless the original
        # is an arg or a kwarg.
        if len(consumer_of_orig_tensor) == 1:
            continue

        if prod_bsym.sym in ltorch._syms_that_may_return_views:
            if prod_bsym.sym == ltorch.to:
                if not bsym_of_to_return_self(prod_bsym):
                    continue
            else:
                check(
                    prod_bsym.sym not in ltorch._syms_returning_views,
                    lambda: (
                        f"in-place op of `{bsym.sym.id}` to `{prod_bsym.sym.id}` output `{in_tensor}` is not "
                        f"supported. It's unclear if the output of "
                        f"{tuple(s.id for s in ltorch._syms_that_may_return_views)} is "
                        f"a copy, a view, or the input itself, as per https://pytorch.org/docs/stable/tensor_view.html"
                    ),
                    NotImplementedError,
                )
        if prod_bsym.sym not in ltorch._syms_returning_views:
            continue

        check(
            orig_tensor.numel == in_tensor.numel,
            lambda: (
                f"in-place op of `{bsym.sym.id}` to `{in_tensor}`, a view tensor of "
                f"`{orig_tensor}` is not supported because {in_tensor.numel} != {orig_tensor.numel}"
            ),
            NotImplementedError,
        )

        swap_map[variableify(orig_tensor)] = in_tensor
    return swap_map


def is_functionalizable(bsym: BoundSymbol) -> bool:
    """Has `OpTags.IN_PLACE` and its args are NOT ``computation_trace.args`` nor ``computation_trace.kwargs``."""
    from thunder.torch import _inplace_to_out_of_place

    return (
        bsym.sym in _inplace_to_out_of_place and bsym.subsymbols and bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_
    )


def _get_first_tensor_arg(bsym: BoundSymbol) -> TensorProxy | None:
    tensor_args = list(filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args))
    if not tensor_args:
        return None
    return tensor_args[0]


def _get_prod_bsym_with_arg(
    t: TensorProxy,
    producer_map: ProxyDict,
    trace_args: ProxyDict,
    *,
    orig_bsym_of_in_tensor: BoundSymbol,
    in_tensor: TensorProxy,
) -> BoundSymbol | None:
    from thunder.torch import _syms_returning_views
    from thunder.torch import _syms_that_may_return_views

    def inplace_or_view(bsym) -> bool:
        import thunder.torch as ltorch

        sym = bsym.sym
        if sym == ltorch.to:
            return bsym_of_to_return_self(bsym)
        check(
            sym not in _syms_that_may_return_views,
            lambda: (
                f"in-place op of `{orig_bsym_of_in_tensor.sym.id}` to `{bsym.sym.id}` output `{in_tensor.name}` is not "
                f"supported. It's unclear if `{in_tensor.name}`, the output of "
                f"{tuple(s.id for s in _syms_that_may_return_views)} is "
                "a copy, a view, or the input itself, as per https://pytorch.org/docs/stable/tensor_view.html\n"
                "Please use `torch.view` to create a view. Cloning the reshaped tensor before the in-place op is not currently supported.\n"
                "This error can be skipped with `skip_inplace_functionalization=True` passed to `thunder.jit`.\n"
                "If you believe this is a bug, please report it to the Lightning Thunder team in "
                "https://github.com/Lightning-AI/lightning-thunder/issues/957."
            ),
            NotImplementedError,
        )
        return (sym.tags and prims.OpTags.IN_PLACE in sym.tags) or sym in _syms_returning_views

    if t not in producer_map:
        return None
    prod_bsym: BoundSymbol = producer_map[t]
    first_tensor = _get_first_tensor_arg(prod_bsym)
    if first_tensor is None:
        return None

    if inplace_or_view(prod_bsym):
        if first_tensor in trace_args:
            return prod_bsym
        else:
            return _get_prod_bsym_with_arg(
                first_tensor,
                producer_map,
                trace_args,
                orig_bsym_of_in_tensor=orig_bsym_of_in_tensor,
                in_tensor=in_tensor,
            )
    else:
        return None


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
                bsyms[-1].header = (
                    f"[alias tensor args] `{list(replaced_args_map.keys())[0]}` is replaced by `{list(replaced_args_map.values())[0]}`"
                )
            else:
                bsyms[-1].header = (
                    f"[alias tensor args] {list(replaced_args_map.keys())} are replaced by {list(replaced_args_map.values())}, respectively"
                )
        else:
            bsyms.append(bsym)
    no_implicit_alias_trace = from_trace(computation_trace)
    no_implicit_alias_trace.bound_symbols = bsyms
    str_map = {unvariableify(k).name: v.name for k, v in swap_map_for_aliases.items()}
    no_implicit_alias_trace.set_provenance(TraceProvenance(f"Duplicate alias args using {str_map}"))
    return no_implicit_alias_trace, swap_map_for_aliases


def canonicalize_bsym_args(
    computation_trace: Trace,
    orig_to_view_swap_map: dict[VariableInterface, TensorProxy],
) -> tuple[Trace, dict[BoundSymbol, dict[VariableInterface, TensorProxy]]]:
    """Avoid ``TensorProxy`` being consumed by more than one mathematical ops and/or distributed comms."""
    bsym: BoundSymbol
    swap_map: dict[VariableInterface, TensorProxy] = {}
    reverse_swap_map: dict[BoundSymbol, dict[VariableInterface, TensorProxy]] = {}
    bsyms: list[BoundSymbol] = []
    tensors_observed: set[VariableInterface] = set()
    for bsym in computation_trace.bound_symbols:
        # update args/kwargs so that any tensors are not consumed by multiple math / comm ops.
        new_bsym = bsym.from_bsym_swap_proxies(swap_map, skip_output=True)
        if swap_map:
            if replaced_args_map := {
                variableify(x): swap_map[variableify(x)]
                for x in filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args)
                if variableify(x) in swap_map
            }:
                keys = [unvariableify(k).name for k in replaced_args_map.keys()]
                values = [v.name for v in replaced_args_map.values()]
                if len(keys) == 1:
                    new_bsym.header = f"`{keys[0]}` is replaced by `{values[0]}`"
                else:
                    new_bsym.header = f"{keys} are replaced by {values}, respectively"
                if bsym.header.startswith("[alias tensor args]"):
                    reverse_swap_map[new_bsym] = {
                        variableify(v): unvariableify(k) for k, v in replaced_args_map.items()
                    }

        cur_orig_to_view_swap_map: dict[VariableInterface, TensorProxy] = {}
        for t in filter(lambda p: isinstance(p, TensorProxy), new_bsym.flat_args):
            if (var_t := variableify(t)) not in tensors_observed:
                tensors_observed.add(var_t)
            else:
                if var_t in orig_to_view_swap_map:
                    var_view_t = variableify(orig_to_view_swap_map[var_t])
                    check(var_view_t in swap_map, lambda: f"{var_view_t} not in {swap_map}, {orig_to_view_swap_map = }")
                    cur_orig_to_view_swap_map[var_t] = swap_map[var_view_t]
        if cur_orig_to_view_swap_map:
            with tracectx(computation_trace):
                for var_orig, view in cur_orig_to_view_swap_map.items():
                    view_of_orig_shape = prims.reshape.meta(view, unvariableify(var_orig).shape)
                    reshape_bsym = prims.reshape.bind(view, unvariableify(var_orig).shape, output=view_of_orig_shape)
                    cur_orig_to_view_swap_map[var_orig] = view_of_orig_shape
                    bsyms.append(reshape_bsym)
            new_bsym = new_bsym.from_bsym_swap_proxies(cur_orig_to_view_swap_map, skip_output=True)

        # non in-place bsyms would only need to update its outputs.
        # but in-place bsyms would, as it needs to make sure they return `prims.copy_`'s return value.
        if not is_functionalizable(new_bsym):
            bsyms.append(new_bsym)
        else:
            copy_bsym = bsym.subsymbols[-1]
            copy_out = copy_bsym.flat_proxy_outs[0]
            copy_dst = copy_bsym.flat_proxy_args[1]
            swap_map[variableify(copy_dst)] = copy_out
            # make sure an in-place bsym returns `prims.copy_` output
            new_bsym = new_bsym.from_bsym_swap_proxies(swap_map, skip_inputs=True, skip_subsymbols=True)
            bsyms.append(new_bsym)
        if cur_orig_to_view_swap_map:
            bsyms[-1].header = (
                f"Replace {[unvariableify(k) for k in cur_orig_to_view_swap_map]} with {[list(cur_orig_to_view_swap_map.values())]}"
            )

    intermediate_trace = from_trace(computation_trace)
    intermediate_trace.bound_symbols = bsyms
    intermediate_trace.set_provenance(TraceProvenance("Intermediate trace of `functionalize_inplace_ops`"))

    intermediate_trace.bound_symbols[-1] = intermediate_trace.bound_symbols[-1].from_bsym_swap_proxies(swap_map)
    return_bsym = intermediate_trace.bound_symbols[-1]
    for t in filter(lambda p: isinstance(p, TensorProxy), return_bsym.flat_args):
        check(
            (var_t := variableify(t)) not in swap_map,
            lambda: f"{return_bsym.flat_args=}. `{t}` should have been replaced by `{swap_map[var_t]}`",
        )
    return intermediate_trace, reverse_swap_map


def create_functional_bsym_from(inplace_bsym: BoundSymbol) -> BoundSymbol:
    from thunder.torch import _inplace_to_out_of_place, setitem_, setitem

    functional_sym, optional_inplace_arg_index = _inplace_to_out_of_place[inplace_bsym.sym]
    args, kwargs = inplace_bsym.args, inplace_bsym.kwargs
    if optional_inplace_arg_index > -1:
        # Update `inplace` from `True` to `False`. e.g. `relu(x, inplace=True)` -> `relu(x, inplace=False)`
        flat_args, flat_args_spec = tree_flatten((args, kwargs))
        flat_args[optional_inplace_arg_index] = False
        args, kwargs = tree_unflatten(flat_args, flat_args_spec)
    functional_output = inplace_bsym.output
    if inplace_bsym.sym is setitem_:
        # setitem does not return a value, take the output of the setitem subsymbol
        assert inplace_bsym.subsymbols[0].sym is setitem
        functional_output = inplace_bsym.subsymbols[0].output
    functional_bsym = functional_sym.bind(
        *args,
        **kwargs,
        output=functional_output,
        subsymbols=inplace_bsym.subsymbols,
        _call_ctx=inplace_bsym._call_ctx,
    )
    if functional_bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_:
        functional_bsym.subsymbols = functional_bsym.subsymbols[:-1]
    if len(functional_bsym.subsymbols) == 1 and functional_bsym.rhs == functional_bsym.subsymbols[0].rhs:
        functional_bsym.subsymbols = functional_bsym.subsymbols[0].subsymbols
    return functional_bsym


def needs_replace_of_sibling_views(
    var_copy_to: VariableInterface,
    bsym: BoundSymbol,
    reverse_swap_map_for_canonicalization: dict[BoundSymbol, dict[VariableInterface, TensorProxy]],
) -> dict[VariableInterface, TensorProxy]:
    if (
        (swap_map_from_canonicalization := reverse_swap_map_for_canonicalization.get(bsym, None))
        and swap_map_from_canonicalization is not None
        and var_copy_to in swap_map_from_canonicalization
    ):
        return swap_map_from_canonicalization
    else:
        return {}


def apply_functionalization_to_canonicalized_trace(
    canonicalized_trace: Trace,
    reverse_swap_map_for_canonicalization: dict[BoundSymbol, dict[VariableInterface, TensorProxy]],
) -> Trace:
    from thunder.torch import _syms_returning_views

    def _check_numel(src: TensorProxy, dst: TensorProxy) -> None:
        check(
            src.numel == dst.numel,
            lambda: (
                f"Fail to propagate the in-place change of `{src.name}` to `{dst.name}` "
                f"because of the different number of elements: {src.numel} and {dst.numel}"
            ),
        )

    def _reshape_bsym_ctor(src: TensorProxy, dst: TensorProxy, trace: Trace) -> tuple[BoundSymbol | None, TensorProxy]:
        target = dst
        if src.shape == dst.shape:
            return None, target
        with tracectx(trace):
            target = prims.reshape.meta(src, dst.shape)
            reshape_bsym = prims.reshape.bind(src, dst.shape, output=target)
            reshape_bsym.header = f"`{new_t.name}` replaces `{dst.name}` due to in-place op into `{dst.name}`"
        return reshape_bsym, target

    swap_map: dict[VariableInterface, TensorProxy] = {}
    flat_args = tuple(
        filter(
            lambda p: isinstance(p, TensorProxy),
            tree_flatten((canonicalized_trace.args, canonicalized_trace.kwargs))[0],
        )
    )
    arg_to_copy_bsyms = ProxyDict()
    for a in flat_args:
        arg_to_copy_bsyms[a] = None
    bsym_to_idx = {bsym: i for i, bsym in enumerate(canonicalized_trace.bound_symbols)}
    consumer_map_of_intermediate_trace = consumers(canonicalized_trace)
    producer_map_of_intermediate_trace = producers(canonicalized_trace)
    copy_from_to_copy_bsyms: dict[VariableInterface, list[BoundSymbol]] = {}
    base_to_views: dict[VariableInterface, list[TensorProxy]] = defaultdict(list)
    view_to_base: dict[VariableInterface, TensorProxy] = {}
    bsym_to_trigger_inplace_propagation: dict[BoundSymbol, tuple[TensorProxy, ...]] = {}

    arg_to_copy_froms: dict[VariableInterface, list[TensorProxy]] = {}
    swap_map_for_tensor_alias_child: dict[VariableInterface, TensorProxy] = {}

    new_bsyms: list[BoundSymbol] = []
    for idx, bsym in enumerate(canonicalized_trace.bound_symbols):
        # `new_bsym` is new in the sense that its args/kwargs do not use a tensor proxy
        # returned from `prims.copy_`, at this point.
        new_bsym = bsym.from_bsym_swap_proxies(swap_map, skip_output=True)

        if new_bsym.sym in _syms_returning_views:
            views = list(filter(lambda p: isinstance(p, TensorProxy), new_bsym.flat_outs))
            base = _get_first_tensor_arg(new_bsym)
            var_base = variableify(base)
            if (orig_base := view_to_base.get(var_base, None)) is not None:
                base = orig_base
                var_base = variableify(base)
            for v in views:
                view_to_base[variableify(v)] = base
            base_to_views[var_base].extend(views)

        if not is_functionalizable(new_bsym):
            new_bsyms.append(new_bsym)
            if replaced_args_map := {
                variableify(x): swap_map[variableify(x)]
                for x in filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args)
                if variableify(x) in swap_map
            }:
                keys = [unvariableify(k).name for k in replaced_args_map.keys()]
                values = [v.name for v in replaced_args_map.values()]
                if len(keys) == 1:
                    new_bsym.header = f"`{keys[0]}` is replaced by `{values[0]}`"
                else:
                    new_bsym.header = f"{keys} are replaced by {values}, respectively"
            continue

        # If `bsym` is functionalizable, i.e., its last subsymbol is `prims.copy_`,
        # this transform creates a semantically equivalent functional bsym that's different from
        # `new_bsym` / `bsym` in
        #     - does not have `prims.copy_` as one of its subsymbols
        #     - Output tensor is `copy_from` i.e., the output of last subsymbol

        # We use `bsym.subsymbols[-1]` instead of `new_bsym.subsymbols[-1]` because the latter
        # would be modified using `swap_map`, i.e., the signature could be broken.
        copy_bsym = bsym.subsymbols[-1]
        copy_return = copy_bsym.flat_proxy_outs[0]
        copy_from = copy_bsym.flat_proxy_args[0]
        copy_to = copy_bsym.flat_proxy_args[1]
        swap_map[variableify(copy_return)] = copy_from

        # The last subsymbol is `prims.copy_`, so the new_bsym shouldn't have it.
        new_bsym = new_bsym.from_bsym_swap_proxies(swap_map)
        functional_bsym = create_functional_bsym_from(new_bsym)
        out = ",".join([a.name if isinstance(a, ProxyInterface) else str(a) for a in bsym.flat_outs])
        args = ",".join([a.name if isinstance(a, ProxyInterface) else str(a) for a in bsym.flat_args])
        functional_bsym.header = f"Functionalized from `{out} = {bsym.sym.name}({args})`"
        new_bsyms.append(functional_bsym)

        # If trace's arguments and/or their views are consumed by an in-place op,
        # we'd have to have pairs of `prims.copy_` and auxiliary `prims.reshape` in the functionalized trace
        # in order to preserve the semantics of the original trace.
        #
        # If the modified operand is a function input, we just reuse the copy bsym removed above. -- `if`
        # If the modified operand is a view of a function input, we'd need to create a new copy bsym.
        # We might have to create an auxiliary reshape bsym as well
        # if the shape of the source is different from the function input tensor. -- `elif`
        #
        # On top of that, when the operand is either an intermediate tensor which has one or more views
        # or a view whose base has other view tensors (sibling views), we'd need to propagate
        # the value to them in any way. Here, we realize that by replacing each of them
        # with the functionalized bsym's output after a reshape is applied. -- `else`
        arg_copy_dst: TensorProxy
        copy_bsyms: list[BoundSymbol] = []
        if copy_to in arg_to_copy_bsyms:
            copy_bsyms.append(copy_bsym)
            arg_copy_dst = copy_to
        elif (
            optional_prod_bsym := _get_prod_bsym_with_arg(
                copy_to,
                producer_map_of_intermediate_trace,
                arg_to_copy_bsyms,
                orig_bsym_of_in_tensor=new_bsym,
                in_tensor=copy_to,
            )
        ) is not None:
            new_copy_to = _get_first_tensor_arg(optional_prod_bsym)
            arg_copy_dst = new_copy_to

            with tracectx(canonicalized_trace):
                copy_from_for_new_copy: TensorProxy
                if copy_from.shape != new_copy_to.shape:
                    dst_shape = new_copy_to.shape
                    reshaped_copy_from = prims.reshape.meta(copy_from, dst_shape)
                    reshape_bsym = prims.reshape.bind(copy_from, dst_shape, output=reshaped_copy_from)
                    copy_bsyms.append(reshape_bsym)

                    copy_from_for_new_copy = reshaped_copy_from
                else:
                    copy_from_for_new_copy = copy_from
                new_copy_return = prims.copy_.meta(copy_from_for_new_copy, new_copy_to)
                new_copy_bsym = prims.copy_.bind(copy_from_for_new_copy, new_copy_to, output=new_copy_return)
                copy_bsyms.append(new_copy_bsym)
        else:
            var_copy_to = variableify(copy_to)
            if var_copy_to in base_to_views or var_copy_to in view_to_base:
                base: TensorProxy
                views_to_replace: list[TensorProxy]
                if var_copy_to in base_to_views:
                    base = unvariableify(var_copy_to)
                    views_to_replace = base_to_views[var_copy_to]
                else:
                    base = view_to_base[var_copy_to]
                    views = base_to_views[variableify(base)]
                    views_to_replace = list(filter(lambda t: variableify(t) != var_copy_to, views))
                if [bsym for bsym in consumer_map_of_intermediate_trace[base] if bsym_to_idx[bsym] > idx]:
                    _check_numel(copy_from, base)
                    optional_reshape_bsym, new_t = _reshape_bsym_ctor(copy_from, base, canonicalized_trace)
                    if optional_reshape_bsym is not None:
                        new_bsyms.append(optional_reshape_bsym)
                    swap_map[variableify(base)] = new_t
                for v in views_to_replace:
                    if v in consumer_map_of_intermediate_trace and [
                        bsym for bsym in consumer_map_of_intermediate_trace[v] if bsym_to_idx[bsym] > idx
                    ]:
                        _check_numel(copy_from, v)
                        new_t: TensorProxy = copy_from
                        optional_reshape_bsym, new_t = _reshape_bsym_ctor(copy_from, v, canonicalized_trace)
                        if optional_reshape_bsym is not None:
                            new_bsyms.append(optional_reshape_bsym)
                        swap_map[variableify(v)] = new_t
                bsym_to_trigger_inplace_propagation[functional_bsym] = (copy_to, base, views_to_replace)

        if copy_bsyms:
            if arg_copy_dst in arg_to_copy_bsyms and (value := arg_to_copy_bsyms[arg_copy_dst]) is not None:
                prev_functional_bsym, prev_copy_bsyms = value
                prev_copy_from = _get_first_tensor_arg(prev_copy_bsyms[0])
                del copy_from_to_copy_bsyms[variableify(prev_copy_from)]
            arg_to_copy_bsyms[arg_copy_dst] = functional_bsym, copy_bsyms
            copy_from_for_copy_bsyms = _get_first_tensor_arg(copy_bsyms[0])
            copy_from_to_copy_bsyms[variableify(copy_from_for_copy_bsyms)] = copy_bsyms

            # Handling of in-place ops to alias tensor args.
            if (var_arg_copy_dst := variableify(arg_copy_dst)) not in arg_to_copy_froms:
                arg_to_copy_froms[var_arg_copy_dst] = [copy_from]
            else:
                arg_to_copy_froms[var_arg_copy_dst].append(copy_from)
                for v in arg_to_copy_froms[var_arg_copy_dst]:
                    if v.name == copy_from.name:
                        continue
                    _check_numel(copy_from, v)
                    reshaped_copy_from_for_aliases = copy_from
                    optional_reshape_bsym, new_dst = _reshape_bsym_ctor(copy_from, v, canonicalized_trace)
                    if optional_reshape_bsym is not None:
                        reshaped_copy_from_for_aliases = new_dst
                        new_bsyms.append(optional_reshape_bsym)
                    swap_map_for_tensor_alias_child[variableify(v)] = reshaped_copy_from_for_aliases
                    swap_map[variableify(v)] = reshaped_copy_from_for_aliases
                arg_to_copy_froms[var_arg_copy_dst].append(copy_from)

        for k in swap_map:
            v = swap_map[k]
            cur_k = k
            cur_v = v
            while cur_k in swap_map:
                cur_v = swap_map[cur_k]
                cur_k = variableify(cur_v)
            if v.name != cur_v.name:
                swap_map[k] = cur_v

    # For nvfuser to be comfortably create fusion regions, we put each `prims.copy_` after the last
    # use of `copy_from`. We don't take the return value of `prims.copy_` because it's already
    # obviated by the functionalization above.
    consumer_map_of_functionalized_bsyms = consumers(new_bsyms)
    producer_map_of_functionalized_bsyms = producers(new_bsyms)
    bsym_to_copy_bsyms: dict[BoundSymbol, list[BoundSymbol]] = defaultdict(list)
    for var_copy_from, copy_bsyms in copy_from_to_copy_bsyms.items():
        copy_from = unvariableify(var_copy_from)
        key_bsym: BoundSymbol = producer_map_of_functionalized_bsyms[copy_from]
        # Make sure `copy_from` has no consumers other than `prims.return`.
        if copy_from in consumer_map_of_functionalized_bsyms:
            consumer_bsyms = list(
                filter(
                    lambda bsym: bsym.sym.id != prims.PrimIDs.RETURN, consumer_map_of_functionalized_bsyms[copy_from]
                )
            )
            if consumer_bsyms:
                check(
                    all(bsym.sym.id != prims.PrimIDs.COPY_ for bsym in consumer_bsyms),
                    lambda: f"Unexpected `prims.copy_` found in {[bsym.sym for bsym in consumer_bsyms]} for {var_copy_from}",
                )
                key_bsym = consumer_bsyms[-1]
        bsym_to_copy_bsyms[key_bsym].extend(copy_bsyms)

    functionalized_computation_trace = from_trace(canonicalized_trace)
    functionalized_computation_trace.set_provenance(TraceProvenance("Functionalize in-place ops"))

    swap_map_for_return: dict[VariableInterface, TensorProxy] = {}
    functionalized_bsyms: list[BoundSymbol] = []
    for bsym in new_bsyms[:-1]:
        functionalized_bsyms.append(bsym)
        if bsym in bsym_to_copy_bsyms:
            functionalized_bsyms.extend(bsym_to_copy_bsyms[bsym])
            copy_bsym = functionalized_bsyms[-1]
            # wrap_return_value_together_with_arguments places all the arguments in the return value
            # We swap these arguments in the return value with the outputs of copies onto them
            # This prevents subsequent transforms from ordering the return statement before those copies
            swap_map_for_return[variableify(copy_bsym.flat_proxy_args[0])] = copy_bsym.flat_proxy_outs[0]
    functionalized_bsyms.append(new_bsyms[-1].from_bsym_swap_proxies(swap_map_for_return))

    functionalized_computation_trace.bound_symbols = functionalized_bsyms
    return functionalized_computation_trace


def functionalize_inplace_ops(
    computation_trace: Trace,
    orig_to_view_swap_map: dict[VariableInterface, TensorProxy],
    alias_tensor_indices: list[list[int]],
) -> list[Trace]:
    r"""Functionalize in-place ops in ``computation_trace``.

    This function is skipped if kwarg of ``skip_inplace_functionalization=True`` is passed to :func:`thunder.jit`.

    In thunder, an in-place is a pair of an out-of-place or functional op followed by :func:`thunder.core.prims.copy_`.
    This function replaces in-place ops with out-of-place ops.

    This function returns an empty list if no in-place ops are found in ``computation_trace``.
    If any are found, functionalization has two steps and another optional step.
    The optional step chimes in if ``alias_tensor_indices`` are not null. This step is to consolidate
    the use of aliases accordingly to ``alias_tensor_indices`` which is a list of lists of indices where
    each list represents the group of tensors of the same underlying storage.
    The second step is to canonicalize the trace by making sure that any operands of in-place ops have
    only one consumer.
    The final step is to replace in-place ops with their out-of-place versions, i.e.,
    to remove ``prims.copy_`` from the trace as much as possible.
    This step also inserts the required ``prims.copy_``\s right after its destination's last use.
    The inserted ``prims.copy_``\s are ones whose destination is ``computation_trace``'s args/kwargs.


    .. seealso::

        `PyTorch Docs - Tensor Views <https://pytorch.org/docs/stable/tensor_view.html>`_

        `PyTorch Docs - torch.func.functionalization <https://pytorch.org/docs/main/generated/torch.func.functionalize.html>`_


    Let's take a look at how the functionalization is working for the following ``f``.
    This function applies in-place ``exp`` to ``a`` which does not share
    the storage of the argument ``x``, thus the functionalized trace would be
    free from ``prims.copy_``.

    .. code-block:: python
        :name: input-func

        def f(x):
            a = x.sin()
            a.exp_()
            return a.cos()

    The input trace which is the first of many computation traces is as follows. Notice that ``a`` is
    consumed by not only ``ltorch.exp_`` but also ``ltorch.cos``.

    .. code-block:: python
        :name: initial-trace

        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.sin(x)  # a: "cpu f32[2, 2]"
            # a = prims.sin(x)  # a: "cpu f32[2, 2]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[2, 2]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[2, 2]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[2, 2]"

          t3 = ltorch.cos(a)  # t3: "cpu f32[2, 2]"
            # t3 = prims.cos(a)  # t3: "cpu f32[2, 2]"
          return t3

    The output of the first step ("canonicalization") is as follows. Notice that now ``ltorch.cos`` takes
    ``t2`` which is the return value of ``ltorch.exp_(a)``.

    .. code-block:: python
        :name: canonicalized-traces

        # Constructed by Intermediate trace of `functionalize_inplace_ops`
        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.sin(x)  # a: "cpu f32[2, 2]"
            # a = prims.sin(x)  # a: "cpu f32[2, 2]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[2, 2]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[2, 2]"
              # t1 = prims.exp(a)  # t1: "cpu f32[2, 2]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[2, 2]"

          # `a` is replaced by `t2`
          t3 = ltorch.cos(t2)  # t3: "cpu f32[2, 2]"
            # t3 = prims.cos(t2)  # t3: "cpu f32[2, 2]"
          return t3

    The functionalized trace is as follows. Notice that this trace has no ``prims.copy_``\s
    and the operand of ``ltorch.cos`` is updated to ``t1`` from ``t2``.

    .. code-block:: python
        :name: functionalized-traces

        # Constructed by Functionalize in-place ops
        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.sin(x)  # a: "cpu f32[2, 2]"
            # a = prims.sin(x)  # a: "cpu f32[2, 2]"
          # Functionalized from `t2 = exp_(a)`
          t1 = ltorch.exp(a)  # t1: "cpu f32[2, 2]"
            # t1 = prims.exp(a)  # t1: "cpu f32[2, 2]"

          # `t2` is replaced by `t1`
          t3 = ltorch.cos(t1)  # t3: "cpu f32[2, 2]"
            # t3 = prims.cos(t1)  # t3: "cpu f32[2, 2]"
          return t3


    Another example to take a look at would be a function with one or more in-place ops to
    its argument(s) and/or their views.
    The ``g`` defined below cannot be functionalized with appropriate ``prims.copy_`` to ``x`` because ``a`` shares
    its storage with ``x``.

    .. code-block:: python
        :name: input-func-with-multiple-inplace

        def g(x):
            a = x.view(-1)
            a.exp_()
            a.sin_()
            return a.cos()

    The input trace to this function is as follows.

    .. code-block:: python
        :name: input-trace-with-multiple-inplace

        def computation(x):
          # x: "cpu f32[2, 2]"

          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
            # a = ltorch.reshape(x, (-1,))  # a: "cpu f32[4]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[4]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[4]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[4]"

          t4 = ltorch.sin_(a)  # t4: "cpu f32[4]"
            # t3 = ltorch.sin(a)  # t3: "cpu f32[4]"
            # t4 = prims.copy_(t3, a)  # t4: "cpu f32[4]"

          t5 = ltorch.cos(a)  # t5: "cpu f32[4]"
            # t5 = prims.cos(a)  # t5: "cpu f32[4]"
          return t5

    For ``g``, the copy from ``a.sin_()`` to ``x`` is critical. If it's missing,
    the consumers of ``x`` including ``g`` itself would get broken.
    Below are the outputs of this function. The top is "canonicalized" (see the change
    of ``ltorch.sin_``'s operand).
    The bottom is the functionalized trace. Since ``a`` has a different shape from ``x``,
    there are ``prims.reshape(t3, (2, 2))`` and ``prims.copy_(t8, x)``.

    .. code-block:: python
        :name: functionalization-output-traces

        # Constructed by Intermediate trace of `functionalize_inplace_ops`
        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
            # a = ltorch.reshape(x, (-1,))  # a: "cpu f32[4]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[4]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[4]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[4]"

          # `a` is replaced by `t2`
          t4 = ltorch.sin_(t2)  # t4: "cpu f32[4]"
            # t3 = ltorch.sin(t2)  # t3: "cpu f32[4]"
            # t4 = prims.copy_(t3, t2)  # t4: "cpu f32[4]"

          # `a` is replaced by `t4`
          t5 = ltorch.cos(t4)  # t5: "cpu f32[4]"
            # t5 = prims.cos(t4)  # t5: "cpu f32[4]"
          return t5


        # Constructed by Functionalize in-place ops
        def computation(x):
          # x: "cpu f32[2, 2]"

          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
            # a = ltorch.reshape(x, (-1,))  # a: "cpu f32[4]"
              # a = prims.reshape(x, (4,))  # a: "cpu f32[4]"
          # Functionalized from `t2 = exp_(a)`
          t1 = ltorch.exp(a)  # t1: "cpu f32[4]"
            # t1 = prims.exp(a)  # t1: "cpu f32[4]"
          # Functionalized from `t4 = sin_(t2)`
          t3 = ltorch.sin(t1)  # t3: "cpu f32[4]"
            # t3 = prims.sin(t1)  # t3: "cpu f32[4]"

          # `t4` is replaced by `t3`
          t5 = ltorch.cos(t3)  # t5: "cpu f32[4]"
            # t5 = prims.cos(t3)  # t5: "cpu f32[4]"
          t8 = prims.reshape(t3, (2, 2))  # t8: "cpu f32[2, 2]"
          t9 = prims.copy_(t8, x)  # t9: "cpu f32[2, 2]"

          return t5


    Last but not least, let's see what would happen when we pass one tensor to the two args of ``f``
    as in the following snippet:

    .. code-block:: python

        @thunder.jit
        def f(a, b):
            c = a.exp_()
            d = b.exp_()
            return c + d

        a = torch.randn((2, 2))
        y = f(a, a)

    The expected result would be ``2 * a.exp().exp()`` because the both of addition's operands share
    the same storage and the right-hand side operand is the result of ``a.exp()_.exp()_``. ``a`` is
    supposed to be the result of ``a.exp().exp()``.
    Since arguments are aliases, the optional first step is called and its output is as follows:

    .. code-block:: python

        # Constructed by Duplicate alias args using {'b': 'a'}
        def computation(a, b):
          # a: "cpu f32[2, 2]"
          # b: "cpu f32[2, 2]"

          # inplace_and_aliases.py:7:         c = a.exp_()
          c = ltorch.exp_(a)  # t1: "cpu f32[2, 2]"
            # t0 = ltorch.exp(a)  # t0: "cpu f32[2, 2]"
              # t0 = prims.exp(a)  # t0: "cpu f32[2, 2]"
            # c = prims.copy_(t0, a)  # t1: "cpu f32[2, 2]"
          # inplace_and_aliases.py:8:         d = b.exp_()
          # [alias tensor args] `b` is replaced by `a`
          d = ltorch.exp_(a)  # t3: "cpu f32[2, 2]"
            # t2 = ltorch.exp(a)  # t2: "cpu f32[2, 2]"
              # t2 = prims.exp(a)  # t2: "cpu f32[2, 2]"
            # d = prims.copy_(t2, a)  # t3: "cpu f32[2, 2]"
          # inplace_and_aliases.py:9:         return c + d
          result = ltorch.add(c, d, alpha=None)  # result: "cpu f32[2, 2]"
            # result = prims.add(c, d)  # result: "cpu f32[2, 2]"
          return result

    As the header says, ``ltorch.exp_`` takes ``a`` instead of ``b`` based on the alias info.
    This trace is canonicalized into:

    .. code-block:: python

        # Constructed by Intermediate trace of `functionalize_inplace_ops`
        def computation(a, b):
          # a: "cpu f32[2, 2]"
          # b: "cpu f32[2, 2]"

          # inplace_and_aliases.py:7:         c = a.exp_()
          t1 = ltorch.exp_(a)  # t1: "cpu f32[2, 2]"
            # t0 = ltorch.exp(a)  # t0: "cpu f32[2, 2]"
              # t0 = prims.exp(a)  # t0: "cpu f32[2, 2]"
            # t1 = prims.copy_(t0, a)  # t1: "cpu f32[2, 2]"
          # inplace_and_aliases.py:8:         d = b.exp_()
          # `a` is replaced by `c`
          d = ltorch.exp_(c)  # t3: "cpu f32[2, 2]"
            # t2 = ltorch.exp(c)  # t2: "cpu f32[2, 2]"
              # t2 = prims.exp(c)  # t2: "cpu f32[2, 2]"
            # d = prims.copy_(t2, c)  # t3: "cpu f32[2, 2]"
          # inplace_and_aliases.py:9:         return c + d
          result = ltorch.add(c, d, alpha=None)  # result: "cpu f32[2, 2]"
            # result = prims.add(c, d)  # result: "cpu f32[2, 2]"
          return result

    Then the functionalized trace is:

    .. code-block:: python

        # Constructed by Functionalize in-place ops
        def computation(a, b):
          # a: "cpu f32[2, 2]"
          # b: "cpu f32[2, 2]"
          # Functionalized from `c = exp_(a)`
          t0 = ltorch.exp(a)  # t0: "cpu f32[2, 2]"
            # t0 = prims.exp(a)  # t0: "cpu f32[2, 2]"
          # Functionalized from `d = exp_(c)`
          t2 = ltorch.exp(t0)  # t2: "cpu f32[2, 2]"
            # t2 = prims.exp(t0)  # t2: "cpu f32[2, 2]"

          # [`c`, `d`] are replaced by [`t2`, `t2`], respectively
          result = ltorch.add(t2, t2, alpha=None)  # result: "cpu f32[2, 2]"
            # result = prims.add(t2, t2)  # result: "cpu f32[2, 2]"
          t5 = prims.copy_(t2, a)  # t5: "cpu f32[2, 2]"

          return result

    Args:
        computation_trace: A computation trace created by ``interpreter``.
        orig_to_view_swap_map:
        alias_tensor_indices: Nested list of integers. Each int represents the index of
            ``tree_flatten((computation_trace.args, computation_trace.kwargs))`` and each list
            represents the group of args that have the same :func:`torch.Tensor.data_ptr`.
    """

    if not any(is_functionalizable(bsym) for bsym in computation_trace.bound_symbols):
        return []

    # Step 0:
    no_implicit_alias_trace, swap_map_for_aliases = replace_args_with_alias_map(computation_trace, alias_tensor_indices)

    # Step 1: make sure each tensor is consumed only once.
    intermediate_trace, reverse_swap_map_for_canonicalization = canonicalize_bsym_args(
        no_implicit_alias_trace, orig_to_view_swap_map
    )

    # Step 2: Remove `prims.copy_` if it's the last one of `bsym.subsymbols`,
    # unless `copy_to` is `computation_trace.args` or `computation_trace.kwargs`
    functionalized_computation_trace = apply_functionalization_to_canonicalized_trace(
        intermediate_trace,
        reverse_swap_map_for_canonicalization,
    )
    if not swap_map_for_aliases:
        return [intermediate_trace, functionalized_computation_trace]
    return [no_implicit_alias_trace, intermediate_trace, functionalized_computation_trace]
