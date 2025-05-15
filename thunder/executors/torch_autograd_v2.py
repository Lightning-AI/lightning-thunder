from inspect import Parameter, Signature

import thunder
import thunder.core.prims as prims
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.transforms import augmented_forward_pass, backward_pass


# The entire contents of this file will be replaced by Tom's work.
def grad_transform_on_trace(ff_trace):
    grad_required = len(ff_trace.output["output"]) != 0

    def joint_forward_and_backward(*args, **kwargs):
        python_return = ff_trace.bound_symbols.pop(-1)
        with thunder.core.trace.tracectx(ff_trace):
            # remove the input that is also returned as part of the output
            prims.python_return((python_return.args[0]["output"],))
        output, all_intermediates = augmented_forward_pass(*args, trace=ff_trace, **kwargs)
        # !!! need to filter out the tensors that don't require grad
        output_grad = tree_map(prims.get_grad, output)
        # restore the args to the result
        result = {"flat_args": args, "output": output[0]}
        if grad_required:
            backward_result = backward_pass(all_intermediates, ff_trace, output_grad)
            result.update({"grad_flat_args": backward_result})
        return result

    params = [Parameter(arg.name, Parameter.POSITIONAL_OR_KEYWORD) for arg in ff_trace.args]
    joint_forward_and_backward.__signature__ = Signature(params)
    #  the new check_trace was failing with weird errors of unknown variables
    #  dce eliminates that
    #  symbolic caching was running into naming collisions because of the renaming and re_unpacking of the args
    return thunder.trace(use_dce=True, rename_proxies=False)(
        joint_forward_and_backward, *ff_trace.args, **ff_trace.kwargs
    )


def split_forward_backward(joint_trace):
    # !!!
    if "grad_flat_args" not in joint_trace.output:
        return joint_trace, None

    forward_part_bsyms = []
    backward_part_bsyms = []

    return_bsym = joint_trace.bound_symbols[-1]
    assert return_bsym.sym == prims.python_return

    fw_output = return_bsym.args[0]["output"]
    assert isinstance(fw_output, tuple)

    grad_outs = [None for _ in fw_output]

    output_pos = {o.name: i for i, o in enumerate(fw_output) if isinstance(o, thunder.TensorProxy)}
    forward_proxy_names = {o.name for o in thunder.core.pytree.tree_iter(fw_output) if isinstance(o, thunder.Proxy)}

    # for inplace, we need to update this (or have flat args be the right thing?...)
    forward_proxy_names.update(a.name for a in return_bsym.args[0]["flat_args"] if isinstance(a, thunder.Proxy))

    for bsym in reversed(joint_trace.bound_symbols):
        if bsym.sym == prims.python_return:
            continue

        # unpack_trivial is always a (forward trace) argument
        if any((o.name in forward_proxy_names) for o in bsym.flat_proxy_outs) or bsym.sym == prims.unpack_trivial:
            forward_part_bsyms.insert(0, bsym.from_bsym())
            forward_proxy_names.update(a.name for a in bsym.flat_proxy_args)
            forward_proxy_names.update(o.name for o in bsym.flat_proxy_outs)
            continue

        if bsym.sym == prims.get_grad:
            grad_outs[output_pos[bsym.args[0].name]] = bsym.output
            continue

        # !!!
        if (bsym.sym == prims.copy_ or bsym.sym.name == "copy_") and bsym.args[1].name in forward_proxy_names:
            # todo: should we also handle ltorch.copy_ ?
            forward_part_bsyms.insert(0, bsym.from_bsym())
            forward_proxy_names.update(a.name for a in bsym.flat_proxy_args)
            continue

        backward_part_bsyms.insert(0, bsym.from_bsym())

    # collect needed computation
    saved_for_backward = {}
    for bsym in backward_part_bsyms:
        saved_for_backward.update(
            (a.name, a)
            for a in bsym.flat_proxy_args
            if a.name in forward_proxy_names and a.name not in saved_for_backward
        )
    saved_for_backward_tensors = [p for p in saved_for_backward.values() if isinstance(p, thunder.TensorProxy)]
    saved_for_backward_other = [p for p in saved_for_backward.values() if not isinstance(p, thunder.TensorProxy)]

    forward_trace = thunder.core.trace.from_trace(joint_trace)
    forward_trace.tags.add(thunder.core.trace.TraceTag.AUGMENTED_FORWARD)

    forward_trace.names = forward_trace.names.copy()  ## ehem
    forward_trace.bound_symbols += forward_part_bsyms

    fw_output_dict = {k: v for k, v in return_bsym.args[0].items() if k != "grad_flat_args"}

    flat_output, _ = thunder.core.pytree.tree_flatten_with_dataclass(fw_output)
    fw_output_dict["flat_output"] = tuple(flat_output)

    with thunder.core.trace.tracectx(forward_trace):
        prims.python_return(fw_output_dict, (saved_for_backward_tensors, saved_for_backward_other))

    # !!!
    if len(backward_part_bsyms) == 0:
        return forward_trace, None

    def backward_fn(saved_for_backward, cotangents):
        pass

    backward_trace = thunder.core.trace.TraceCtx(fn=backward_fn)
    backward_trace.names = forward_trace.names
    backward_trace.name_ctr = forward_trace.name_ctr

    # TODO: make this name-agnostic
    backward_trace.names.discard("saved_for_backward")
    backward_trace.names.discard("cotangents")

    with thunder.core.trace.tracectx(backward_trace):
        p_C0 = thunder.core.proxies.CollectionProxy(list(saved_for_backward_tensors), name="C0")
        p_C1 = thunder.core.proxies.CollectionProxy(list(saved_for_backward_other), name="C1")
        p_saved_for_backward = thunder.core.proxies.CollectionProxy([p_C0, p_C1], name="saved_for_backward")
        p_cotangents = thunder.core.proxies.CollectionProxy(grad_outs, name="cotangents")
        backward_trace.args = (p_saved_for_backward, p_cotangents)

        prims.unpack_trivial(p_saved_for_backward, name="saved_for_backward")
        prims.unpack_trivial(p_cotangents, name="cotangents")
        prims.unpack_sequence(p_saved_for_backward, len(p_saved_for_backward.coll))
        prims.unpack_sequence(p_cotangents, len(p_cotangents.coll))
        prims.unpack_sequence(p_C0, len(p_C0.coll))
        prims.unpack_sequence(p_C1, len(p_C1.coll))

    backward_trace.bound_symbols += backward_part_bsyms

    with thunder.core.trace.tracectx(backward_trace):
        prims.python_return(tuple(return_bsym.args[0]["grad_flat_args"]))

    # !!! args are CollectionProxies, which is not the case in the current implementation
    # !!! also, this is gross
    backward_trace.args = tree_map(
        lambda x: x.coll if isinstance(x, thunder.core.proxies.CollectionProxy) else x, backward_trace.args
    )
    backward_trace.args = (
        tree_map(
            lambda x: x.coll if isinstance(x, thunder.core.proxies.CollectionProxy) else x, backward_trace.args[0]
        ),
        backward_trace.args[1],
    )

    return forward_trace, backward_trace
