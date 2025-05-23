import thunder.core.transforms
from thunder.core.transforms import ForwardBackwardTraces

from thunder.core import prims, utils
from thunder.core.transforms import (
    is_constant_for_vjp,
    _get_gradfn_and_executor,
    augmented_forward_impls,
    backward_impls,
    recompute_saved_for_backward,
)
from thunder.core.proxies import ProxyTag
from thunder.core.symbol import BoundSymbol, BoundSymbolTag
from thunder.core.vjp_utils import make_aug_forward_and_backward
from thunder.core.pytree import tree_map
import thunder
import time


def _should_recompute_bsym_in_backward(bsym):
    return BoundSymbolTag.RECOMPUTE_IN_BACKWARD in bsym.tags or any(
        (ProxyTag.RECOMPUTE_IN_BACKWARD in o.tags) for o in bsym.flat_proxy_outs
    )


# Transforms a trace by determining which grad transforms to call given the list of executors in priority order
# This pass tries to preserve the original trace and proxies.
def grad_transform_on_trace(trace, /, *args, **kwargs):
    # This processes the bsyms to map symbols to operator executors:
    # - in the order of the executor list
    #   - if the executor defines a grad transform, call that to
    #     get fw + bw,
    # - if there is a augmented forward registered, use that and the backward registered,
    #   to construct a grad transform to get fw + bw
    # - if neither of the above apply, and the symbol has subsymbols, push the decomposition
    #   to the front of the queue
    # - if none of the above apply and we have a prim, raise an error
    class AugmentedForwardProcessor(thunder.core.trace_interpreter.TraceSubstitutionProcessor):
        def __init__(self, trace):
            super().__init__(trace)
            self.collected_bw_part_bsyms = []

        def process_bsym(self, bsym: thunder.core.symbol.BoundSymbol) -> None:
            if bsym.sym is prims.python_return:
                # BEGINNING of return handling (and putting the backward computation in the joint trace)
                # This is big (and a bit messy):
                # the return (of the input trace) signals the end of the forward processing
                # all the backwards will have been put in self.collected_bw_part_syms
                # special symbols are
                # - get_grad "reads" the gradient of a variable from the forward to compute with it
                #   (if you are used to torch.autograd.Function, this is a "grad_out" in the arguments
                #    of the backward static method)
                # - put_grad "writes" a computed gradients (similar to returning the "grad_in" in torch.autograd.Function)
                # The backward computation in self.collected_bw_part_syms is in the order of the
                # backward computation.
                #
                input_proxy_names = {p.name for p in bsym.args[0]["flat_args"] if isinstance(p, thunder.Proxy)}
                output_proxy_names = set()
                for o in thunder.core.pytree.tree_iter(bsym.args[0]["output"]):
                    if isinstance(o, thunder.Proxy):
                        output_proxy_names.add(self.read(o).name)
                grad_proxy_map = {}

                # which names we saw get_grad for, we cannot do put_grad on them anymore
                names_seen_get_grad = set()

                # we iterate through the collected symbols backward in order (from the generation of the list, this
                # is the backward of the last forward instruction first, then then the backward of the second to last etc.
                for bw_bsyms in self.collected_bw_part_bsyms:
                    # if we don't have and grad_outs, we know that the grad_ins are not needed, so we don't compute them
                    # (e.g. for `where(a * b > 0, a, b)` we do not need the grad components of a * b propagated
                    if any(
                        ((gg.name in grad_proxy_map) or (gg.name in output_proxy_names))
                        for gg in (bs.args[0] for bs in bw_bsyms if bs.sym == prims.get_grad)
                    ):
                        for nbsym in bw_bsyms:
                            if nbsym.sym == prims.put_grad:
                                # put_grad adds something to the gradients
                                # - if it is an output of the function, we also need to get the
                                #   "grad_output" (but only once)
                                # - if we already had a gradient, we need to add the new component
                                # - structurally (for correctness), there should be no put_grad
                                #   after a get_grad
                                p = self.env.get(nbsym.args[0].name, nbsym.args[0])
                                assert p.name not in names_seen_get_grad, f"put_grad after get_grad on {p.name}"
                                current_grad = grad_proxy_map.get(p.name)
                                new_grad = nbsym.args[1]
                                if current_grad is None and p.name in output_proxy_names:
                                    self.new_trace.push_scope([])
                                    current_grad = prims.get_grad(p)
                                    self.add_processed_bsyms(self.new_trace.pop_scope())

                                if current_grad is not None:
                                    new_grad = self.add_bsyms_from_function(thunder.torch.add, current_grad, new_grad)

                                grad_proxy_map[p.name] = new_grad
                                self.write(new_grad, new_grad)
                            elif nbsym.sym == prims.get_grad:
                                # get grad reads a gradient, this could be
                                # - the gradient of an intermediate (possibly combined with an output) form put_grad above
                                #   note that all put_grads are happen before all get_grads from the dataflow of the forward,
                                #   in this case we use the gradient from the put_grad(s)
                                # - an output of the forward function (that has not been combined with an intermediate in put grad),
                                #   then the gradient is like the "grad_out" in the parameter list of a autograd.Function.backward
                                #   in this case we keep a get_grad in the trace.
                                # - an intermediate that does not have a gradient computed. In this case the gradient is 0 in the shape
                                #   of the get_grad argument. This happens e.g. if we have something like "a > 0" for a float a then
                                #   a has grad 0.
                                #   TODO: We could (and should) optimize this (see below).
                                p = nbsym.args[0]

                                # TODO: set proxy requires grad here!
                                name = p.name
                                names_seen_get_grad.add(name)
                                current_grad = grad_proxy_map.get(name)
                                if current_grad is not None:
                                    # do we also need to map?
                                    self.write(nbsym.output, current_grad)
                                    # replace_output_with_current_grad
                                    self.swap_map[thunder.core.proxies.variableify(nbsym.output)] = current_grad
                                elif name in output_proxy_names:
                                    # output here???
                                    new_bsym = nbsym.from_bsym()
                                    self.add_processed_bsyms([new_bsym])
                                    grad_proxy_map[name] = new_bsym.output
                                else:
                                    # TODO: mark this? if all inputs to the backward formula are unused, we would not want to compute it.
                                    new_bsym = thunder.torch.zeros.bind(
                                        *p.shape, device=p.device, dtype=p.dtype, output=nbsym.output
                                    )
                                    self.add_processed_bsyms([new_bsym])
                                    grad_proxy_map[name] = new_bsym.output
                                    self.write(nbsym.output, new_bsym.output)
                            else:
                                # all other symbols are just computation
                                self.add_processed_bsyms([nbsym.from_bsym()])
                self.collected_bw_part_bsyms.clear()

                # we collect the gradients fo the flat args in the output dictionary's 'grad_flat_args'
                # this means that the joint trace has them as outputs, which is good.
                grad_flat_args = []
                for p in bsym.args[0]["flat_args"]:
                    # or p = self.read(p) here?
                    if isinstance(p, thunder.TensorProxy) and p.requires_grad and p.name in grad_proxy_map:
                        # is it always OK if we don't have a gradient? (one case: unused input)
                        # result of put_grad???
                        grad_flat_args.append(grad_proxy_map[p.name])
                    else:
                        grad_flat_args.append(None)

                bsym.args[0]["grad_flat_args"] = grad_flat_args
                self.add_processed_bsyms([bsym.from_bsym()])
                self.set_result(bsym.output)
                return
                # END of return handling (and putting the backward computation in the joint trace)

            # we now handle various bound symbols by collecting augmented forward and backward symbols

            # 1. constant for gradients (no grad required)
            #    executing synchronize here terrible hack to cope with non-grad-needing sharded tensors
            #    as required by LoRA. We should have a symbol tag "always compute grad" instead.
            if is_constant_for_vjp(bsym) and not bsym.sym.name == "synchronize":
                self.add_processed_bsyms([bsym.from_bsym()])
                self.set_result(bsym.output)
                return

            # 2. Special case the thunder.torch.checkpoint higher order function
            if bsym.sym == thunder.torch.checkpoint:
                # Tag all intermediate outputs as to be recomputed.
                function_arg_names = {a.name for a in bsym.flat_proxy_args}

                for subsym in bsym.subsymbols:
                    subsym.tags.add(BoundSymbolTag.RECOMPUTE_IN_BACKWARD)
                    for o in subsym.flat_proxy_outs:
                        if o.name not in function_arg_names:
                            o.tags.add(ProxyTag.RECOMPUTE_IN_BACKWARD)

                # decompose
                decomposed_bsyms = bsym.subsymbols[:]
                # shallow copies that need to be not recomputed and need their output replaced.
                for x in bsym.flat_proxy_outs:
                    x.tags.discard(ProxyTag.RECOMPUTE_IN_BACKWARD)
                    decomposed_bsyms.append(prims.shallow_copy.bind(x, output=x))

                self.add_unprocessed_bsyms(decomposed_bsyms)
                return

            # here we do the copy for the args form above
            if bsym.sym == prims.shallow_copy and bsym.output is bsym.args[0]:
                # this is a bit of a hack in order to only replace the output,
                # not the input
                (a,) = bsym.args
                a_inp = self.swap_map.get(thunder.core.proxies.variableify(a), a)
                with thunder.core.trace.tracectx(self.new_trace):
                    o = prims.shallow_copy(a_inp)
                self.add_to_swap_map(a, o)
                self.add_to_swap_map(a_inp, o)
                self.write(a_inp, o)

                self.new_trace.push_scope([])
                with thunder.core.trace.tracectx(self.new_trace):
                    prims.put_grad(a_inp, prims.get_grad(o))
                backward_part_bsyms = self.new_trace.pop_scope()
                self.collected_bw_part_bsyms.insert(0, backward_part_bsyms)
                return

            # 3a. see if we have a grad_transform (e.g. from OperatorExecutor.register_grad_transform)

            # executor or global grad transform
            joint_forward_backward, _ = _get_gradfn_and_executor(bsym)

            if joint_forward_backward is None:
                # 3b. if we don't have a grad_transform for a bsym, maybe we have old style augmented forward and backward
                # registered. In this case, we make the joint_forward_backward which works like a grad_gransform.
                # registered augmented forward impl (where aug fwd and backward are registered separately)
                aug_fwd_impl = augmented_forward_impls.get(bsym.sym.id)
                if aug_fwd_impl is not None:
                    bwd_impl = backward_impls.get(bsym.sym.id)

                    # this is our ad hoc combined forward and backward function ("grad_transform")
                    def joint_forward_backward(*args, **kwargs):
                        arg_proxy_names = {a.name for a in bsym.flat_proxy_args}

                        # run the augmented forward res
                        aug_fwd_res = aug_fwd_impl(*bsym.args, **bsym.kwargs)

                        # aug_fwd_res could be either VJPDual or just a tuple, in any case, we can decompose it.
                        res, saved_for_backward = aug_fwd_res

                        # we need to shallow copy inputs that are returned for "get_grad" and "put_grad" to properly work
                        # (this shallow copy is the equivalent because we of an "edge" in the PyTorch autograd graph)
                        def shallow_copy_if_input(p):
                            if isinstance(p, thunder.TensorProxy) and p.name in arg_proxy_names:
                                return thunder.core.prims.shallow_copy(p)
                            return p

                        res = tree_map(shallow_copy_if_input, res)

                        # now we need the backward. it starts by getting the grad_outs
                        grad_outs = []
                        for r in thunder.core.pytree.tree_iter(res):
                            if isinstance(r, thunder.TensorProxy):
                                grad_outs.append(prims.get_grad(r))

                        # The backward computes the grad_inps of the bsym from the grad_outs
                        # TODO: non-grad outputs of bwd?
                        grad_inps = bwd_impl(*saved_for_backward, *grad_outs)
                        if isinstance(grad_inps, thunder.Proxy):
                            grad_inps = [grad_inps]

                        # match the grad_inps to the inputs of the boudnd symbol and put the grads

                        flat_inps = args
                        # for autograd_function_apply, skip the function args
                        # TODO: fix the returned gradients to include two None?.
                        if bsym.sym == thunder.torch.autograd_function_apply:
                            flat_inps = args[2:]

                        # there may be non-gradient requiring additional args (todo: maybe only support this for non-tensor ones?)
                        num_flat_tensor_inps = sum(isinstance(i, thunder.TensorProxy) for i in flat_inps)
                        utils.check(
                            num_flat_tensor_inps <= len(grad_inps),
                            lambda: f"Backward for {bsym.sym.id} returned {len(grad_inps)} value(s), but expected {num_flat_tensor_inps}",
                        )

                        assert len(grad_inps) <= len(flat_inps)
                        for i, gi in zip(flat_inps, grad_inps):
                            # for integer proxies etc. we expect gi to be None
                            if isinstance(i, thunder.TensorProxy) and gi is not None:
                                prims.put_grad(i, gi)
                        return res

            # 3c if we have a joint forward backward (either grad_transform or the ad hoc constructed one above), we can comptue the gradient
            if joint_forward_backward is not None:
                self.new_trace.push_scope([])
                result = joint_forward_backward(*bsym.args, **bsym.kwargs)

                # Check that the number of outputs of the original forward function is the
                # same as the number of primal outputs of the augmented forward trace
                utils.check(
                    len(utils.sequencify(bsym.output)) == len(utils.sequencify(result)),
                    lambda: f"While generating forward and backward functions for {bsym.sym.name}, encountered an error.\n"
                    "The number of outputs of the gradient transform function must be the same as the number of outputs of the original forward function.\n"
                    f"Number of outputs of the original forward function: {len(utils.sequencify(bsym.output))}\n"
                    f"Number of primal outputs of the gradient transform / augmented forward: {len(utils.sequencify(result))}\n"
                    "Please check the forward function and the gradient transform function / augmented forward to ensure that they have the same number of outputs.",
                )
                self.set_result(result)
                new_bsyms = self.new_trace.pop_scope()

                # Let the new bound symbols inherit tags, in particular RECOMPUTE_IN_BACKWARD
                for nbsym in new_bsyms:
                    nbsym.tags |= bsym.tags

                # simple splitting: only compute in forward what is needed for the output
                forward_part_proxy_names = {
                    o.name for o in thunder.core.pytree.tree_iter(result) if isinstance(o, thunder.Proxy)
                }
                forward_part_bsyms = []
                backward_part_bsyms = []
                for nbsym in reversed(new_bsyms):
                    # argnames = {p.name for p in nbsym.flat_proxy_args}
                    if nbsym.sym in {prims.get_grad, prims.put_grad}:
                        backward_part_bsyms.insert(0, nbsym)
                        continue
                    assert nbsym.sym != prims.unpack_trivial  # can unpack trivial happen here?
                    if any((o.name in forward_part_proxy_names) for o in nbsym.flat_proxy_outs):
                        forward_part_bsyms.insert(0, nbsym)
                        bsym_arg_names = {a.name for a in nbsym.flat_proxy_args}
                        bsym_out_names = {o.name for o in nbsym.flat_proxy_outs if o.name not in bsym_arg_names}
                        forward_part_proxy_names.update(bsym_arg_names)
                        forward_part_proxy_names.update(bsym_out_names)
                        continue
                    backward_part_bsyms.insert(0, nbsym)
                self.add_processed_bsyms(forward_part_bsyms)

                self.collected_bw_part_bsyms.insert(0, backward_part_bsyms)
                return

            # 4. No gradient transform found, need to decompose the symbol

            # error if this is a primitive
            utils.check(not bsym.sym.is_prim, lambda: f"Failed to find a gradient transform for bound symbol {bsym=}")

            # TODO: check if this is needed: the old impl checked whether len(bsym.subsymbols) > 0 except for the special case "torch.nn.functional.dropout" with p=0...
            # add the decomposition (= the subsymbols) to the front of the symbols to be processed

            # Let the decomposition inherit tags, in particular RECOMPUTE_IN_BACKWARD
            for nbsym in bsym.subsymbols:
                nbsym.tags |= bsym.tags

            self.add_unprocessed_bsyms(bsym.subsymbols[:])

            # end of 4 and end of the bsym processing loop.

    start_time_ns = time.perf_counter_ns()

    # run the trace through the processor
    trace, _ = AugmentedForwardProcessor(trace)()
    # run through DCE in case some of the gradients of intermediates are not needed.
    trace = thunder.core.transform_common.dce(trace)

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    trace.set_provenance(
        thunder.core.trace.TraceProvenance(f"Grad transform pass (took {elapsed_time_millis} milliseconds)")
    )
    return trace


def split_into_forward_and_backward(joint_trace):
    """split a joint trace for forward and backward into separate ones, including recomputation (aka activation checkpointing)"""

    # the joint trace will have the forward computation at the beginning and then the backward computation
    # from how it is constructed.
    # we split the trace:
    # - forward symbols go into forward_part_bsyms
    # - all symbols not in the forward go into backward_part_bsyms
    # - for recomputation (aka activation checkpointing), we want to insert symbols going into the forward also into the
    #   backward, but we want to do so "just in time". To this end, we gather the symbols in a dict and later
    #   insert it when their respective outputs are needed. This is in backward_part_bsyms_recomputed
    #   The just in time recomputation is a heuristic to save memory mimicking checkpointing: e.g. for a checkpointed
    #   block, the forward would be recomputed just before computing the gradient.
    # the splitting is done in the reverse order of the bound symbols, and works out which bits are needed for the forward
    # from there.
    forward_part_bsyms = []
    backward_part_bsyms = []
    backward_part_bsyms_recomputed: dict[str, tuple[BoundSymbol, set[str]]] = {}  # name -> (bsym, name_set)

    # return has a dict, it contains the forward outputs in "output" and the backward outputs in "grad_flat_args", corresponding
    # to "flat_args"
    return_bsym = joint_trace.bound_symbols[-1]
    assert return_bsym.sym == prims.python_return
    fw_output = return_bsym.args[0]["output"]
    assert isinstance(fw_output, tuple)

    grad_outs = [None for _ in fw_output]
    output_pos = {o.name: i for i, o in enumerate(fw_output) if isinstance(o, thunder.TensorProxy)}

    # the proxies we need to compute in the forward - we start with the outputs of the forward
    forward_proxy_names = {o.name for o in thunder.core.pytree.tree_iter(fw_output) if isinstance(o, thunder.Proxy)}
    # we also have the inputs available, so we add flat_args.
    # for inplace, we need to update this (or have flat args be the right thing?...)
    forward_proxy_names.update(a.name for a in return_bsym.args[0]["flat_args"] if isinstance(a, thunder.Proxy))

    # We keep track of the names of proxies we recompute in the backward as those will not need to be part of the
    # ones saved in the forward for the backward
    backward_recomputed_proxy_names = set()

    # loop over the bound symbols in reverse (see above)
    for bsym in reversed(joint_trace.bound_symbols):
        if bsym.sym == prims.python_return:  # we will re-do the return statements below, so we skip it here
            continue

        # unpack_trivial is always a (forward trace) argument, the backward inputs are from get_grad
        # anything that outputs something needed for the forward (the forward_proxy_names) is part of the forward
        if any((o.name in forward_proxy_names) for o in bsym.flat_proxy_outs) or bsym.sym == prims.unpack_trivial:
            forward_part_bsyms.insert(0, bsym.from_bsym())
            bsym_arg_names = {a.name for a in bsym.flat_proxy_args}
            bsym_out_names = {o.name for o in bsym.flat_proxy_outs if o.name not in bsym_arg_names}
            forward_proxy_names.update(bsym_arg_names)
            forward_proxy_names.update(bsym_out_names)

            # if we need to recompute this, we also add the bound symbol to backward_part_bsyms_recomputed
            if _should_recompute_bsym_in_backward(bsym):
                backward_recomputed_proxy_names.update(bsym_out_names)
                bsym_rec = (bsym.from_bsym(), bsym_out_names)
                backward_part_bsyms_recomputed.update((n, bsym_rec) for n in bsym_out_names)

            continue

        # get grad is always part of the input, record the grad_out (will be part of the "cotangents" list)
        if bsym.sym == prims.get_grad:
            grad_outs[output_pos[bsym.args[0].name]] = bsym.output
            continue

        # copy_ updating a forward proxy is special regardless of the output
        if bsym.sym == prims.copy_ and bsym.args[1].name in forward_proxy_names:
            # todo: should we also handle ltorch.copy_ ?
            forward_part_bsyms.insert(0, bsym.from_bsym())
            forward_proxy_names.update(a.name for a in bsym.flat_proxy_args)
            continue

        # if we don't need to have it in the forward, it is part of the backward
        backward_part_bsyms.insert(0, bsym.from_bsym())

    # we insert the recomputed symbols just before where they are needed.
    # as this inserts into the list during processing, we use a while loop rather than
    # a for loop
    bw_idx = 0
    while bw_idx < len(backward_part_bsyms):
        bsym = backward_part_bsyms[bw_idx]
        modified = False
        for n in reversed(bsym.flat_proxy_args):
            recomp_bsym_rec = backward_part_bsyms_recomputed.get(n.name)
            if recomp_bsym_rec is not None:
                modified = True
                recomp_bsym, recomp_output = recomp_bsym_rec
                backward_part_bsyms.insert(bw_idx, recomp_bsym)
                for nn in recomp_output:
                    del backward_part_bsyms_recomputed[nn]
        if not modified:
            bw_idx += 1

    # collect proxies needing to be saved in the forward for the backward
    saved_for_backward = {}
    for bsym in backward_part_bsyms:
        saved_for_backward.update(
            (a.name, a)
            for a in bsym.flat_proxy_args
            if a.name in forward_proxy_names and a.name not in backward_recomputed_proxy_names
        )
    saved_for_backward_tensors = [p for p in saved_for_backward.values() if isinstance(p, thunder.TensorProxy)]
    saved_for_backward_other = [p for p in saved_for_backward.values() if not isinstance(p, thunder.TensorProxy)]

    # we build the forward trace
    forward_trace = thunder.core.trace.from_trace(joint_trace)
    forward_trace.tags.add(thunder.core.trace.TraceTag.AUGMENTED_FORWARD)
    forward_trace.names = forward_trace.names.copy()  ## ehem
    forward_trace.bound_symbols += forward_part_bsyms

    # now we create the return value and return bound symbol for the forward
    fw_output_dict = {k: v for k, v in return_bsym.args[0].items() if k != "grad_flat_args"}
    flat_output, _ = thunder.core.pytree.tree_flatten_with_dataclass(fw_output)
    fw_output_dict["flat_output"] = tuple(flat_output)
    with thunder.core.trace.tracectx(forward_trace):
        prims.python_return(fw_output_dict, (saved_for_backward_tensors, saved_for_backward_other))

    # then we construct the backward trace, unpacking saved_for_backward and cotangents lists
    def backward_fn(saved_for_backward, cotangents):
        pass

    backward_trace = thunder.core.trace.TraceCtx(fn=backward_fn)
    backward_trace.names = forward_trace.names
    backward_trace.name_ctr = forward_trace.name_ctr

    # TODO: make this name-agnostic
    backward_trace.names.discard("saved_for_backward")
    backward_trace.names.discard("cotangents")

    # set up the inputs of the backward properly (args and unpacking)
    with thunder.core.trace.tracectx(backward_trace):
        p_C0 = thunder.core.proxies.CollectionProxy(list(saved_for_backward_tensors), name="C0")
        p_C1 = thunder.core.proxies.CollectionProxy(list(saved_for_backward_other), name="C1")
        p_saved_for_backward = thunder.core.proxies.CollectionProxy([p_C0, p_C1], name="saved_for_backward")
        p_cotangents = thunder.core.proxies.CollectionProxy(grad_outs, name="cotangents")

        # set the args (which currently don't use the collection proxies but the collections directly)
        saved_for_backward_tuple = [p_C0.collection(), p_C1.collection()]
        backward_trace.args = (saved_for_backward_tuple, p_cotangents.collection())

        # unpacking symbols
        prims.unpack_trivial(p_saved_for_backward, name="saved_for_backward")
        prims.unpack_trivial(p_cotangents, name="cotangents")
        prims.unpack_sequence(p_saved_for_backward, len(p_saved_for_backward.coll))
        prims.unpack_sequence(p_cotangents, len(p_cotangents.coll))
        prims.unpack_sequence(p_C0, len(p_C0.coll))
        prims.unpack_sequence(p_C1, len(p_C1.coll))

    # add the backward computation
    backward_trace.bound_symbols += backward_part_bsyms

    # and finally the backward return statement
    with thunder.core.trace.tracectx(backward_trace):
        prims.python_return(tuple(return_bsym.args[0]["grad_flat_args"]))

    return forward_trace, backward_trace


def forward_and_backward_from_trace(trace: thunder.core.trace.TraceCtx, torch_autograd=False) -> ForwardBackwardTraces:
    if not torch_autograd:
        return thunder.core.transforms.forward_and_backward_from_trace(trace, torch_autograd=torch_autograd)
    joint_trace = grad_transform_on_trace(trace)

    forward_trace, backward_trace = split_into_forward_and_backward(joint_trace)
    return ForwardBackwardTraces(forward_trace, backward_trace)
