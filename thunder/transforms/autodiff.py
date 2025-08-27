import time

from thunder.core import prims, utils

from thunder.core.pytree import tree_map, tree_iter, tree_flatten_with_dataclass
from thunder.core.proxies import TensorProxy, ProxyTag, Proxy, CollectionProxy, variableify
from thunder.core.symbol import BoundSymbol, BoundSymbolTag
from thunder.core.trace import TraceProvenance, tracectx, TraceCtx, from_trace, TraceTag
from thunder.core.trace_interpreter import TraceSubstitutionProcessor
from thunder.core.transforms import (
    dce,
    is_constant_for_vjp,
    _get_gradfn_and_executor,
    augmented_forward_impls,
    backward_impls,
)
import thunder.torch as ltorch


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
    class AugmentedForwardProcessor(TraceSubstitutionProcessor):
        def __init__(self, trace):
            super().__init__(trace)
            self.collected_bw_part_bsyms = []

        def process_bsym(self, bsym: BoundSymbol) -> None:
            if _should_recompute_bsym_in_backward(bsym) and bsym.sym is not prims.python_return:
                # potentially taking a recompute proxy tag and making it a bound symbol tag
                bsym.tags.add(BoundSymbolTag.RECOMPUTE_IN_BACKWARD)

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

                output_proxy_names = set()
                for o in tree_iter(bsym.args[0]["output"]):
                    if isinstance(o, Proxy):
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
                                    new_bsyms = self.new_trace.pop_scope()
                                    for nbsym in new_bsyms:
                                        nbsym.tags.add(BoundSymbolTag.BACKWARD)
                                    self.add_processed_bsyms(new_bsyms)

                                if current_grad is not None:
                                    new_grad = self.add_bsyms_from_function(
                                        ltorch.add, current_grad, new_grad, tags={BoundSymbolTag.BACKWARD}
                                    )

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
                                    self.swap_map[variableify(nbsym.output)] = current_grad
                                elif name in output_proxy_names:
                                    # output here???
                                    new_bsym = nbsym.from_bsym()
                                    new_bsym.tags.add(BoundSymbolTag.BACKWARD)
                                    self.add_processed_bsyms([new_bsym])
                                    grad_proxy_map[name] = new_bsym.output
                                else:
                                    # TODO: mark this? if all inputs to the backward formula are unused, we would not want to compute it.
                                    new_bsym = ltorch.zeros.bind(
                                        *p.shape, device=p.device, dtype=p.dtype, output=nbsym.output
                                    )
                                    new_bsym.tags.add(BoundSymbolTag.BACKWARD)
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
                    if isinstance(p, TensorProxy) and p.requires_grad and p.name in grad_proxy_map:
                        # is it always OK if we don't have a gradient? (one case: unused input)
                        # result of put_grad???
                        grad_flat_args.append(grad_proxy_map[p.name])
                    else:
                        grad_flat_args.append(None)

                # Store the outputs from the forward trace in fw_flat_out what will be used in
                # the split logic to create the return for the forward trace.
                new_return_args = {
                    **bsym.args[0],
                    "fw_flat_out": bsym.args[0]["output"],
                    "output": tuple(grad_flat_args),
                }

                new_return_bsym = bsym.from_bsym(args=(new_return_args,))
                self.add_processed_bsyms([new_return_bsym])

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

            # 2. Special case the ltorch.checkpoint higher order function
            if bsym.sym == ltorch.checkpoint:
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

            # here we do the copy for the args from above
            if bsym.sym == prims.shallow_copy and bsym.output is bsym.args[0]:
                # this is a bit of a hack in order to only replace the output,
                # not the input
                (a,) = bsym.args
                a_inp = self.swap_map.get(variableify(a), a)
                with tracectx(self.new_trace):
                    o = prims.shallow_copy(a_inp)
                self.add_to_swap_map(a, o)
                self.add_to_swap_map(a_inp, o)
                self.write(a_inp, o)

                self.new_trace.push_scope([])
                with tracectx(self.new_trace):
                    prims.put_grad(a_inp, prims.get_grad(o))
                backward_part_bsyms = self.new_trace.pop_scope()
                for nbsym in backward_part_bsyms:
                    nbsym.tags.add(BoundSymbolTag.BACKWARD)
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
                            if isinstance(p, TensorProxy) and p.name in arg_proxy_names:
                                return prims.shallow_copy(p)
                            return p

                        res = tree_map(shallow_copy_if_input, res)

                        # now we need the backward. it starts by getting the grad_outs
                        grad_outs = []
                        for r in tree_iter(res):
                            if isinstance(r, TensorProxy):
                                grad_outs.append(prims.get_grad(r))

                        # The backward computes the grad_inps of the bsym from the grad_outs
                        # TODO: non-grad outputs of bwd?
                        grad_inps = bwd_impl(*saved_for_backward, *grad_outs)
                        if isinstance(grad_inps, Proxy):
                            grad_inps = [grad_inps]

                        # match the grad_inps to the inputs of the boudnd symbol and put the grads

                        flat_inps = args
                        # for autograd_function_apply, skip the function args
                        # TODO: fix the returned gradients to include two None?.
                        if bsym.sym == ltorch.autograd_function_apply:
                            flat_inps = args[2:]

                        # there may be non-gradient requiring additional args (todo: maybe only support this for non-tensor ones?)
                        num_flat_tensor_inps = sum(isinstance(i, TensorProxy) for i in flat_inps)
                        utils.check(
                            num_flat_tensor_inps <= len(grad_inps),
                            lambda: f"Backward for {bsym.sym.id} returned {len(grad_inps)} value(s), but expected {num_flat_tensor_inps}",
                        )

                        assert len(grad_inps) <= len(flat_inps)
                        for i, gi in zip(flat_inps, grad_inps):
                            # for integer proxies etc. we expect gi to be None
                            if isinstance(i, TensorProxy) and gi is not None:
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
                forward_part_proxy_names = {o.name for o in tree_iter(result) if isinstance(o, Proxy)}
                forward_part_bsyms = []
                backward_part_bsyms = []
                for nbsym in reversed(new_bsyms):
                    # argnames = {p.name for p in nbsym.flat_proxy_args}
                    if nbsym.sym in {prims.get_grad, prims.put_grad}:
                        nbsym.tags.add(BoundSymbolTag.BACKWARD)
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
                    nbsym.tags.add(BoundSymbolTag.BACKWARD)
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

    # This processes the bsyms to add duplicate symbols in the backward region for those that
    # are tagged for recomputation.
    class InsertRecomputationsProcessor(TraceSubstitutionProcessor):
        def __init__(self, trace):
            super().__init__(trace)
            self.backward_part_bsyms_recomputed: dict[str, BoundSymbol] = {}
            self.already_processed_recomputations = set()

        def find_recomputation_symbols_for_bsym(self, bsym) -> list[tuple[str, BoundSymbol]]:
            need_sorting: dict[str, BoundSymbol] = {}
            queue: list[Proxy] = list(
                filter(lambda p: p.name in self.backward_part_bsyms_recomputed, bsym.flat_proxy_args)
            )
            visited = set(map(lambda p: p.name, queue))
            while queue:
                proxy = queue.pop()
                producer = self.backward_part_bsyms_recomputed.get(proxy.name, None)
                if producer is not None:
                    need_sorting[proxy.name] = producer
                    for arg in producer.flat_proxy_args:
                        arg_name = arg.name
                        if arg_name not in visited:
                            queue.append(arg)
                            visited.add(arg_name)

            sorted_recomputation = []

            while need_sorting:
                sorted_recomputation_names = []
                for name, producer in need_sorting.items():
                    ready = True
                    for dep in producer.flat_proxy_args:
                        if dep.name in need_sorting:
                            ready = False
                            break
                    if ready:
                        sorted_recomputation_names.append(name)
                for name in sorted_recomputation_names:
                    sorted_recomputation.append((name, need_sorting[name]))
                    del need_sorting[name]

            return sorted_recomputation

        def process_bsym(self, bsym: BoundSymbol) -> None:
            processed_bsyms = []
            if _should_recompute_bsym_in_backward(bsym) and BoundSymbolTag.BACKWARD not in bsym.tags:
                nbsym = bsym.from_bsym()
                nbsym.tags.add(BoundSymbolTag.BACKWARD)
                self.backward_part_bsyms_recomputed.update({arg.name: nbsym for arg in nbsym.flat_proxy_outs})

            elif BoundSymbolTag.BACKWARD in bsym.tags:
                sorted_recomputed_bsyms: list[tuple[str, BoundSymbol]] = self.find_recomputation_symbols_for_bsym(bsym)

                for name, rec_bsym in sorted_recomputed_bsyms:
                    if name in self.already_processed_recomputations:
                        continue
                    # To avoid name clashes, we create new output proxies.
                    # This relies on the fact that all backward operations occur after get_grad,
                    # which is no longer true after fusion passes.
                    with tracectx(self.new_trace):
                        for output in rec_bsym.flat_proxy_outs:
                            self.already_processed_recomputations.add(output.name)
                            new = output.replace_name("bw_" + output.name)

                            self.add_to_swap_map(output, new)

                    processed_bsyms.append(rec_bsym)

            processed_bsyms.append(bsym.from_bsym())

            self.add_processed_bsyms(processed_bsyms)
            self.set_result(processed_bsyms[-1].output)

    start_time_ns = time.perf_counter_ns()

    # run the trace through the processor
    joint_trace, _ = AugmentedForwardProcessor(trace)()
    joint_trace, _ = InsertRecomputationsProcessor(joint_trace)()

    # run through DCE in case some of the gradients of intermediates are not needed.
    joint_trace = dce(joint_trace)
    # group get_grad symbols together for torch compile fusions and to make clear boundary for cse
    _group_get_grad_bsyms(joint_trace)

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    joint_trace.set_provenance(TraceProvenance(f"Grad transform pass (took {elapsed_time_millis} milliseconds)"))
    return joint_trace


def _group_get_grad_bsyms(trace):
    i = 0
    n = len(trace.bound_symbols)
    while i < n and trace.bound_symbols[i].sym != prims.get_grad:
        i += 1
    if i == n:
        return
    get_grad_bsyms = list(filter(lambda bsym: bsym.sym == prims.get_grad, trace.bound_symbols))
    bsyms = list(filter(lambda bsym: bsym.sym != prims.get_grad, trace.bound_symbols))
    bsyms = bsyms[:i] + list(get_grad_bsyms) + bsyms[i:]
    trace.bound_symbols = bsyms


def split_into_forward_and_backward(joint_trace: TraceCtx):
    """split a joint trace for forward and backward into separate ones"""

    # the joint trace will have the forward computation at the beginning and then the backward computation
    # from how it is constructed.
    # we split the trace:
    # - forward symbols go into forward_part_bsyms
    # - all symbols not in the forward go into backward_part_bsyms
    # the splitting is done in the reverse order of the bound symbols, and works out which bits are needed for the forward
    # from there.
    forward_part_bsyms = []
    backward_part_bsyms = []

    # return has a dict, it contains the forward outputs in "output" and the backward outputs in "grad_flat_args", corresponding
    # to "flat_args"
    return_bsym = joint_trace.bound_symbols[-1]
    assert return_bsym.sym == prims.python_return
    fw_output = return_bsym.args[0]["fw_flat_out"]
    assert isinstance(fw_output, tuple)

    grad_outs = [None for _ in fw_output]
    output_pos = {}
    for i, o in enumerate(fw_output):
        if isinstance(o, TensorProxy):
            output_pos.setdefault(o.name, []).append(i)

    # the proxies we need to compute in the forward - we start with the outputs of the forward
    forward_proxy_names = {o.name for o in tree_iter(fw_output) if isinstance(o, Proxy)}
    # we also have the inputs available, so we add flat_args.
    # for inplace, we need to update this (or have flat args be the right thing?...)
    forward_proxy_names.update(a.name for a in return_bsym.args[0]["flat_args"] if isinstance(a, Proxy))

    # loop over the bound symbols in reverse (see above)
    for bsym in reversed(joint_trace.bound_symbols):
        if bsym.sym == prims.python_return:  # we will re-do the return statements below, so we skip it here
            continue

        # get grad is always part of the input, record the grad_out (will be part of the "cotangents" list)
        if bsym.sym == prims.get_grad:
            grad_outs[output_pos[bsym.args[0].name].pop(0)] = bsym.output
            continue

        # unpack_trivial is always a (forward trace) argument, the backward inputs are from get_grad
        # anything that outputs something needed for the forward (the forward_proxy_names) is part of the forward
        if any((o.name in forward_proxy_names) for o in bsym.flat_proxy_outs) or bsym.sym == prims.unpack_trivial:
            forward_part_bsyms.insert(0, bsym.from_bsym())
            bsym_arg_names = {a.name for a in bsym.flat_proxy_args}
            bsym_out_names = {o.name for o in bsym.flat_proxy_outs if o.name not in bsym_arg_names}
            forward_proxy_names.update(bsym_arg_names)
            forward_proxy_names.update(bsym_out_names)

            continue

        # copy_ updating a forward proxy is special regardless of the output
        if (bsym.sym == prims.copy_ or bsym.sym.name == "copy_") and bsym.args[1].name in forward_proxy_names:
            # todo: should we also handle ltorch.copy_ ?
            forward_part_bsyms.insert(0, bsym.from_bsym())
            forward_proxy_names.update(a.name for a in bsym.flat_proxy_args)
            continue

        # if we don't need to have it in the forward, it is part of the backward
        backward_part_bsyms.insert(0, bsym.from_bsym())

    # collect proxies needing to be saved from the forward for the backward
    saved_for_backward = {}
    for bsym in backward_part_bsyms:
        saved_for_backward.update((a.name, a) for a in bsym.flat_proxy_args if a.name in forward_proxy_names)
    saved_for_backward_tensors = tuple(p for p in saved_for_backward.values() if isinstance(p, TensorProxy))
    saved_for_backward_other = tuple(p for p in saved_for_backward.values() if not isinstance(p, TensorProxy))

    # we build the forward trace
    forward_trace = from_trace(joint_trace)
    forward_trace.tags.add(TraceTag.AUGMENTED_FORWARD)
    forward_trace.names = forward_trace.names.copy()  ## ehem
    forward_trace.bound_symbols += forward_part_bsyms

    # now we create the return value and return bound symbol for the forward
    fw_output_dict = {k: v for k, v in return_bsym.args[0].items() if k != "fw_flat_out"}
    # replace the backward output with the forward one for the forward trace
    fw_output_dict.update({"output": return_bsym.args[0]["fw_flat_out"]})

    flat_output, _ = tree_flatten_with_dataclass(fw_output)
    fw_output_dict["flat_output"] = tuple(flat_output)
    with tracectx(forward_trace):
        prims.python_return(fw_output_dict, (saved_for_backward_tensors, saved_for_backward_other))

    if len(backward_part_bsyms) == 0 and not any(
        [True if arg is not None else False for arg in return_bsym.args[0]["output"]]
    ):
        return forward_trace, None

    # then we construct the backward trace, unpacking saved_for_backward and cotangents lists
    def backward_fn(saved_for_backward, cotangents):
        pass

    backward_trace = TraceCtx(fn=backward_fn)
    backward_trace.names = forward_trace.names
    backward_trace.name_ctr = forward_trace.name_ctr

    # TODO: make this name-agnostic
    backward_trace.names.discard("saved_for_backward")
    backward_trace.names.discard("cotangents")

    # set up the inputs of the backward properly (args and unpacking)
    with tracectx(backward_trace):
        p_C0 = CollectionProxy(list(saved_for_backward_tensors), name="C0")
        p_C1 = CollectionProxy(list(saved_for_backward_other), name="C1")
        p_saved_for_backward = CollectionProxy([p_C0, p_C1], name="saved_for_backward")
        p_cotangents = CollectionProxy(grad_outs, name="cotangents")

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
    with tracectx(backward_trace):
        prims.python_return(tuple(return_bsym.args[0]["output"]))

    backward_trace = dce(backward_trace)

    # Importing here to avoid cyclical dependencies in future.
    # NOTE: This is required only for v1 executor.
    from thunder.executors.transformer_engineex import transformer_engine_v1_bwd_fp8_meta_sync

    transformer_engine_v1_bwd_fp8_meta_sync(forward_trace, backward_trace)

    # We only want to apply it on backward trace.
    from thunder.torch.experimental.dtensor_utils import check_dtensor_cotangent_metadata_in_backward

    backward_trace = check_dtensor_cotangent_metadata_in_backward(backward_trace)

    return forward_trace, backward_trace
