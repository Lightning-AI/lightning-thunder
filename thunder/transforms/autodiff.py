import thunder.core.transforms
from thunder.core.transforms import ForwardBackwardTraces

from thunder.core import prims
from thunder.core.transforms import (
    is_constant_for_vjp,
    _get_gradfn_and_executor,
    augmented_forward_impls,
    backward_impls,
)
from thunder.core.utils import check
from thunder.core.vjp_utils import make_aug_forward_and_backward
from thunder.core.pytree import tree_map
import thunder
import time


# Transforms a trace by determining which grad transforms to call given the list of executors in priority order
# This pass tries to preserve the original trace and proxies.
def grad_transform_on_trace(trace, /, *args, **kwargs):
    # This processes the bsyms to map symbols to operator executors:
    # - in the order of the executor list
    #   - if the executor defines a grad transform, call that to
    #     get fw + bw,
    # - if there is a augmented forward registered, use that and the backwared registered,
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
                # This is big (and a big messy):
                # the return (of the input trace) signals the end of the forward processing
                # all the backwards will have been put in self.collected_bw_part_syms
                # special symbols are
                # - get_grad "reads" the gradient of a variable from the forward to compute with it
                #   (if you are used to torch.autograd.Function, this is a "grad_out" in the arguments
                #    of the backward static method)
                # - put_grad "writes" a computed gradients (similar to returning the "grad_in" in t.a.F)
                # The backward computation in self.collected_bw_part_syms is in the oder of th
                # backward computation.
                #
                input_proxy_names = {p.name for p in bsym.args[0]["flat_args"] if isinstance(p, thunder.Proxy)}
                output_proxy_names = set()
                for o in thunder.core.pytree.tree_iter(bsym.args[0]["output"]):
                    if isinstance(o, thunder.Proxy):
                        output_proxy_names.add(self.read(o).name)
                grad_proxy_map = {}

                # we iterate through the collected symbols in order
                for bw_bsyms in self.collected_bw_part_bsyms:
                    # if we don't have and grad_outs, we know that the grad_ins are not needed, so we don't compute them
                    # (e.g. for `where(a * b > 0, a, b)` we do not need the grad components of a * b propagated
                    if any(
                        ((gg.name in grad_proxy_map) or (gg.name in output_proxy_names))
                        for gg in (bs.args[0] for bs in bw_bsyms)
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
                                p = nbsym.args[0]
                                # TODO: set requires grad here!
                                name = p.name
                                current_grad = grad_proxy_map.get(name)
                                if current_grad is not None:
                                    # do we also need to map?
                                    self.write(nbsym.output, current_grad)
                                    self.swap_map[thunder.core.proxies.variableify(nbsym.output)] = current_grad
                                    # XXX_replace_output_with_current_grad
                                elif name in output_proxy_names:
                                    new_grad = self.add_processed_bsyms([nbsym.from_bsym()])
                                    grad_proxy_map[name] = new_grad
                                # elif KNOWN proxy:
                                #    ### TODO: it can happen that an intermediate does not have put_grad, e.g. first arg to where
                                else:
                                    raise RuntimeError(f"grad of non-output and non-intermediate {name} requested")
                            else:
                                self.add_processed_bsyms([nbsym.from_bsym()])
                self.collected_bw_part_bsyms.clear()

                grad_flat_args = []
                for p in bsym.args[0]["flat_args"]:
                    # or p = self.read(p) here?
                    if isinstance(p, thunder.Proxy) and p.requires_grad and p.name in grad_proxy_map:
                        # is it always OK if we don't have a gradient? (one case: unused input)
                        # result of put_grad???
                        grad_flat_args.append(grad_proxy_map[p.name])
                    else:
                        grad_flat_args.append(None)

                bsym.args[0]["grad_flat_args"] = grad_flat_args
                self.add_processed_bsyms([bsym.from_bsym()])
                self.set_result(bsym.output)
                return

            # constant for gradients (no grad required)
            if is_constant_for_vjp(bsym):
                if bsym.sym.name == "synchronize":
                    # This is a *really* terrible hack to cope with non-grad-needing sharded tensors
                    # as required by LoRA.
                    from thunder.distributed.prims import all_gather

                    def synchronize_impl(a, group):
                        return all_gather(a, group, True).wait()

                    self.add_bsyms_from_function(synchronize_impl, *bsym.args, **bsym.kwargs)
                    return

                self.add_processed_bsyms([bsym.from_bsym()])
                self.set_result(bsym.output)
                return

            # executor or global grad transform
            joint_forward_backward, _ = _get_gradfn_and_executor(bsym)

            if joint_forward_backward is None:
                # registered augmented forward impl (where aug fwd and backward are registered separately)
                aug_fwd_impl = augmented_forward_impls.get(bsym.sym.id)
                if aug_fwd_impl is not None:
                    bwd_impl = backward_impls.get(bsym.sym.id)

                    def joint_forward_backward(*args, **kwargs):
                        aug_fwd_res = aug_fwd_impl(*bsym.args, **bsym.kwargs)
                        if isinstance(aug_fwd_res, thunder.core.transforms.VJPDual):
                            res, saved_for_backward = aug_fwd_res
                        else:
                            res, saved_for_backward = aug_fwd_res
                        grad_outs = []
                        for r in thunder.core.pytree.tree_iter(res):
                            if isinstance(r, thunder.TensorProxy):
                                grad_outs.append(prims.get_grad(r))
                        # to do non-grad outputs of bwd
                        grad_inps = bwd_impl(*saved_for_backward, *grad_outs)
                        if isinstance(grad_inps, thunder.Proxy):
                            grad_inps = [grad_inps]

                        flat_inps = args
                        # there may be non-gradient requiring additional args (todo: maybe only support this for non-tensor ones?)
                        assert len(grad_inps) <= len(flat_inps)
                        for i, gi in zip(flat_inps, grad_inps):
                            if isinstance(i, thunder.TensorProxy):
                                prims.put_grad(i, gi)
                        return res

            if joint_forward_backward is not None:
                self.new_trace.push_scope([])
                result = joint_forward_backward(*bsym.args, **bsym.kwargs)
                self.set_result(result)
                new_bsyms = self.new_trace.pop_scope()

                forward_part_bsyms = []
                backward_part_bsyms = []
                backward_proxy_names = set()
                for nbsym in new_bsyms:
                    argnames = {p.name for p in nbsym.flat_proxy_args}
                    if nbsym.sym in {prims.get_grad, prims.put_grad} or any(
                        n in backward_proxy_names for n in argnames
                    ):
                        backward_proxy_names.update(p.name for p in nbsym.flat_proxy_outs if p.name not in argnames)
                        backward_part_bsyms.append(nbsym)
                    else:
                        forward_part_bsyms.append(nbsym)

                self.add_processed_bsyms(forward_part_bsyms)

                self.collected_bw_part_bsyms.insert(0, backward_part_bsyms)
                return

            # No gradient transform found, need to descend
            check(not bsym.sym.is_prim, lambda: f"Failed to find a gradient transform for bound symbol {bsym=}")

            # TODO: check if this is needed: the old impl checked whether len(bsym.subsymbols) > 0 except for the special case "torch.nn.functional.dropout" with p=0...

            # OUTPUTS to map
            self.add_unprocessed_bsyms(bsym.subsymbols[:])

    start_time_ns = time.perf_counter_ns()

    processor = AugmentedForwardProcessor(trace)
    trace, _ = processor()
    trace = thunder.core.transform_common.dce(trace)

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    trace.set_provenance(
        thunder.core.trace.TraceProvenance(f"Grad transform pass (took {elapsed_time_millis} milliseconds)")
    )
    return trace


def split_into_forward_and_backward(joint_trace):
    forward_part_bsyms = []
    backward_part_bsyms = []

    return_bsym = joint_trace.bound_symbols[-1]
    assert return_bsym.sym == prims.python_return

    fw_output = return_bsym.args[0]["output"]
    assert isinstance(fw_output, tuple)

    grad_outs = [None for _ in fw_output]

    output_pos = {o.name: i for i, o in enumerate(fw_output) if isinstance(o, thunder.TensorProxy)}
    forward_proxy_names = {o.name for o in thunder.core.pytree.tree_iter(fw_output) if isinstance(o, thunder.Proxy)}

    for bsym in reversed(joint_trace.bound_symbols):
        if bsym.sym == prims.python_return:
            continue

        if any((o.name in forward_proxy_names) for o in bsym.flat_proxy_outs):
            forward_part_bsyms.insert(0, bsym.from_bsym())
            forward_proxy_names.update(a.name for a in bsym.flat_proxy_args)
            forward_proxy_names.update(o.name for o in bsym.flat_proxy_outs)
            continue

        if bsym.sym == prims.get_grad:
            grad_outs[output_pos[bsym.args[0].name]] = bsym.output
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
        prims.python_return(return_bsym.args[0]["grad_flat_args"])

    return forward_trace, backward_trace


def forward_and_backward_from_trace(trace: thunder.core.trace.TraceCtx, torch_autograd=False) -> ForwardBackwardTraces:
    if not torch_autograd:
        return thunder.core.transforms.forward_and_backward_from_trace(trace, torch_autograd=torch_autograd)

    joint_trace = grad_transform_on_trace(trace)

    forward_trace, backward_trace = split_into_forward_and_backward(joint_trace)
    return ForwardBackwardTraces(forward_trace, backward_trace)
