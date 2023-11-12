import copy
import inspect
from inspect import Parameter, Signature
from itertools import chain
from collections.abc import Callable

from thunder.core import prims, utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import variableify, Proxy
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceCtx
from thunder.core.transform_common import dce


def make_aug_forward_and_backward(bsym: BoundSymbol) -> tuple[Callable, Callable]:
    """
    Given a bound symbol, return a pair of forward and backward functions
    implementing the forward and backward passes for the given bound symbol.

    The forward function is the same as the original bound symbol's function,
    but with the addition of saving the intermediates from the forward function
    in a tuple and returning that tuple as the second output.

    The backward function is a pure function that takes as inputs the saved
    intermediates from the forward function and the inputs to the backward
    function, and returns the gradients of the inputs to the forward function.

    Args:
        bsym: The bound symbol to split into forward and backward functions.

    Returns:
        A pair of forward and backward functions.
    """
    import thunder
    from thunder.core.transforms import _grad_fn_map

    joint_forward_backward = _grad_fn_map.get(bsym.sym.id, None)
    utils.check(
        joint_forward_backward is not None,
        lambda: f"Cannot generate forward and backward functions for {bsym.sym.name}",
    )
    joint_trace = thunder.trace(inline_trace=False, use_dce=False)(joint_forward_backward, *bsym.args, **bsym.kwargs)
    consumers = utils.consumers(joint_trace)

    def find_backward_input(forward_output):
        output_consumers = consumers.get(forward_output, None)
        if output_consumers is None or not output_consumers:
            return None
        get_grad_bsym = next(
            filter(lambda bsym: bsym.sym.id == PrimIDs.GET_GRAD, output_consumers),
            None,
        )
        return get_grad_bsym.output if get_grad_bsym is not None else None

    def find_backward_output(forward_input):
        forward_input_consumers = consumers.get(forward_input, None)
        if forward_input_consumers is None or not forward_input_consumers:
            return None
        put_grad_bsym = next(
            filter(lambda bsym: bsym.sym.id == PrimIDs.PUT_GRAD, forward_input_consumers),
            None,
        )
        return put_grad_bsym.args[1] if put_grad_bsym is not None else None

    bw_inputs = tree_map(find_backward_input, utils.sequencify(joint_trace.output))
    bw_outputs_args = tree_map(find_backward_output, joint_trace.args)
    bw_outputs_kwargs = tree_map(find_backward_output, joint_trace.kwargs)
    meta_parameters = inspect.signature(bsym.sym.meta).parameters
    bw_outputs = {name: bw_output for name, bw_output in utils.safe_zip(meta_parameters, bw_outputs_args)}
    bw_outputs = bw_outputs | bw_outputs_kwargs
    flat_bw_outputs, _ = tree_flatten(bw_outputs)

    backward_bsyms = utils.find_producer_symbols(joint_trace, flat_bw_outputs, bw_inputs)
    unpacking_ops = (
        prims.PrimIDs.UNPACK_EMPTY_DICT,
        prims.PrimIDs.UNPACK_KEY,
        prims.PrimIDs.UNPACK_SEQUENCE,
        prims.PrimIDs.UNPACK_TRIVIAL,
    )
    backward_bsyms = [bsym for bsym in backward_bsyms if bsym.sym.id not in unpacking_ops]
    backward_bsyms.append(prims.python_return.bind(bw_outputs, output=()))

    # Find required info from forward trace for backward trace
    backward_producers = utils.producers(backward_bsyms)
    saved_for_backward = []
    for backward_bsym in backward_bsyms:
        for arg in chain(backward_bsym._flat_args, backward_bsym._flat_kwargs):
            if not isinstance(arg, Proxy):
                continue
            if arg not in backward_producers and variableify(arg) not in map(variableify, tree_flatten(bw_inputs)[0]):
                saved_for_backward.append(arg)

    backward_params = [Parameter(x.name, Parameter.POSITIONAL_OR_KEYWORD) for x in chain(saved_for_backward, bw_inputs)]
    backward_signature = Signature(backward_params)

    def backward_fn():
        pass

    backward_fn.__signature__ = backward_signature
    backward_fn.__name__ = bsym.sym.name + "_backward"

    # Finally, build the backward trace
    backward_trace = TraceCtx(backward_fn)
    backward_trace.args = (*saved_for_backward, *bw_inputs)
    backward_trace.kwargs = {}
    backward_trace.bound_symbols = backward_bsyms

    # Augment forward trace to include saved_for_backward as output
    augmented_forward_trace = from_trace(joint_trace)
    augmented_forward_trace.bound_symbols = copy.copy(joint_trace.bound_symbols)
    return_bsym = augmented_forward_trace.bound_symbols[-1]
    assert return_bsym.sym.id == PrimIDs.RETURN
    augmented_forward_trace.bound_symbols[-1] = prims.python_return.bind(
        (joint_trace.output, saved_for_backward), output=()
    )
    # Remove put/get grad from augmented forward trace
    augmented_forward_trace = dce(augmented_forward_trace)

    return augmented_forward_trace.python_callable(), backward_trace.python_callable()
