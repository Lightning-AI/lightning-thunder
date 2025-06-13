import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from inspect import Parameter, Signature
from itertools import chain

from thunder.core import prims, utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import Proxy, variableify, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceCtx, TraceTag
from thunder.core.transform_common import dce


_cache = {}


def disable_caching_split_forward_and_backward(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._disable_caching = True

    return wrapper


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
    from thunder.common import _make_cache_key
    from thunder.core.transforms import _get_gradfn_and_executor, eval_trace

    joint_forward_backward, executor = _get_gradfn_and_executor(bsym)
    utils.check(
        joint_forward_backward is not None,
        lambda: f"Cannot generate forward and backward functions for {bsym.sym.name}",
    )
    key = (bsym.sym, executor, subkey := _make_cache_key(bsym.args, bsym.kwargs))
    cached_result = _cache.get(key, None) if subkey is not None else None
    if cached_result is not None and not getattr(joint_forward_backward, "_disable_caching", False):
        return cached_result

    # dce is necessary to remove duplicated shape queries, otherwise the trace might overwritten NumberProxy variables
    joint_trace = thunder.trace(inline_trace=False, use_dce=True)(joint_forward_backward, *bsym.args, **bsym.kwargs)
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
    meta_parameters = {
        name: param
        for name, param in meta_parameters.items()
        if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY)
    }
    bw_outputs = {name: bw_output for name, bw_output in utils.safe_zip(meta_parameters, bw_outputs_args)}
    bw_outputs = bw_outputs | bw_outputs_kwargs
    flat_bw_outputs, _ = tree_flatten(bw_outputs)

    backward_bsyms = utils.find_producer_symbols(joint_trace, flat_bw_outputs, tree_flatten(bw_inputs)[0])
    skip = (
        prims.PrimIDs.UNPACK_EMPTY_DICT,
        prims.PrimIDs.UNPACK_KEY,
        prims.PrimIDs.UNPACK_SEQUENCE,
        prims.PrimIDs.UNPACK_TRIVIAL,
        prims.PrimIDs.GET_GRAD,
    )
    backward_bsyms = [bsym for bsym in backward_bsyms if bsym.sym.id not in skip]
    backward_bsyms.append(prims.python_return.bind(bw_outputs, output=None))

    forward_input_proxies = tree_flatten((joint_trace.args, joint_trace.kwargs))[0]
    forward_input_proxies = [arg for arg in forward_input_proxies if isinstance(arg, Proxy)]
    forward_bsyms = utils.find_producer_symbols(joint_trace, tree_flatten(joint_trace.output)[0], forward_input_proxies)
    backward_bsyms = [bsym for bsym in backward_bsyms if bsym not in forward_bsyms]

    # Find required info from forward trace for backward trace
    backward_producers = utils.producers(backward_bsyms)
    saved_for_backward = []
    for backward_bsym in backward_bsyms:
        for arg in backward_bsym.flat_args:
            if not isinstance(arg, Proxy):
                continue
            if arg not in backward_producers and variableify(arg) not in map(variableify, tree_flatten(bw_inputs)[0]):
                saved_for_backward.append(arg)

    saved_for_backward = list({variableify(arg): arg for arg in saved_for_backward}.values())

    # Augment forward trace to include saved_for_backward as output
    augmented_forward_trace = from_trace(joint_trace)
    augmented_forward_trace.bound_symbols = [
        b for b in joint_trace.bound_symbols if b.sym.id not in (PrimIDs.PUT_GRAD, PrimIDs.GET_GRAD)
    ]
    return_bsym = augmented_forward_trace.bound_symbols[-1]
    assert return_bsym.sym.id == PrimIDs.RETURN
    augmented_forward_trace.bound_symbols[-1] = prims.python_return.bind(
        (joint_trace.output, saved_for_backward), output=None
    )
    # Remove put/get grad and backward symbols from augmented forward trace
    augmented_forward_trace = dce(augmented_forward_trace)

    # Check that the number of outputs of the original forward function is the
    # same as the number of primal outputs of the augmented forward trace
    utils.check(
        len(utils.sequencify(bsym.output)) == len(utils.sequencify(augmented_forward_trace.output[0])),
        lambda: f"While generating forward and backward functions for {bsym.sym.name}, encountered an error.\n"
        "The number of outputs of the original forward function must be the same as the number of primal outputs of the augmented forward trace.\n"
        f"Number of outputs of the original forward function: {len(utils.sequencify(bsym.output))}\n"
        f"Number of primal outputs of the augmented forward trace: {len(utils.sequencify(augmented_forward_trace.output[0]))}\n"
        "Please check the forward function and the augmented forward trace to ensure that they have the same number of outputs.",
    )

    # Check if any of the bound symbols in the backward trace are also in the
    # augmented forward trace
    # If so, remove them from the backward trace
    same_bsyms = set(augmented_forward_trace.bound_symbols) & set(backward_bsyms)
    if same_bsyms:
        backward_bsyms = [bsym for bsym in backward_bsyms if bsym not in same_bsyms]
        additional_saved = [o for bsym in same_bsyms for o in bsym.flat_proxy_outs]
        saved_for_backward += list({variableify(arg): arg for arg in additional_saved}.values())
        augmented_forward_trace.bound_symbols[-1] = prims.python_return.bind(
            (joint_trace.output, saved_for_backward), output=None
        )

    backward_params = [
        Parameter(getattr(x, "name", f"arg{i}"), Parameter.POSITIONAL_OR_KEYWORD)
        for i, x in enumerate(chain(saved_for_backward, bw_inputs))
    ]
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

    # Creating new functions instead of using partial to avoid limitations in
    # codeutils.get_siginfo
    # https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/core/codeutils.py#L349-L353
    def fw_fn(*args, **kwargs):
        return eval_trace(augmented_forward_trace, *args, **kwargs)

    def bw_fn(*args, **kwargs):
        return eval_trace(backward_trace, *args, **kwargs)

    _cache[key] = fw_fn, bw_fn

    return fw_fn, bw_fn


def get_saved_for_backward_tensors(trace: TraceCtx) -> tuple[TensorProxy]:
    """
    Given a trace, return the tensors that are saved for backward in the trace.

    Args:
        trace: The trace to extract saved tensors from.

    Example:
        >>> import thunder
        >>> import torch
        >>> from thunder.core.vjp_utils import get_saved_for_backward_tensors
        >>> def forward(a, b): return a * b
        >>> a = torch.tensor(1.0, requires_grad=True)
        >>> b = torch.tensor(2.0, requires_grad=True)
        >>> jitted_forward = thunder.jit(forward)
        >>> jitted_forward(a, b)
        tensor(2., grad_fn=<ThunderFunctionBackward>)
        >>> trace = thunder.last_traces(jitted_forward)[-1]
        >>> saved_tensors = get_saved_for_backward_tensors(trace)
        >>> print(saved_tensors)
        (<TensorProxy(name="a", dtype=thunder.dtypes.float32, shape=())>, <TensorProxy(name="b", dtype=thunder.dtypes.float32, shape=())>)
    """
    # First let's check if the trace was generated by Thunder's automatic differentiation
    utils.check(
        TraceTag.AUGMENTED_FORWARD in trace.tags,
        lambda: f"The trace must be generated by Thunder's automatic differentiation, not {trace.get_provenance()}",
    )
    # This location might change if the implementation of the automatic
    # differentiation transform changes. The saved tensors are the second output
    # of the return statement. There's a prototype changing the saved tensors to
    # be part of the output of a special symbol
    # https://github.com/Lightning-AI/lightning-thunder/pull/214
    saved_tensors = trace.output[1][0]
    utils.check(
        all(isinstance(t, TensorProxy) or t is None for t in saved_tensors),
        lambda: "All saved tensors must be TensorProxy or None",
    )
    return tuple(saved_tensors)


def set_saved_for_backward_tensors(trace: TraceCtx, saved_tensors: Sequence[TensorProxy]):
    """
    Given a trace, return the tensors that are saved for backward in the trace.

    Args:
        trace: The trace to set saved tensors for.
        saved_tensors: proxies for the tensors to save.
    """
    utils.check(
        all(isinstance(t, TensorProxy) or t is None for t in saved_tensors),
        lambda: "All saved tensors must be TensorProxy or None",
    )
    ret_node = trace.bound_symbols.pop(-1)
    assert ret_node.sym == prims.python_return
    output = ret_node.args
    output = (output[0], (tuple(saved_tensors), *output[1][1:]), *output[2:])
    trace.bound_symbols.append(ret_node.from_bsym(args=output))
