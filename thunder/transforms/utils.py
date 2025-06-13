import thunder
from thunder.core import utils
from thunder.core import prims
from thunder.core.trace import TraceCtx
from thunder.core.pytree import tree_map


def get_checks(prologue_trace):
    # returns a dictionary mapping model param names to (check bsym, get param bsym
    check_dict = {}
    prologue_producers, prologue_consumers = utils.producers_and_consumers(prologue_trace)
    for bsym in prologue_trace.bound_symbols:
        if bsym.sym == prims.unpack_parameter or bsym.sym == prims.unpack_buffer:
            param_thunder_module, param_name = bsym.args
            checks = [
                bsym2 for bsym2 in prologue_consumers[bsym.output] if bsym2.sym == prims.check_tensor_shape_and_metadata
            ]
            assert len(checks) == 1, (
                f"expected each parameter and buffer to have exactly one checker, but {bsym.output} has {len(checks)}"
            )
            assert isinstance(param_name, str)
            check_dict[param_name] = (checks[0], bsym)
    return check_dict


def add_trace_output(trace, output, subindex: int | None = None):
    ret_node = trace.bound_symbols[-1]
    assert ret_node.sym == prims.python_return
    assert len(ret_node.args) == 1
    (ret_args,) = ret_node.args

    if subindex is None:
        ret_args = (*ret_args, output)
    else:
        assert isinstance(ret_args[subindex], tuple)
        ret_args = (*ret_args[:subindex], (*ret_args[subindex], output), *ret_args[subindex + 1 :])

    ret_node.args = (ret_args,)


def trace_with_replaced_proxy_metadata(trace: TraceCtx, proxy_replacement_metadata) -> TraceCtx:
    t = TraceCtx(trace.fn)

    proxymap: dict[str, thunder.Proxy] = {}

    def map_proxy(p):
        if isinstance(p, thunder.Proxy):
            return proxymap[p.name]
        return p

    def create_proxy(p):
        if isinstance(p, thunder.Proxy):
            if p.name in proxymap:  # happens with subsymbols
                return proxymap[p.name]
            with thunder.core.trace.tracectx(t):
                np = p.replace(**proxy_replacement_metadata.get(p.name, {}))
                proxymap[p.name] = np
                return np
        return p

    def process_bound_symbols(src_bound_symbols, target_bound_symbols):
        for bsym in src_bound_symbols:
            new_args = tree_map(map_proxy, bsym.args)
            new_kwargs = tree_map(map_proxy, bsym.kwargs)
            new_output = tree_map(create_proxy, bsym.output)
            new_bsym = bsym.from_bsym(output=new_output, args=new_args, kwargs=new_kwargs, subsymbols=[])
            target_bound_symbols.append(new_bsym)
            if len(bsym.subsymbols) > 0:
                process_bound_symbols(bsym.subsymbols, new_bsym.subsymbols)

    process_bound_symbols(trace.bound_symbols, t.bound_symbols)

    t.args = tree_map(map_proxy, trace.args)
    t.kwargs = tree_map(map_proxy, trace.kwargs)
    t._siginfo = trace._siginfo
    return t
