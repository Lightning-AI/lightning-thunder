from thunder.core import utils
from thunder.core import prims


def get_orig_and_thunder_module_proxies_from_prologue(prologue_trace):
    modules_and_thunder_modules = [
        (bsym.args[0], bsym.output) for bsym in prologue_trace.bound_symbols if bsym.sym is prims.unpack_thunder_module
    ]

    if len(modules_and_thunder_modules) != 1:
        raise NotImplementedError("cannot deal with modules other than the compiled module")

    ((orig_module_proxy, thunder_module_proxy),) = modules_and_thunder_modules
    if prologue_producers[orig_module_proxy].sym is not prims.unpack_function_obj:
        raise NotImplementedError("original module does not match the compiled module")

    return orig_module_proxy, thunder_module_proxy


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
            assert (
                len(checks) == 1
            ), f"expected each parameter and buffer to have exactly one checker, but {bsym.output} has {len(checks)}"
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
