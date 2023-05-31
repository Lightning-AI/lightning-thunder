from typing import Any, Callable

import thunder
import thunder.core.script as script
from thunder.torch import _torch_to_thunder_function_map

import torch


# TODO Maybe make collect_into a set?
class CollectFunctionsUsed(torch.overrides.TorchFunctionMode):
    def __init__(self, collect_into: list):
        self.functions = set()
        self.collect_into = collect_into

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        qn = getattr(func, "__qualname__", None)
        if qn.startswith("getset_descriptor."):
            qn = getattr(func.__self__, "__qualname__", qn)
        mod = getattr(func, "__module__", None)
        if mod is not None and qn is not None and not qn.startswith(mod):
            modstr = " of " + mod
        else:
            modstr = ""
        self.functions.add((f"{qn or func}{modstr}", func))
        return func(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.collect_into.extend(sorted(self.functions))
        super().__exit__(exc_type, exc_value, traceback)


# TODO Refactor preprocessing function so there's one definition used by compile and this
def preprocess(fn, is_module):
    gr = script.frontend.acquire_method(fn.forward if is_module else fn)
    script.passes.unroll_for_loops_and_inline_modules(gr)
    if is_module:
        additional_param_names, additional_param_values = script.passes.module_to_function(gr)
    script.passes.strongly_inline_functions(gr)
    script.passes.torch_to_thunder(gr)

    thunder_fn = script.python_ir.generate_function(gr)
    if is_module:
        thunder_fn._additional_param_names = additional_param_names
        thunder_fn._additional_param_values = additional_param_values
    else:
        thunder_fn._additional_param_names = None
        thunder_fn._additional_param_values = None
    return thunder_fn


# TODO Maybe have this print additional information and return more metadata?
# TODO Add option to disable attempted preprocessing
# TODO Accept kwargs for compile_with_info (like langctx)
# TODO Add profiling (or profiling option) to determine if we have a slowdown
# TODO If an error occurs, try to minify the program to produce a smaller sample to reproduce the error
def examine(fn: Callable, *args, **kwargs):
    # Step 0, runs the operation with our torch function mode to collection information
    #   and ensure the operation itself is working correctly
    collected_ops = []
    torch_result: Any
    with CollectFunctionsUsed(collected_ops):
        try:
            torch_result = fn(*args, **kwargs)
        except Exception as e:
            print("Found an issue running the function with torch tensors!")
            raise e

    # Step 1 Identifies supported (and unsupported) operations
    supported_ops = set()
    for name, op in collected_ops:
        if op in _torch_to_thunder_function_map:
            supported_ops.add((name, op))
    unsupported_ops = set(collected_ops) - supported_ops

    print(
        f"Found {len(collected_ops)} distinct operations, of which {len(supported_ops)} ({len(supported_ops) / len(collected_ops) * 100}%) are supported"
    )

    # Terminates early if there are unsupported operations
    if len(unsupported_ops) > 0:
        print(
            "Please file an issue requesting the following operators here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )

        for name, op in unsupported_ops:
            print(f"{name}")

        return

    # Step 2 Attempts to preprocess the function
    try:
        preprocess(fn, is_module=isinstance(fn, torch.nn.Module))
    except Exception as e:
        print("Encountered an error while preprocessing the function")
        print(
            "Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )
        raise e

    # Step 3 Attempts to compile the function using lightning.compile
    try:
        cfn = thunder.compile_with_info(fn)
    except Exception as e:
        print("Encountered an error while compiling the function")
        print(
            "Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )
        raise e

    # Step 4 Attemps to execute the function using lightning.compile
    lc_result: Any
    try:
        lc_result = cfn(*args, **kwargs)
    except Exception as e:
        print("Encountered an error while running the compiled function")
        print(
            "Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )
        # TODO On failure, try to identify where the failure occurred and produce a constructive error
        #   message -- did it happen during caching, unpacking, transformation, callable construction,
        #   or executing the callable?
        raise e

    # TODO Consider comparing the torch_result and lc_result -- they might reasonably be different but we
    #   warn about this

    # TODO Consider returning additional information
    print(f"The function appears to be working as expected")
