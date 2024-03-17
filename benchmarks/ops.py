import argparse
from itertools import product

# These benchmarks currenly rely on PyTorch
import torch
from torch.utils.benchmark import Timer

import thunder
import thunder.core.dtypes as datatypes
import thunder.tests as testing

# Programmatically generated operator benchmarks


# TODO: add choice of executors
# TODO: allow setting a specific device, not just the devicetype
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", action="append", help="add, var, ...")
    parser.add_argument("--devicetypes", action="append", help="cpu, cuda")
    parser.add_argument("--dtypes", action="append", help="float32, int64, ...")
    return parser.parse_args()


def _resolve_op(name):
    """Tries to extract the operator name from the thunder namespace directly, and if that fails assumes it's a core
    language operator."""
    try:
        getattr(thunder, name)
    except AttributeError:
        return getattr(thunder.core.lang, name)


def _timer_helper(fn, args, kwargs=None, iters=10):
    """A wrapper around PyTorch's timer.

    Returns a Measurement object, described here:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/benchmark/utils/common.py
    """
    kwargs = kwargs or {}
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt


# TODO: make whether using cached or non-cached executors a flag
# TODO: what statistics do we want to return here, and in what return format?
def _benchmark_op_helper(opinfo, devicetype, dtype, executor):
    op = executor.make_callable(opinfo.op)

    for name, benchmark in opinfo.benchmarks(devicetype, dtype):
        tt = _timer_helper(op, args=benchmark.args, kwargs=benchmark.kwargs)
        print(f"{name}, {tt.median}")


# TODO: make a return format so this is independently callable
def benchmark_op(*, opinfo, devicetypes, dtypes, executors):
    print(f"op={opinfo.name}")
    for devicetype, dtype, executor in product(devicetypes, dtypes, executors):
        if not executor.supports_devicetype(devicetype) or not executor.supports_dtype(dtype):
            continue

        print(f"devicetype={devicetype}, dtype={dtype}, executor={executor.name}")

        _benchmark_op_helper(opinfo, devicetype, dtype, executor)


if __name__ == "__main__":
    args = parse_args()

    # Acquires ops and their opinfos
    ops = tuple(_resolve_op(op) for op in args.ops)
    opinfos = tuple(opinfo for opinfo in testing.opinfos if opinfo.op in ops)

    for opinfo in opinfos:
        if opinfo.benchmark_generator is None:
            raise ValueError(f"Op {opinfo.name} doesn't have a benchmark generator!")

    # Acquires devicetypes
    devicetypes = set(testing.available_device_types())
    if args.devicetypes is not None:
        devicetypes = devicetypes.intersection(set(args.devicetypes))

    # Acquires dtypes
    if args.dtypes is None or len(args.dtypes) == 0:
        raise ValueError("Expected at least one dtype, specified with --dtypes ")

    dtypes = tuple(getattr(thunder, dtype) for dtype in args.dtypes)
    for dtype in dtypes:
        if not thunder.dtypes.is_dtype(dtype):
            raise ValueError(f"Unknown dtype {dtype} specified!")

    # TODO: allow the user to set this in the future
    executors = testing.benchmark_executors()

    for opinfo in opinfos:
        benchmark_op(opinfo=opinfo, devicetypes=devicetypes, dtypes=dtypes, executors=executors)
