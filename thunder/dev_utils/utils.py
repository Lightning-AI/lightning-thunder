import collections

from torch.utils.benchmark import Timer
from thunder.core.prims import PrimIDs
from thunder.core.symbol import BoundSymbol
from thunder.executors.torch_compile import make_compiled as make_torch_compile_callable

NON_COMPUTATION_PRIMS = (
    PrimIDs.ASSERT_TENSOR_METADATA,
    PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA,
    PrimIDs.CHECK_NONE,
    PrimIDs.CHECK_EMPTY,
    PrimIDs.CHECK_LITERAL_LIKE,
    PrimIDs.CHECK_TYPE,
    PrimIDs.CHECK_INSTANCE,
    PrimIDs.CHECK_NUMBER_TYPE_AND_VALUE,
    PrimIDs.CHECK_BOOL_CONVERSION,
    PrimIDs.CHECK_STRING_VALUE,
    PrimIDs.CHECK_LEN,
    PrimIDs.ASSERT_COMPARE,
    PrimIDs.PYTHON_VARS,
    PrimIDs.UNPACK_FUNCTION_OBJ,
    PrimIDs.UNPACK_CACHE_INFO,
    PrimIDs.UNPACK_ATTR,
    PrimIDs.UNPACK_GETITEM,
    PrimIDs.UNPACK_EMPTY_DICT,
    PrimIDs.UNPACK_ITER,
    PrimIDs.UNPACK_NEXT,
    PrimIDs.UNPACK_KEY,
    PrimIDs.UNPACK_SEQUENCE,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_TUPLE,
    PrimIDs.UNPACK_LIST,
    PrimIDs.UNPACK_DICT_KEY,
    PrimIDs.CONSTRUCT_TUPLE,
    PrimIDs.PACK_LIST,
    PrimIDs.PACK_SETITEM,
    # TODO: UNPACK_SET
    # Utility prims
    PrimIDs.COMMENT,
    PrimIDs.DEL,
    PrimIDs.PRINT,
    PrimIDs.RETURN,
)

BenchmarkComparisonData = collections.namedtuple(
    "BenchmarkComparisonData",
    [
        "nvfuser_walltime",
        "torch_compile_walltime",
        "nvfuser_kernel_time",
        "torch_compile_kernel_time",
        "nvfuser_profiler_data",
    ],
)


def _benchmark_fusion_region_with_nvfuser_and_torch_compile(bsym: BoundSymbol) -> BenchmarkComparisonData:
    """
    Benchmark the performance of nvFuser and torch.compile for a given fusion region.

    This function takes a BoundSymbol generated from nvFuser and performs the following:
    1. Executes the fusion region using both nvFuser and torch.compile.
    2. Measures wall time and kernel time for both implementations.
    3. Collects profiling data for the nvFuser implementation.

    Args:
        bsym (BoundSymbol): A BoundSymbol generated from nvFuser.

    Returns:
        BenchmarkComparisonData: A named tuple containing:
            - nvfuser_walltime: Wall time for nvFuser execution using `torch.utils.benchmark.Timer`.
            - torch_compile_walltime: Wall time for torch.compile execution using `torch.utils.benchmark.Timer`.
            - nvfuser_kernel_time: Kernel time for nvFuser execution using `triton.testing.do_bench`.
            - torch_compile_kernel_time: Kernel time for torch.compile execution using `triton.testing.do_bench`.
            - nvfuser_profiler_data: Profiling data for the nvFuser implementation by calling `fusion_defition.profile`.

    .. note:: The function assumes that the fusion has been previously executed and inputs are recorded.
    """
    assert "nvFusion" in bsym.sym.name, "Expected the BoundSymbol to be generated from nvFuser"
    import triton  # Import triton here as it may not be available in CPU only setting.

    nvfuser_callable = bsym._call_ctx[bsym.sym.name]
    inputs = nvfuser_callable.last_inputs
    if nvfuser_callable.last_used is None:
        raise RuntimeError(
            "Fusion definition needs to be executed to record the inputs. You must execute the fusion first before you can query the repro."
        )

    if nvfuser_callable.last_inputs is None:
        raise RuntimeError(
            "Fusion definition inputs need to be recorded. Use compile option 'nv_store_fusion_inputs=True' while tracing."
        )

    torch_compile_callable = make_torch_compile_callable(bsym.subsymbols, bsym.flat_args, bsym.flat_outs)

    nvfuser_callable(*inputs)
    torch_compile_callable(*inputs)

    nvfuser_timer = Timer("nvfuser_callable(*inputs)", globals={"nvfuser_callable": nvfuser_callable, "inputs": inputs})
    tc_timer = Timer(
        "torch_compile_callable(*inputs)", globals={"torch_compile_callable": torch_compile_callable, "inputs": inputs}
    )

    # Wall times
    wall_time_nvfuser = nvfuser_timer.blocked_autorange(min_run_time=2)
    wall_time_tc = tc_timer.blocked_autorange(min_run_time=2)

    # Kernel Times
    kernel_time_nvfuser = triton.testing.do_bench(lambda: nvfuser_callable(*inputs), return_mode="median")
    kernel_time_tc = triton.testing.do_bench(lambda: torch_compile_callable(*inputs), return_mode="median")

    # nvFuser's profiling utility.
    fd = nvfuser_callable.get_fd(nvfuser_callable.to_descriptors(inputs))
    fd.execute(inputs, profile=True)
    nvfuser_prof = fd.profile()

    return BenchmarkComparisonData(wall_time_nvfuser, wall_time_tc, kernel_time_nvfuser, kernel_time_tc, nvfuser_prof)
