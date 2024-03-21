from collections.abc import Callable

from thunder.core.prims import PrimIDs
from thunder.core.proxies import FutureTensorProxy, Proxy, TensorProxy
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx
from thunder.core.utils import check_type, ProxyDict

memory_calculate_skip_list = (PrimIDs.RETURN, "clear_collection")

# List of operators that considered no memory changes occurred in Thunder.
# NOTE: for the operators have different input and output shape, such as expand,
# the final del statement always reduce the memory size by the initial size of the input tensor
thunder_alias_operator_list = (
    "unsqueeze",
    "squeeze",
    "transpose",
    "expand",
    "reshape",
    "permute",
    "contiguous",
    "torch_wait_prim_impl",
)

# A registry of symbols that require special memory calculation;
# if not registered, the default memory calculation function is used.
memory_calculate_impls: dict[Symbol, Callable] = dict()


def register_memory_calculate_function(*ops):
    """Decorator to register memory calculation function for a list of symbol."""

    def decorator(func):
        for op in ops:
            memory_calculate_impls[op] = func
        return func

    return decorator


class MemoryData:
    """
    Helper class to track the reference count of a Proxy
    """

    def __init__(self, n: int, proxy: Proxy):
        self._cnt = n
        self._proxy = proxy

    def incr_ref(self):
        self._cnt += 1
        return self._cnt

    def decr_ref(self):
        if self._cnt == 0:
            raise ValueError(f"Refernce count of {self._proxy} cannot be negative")
        self._cnt -= 1
        return self._cnt

    def get_memory_size(self):
        check_type(self._proxy, Proxy)
        return self._proxy.numel * self._proxy.dtype.bytes


def default_alloc_memory(
    bsym: BoundSymbol, tensor_to_memory_data: ProxyDict, name_to_alloc_memory: dict[str, int]
) -> int:
    """
    The default function to calculate the memory usage, covers the most common situation where memory is increased by the size of the output tensors.

    Args:
        bsym (BoundSymbol): The bound symbol that needs to calculate the memory usage
        tensor_to_memory_data (ProxyDict): A dict to record the reference count of proxy
        name_to_alloc_memory (dict): A dict to record the bound symbol's name and the proxy names as key and the size of memory change as value

    Returns:
        int: The size of memory change caused by the input bsym
    """
    # Skip CollectionProxy(output of input unpacking operators, such as unpack_sequence)
    # and other negligible scalar types
    tensor_outs = [x for x in bsym.flat_proxy_outs if isinstance(x, (TensorProxy, FutureTensorProxy))]
    result = sum(t.numel * t.dtype.bytes for t in tensor_outs)
    for x in tensor_outs:
        # skip when the function returns its own input
        if x not in tensor_to_memory_data:
            tensor_to_memory_data[x] = MemoryData(n=1, proxy=x)

    name_to_alloc_memory[f"{bsym.sym.name} {', '.join(t.name for t in tensor_outs)}"] = result
    return result


# The reference count and original proxy of the aliases are tracked in tensor_to_memory_data
@register_memory_calculate_function(*thunder_alias_operator_list)
def track_alias_op_memory(
    bsym: BoundSymbol, tensor_to_memory_data: ProxyDict, name_to_alloc_memory: dict[str, int]
) -> int:
    inp = bsym.flat_proxy_args[0]
    out = bsym.flat_proxy_outs[0]
    assert inp in tensor_to_memory_data
    tensor_to_memory_data[inp].incr_ref()
    tensor_to_memory_data[out] = tensor_to_memory_data[inp]
    return 0


@register_memory_calculate_function(PrimIDs.DEL)
def del_op_memory(bsym: BoundSymbol, tensor_to_memory_data: ProxyDict, name_to_alloc_memory: dict[str, int]) -> int:
    tensor_args = (x for x in bsym.flat_proxy_args if isinstance(x, (TensorProxy, FutureTensorProxy)))
    memory_size = 0
    for a in tensor_args:
        assert a in tensor_to_memory_data
        cnt_a = tensor_to_memory_data[a].decr_ref()

        if cnt_a == 0:
            size_a = tensor_to_memory_data[a].get_memory_size()
            memory_size -= size_a
            name_to_alloc_memory[f"{bsym.sym.name} {a.name}"] = -size_a
    return memory_size


def get_alloc_memory(trc: TraceCtx) -> tuple[int, dict[str, int]]:
    """
    Calculate the memory usage based on the executable trace.
    The memory calculation is based only on the compile-time trace, i.e. the input and output shape
    and type of the operator, without taking into account any kind of caching e.g. CudaCachedAllocator,
    or any allocation for the intermediate within the operator (e.g.: nvFusion ops).
    Note that the trace doesn't contain strides. Only the operators in `thunder_alias_operator_list` are assumed to
    cause no memory changes, this is different from the actual memory allocation for Torch view operators, such as reshape.
    So the estimated memory usage may differ from the runtime memory usage.


    Args:
        trc (TraceCtx): The executable Thunder trace.

    Returns:
        tuple[int, dict[str, int]]: The peak memory usage and a dict mapping the symbol name and
        associated tensor names to the number of bytes of memory.
    """
    name_to_alloc_memory: dict[str, int] = dict()
    max_allocated = 0
    allocated = 0

    tensor_to_memory_data = ProxyDict()
    for bsym in trc.bound_symbols:
        if bsym.sym.id in memory_calculate_skip_list:
            continue
        impl = memory_calculate_impls.get(bsym.sym.id, default_alloc_memory)
        allocated += impl(bsym, tensor_to_memory_data, name_to_alloc_memory)

        max_allocated = max(max_allocated, allocated)

    return max_allocated, name_to_alloc_memory
