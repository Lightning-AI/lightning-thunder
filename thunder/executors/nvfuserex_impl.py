from dataclasses import dataclass
from functools import partial, lru_cache
from numbers import Number
from typing import Union, List, Any, Optional, Dict, Set, Tuple, Type
from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Sequence
import time
from copy import copy
from itertools import chain

from looseversion import LooseVersion
import torch

import thunder.core.dtypes as dtypes
import thunder.torch as ltorch
from thunder.core import prims, utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy, variableify, unvariableify, Variable, pyval
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.utils import OrderedSet, check
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.symbol import BoundSymbol, Symbol, has_tags
from thunder.core.devices import Device, DeviceType
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable
from thunder.core.transform_common import dce

from thunder.executors.utils import Region
from thunder.extend import FusionExecutor, register_executor, add_default_executor

# NOTE This impl file is here because nvFuser may not be available, so it's imported conditionally
#   by nvfuserex.py when nvFuser is available.
import nvfuser
from nvfuser import DataType, FusionDefinition, compute_contiguity as nv_compute_contiguity

nvTensor = nvfuser._C.Tensor
nvNumber = nvfuser._C.Scalar
nv_version: LooseVersion = LooseVersion(nvfuser.version())

#
# Helper functions
#

_lcdtype_to_nvdtype_map: dict[None | type | dtypes.dtype, DataType] = {
    dtypes.complex128: DataType.ComplexDouble,
    dtypes.complex64: DataType.ComplexFloat,
    dtypes.float64: DataType.Double,
    dtypes.float32: DataType.Float,
    dtypes.float16: DataType.Half,
    dtypes.bfloat16: DataType.BFloat16,
    dtypes.int64: DataType.Int,
    dtypes.int32: DataType.Int32,
    dtypes.bool8: DataType.Bool,
    dtypes.complex128_: DataType.ComplexDouble,
    dtypes.complex64_: DataType.ComplexFloat,
    dtypes.float64_: DataType.Double,
    dtypes.float32_: DataType.Float,
    dtypes.float16_: DataType.Half,
    dtypes.bfloat16_: DataType.BFloat16,
    dtypes.int64_: DataType.Int,
    dtypes.int32_: DataType.Int32,
    dtypes.bool8_: DataType.Bool,
    # Number types
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
    # Null types
    None: DataType.Null,
}


def lcdtype_to_nvdtype(lcdtype: type | dtypes.dtype) -> DataType:
    return _lcdtype_to_nvdtype_map[lcdtype]


# TODO What kind of constants can nvFuser support?
# TODO Is there a better type annotation for an nvConstant?
# TODO Handle devices!
# Helper to map objects to nvFuser fusion definitions
def _define_constant(fd: FusionDefinition, constant: Any) -> Any:
    if isinstance(constant, Number):
        val = pyval(constant)
        nvdtype = lcdtype_to_nvdtype(type(val))
        if nv_version >= LooseVersion("0.0.14"):
            return fd.define_scalar(constant, nvdtype)
        else:
            return fd.define_constant(constant, nvdtype)
    if isinstance(constant, (dtypes.dtype, type)):
        return lcdtype_to_nvdtype(constant)
    if isinstance(constant, Device):
        return None

    utils.check(False, lambda: f"Cannot translate {constant} of type {type(constant)} into an nvFuser constant")


def getnv(x: Any, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    if isinstance(x, Proxy):
        return lc_to_nv_map[x]
    if isinstance(x, (Number, dtypes.dtype, type, Device)):
        return _define_constant(fd, x)

    utils.check(False, lambda: f"Cannot translate {x} of type {type(x)} to an nvFuser object")


# TODO Check the CUDA arch?
def is_supported_device(device: Device) -> bool:
    utils.check_type(device, Device)
    return device.devicetype is DeviceType.CUDA


def is_supported_devicetype(devicetype: DeviceType) -> bool:
    utils.check_type(devicetype, DeviceType)
    return devicetype is DeviceType.CUDA


_low_precision_floats = (dtypes.float16, dtypes.float16_, dtypes.bfloat16, dtypes.bfloat16_)


def is_supported_dtype(dtype: type | dtypes.dtype, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(dtype, (type, dtypes.dtype))

    if not allow_low_precision_floats:
        if dtype in _low_precision_floats:
            return False

    return dtype in _lcdtype_to_nvdtype_map


def is_supported_tensor(a: TensorProxy, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(a, TensorProxy)
    devicetype_supported = a.device.devicetype is DeviceType.CUDA
    dtype_supported = is_supported_dtype(a.dtype)

    if not allow_low_precision_floats:
        if a.dtype in _low_precision_floats:
            return False

    rank_supported = a.ndim <= 8
    return devicetype_supported and dtype_supported and rank_supported


def is_supported_tensor_or_number(a: TensorProxy | Number) -> bool:
    if isinstance(a, Number):
        return True

    return is_supported_tensor(a)


# Returns True when all arguments given are supported tensors
#   Throws an error if any arguments are not tensors
# TODO Add a check for the tensor have > 0 elements?
def are_supported_tensors(*args) -> bool:
    for a in args:
        if not is_supported_tensor(a):
            return False

    return True


# Returns True when all arguments given are supported tensors or numbers
#   Throws an error if any arguments are not numbers or tensors
def are_supported_tensors_or_numbers(*args) -> bool:
    for a in args:
        if not is_supported_tensor_or_number(a):
            return False

    return True


#
# Functions related to creating fusions
#

_translation_map: dict[Hashable, Callable] = {}


def get_translator(bsym: BoundSymbol) -> Callable:
    return _translation_map[bsym.sym.id]


def create_fd(
    bsyms: list[BoundSymbol],
    input_descriptors: Sequence[type | tuple[tuple[int, ...], tuple[bool, ...]]],
    sorted_unique_inputs: list[Proxy],
    sorted_unique_outputs: list[Proxy],
) -> FusionDefinition:
    lc_to_nv_map = utils.ProxyDict()

    # NOTE nvFuser's default max length is 1024 operations at the time of this writing
    #   This arbitrarily increases it to 9999
    # TODO Review splititng very large fusions or removing the max length restriction completely
    #   See https://github.com/Lightning-AI/lightning-thunder/issues/901
    fd = FusionDefinition(max_length=9999)
    with fd:
        # NOTE Adding constants is disabled for the moment in favor of definining them inline
        # 0) Adds constants

        # for c in constants:
        #     nv = _define_constant(fd, c)
        #     lc_to_nv_map[c] = nv

        # 1) Inputs are added and mapped to nvFuser objects

        # NOTE x is the trace's annotation of the input, y is the actual concrete input descriptor at call time
        def add_input(x: Any, y: Any) -> Any:
            nv: Any
            if isinstance(x, NumberProxy):
                utils.check_type(y, type)
                python_type = y
                nvdtype = lcdtype_to_nvdtype(python_type)
                nv = fd.define_scalar(nvdtype)
            elif isinstance(x, TensorProxy):
                utils.check_type(y, tuple)
                symbolic_shape, contiguity, dtype = y
                nvdtype = lcdtype_to_nvdtype(ltorch.to_thunder_dtype(dtype))
                if nv_version >= LooseVersion("0.0.17"):
                    nv = fd.define_tensor(shape=symbolic_shape, contiguity=contiguity, dtype=nvdtype)
                elif nv_version >= LooseVersion("0.0.9"):
                    nv = fd.define_tensor(symbolic_sizes=symbolic_shape, contiguity=contiguity, dtype=nvdtype)
                else:
                    nv = fd.define_tensor(symbolic_sizes=symbolic_shape, contiguous=contiguity, dtype=nvdtype)
            elif isinstance(x, Proxy):
                utils.check(False, lambda: f"Unsupported proxy type {x} in fusion", exception_type=AssertionError)
            else:
                nv = x

            lc_to_nv_map[x] = nv
            return nv

        for pinp, inp in zip(sorted_unique_inputs, input_descriptors):
            add_input(pinp, inp)

        # 2) Translates bound symbols

        def translate_bound_symbol(bsym: BoundSymbol) -> Any:
            translator = get_translator(bsym)
            nvresults = translator(*bsym.args, **bsym.kwargs, fd=fd, lc_to_nv_map=lc_to_nv_map)

            # Updates map
            for out, nvout in zip(utils.sequencify(bsym.output), utils.sequencify(nvresults)):
                # NOTE out can be None if an operation returned multiple results but only some are used,
                #   in which case DCE will replace the unused results with None
                if out is not None and isinstance(out, Proxy):
                    lc_to_nv_map[out] = nvout

        for bsym in bsyms:
            translate_bound_symbol(bsym)

        # 3) Adds outputs
        # TODO Translate numbers to tensors (and provide the information to translate them back to numbers!)
        for out in sorted_unique_outputs:
            nvout = lc_to_nv_map[out]
            fd.add_output(nvout)

    return fd


def compute_symbolic_shape(shape: torch.Size | Sequence[int]) -> tuple[int, ...]:
    """
    Computes the symbolic shape of a tensor using nvFuser's notion of a symbolic
    shape, it's represented by 1s and -1s. 1s represent dimensions that are
    known to be 1, and -1s represent dimensions that are not known to be 1.

    For example, the symbolic shape of a tensor with shape (1, 2, 3) is (1, -1, -1).

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.

    Returns:
        Tuple[int, ...]: The symbolic shape of the tensor.
    """
    return tuple(1 if l == 1 else -1 for l in shape)


def compute_contiguity(shape: torch.Size | Sequence[int], stride: Sequence[int]) -> tuple[bool, ...]:
    """
    Computes the contiguity of a tensor using nvFuser's notion of contiguity, it's
    represented by True, False and None. True represents dimensions that are contiguous,
    and False represents dimensions that are not contiguous, and None represents
    stride-0 or size-1 dimensions.

    For example, the contiguity of a tensor with shape (1, 2, 3) and stride (6, 3, 1)
    is (None, True, True).

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.
        stride (Sequence[int]): The stride of the tensor.

    Returns:
        Tuple[bool, ...]: The contiguity of the tensor.
    """
    return tuple(nv_compute_contiguity(shape, stride))


@lru_cache(maxsize=2048)
def compute_symbolic_shape_and_contiguity(
    shape: torch.Size | Sequence[int], stride: Sequence[int]
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """
    Computes the symbolic shape and contiguity of a tensor using nvFuser's notion of
    symbolic shape and contiguity. See compute_symbolic_shape and compute_contiguity
    for more details.

    This function is caching the results of compute_symbolic_shape and compute_contiguity
    to speed up the computation.

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.
        stride (Sequence[int]): The stride of the tensor.

    Returns:
        Tuple[Tuple[int, ...], Tuple[bool, ...]]: The symbolic shape and contiguity of the tensor.
    """
    return compute_symbolic_shape(shape), compute_contiguity(shape, stride)


def get_symbolic_shape_and_contiguity(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    return compute_symbolic_shape_and_contiguity(t.shape, t.stride())


# NOTE Currently assumes that only numbers and tensors are passed in
# TODO Add check that only numbers and tensors are passed in
# TODO Inline the get_symbolic_shape_and_contiguity call
def to_descriptors(args):
    return tuple(
        type(arg) if isinstance(arg, Number) else (*get_symbolic_shape_and_contiguity(arg), arg.dtype) for arg in args
    )


# TODO Consider making this just a function, because it's faster to call a function than a callable class
@dataclass
class FusionDefinitionWrapper:
    """
    A callable object wrapping a nvFuser fusion definition.
    """

    get_fd: Callable[[tuple[type | tuple[tuple[int, ...], tuple[bool, ...]], ...]], FusionDefinition]
    cache_info: None | Callable = None
    cache_clear: None | Callable = None
    last_used: None | FusionDefinition = None

    def __call__(self, *args):
        fd = self.get_fd(to_descriptors(args))
        self.last_used = fd

        # Set device if set in one of the "factory" methods like full, iota, or uniform
        kwargs = (
            {"device": fd._selected_device}
            if nv_version >= LooseVersion("0.0.13") and hasattr(fd, "_selected_device")
            else {}
        )
        return fd.execute(args, **kwargs)

    def __repr__(self):
        return f"FusionDefinitionWrapper"


# Group bookend meta operations into separate regions
# This function returns a List[Region] which changes the executor of meta regions to torchex
#
# NOTE this function assumes bound_symbols in region is toposorted
def group_bookend_meta_ops(producers, consumers, region: Region) -> list[Region]:
    front_meta_cluster = list()
    middle_cluster = list()
    rear_meta_cluster = list()
    region_inputs = copy(region.inputs)

    # bsym can be moved to the front if all their inputs are direct region inputs
    def can_move_to_front(bsym: BoundSymbol) -> bool:
        # non proxy don't need to be checked here.
        for x in bsym.flat_args:
            if not isinstance(x, Proxy):
                continue

            if variableify(x) not in region_inputs:
                return False

        return True

    # when bsym has no consumer in current region, it can be safely moved to the rear
    def can_move_to_rear(bsym: BoundSymbol) -> bool:
        # check no existing bsym in region depends on current bsym
        for out in bsym.flat_outs:
            if not isinstance(out, Proxy):
                continue

            consumed_by = consumers.get(out, list())
            for consumer in consumed_by:
                # TODO: switch query to set for faster query
                if consumer in middle_cluster:
                    return False
        return True

    def all_tagged(bsym: BoundSymbol, tags: set[prims.OpTags]) -> bool:
        if not has_tags(bsym, tags):
            return False

        for sbsym in bsym.subsymbols:
            if not has_tags(sbsym, tags):
                return False

        return True

    # traversing all bound_symbols in topo order
    for bsym in region.bound_symbols:
        # we look at meta operations that can be moved to the front
        if all_tagged(bsym, {prims.OpTags.SHAPE_OP}) and can_move_to_front(bsym):
            # when we remove a node, we add all the bsym's flat_outs to region_inputs
            front_meta_cluster.append(bsym)
            for out in bsym.flat_outs:
                if isinstance(out, Proxy):
                    region_inputs.add(variableify(out))
        else:
            # otherwise we just keep the bound_symbol in the middle_cluster
            middle_cluster.append(bsym)
    # traversing all bound_symbols in reverse topo order
    for bsym in reversed(copy(middle_cluster)):
        if all_tagged(bsym, {prims.OpTags.SHAPE_OP}) and can_move_to_rear(bsym):
            middle_cluster.remove(bsym)
            rear_meta_cluster.insert(0, bsym)

    return {
        "front_bsyms": front_meta_cluster,
        "fusion": None if len(middle_cluster) == 0 else Region(producers, consumers, middle_cluster),
        "rear_bsyms": rear_meta_cluster,
    }


def create_fusion_definition_wrapper(
    bsyms: list[BoundSymbol], sorted_unique_inputs: list[Proxy], sorted_unique_outputs: list[Proxy]
) -> FusionDefinitionWrapper:
    # NOTE Region Inputs and Outputs
    # The inputs and outputs to a region are represented as sets, which are sorted by name
    #   for determinism. Because they're sets, the inputs and outputs to each region are
    #   unique.
    # It's OK to reorder inputs to regions and outputs from regions, become the dataflow of those
    #   objects is captured by names in the trace.
    # These properties are distinct from the inputs and outputs to the trace itself, which
    #   may contain duplicates and whose order must be preserved.

    tensor_indices = []
    for idx, x in enumerate(sorted_unique_inputs):
        if isinstance(x, TensorProxy):
            tensor_indices.append(idx)

    # NOTE create_fd is an expensive function so we cache using the descriptors of inputs
    # TODO (mruberry) We should think how to express "static fusion" that don't need to use
    #   a cache to improve dispatch performance
    @lru_cache(maxsize=2048)
    def get_fd(input_descriptors) -> FusionDefinition:
        # A closure over local trace and region
        return create_fd(bsyms, input_descriptors, sorted_unique_inputs, sorted_unique_outputs)

    fdw = FusionDefinitionWrapper(get_fd, get_fd.cache_info, get_fd.cache_clear)
    return fdw


class nvFuserExecutor(FusionExecutor):
    def __init__(self):
        super().__init__("nvfuser", version=nvfuser.version())

    def flatten(self, bsym: BoundSymbol) -> list[BoundSymbol]:
        flattened: list[BoundSymbol] = []

        # TODO Maybe make this nonrecursive
        def _flatten(bsym: BoundSymbol):
            nonlocal flattened

            if self.can_execute(bsym):
                flattened.append(bsym)
                return

            # NOTE self.can_execute(bsym) is False
            check(
                len(bsym.subsymbols) > 0,
                lambda: f"nvFuser is trying to flatten {bsym} for execution but it's not supported and has no subsymbols",
                exception_type=AssertionError,
            )

            for ssym in bsym.subsymbols:
                _flatten(ssym)

        _flatten(bsym)

        return flattened

    def has_cuda_input_or_output(self, bsym: BoundSymbol) -> bool:
        for p in chain(bsym.flat_proxy_args, bsym.flat_proxy_outs):
            if isinstance(p, TensorProxy) and p.device.devicetype is DeviceType.CUDA:
                return True

        return False

    def _dce_bsyms(self, output, bsyms: list[BoundSymbol]) -> list[BoundSymbol]:
        trace = TraceCtx(None)
        trace.bound_symbols = bsyms
        bsyms.append(prims.python_return.bind(output, output=()))
        trace = dce(trace)
        return list(filter(lambda x: x.sym != prims.python_return, trace.bound_symbols))

    def fuse(self, region: Region, fusion_counter: int) -> BoundSymbol:
        def keyfn(x: Variable) -> str:
            return x.proxy.name

        sorted_unique_inputs: list[Proxy] = list(unvariableify(x) for x in sorted(region.inputs, key=keyfn))
        sorted_unique_outputs: list[Proxy] = list(unvariableify(x) for x in sorted(region.outputs, key=keyfn))

        flattened_bsyms: list[BoundSymbol] = []
        for bsym in region.bound_symbols:
            flattened_bsyms.extend(self.flatten(bsym))

        flattened_bsyms = self._dce_bsyms(sorted_unique_outputs, flattened_bsyms)

        fdw: FusionDefinitionWrapper = create_fusion_definition_wrapper(
            flattened_bsyms, sorted_unique_inputs, sorted_unique_outputs
        )

        fusion_name = f"nvFusion{fusion_counter}"

        fusion_bsym: BoundSymbol = self.register_temporary_operation(
            fusion_name, fdw, inputs=sorted_unique_inputs, outputs=sorted_unique_outputs, bsyms=flattened_bsyms
        )

        return fusion_bsym

    # TODO Restore fusion logic here -- this just replaces supported operations in isolation at the moment
    def fusion_pass(self, trace: TraceCtx) -> TraceCtx:
        start_time_ns: int = time.time_ns()

        fusedtrace: TraceCtx = from_trace(trace)

        producers, consumers = utils.producers_and_consumers(trace)
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols

        fused_bsyms = []

        # TODO has_cuda_input_or_output is too restrictive a check on what should be fused
        # TODO check whether a function would output a CPU tensor? -- can nvFuser fuse such operations?
        #   ex. device_put to a CPU device from a CUDA device
        #   (mruberry) I don't know if nvFuser even attempts to fuse any operation that can go
        #       cross-device today
        def _should_fuse(a: Node, b: Node):
            def _can_fuse_node(n: Node):
                # if already merged, then node can be fused
                if len(n.group_bsyms) > 1:
                    return True
                bsym: BoundSymbol = n.group_bsyms[0]
                can_fuse: bool = self.can_fuse(bsym)
                cuda_in_or_out: bool = self.has_cuda_input_or_output(bsym)
                return can_fuse and cuda_in_or_out

            return _can_fuse_node(a) and _can_fuse_node(b)

        bound_symbol_groups = fuse_bound_symbols(producers, consumers, trace.bound_symbols, _should_fuse)

        # Counts how many fusions (per executor) have been constructed
        #   (Used to name fusions like nvFusion0, nvFusion1, ...)
        fusion_counter: int = 0
        for bsyms in bound_symbol_groups:
            # TODO The following allows generating single node fusions, which
            #   may be suboptimal for real-world performance.
            #   Provide a mechanism to switch between "test" and "perf" modes
            #   so that we can continue to generate single node fusions when testing.
            # if len(bsyms) > 1:
            region = Region(producers, consumers, bsyms)
            bookend_result = group_bookend_meta_ops(producers, consumers, region)

            if len(bsyms) == 1:
                bsym: BoundSymbol = bsyms[0]
                can_fuse: bool = self.can_fuse(bsym)
                cuda_in_or_out: bool = self.has_cuda_input_or_output(bsym)
                if not can_fuse or not cuda_in_or_out:
                    fused_bsyms.append(bsym)
                    continue

            # TODO bookend_result probably shouldn't return a dict
            prologue: list
            fusion: None | Region
            epilogue: list
            prologue, fusion, epilogue = (
                bookend_result["front_bsyms"],
                bookend_result["fusion"],
                bookend_result["rear_bsyms"],
            )

            fused_bsyms.extend(prologue)
            if fusion is not None:
                fusion_bsym: BoundSymbol = self.fuse(fusion, fusion_counter)
                fusion_counter += 1
                fused_bsyms.append(fusion_bsym)
            fused_bsyms.extend(epilogue)

        # fusion_counter: int = 0
        # fused_bsyms: list[BoundSymbol] = []
        # for bsym in trace.bound_symbols:
        #     # Leaves the bound symbol unchanged if it can't (or shouldn't) be fused
        #     if (
        #         not self.can_fuse(bsym)
        #         or not self.has_cuda_input_or_output(bsym)
        #         or has_tags(bsym, {prims.OpTags.SHAPE_OP})
        #     ):
        #         fused_bsyms.append(bsym)
        #         continue

        #     # NOTE self.can_fuse(bsym) and self.has_cuda_input(bsym)
        #     r = Region(producers, consumers, [bsym])

        #     fusion_bsym: BoundSymbol = self.fuse(r, fusion_counter)
        #     fusion_counter += 1

        #     fused_bsyms.append(fusion_bsym)

        fusedtrace.bound_symbols = fused_bsyms

        fusedtrace = dce(fusedtrace)
        # TODO: Restore this when CSE pass can remove ops inside fusion
        # See https://github.com/Lightning-AI/lightning-thunder/pull/1338
        # OR nvFuser itself can handle fusions with certain common subexpressions
        # See https://github.com/NVIDIA/Fuser/issues/1301
        # fusedtrace = remove_redundant_casts(fusedtrace)

        end_time_ns: int = time.time_ns()
        elapsed_time_ns: int = end_time_ns - start_time_ns
        elapsed_time_millis: int = elapsed_time_ns // 1000000
        fusedtrace.set_provenance(TraceProvenance(f"Fusion (took {elapsed_time_millis} milliseconds)"))

        return fusedtrace


ex = nvFuserExecutor()
register_executor(ex)


def register_supported(id: Hashable, translator: Callable, checker: Callable):
    ex.register_supported(id, checker)
    _translation_map[id] = translator


#
# Data movement operations
#


def _convert_element_type_check(a: TensorProxy | Number, dtype: type | dtypes.dtype) -> bool:
    return is_supported_tensor_or_number(a) and is_supported_dtype(dtype)


# TODO Review conversion of numbers vs. tensors
def convert_element_type(
    a: TensorProxy | Number, dtype: type | dtypes.dtype, *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    return fd.ops.cast(nva, nvdtype)


register_supported(PrimIDs.CONVERT_ELEMENT_TYPE, convert_element_type, _convert_element_type_check)

#
# Tensor creation operations
#


def _select_device(fd: FusionDefinition, device: None | Device):
    """Specify device for function return values.

    The device argument is sometimes provided in factory functions that don't
    take inputs, such as `full` or `uniform`. This argument provides a useful
    hint for inferring which device to run a Fusion in some cases. Note that if
    this is not called, the device will be inferred from inputs or default to
    the first CUDA device. If this function is called with an argument other
    than `None`, then passing inputs residing on other devices is an error, as
    is calling this function with non-`None` arguments multiple times with
    incompatible arguments.
    """
    if device is None:
        return

    utils.check(
        device.devicetype == DeviceType.CUDA,
        lambda: f"If device argument is provided, NVFuser executor requires it to be a CUDA device, but {device=}",
    )

    if device.index is None:
        return

    utils.check(
        not hasattr(fd, "_selected_device") or fd._selected_device == device.index,
        lambda: f"Found multiple requested devices: {fd._selected_device} and {device.index}",
    )

    fd._selected_device = device.index


def _full_check(shape: Sequence[int], fill_value: Number, *, device: Device, dtype: dtypes.dtype) -> bool:
    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Improve device handling
# TODO Materialize shape (if necessary)
# NOTE nvFuser's full prim requires shape to be a sequence of Python numbers
# NOTE nvFuser's full prim requires fill_value be an nvScalar (or nvConstant?)
# NOTE nvFuser's full prim accepts no device argument
def full(
    shape: Sequence[int],
    fill_value: Number,
    *,
    device: Device,
    dtype: dtypes.dtype,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nv_fill_value = getnv(fill_value, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    _select_device(fd, device)

    return fd.ops.full(shape, nv_fill_value, nvdtype)


register_supported(PrimIDs.FULL, full, _full_check)


def _iota_check(length: Number, *, start: Number, step: Number, device: Device, dtype: dtypes.dtype) -> bool:
    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Improve device handling
def iota(
    length: Number,
    *,
    start: Number,
    step: Number,
    device: Device,
    dtype: dtypes.dtype,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nvlength = getnv(length, fd, lc_to_nv_map)
    nvstart = getnv(start, fd, lc_to_nv_map)
    nvstep = getnv(step, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    _select_device(fd, device)

    return fd.ops.iota(nvlength, nvstart, nvstep, nvdtype)


register_supported(PrimIDs.IOTA, iota, _iota_check)


def _uniform_check(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: Device, dtype: dtypes.dtype
) -> bool:
    if nv_version < LooseVersion("0.0.3"):
        return False

    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Add type annotations
# TODO Fix device handling
# NOTE Shape must be a list of nvScalars or nvConstants
def uniform(
    shape, minval, maxval, *, device: Device, dtype: dtypes.dtype, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nvdtype = lcdtype_to_nvdtype(dtype)

    nv_minval = getnv(minval, fd, lc_to_nv_map)
    nv_maxval = getnv(maxval, fd, lc_to_nv_map)

    nvshape = list(getnv(x, fd, lc_to_nv_map) for x in shape)

    _select_device(fd, device)

    return fd.ops.uniform(nv_minval, nv_maxval, nvshape, dtype=nvdtype)


register_supported(PrimIDs.UNIFORM, uniform, _uniform_check)


def _uniform_philox_check(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: Device,
    dtype: dtypes.dtype,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> bool:
    return (
        is_supported_device(device) and is_supported_dtype(dtype) and isinstance(seed, int) and isinstance(offset, int)
    )


def uniform_philox(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: Device,
    dtype: dtypes.dtype,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
    fd: FusionDefinition,
    lc_to_nv_map: dict[Any, Any],
) -> Any:
    nvdtype = lcdtype_to_nvdtype(dtype)

    nv_minval = getnv(minval, fd, lc_to_nv_map)
    nv_maxval = getnv(maxval, fd, lc_to_nv_map)

    nvshape = list(getnv(x, fd, lc_to_nv_map) for x in shape)

    nv_rng_seed = getnv(seed, fd, lc_to_nv_map)
    nv_rng_offset = getnv(offset, fd, lc_to_nv_map)

    _select_device(fd, device)

    return fd.ops.uniform(
        nv_minval,
        nv_maxval,
        nvshape,
        dtype=nvdtype,
        rng_seed=nv_rng_seed,
        rng_offset=nv_rng_offset,
    )


register_supported(PrimIDs.UNIFORM_PHILOX, uniform_philox, _uniform_philox_check)

#
# Shape operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _broadcast_in_dim_check(a: TensorProxy, shape: list[int], broadcast_dimensions: list[int]) -> bool:
    return is_supported_tensor(a)


# TODO Carefully consider how shape and broadcast dimensions being constant here relates to
#   the caching of fusions on stride and contiguity information -- do those things being constant
#   imply these values are constant, too?
# TODO Review translating proxy numbers to actual numbers
def broadcast_in_dim(
    a: TensorProxy, shape: list[int], broadcast_dimensions: list[int], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.broadcast_in_dim(nva, shape, broadcast_dimensions)


register_supported(PrimIDs.BROADCAST_IN_DIM, broadcast_in_dim, _broadcast_in_dim_check)


def _cat_check(tensors: list[TensorProxy], dim: int) -> bool:
    # nvFuser cat fusion is currently disabled due to
    #   https://github.com/Lightning-AI/lightning-thunder/issues/1071
    return False

    # Validates tensors and concatenated dimension lengths
    for t in tensors:
        if not is_supported_tensor(t):
            return False

        # See https://github.com/NVIDIA/Fuser/issues/21
        #   nvFuser cannot concatenate dimensions of length 1
        if t.shape[dim] == 1:
            return False

    return True


# NOTE nvFuser's cat prim accepts dim as a Python Number, not a constant
def cat(tensors: list[TensorProxy], dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nvtensors = list(getnv(t, fd, lc_to_nv_map) for t in tensors)

    return fd.ops.cat(nvtensors, dim)


register_supported(PrimIDs.CAT, cat, _cat_check)


def _stride_order_check(a: TensorProxy, order: Sequence[int]) -> bool:
    if nv_version < LooseVersion("0.0.20"):
        return False

    return is_supported_tensor(a)


def stride_order(a: TensorProxy, order: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.stride_order(nva, order)


register_supported(PrimIDs.STRIDE_ORDER, stride_order, _stride_order_check)


# NOTE nvFuser does not support dilation > 0
def _pad_check(a: TensorProxy, padding_value: Number, padding_config: tuple[int, int, int]) -> bool:
    if a.numel == 0 and nv_version < LooseVersion("0.0.21"):
        return False

    if nv_version < LooseVersion("0.0.6"):
        return False

    if not is_supported_tensor(a):
        return False

    for lo, hi, dilation in padding_config:
        if dilation > 0:
            return False

    return True


# NOTE Translating to nvFuser's pad operation
#   nvFuser's pad op requires pad_widths to be a sequence of Python numbers
#   (lo_n, hi_n, lo_{n-1}, hi_{n-1}, ...) where dimensions are counted in reverse
#   as shown, and dilation is not supported.
#   This is in constrant to lightning.compile's pad primitive, which specifies padding
#   and dilation as an  ndim-length list of (lo, hi, dilation) triples.
# NOTE padding_value must be an nvConstant (or nvScalar?)
def pad(
    a: TensorProxy,
    padding_value: Number,
    padding_config: tuple[int, int, int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nv_padding_value = getnv(padding_value, fd, lc_to_nv_map)

    pad_widths = []

    for lo, hi, dilation in reversed(padding_config):
        pad_widths.extend([lo, hi])

    return fd.ops.pad(nva, pad_widths, nv_padding_value)


register_supported(PrimIDs.PAD, pad, _pad_check)


def _reshape_check(a: TensorProxy, shape: list[int]) -> bool:
    return is_supported_tensor(a)


def reshape(a: TensorProxy, shape: list[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)

    if nv_version < LooseVersion("0.0.22"):
        return fd.ops.reshape(nv_a, a.shape, shape)
    else:
        return fd.ops.reshape(nv_a, shape)


register_supported(PrimIDs.RESHAPE, reshape, _reshape_check)


# NOTE nvFuser's slice operation only supports all strides == 1
def _slice_check(
    a: TensorProxy, start_indices: Sequence[int], end_indices: Sequence[int], strides: Sequence[int] | None = None
) -> bool:
    if nv_version < LooseVersion("0.0.6"):
        return False

    if not is_supported_tensor(a):
        return False

    # Checks that strides are not specified or all are explicitly set to 1
    if strides is not None:
        for stride in strides:
            if stride != 1:
                return False

    return True


def nv_slice(
    a: TensorProxy,
    /,
    start_indices: Sequence[int],
    end_indices: Sequence[int],
    strides: Sequence[int] | None = None,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.slice(nva, start_indices, end_indices, strides)


register_supported(PrimIDs.SLICE, nv_slice, _slice_check)


def _squeeze_check(a: TensorProxy, /, dims: Sequence[int]) -> bool:
    return is_supported_tensor(a)


# NOTE nvFuser's squeeze operation requires the shape of the tensor be specified
def squeeze(a: TensorProxy, /, dims: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    if nv_version >= LooseVersion("0.1.4"):
        return fd.ops.squeeze(nva, dims)
    else:
        return fd.ops.squeeze(nva, a.shape, dims)


register_supported(PrimIDs.SQUEEZE, squeeze, _squeeze_check)

# TAKE is currently disabled
# def _take_check(a: TensorProxy, /, index: TensorProxy, dim: int) -> bool:
#     return are_supported_tensors(a, index)

# def take(a: TensorProxy, /, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
#     nv_a = getnv(a, fd, lc_to_nv_map)
#     nv_index = getnv(index, fd, lc_to_nv_map)

#     return fd.ops.index_select(nv_a, nv_index, dim)
# register_supported(PrimIDs.TAKE, take, _take_check)

# TAKE_ALONG_AXIS is currently disabled
# See https://github.com/NVIDIA/Fuser/issues/458
# # TODO Check that the nvFuser version is >= 0.0.10 when this operator was added
# def take_along_axis(a: TensorProxy, /, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
#     nv_a = getnv(a, fd, lc_to_nv_map)
#     nv_index = getnv(index, fd, lc_to_nv_map)

#     return fd.ops.take_along_axis(nv_a, nv_index, dim)
# register_supported(PrimIDs.TAKE_ALONG_AXIS, take_along_axis, _take_check)


def _transpose_check(a: TensorProxy, /, permutation: Sequence[int]) -> bool:
    return is_supported_tensor(a)


def transpose(a: TensorProxy, /, permutation: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.permute(nva, permutation)


register_supported(PrimIDs.TRANSPOSE, transpose, _transpose_check)

#
# Elementwise unary operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _elementwise_unary_check(
    a: Number | TensorProxy, /, *, version_required: LooseVersion = LooseVersion("0.0.0")
) -> bool:
    return is_supported_tensor_or_number(a) and nv_version > version_required


# NOTE nv_abs to avoid a name conflict with the builin abs
def nv_abs(a: Number | TensorProxy, /, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.abs(nva)


register_supported(PrimIDs.ABS, nv_abs, _elementwise_unary_check)


def acos(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = lc_to_nv_map[a]

    return fd.ops.acos(nva)


register_supported(PrimIDs.ACOS, acos, _elementwise_unary_check)


def acosh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.acosh(nva)


register_supported(PrimIDs.ACOSH, acosh, _elementwise_unary_check)


def asin(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asin(nva)


register_supported(PrimIDs.ASIN, asin, _elementwise_unary_check)


def asinh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asinh(nva)


register_supported(PrimIDs.ASINH, asinh, _elementwise_unary_check)


def atan(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atan(nva)


register_supported(PrimIDs.ATAN, atan, _elementwise_unary_check)


def atanh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atanh(nva)


register_supported(PrimIDs.ATANH, atanh, _elementwise_unary_check)


def bitwise_not(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.bitwise_not(nva)


register_supported(PrimIDs.BITWISE_NOT, bitwise_not, _elementwise_unary_check)


def ceil(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.ceil(nva)


register_supported(PrimIDs.CEIL, ceil, _elementwise_unary_check)


def cos(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cos(nva)


register_supported(PrimIDs.COS, cos, _elementwise_unary_check)


def cosh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cosh(nva)


register_supported(PrimIDs.COSH, cosh, _elementwise_unary_check)


def erf(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erf(nva)


register_supported(PrimIDs.ERF, erf, _elementwise_unary_check)


def erfc(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfc(nva)


register_supported(PrimIDs.ERFC, erfc, _elementwise_unary_check)


def erfcinv(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfcinv(nva)


register_supported(PrimIDs.ERFCINV, erfcinv, _elementwise_unary_check)


def erfinv(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfinv(nva)


register_supported(PrimIDs.ERFINV, erfinv, _elementwise_unary_check)


def exp(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp(nva)


register_supported(PrimIDs.EXP, exp, _elementwise_unary_check)


def exp2(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp2(nva)


register_supported(PrimIDs.EXP2, exp2, _elementwise_unary_check)


def expm1(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.expm1(nva)


register_supported(PrimIDs.EXPM1, expm1, _elementwise_unary_check)


def floor(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.floor(nva)


register_supported(PrimIDs.FLOOR, floor, _elementwise_unary_check)


def isfinite(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.isfinite(nva)


register_supported(PrimIDs.ISFINITE, isfinite, _elementwise_unary_check)


def lgamma(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.lgamma(nva)


register_supported(PrimIDs.LGAMMA, lgamma, _elementwise_unary_check)


def log(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log(nva)


register_supported(PrimIDs.LOG, log, _elementwise_unary_check)


def log10(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log10(nva)


register_supported(PrimIDs.LOG10, log10, _elementwise_unary_check)


def log1p(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log1p(nva)


register_supported(PrimIDs.LOG1P, log1p, _elementwise_unary_check)


def log2(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log2(nva)


register_supported(PrimIDs.LOG2, log2, _elementwise_unary_check)

# nvFuser doesn't have an ndtri operation
# def ndtri(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
#     nva = getnv(a, fd, lc_to_nv_map)

#     return fd.ops.ndtri(nva)
# register_supported(PrimIDs.NDTRI, ndtri, _elementwise_unary_check)


def neg(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.neg(nva)


register_supported(PrimIDs.NEG, neg, _elementwise_unary_check)


def real(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.real(nva)


register_supported(PrimIDs.REAL, real, _elementwise_unary_check)


def reciprocal(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.reciprocal(nva)


register_supported(PrimIDs.RECIPROCAL, reciprocal, _elementwise_unary_check)


# NOTE nv_round to avoid a name conflict with the builtin round
def nv_round(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.round(nva)


register_supported(PrimIDs.ROUND, nv_round, _elementwise_unary_check)


def rsqrt(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.rsqrt(nva)


register_supported(PrimIDs.RSQRT, rsqrt, _elementwise_unary_check)


def sign(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sign(nva)


register_supported(PrimIDs.SIGN, sign, _elementwise_unary_check)


def signbit(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.signbit(nva)


register_supported(PrimIDs.SIGNBIT, signbit, _elementwise_unary_check)


def sin(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sin(nva)


register_supported(PrimIDs.SIN, sin, _elementwise_unary_check)


def sinh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sinh(nva)


register_supported(PrimIDs.SINH, sinh, _elementwise_unary_check)


def sqrt(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sqrt(nva)


register_supported(PrimIDs.SQRT, sqrt, _elementwise_unary_check)


def tan(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tan(nva)


register_supported(PrimIDs.TAN, tan, _elementwise_unary_check)


def tanh(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tanh(nva)


register_supported(PrimIDs.TANH, tanh, _elementwise_unary_check)


def trunc(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.trunc(nva)


register_supported(PrimIDs.TRUNC, trunc, _elementwise_unary_check)


#
# Elementwise binary operations
#
# TODO Review support for all elementwise binary operators, like nextafter


def _elementwise_binary_check(a: Number | TensorProxy, b: Number | TensorProxy) -> bool:
    return are_supported_tensors_or_numbers(a, b)


# TODO Generalize to use an elementwise binary helper or factory?
# TODO Convert Python numbers to constants?
def _add(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.add(nva, nvb)


register_supported(PrimIDs.ADD, _add, _elementwise_binary_check)


def atan2(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.atan2(nva, nvb)


register_supported(PrimIDs.ATAN2, atan2, _elementwise_binary_check)


def bitwise_and(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_and(nva, nvb)


register_supported(PrimIDs.BITWISE_AND, bitwise_and, _elementwise_binary_check)


def bitwise_or(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_or(nva, nvb)


register_supported(PrimIDs.BITWISE_OR, bitwise_or, _elementwise_binary_check)


def bitwise_xor(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_xor(nva, nvb)


register_supported(PrimIDs.BITWISE_XOR, bitwise_xor, _elementwise_binary_check)


# TODO nvFuser's div operation is not equivalent to the div primitive
#   (mruberry) I need to investigate if nvFuser exposes a truncation division operation
def div(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    # TODO nvFuser sometimes generates an innacurate result when dividing by a number
    #   Remove this workaround once the issue is fixed
    #   See: https://github.com/NVIDIA/Fuser/issues/160
    if isinstance(b, Number):
        return fd.ops.mul(nva, fd.ops.reciprocal(nvb))

    # NOTE It's currently significantly faster for nvFuser to multiply the reciprocal than divide
    # return fd.ops.div(nva, nvb)
    return fd.ops.mul(nva, fd.ops.reciprocal(nvb))


register_supported(PrimIDs.DIV, div, _elementwise_binary_check)


def eq(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.eq(nva, nvb)


register_supported(PrimIDs.EQ, eq, _elementwise_binary_check)


def fmod(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.fmod(nva, nvb)


register_supported(PrimIDs.FMOD, fmod, _elementwise_binary_check)


def ge(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.ge(nva, nvb)


register_supported(PrimIDs.GE, ge, _elementwise_binary_check)


def gt(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.gt(nva, nvb)


register_supported(PrimIDs.GT, gt, _elementwise_binary_check)


def le(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.le(nva, nvb)


def lt(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.lt(nva, nvb)


register_supported(PrimIDs.LT, lt, _elementwise_binary_check)


def mul(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.mul(nva, nvb)


register_supported(PrimIDs.MUL, mul, _elementwise_binary_check)


def ne(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.ne(nva, nvb)


register_supported(PrimIDs.NE, ne, _elementwise_binary_check)


def nextafter(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.nextafter(nva, nvb)


register_supported(PrimIDs.NEXTAFTER, nextafter, _elementwise_binary_check)


def pow(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.pow(nva, nvb)


register_supported(PrimIDs.POW, pow, _elementwise_binary_check)


def remainder(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.remainder(nva, nvb)


register_supported(PrimIDs.REMAINDER, remainder, _elementwise_binary_check)


def sub(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.sub(nva, nvb)


register_supported(PrimIDs.SUB, sub, _elementwise_binary_check)

#
# Conditional operations
#


# TODO Check supported dtypes
# TODO Properly implement this check
def _where_check(pred, a, b) -> bool:
    return are_supported_tensors_or_numbers(pred, a, b)


def where(
    pred: TensorProxy | Number,
    a: TensorProxy | Number,
    b: TensorProxy | Number,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nvpred = getnv(pred, fd, lc_to_nv_map)
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.where(nvpred, nva, nvb)


register_supported(PrimIDs.WHERE, where, _where_check)

#
# Reduction operations
#


# TODO Checks that the dtype is supported by nvFuser
def _reduction_check(a: TensorProxy, dims: Sequence[int]) -> bool:
    return is_supported_tensor(a, allow_low_precision_floats=False)


# TODO Review if this accepts empty dim sequences
def amax(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims

    return fd.ops.max(nva, nvdims)


register_supported(PrimIDs.AMAX, amax, _reduction_check)


# TODO Review if this accepts empty dim sequences
def amin(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims

    return fd.ops.min(nva, nvdims)


register_supported(PrimIDs.AMIN, amin, _reduction_check)


# TODO Review if this accepts empty dim sequences
def prod(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims

    return fd.ops.prod(nva, nvdims)


register_supported(PrimIDs.PROD, prod, _reduction_check)


def sum(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims

    # NOTE nvFuser's sum primitive does not accept empty dims sequences
    if len(dims) == 0:
        return nva

    return fd.ops.sum(nva, nvdims)


register_supported(PrimIDs.SUM, sum, _reduction_check)


# NOTE https://github.com/NVIDIA/Fuser/pull/121
#   nvFuser's var operation does not support 0-dim inputs
def _var_check(a: TensorProxy, dims: Sequence[int], *, correction: Number) -> bool:
    return is_supported_tensor(a, allow_low_precision_floats=False) and len(a.shape) > 0


# TODO Add type annotations
# TODO Review translation of dims and correction
def var(a: TensorProxy, dims: Sequence[int], *, correction: Number, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = list(dims)
    nvcorrection = correction

    return fd.ops.var(nva, nvdims, nvcorrection)


register_supported(PrimIDs.VAR, var, _var_check)


# NOTE https://github.com/NVIDIA/Fuser/pull/121
#   nvFuser's var_mean operation does not support 0-dim inputs
# TODO Support complex tensors
#   var(complex) = var(real) + var(imag)
def _var_mean_check(
    a: TensorProxy,
    dim=None,
    *,
    correction: None | int = None,
) -> bool:
    if nv_version < LooseVersion("0.0.7"):
        return False

    if not is_supported_tensor(a, allow_low_precision_floats=False):
        return False

    if len(a.shape) == 0:
        return False

    if dtypes.is_complex_dtype(dtypes.to_dtype(a)):
        return False

    return True


# NOTE nvFuser's var_mean op has the signature (tensor, dims, correction, keepdim)
def var_mean(
    a: TensorProxy,
    dim,
    *,
    correction: int,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = list(dim)
    return fd.ops.var_mean(nva, nvdims, correction)


register_supported(PrimIDs.VAR_MEAN, var_mean, _var_mean_check)


# Removes excessive float casts, like those that occur when autocasting
# NOTE This passes actually changes a program's semantics, because it will take a sequence like
#   fp32 -> fp16 -> fp32 and remove all the operations, but casting fp32 values to fp16 can
#   changes the values (because most fp32 values are not representable in fp16)
# NOTE This only handles conversions performed by CONVERT_ELEMENT_TYPE, and not conversions caused
#   by other Symbols, like torch.to, which may be unflattened
# TODO This could be extended to non-float conversions, like complex -> complex conversions
def remove_redundant_casts(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    start_time_ns = time.time_ns()

    rrctrace = from_trace(trace)

    # Returns a tuple (is proxy float->float conversion?, object to convert, dtype to convert to)
    def is_eligible_cast(bsym: BoundSymbol) -> tuple[bool, Any, Any]:
        # Ignores operations other than CONVERT_ELEMENT_TYPE
        if bsym.sym.id is not prims.PrimIDs.CONVERT_ELEMENT_TYPE:
            return False, None, None

        # Parses arguments
        # TODO We should consider canonicalizing how BoundSymbols express their arguments
        a: Any
        dtyp: dtypes.dtype

        if len(bsym.args) == 2:
            a, dtyp = bsym.args
        elif len(bsym.args) == 1:
            utils.check(len(bsym.kwargs) == 1, lambda: f"Expected two arguments for convert element type")
            (a,) = bsym.args
            dtyp = bsym.kwargs["dtype"]
        else:
            a = bsym.kwargs["a"]
            dtyp = bsym.kwargs["dtype"]

        if not isinstance(a, Proxy):
            return False, None, None

        is_float_to_float_conversion = dtypes.is_float_dtype(dtypes.to_dtype(a)) and dtypes.is_float_dtype(dtyp)

        return is_float_to_float_conversion, a, dtyp

    # Updates intermediate conversions, identifies no-ops, and updates no-op consumers
    # NOTE These are separate maps. A no-op in this context is a cast from the
    #   input's dtype to itself, like the following:
    #
    #   b = prims.convert_element_type(a, float32)  # a: f32
    #
    #   For these operations, everywhere b is consumed can be replaced with a.
    #
    #   When there is an intermediate conversion, however, we don't want to replace all uses
    #   of its output with its input. For example, the dtype modified output could
    #   actually be consumed by non-cast operations.

    # TODO This is intentionally commented out. See TODO below on consumer analysis.
    # consumers = cutils.consumers(trace)
    def _remove_redundant_casts(
        bsym: BoundSymbol,
        nbsyms: Sequence[BoundSymbol],
        replacement_map: dict[Variable, Proxy],
        intermediate_map: dict[Variable, Proxy],
    ) -> None:
        is_proxy_f2f_conversion, a, dtyp = is_eligible_cast(bsym)

        # Replaces inputs due to no-op casts for all operations
        if not is_proxy_f2f_conversion:
            nbsym = bsym
            if bsym.has_input(replacement_map):
                nbsym = bsym.from_bsym_swap_proxies(replacement_map, skip_inputs=False, skip_output=True)
            nbsyms.append(nbsym)
            return

        # NOTE is_proxy_f2f_conversion is True
        va = variableify(a)
        vo = variableify(bsym.output)

        # Identifies updated input
        orig = intermediate_map.get(va, a)
        orig_dtype = dtypes.to_dtype(orig)

        # Elides no-ops, marking their outputs for replacement
        if orig_dtype == dtyp:
            replacement_map[vo] = orig
            intermediate_map[vo] = orig
            return

        # NOTE In this case there is a more original input

        # Only marks this output for replacement with the more original input if it's
        #   not consumed by a non-cast operation
        has_non_cast_consumer = False
        # TODO (mruberry) I'm not sure whether the following is worthwhile, although
        #   I'm leaving it as a comment because we may want to revive it in the future.
        #   Essentially, this would be a heuristic that says: "if x is being consumed,
        #   don't bother finding the precursor of x to cast, just cast x itself."
        #   That may improve data locality, but it could also lead to excessive
        #   casts.
        # for consumer in consumers.get(bsym.output, ()):
        #     if consumer.sym.id is not prims.PrimIDs.CONVERT_ELEMENT_TYPE:
        #         has_non_cast_consumer = True
        #         break

        # When this operation has non-cast consumers, later conversion operations
        #   might as well consume its output to try and improve data locality and
        #   not have to preserve the original tensor for so long
        if has_non_cast_consumer:
            intermediate_map[vo] = bsym.output
        else:
            intermediate_map[vo] = orig

        # Possibly creates a new BoundSymbol consuming the original instead of the current input
        if orig is a:
            nbsyms.append(bsym)
        else:
            # NOTE This is faster than using from_bsym_swap_proxies, and relies on us only working
            #   with prims.convert_element_type
            nbsym = bsym.from_bsym(args=(orig, dtyp), kwargs={})
            nbsyms.append(nbsym)
            utils.check(
                nbsym.subsymbols is None or len(nbsym.subsymbols) == 0,
                lambda: f"Expected no subsymbols when creating a new BoundSymbol in the remove redundant casts pass",
                exception_type=AssertionError,
            )

    replacement_map = {}
    intermediate_map = {}
    nbsyms = []
    for bsym in trace.bound_symbols:
        if bsym.sym.is_fusion:
            nbsym = bsym
            if bsym.has_input(replacement_map):
                nbsym = bsym.from_bsym_swap_proxies(
                    replacement_map, skip_inputs=False, skip_output=True, skip_subsymbols=False
                )
            nvfuser_replacement_map = {}
            nvfuser_intermediate_map = {}
            nvfuser_subbsyms = []
            for subbsym in nbsym.subsymbols:
                _remove_redundant_casts(subbsym, nvfuser_subbsyms, nvfuser_replacement_map, nvfuser_intermediate_map)
            nbsym = nbsym.from_bsym(subsymbols=nvfuser_subbsyms)

            def map_inside_replacement(x: Any) -> None:
                vx = variableify(x)
                if vx in nvfuser_replacement_map:
                    replacement_map[vx] = nvfuser_replacement_map[vx]

            tree_map(map_inside_replacement, nbsym.output)
            nbsym = nbsym.from_bsym_swap_proxies(
                nvfuser_replacement_map, skip_inputs=True, skip_output=False, skip_subsymbols=True
            )
            dedup_output = list({variableify(x): x for x in nbsym.output}.values())
            nbsym = nbsym.from_bsym(output=dedup_output)
            nbsyms.append(nbsym)
        else:
            _remove_redundant_casts(bsym, nbsyms, replacement_map, intermediate_map)

    rrctrace.bound_symbols = nbsyms

    # update call_ctx information in nvFusion
    from thunder.core.rematerialization import _update_nvfusion_call_ctx

    for idx, bsym in enumerate(rrctrace.bound_symbols):
        if bsym.sym.is_fusion:
            rrctrace.bound_symbols[idx] = _update_nvfusion_call_ctx(rrctrace, bsym)

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    rrctrace.set_provenance(TraceProvenance(f"Remove redundant casts (took {elapsed_time_millis} milliseconds)"))
    return rrctrace
