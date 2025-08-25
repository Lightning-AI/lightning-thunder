from dataclasses import dataclass, replace
from functools import partial, lru_cache
from numbers import Number
from typing import Any
from collections.abc import Callable, Mapping, Hashable, Sequence
import os
import time
from copy import copy
from itertools import chain, filterfalse
import warnings
from typing import cast

from looseversion import LooseVersion
import torch
from torch import Tensor

IS_TORCH_DISTRIBUTED_AVAILABLE = torch.distributed.is_available()
if IS_TORCH_DISTRIBUTED_AVAILABLE:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
    import torch.distributed as dist

import thunder.core.dtypes as dtypes
import thunder.torch as ltorch
from thunder.torch import TensorLike

from thunder.core import prims, utils
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.prims import PrimIDs
from thunder.core.proxies import (
    NumberProxy,
    Proxy,
    TupleProxy,
    TensorProxy,
    variableify,
    unvariableify,
    Variable,
    pyval,
)
from thunder.core.pytree import tree_map
from thunder.core.rematerialization import rematerialize
from thunder.core.utils import check
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.symbol import BoundSymbol, BoundSymbolRHS, Symbol, has_tags
from thunder.core.devices import Device, DeviceType, cpu
from thunder.core.transform_common import dce, cse_single_bsym, replace_redundant_inputs
from thunder.core.profile import annotate_for_profile
from thunder.core.compile_data import get_compile_option
from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_mul_prim, dtensor_reshape_prim
from thunder.torch.experimental.dtensor_proxy import DTensorProxy

from thunder.core.transforms import (
    get_grad,
    put_grads,
)

from nvfuser.pytorch_utils import (
    torch_dtype_to_nvfuser_dtype,
)


from thunder.executors.utils import (
    Region,
    _input_dtype_check_fused_scaled_dot_product_attention,
    _input_shape_check_fused_scaled_dot_product_attention,
    _fused_sdp_choice,
    SpdaBackend,
)

from thunder.executors.passes import update_fusion_call_ctx
from thunder.extend import FUEL_LEVEL, FusionExecutor, register_executor
from thunder.executors.nvfuserex import nvfuser_version


DTENSOR_SUPPORTED_VERSION = LooseVersion("0.2.28")
if nvfuser_version() >= DTENSOR_SUPPORTED_VERSION:
    import nvfuser_direct as nvfd
    from nvfuser_direct import FusionDefinition as DirectFusionDefinition

# NOTE This impl file is here because nvFuser may not be available, so it's imported conditionally
#   by nvfuserex.py when nvFuser is available.
import nvfuser
from nvfuser import DataType, FusionDefinition

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

if nvfuser_version() >= LooseVersion("0.2.27"):
    _lcdtype_to_nvdtype_map.update(
        {
            dtypes.uint64: DataType.UInt64,
            dtypes.uint64_: DataType.UInt64,
        }
    )

_lcfp8_to_nvfp8_map: dict[dtypes.dtype, DataType] = {
    dtypes.float8_e5m2: DataType.Float8_e5m2,
    dtypes.float8_e5m2_: DataType.Float8_e5m2,
    dtypes.float8_e4m3fn: DataType.Float8_e4m3fn,
    dtypes.float8_e4m3fn_: DataType.Float8_e4m3fn,
}


_lcdtype_to_nvdtype_map.update(_lcfp8_to_nvfp8_map)


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
        return fd.define_scalar(constant, nvdtype)
    if isinstance(constant, (dtypes.dtype, type)):
        return lcdtype_to_nvdtype(constant)
    if isinstance(constant, Device):
        return None

    utils.check(False, lambda: f"Cannot translate {constant} of type {type(constant)} into an nvFuser constant")


# inline_number allows returning Number as-is, instead of wrap it as an nvfuser constant.
def getnv(x: Any, fd: FusionDefinition, lc_to_nv_map: dict, inline_number: bool = False) -> Any:
    if inline_number and isinstance(x, Number):
        return x
    elif isinstance(x, Proxy):
        return lc_to_nv_map[x]
    elif isinstance(x, (Number, dtypes.dtype, type, Device)):
        return _define_constant(fd, x)
    elif isinstance(x, Sequence):
        return tuple(getnv(i, fd, lc_to_nv_map, inline_number) for i in x)

    utils.check(False, lambda: f"Cannot translate {x} of type {type(x)} to an nvFuser object")


# TODO Check the CUDA arch?
def is_supported_device(device: Device) -> bool:
    utils.check_type(device, Device)
    return device.devicetype is DeviceType.CUDA


def is_supported_devicetype(devicetype: DeviceType) -> bool:
    utils.check_type(devicetype, DeviceType)
    return devicetype is DeviceType.CUDA


_low_precision_floats = (dtypes.float16, dtypes.float16_, dtypes.bfloat16, dtypes.bfloat16_) + tuple(
    _lcfp8_to_nvfp8_map.keys()
)


def device_supports_fp8() -> bool:
    cuda_major, _ = torch.cuda.get_device_capability()
    return cuda_major > 8


def is_supported_dtype(dtype: type | dtypes.dtype, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(dtype, (type, dtypes.dtype))

    if not allow_low_precision_floats:
        if dtype in _low_precision_floats:
            return False

    return dtype in _lcdtype_to_nvdtype_map and (device_supports_fp8() if dtype in _lcfp8_to_nvfp8_map else True)


def is_supported_tensor(a: TensorProxy, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(a, TensorProxy)
    devicetype_supported = a.device.devicetype is DeviceType.CUDA or utils.is_cpu_scalar_tensor(a)
    dtype_supported = is_supported_dtype(a.dtype)

    if not allow_low_precision_floats:
        if a.dtype in _low_precision_floats:
            return False

    rank_supported = a.ndim <= 8
    return devicetype_supported and dtype_supported and rank_supported


def is_supported_tensor_or_number(a: TensorProxy | Number) -> bool:
    if isinstance(a, (Number, NumberProxy)):
        return True

    return is_supported_tensor(a)


# Returns True when all arguments given are supported tensors
#   Throws an error if any arguments are not tensors
# TODO Add a check for the tensor have > 0 elements?
def are_supported_tensors(*args) -> bool:
    return all(is_supported_tensor(arg) for arg in args)


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


def register_dtensor_supported(prim_id: int, fn: Callable, checker_fn: Callable) -> None:
    if nvfuser_version() < DTENSOR_SUPPORTED_VERSION:
        # Only register dtensor ops if supported version is available.
        return

    register_supported(prim_id, fn, checker_fn)


def multidevice_schedule(fd: FusionDefinition, in_dtensors: list[Proxy]) -> None:
    for in_tv, in_dtensor in zip(fd.fusion.inputs(), in_dtensors):
        assert isinstance(in_dtensor, DTensorProxy)
        # Set the device mesh.
        assert in_dtensor.device_mesh.ndim == 1, "nvFuser's Python API only supports 1D meshes."
        mesh = nvfd.multidevice.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

        in_tv.set_device_mesh(mesh)

        assert len(in_dtensor.placements) == 1, "nvFuser's Python API only supports 1D meshes."

        # Split and parallelize.
        # When the mesh is multi-dimensional, iterate through the
        # placements in descending order of Placement.dim.
        placement: Placement = in_dtensor.placements[0]
        if placement.is_shard():
            dim = cast(Shard, placement).dim
            in_tv.split(dim, mesh.size, inner_split=False)
            in_tv.axis(dim).parallelize(nvfd.ParallelType.mesh_x)
            in_tv.set_allocation_domain(in_tv.get_loop_domain(), new_contiguity=True)


def create_fd(
    bsyms: list[BoundSymbol],
    input_descriptors: Sequence[type | tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]]],
    sorted_unique_inputs: list[Proxy],
    sorted_unique_outputs: list[Proxy],
) -> FusionDefinition:
    lc_to_nv_map = utils.ProxyDict()

    def definition(fd):
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
                nvdtype = lcdtype_to_nvdtype(x.python_type)
                nv = fd.define_scalar(nvdtype)
                lc_to_nv_map[x] = nv
            elif isinstance(x, TensorProxy):
                utils.check_type(y, tuple)
                contiguity, stride_order, dtensor_metadata = y
                symbolic_shape = compute_symbolic_shape(x._shape, x._shape)
                nvdtype = lcdtype_to_nvdtype(x.dtype)
                is_cpu = x.device == cpu
                nv = fd.define_tensor(
                    shape=symbolic_shape, contiguity=contiguity, dtype=nvdtype, stride_order=stride_order, is_cpu=is_cpu
                )
                lc_to_nv_map[x] = nv

                for idx, s in enumerate(x.shape):
                    if isinstance(s, Proxy):
                        lc_to_nv_map[s] = nv.size(idx)
            elif isinstance(x, TupleProxy):
                # TODO: discuss the contract here on baked in number from a tuple
                # TODO: validate x is a tuple of int
                nv = fd.define_vector(len(x._value))
                lc_to_nv_map[x] = nv
            elif isinstance(x, Proxy):
                utils.check(False, lambda: f"Unsupported proxy type {type(x)} in fusion", exception_type=AssertionError)
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

    MAX_LENGTH = 9999

    if any(isinstance(t, DTensorProxy) for t in sorted_unique_inputs):
        # multi-GPU path
        utils.check(
            all(isinstance(t, DTensorProxy) for t in sorted_unique_inputs),
            lambda: "nvfuser: Currently we only support Fusion region with all DTensor inputs or all Tensor inputs but not a mix",
        )

        def check_dtensor_tracing_and_runtime_metadata(inp):
            x, y = inp
            _, _, dtensor_metadata = y
            runtime_device_mesh_repr = dtensor_metadata[0]
            runtime_placements_repr = dtensor_metadata[1]
            return x.device_mesh == runtime_device_mesh_repr and x.placements == runtime_placements_repr

        utils.check(
            all(map(check_dtensor_tracing_and_runtime_metadata, zip(sorted_unique_inputs, input_descriptors))),
            lambda: "nvfuser: Expected runtime and tracing metadata to be the same for DTensor.",
        )

        fd = DirectFusionDefinition()
        # Device may be set in one of the "factory" methods like full, iota, or uniform
        # NOTE: This should be called before defining because a factory method may look-up at `_selected_device` while being defined.
        fd._selected_device = None
        with fd:
            definition(fd)
            multidevice_schedule(fd, sorted_unique_inputs)
    else:
        # NOTE nvFuser's default max length is 1024 operations at the time of this writing
        #   This arbitrarily increases it to 9999
        # TODO Review splititng very large fusions or removing the max length restriction completely
        #   See "Very large nvFuser fusions hit max_length"
        fd = FusionDefinition(max_length=MAX_LENGTH)
        # Device may be set in one of the "factory" methods like full, iota, or uniform
        # NOTE: This should be called before defining because a factory method may look-up at `_selected_device` while being defined.
        fd._selected_device = None
        with fd:
            definition(fd)

    return fd


def compute_symbolic_shape(
    proxy_shape: Sequence[int | NumberProxy], shape: torch.Size | Sequence[int]
) -> tuple[int, ...]:
    """
    Computes the symbolic shape of a tensor using nvFuser's notion of a symbolic
    shape:
        -1s represent symbolic shape in nvfuser;
        1s represent broadcast dimensions;
        other value represent static shapes in program.

    Since nvfuser specializes on size-1 dimension for broadcast, we cannot allow
    all dimension to be dynamic. This function looks at TensorProxy.shape as
    well as Tensor.shape, and it tries to translate that for nvfuser's
    FusionDefinition:
        1. if the Tensor.shape entry has value `1`, we translate it as a
           constant `1`;
        2. else:
           2.1 if the corresponding proxy_shape entry is a NumberProxy, we mark
               the dimension as dynamic `-1`,
           2.2. otherwise, Tensor.shape is translated as a static shape.

    Args:
        proxy_shape (Sequence[int | NumberProxy]]): The shape property of the
        TensorProxy.
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.

    Returns:
        Tuple[int, ...]: The shape of the tensor for FusionDefinition.
    """
    nvf_shape = []
    for p_l, l in zip(proxy_shape, shape):
        # loudly raise exception when runtime shape violates proxy_shape in the
        # trace, which indicates issues with the cache. This isn't necessarily
        # an exception.
        check(
            isinstance(p_l, NumberProxy) or p_l == l,
            lambda: f"inconsistent fusion definition with runtime shape {shape} and trace shape {proxy_shape}",
            exception_type=AssertionError,
        )

        # broadcast is specialized in FusionDefinition, preserve it for correct broadcast semantics
        if l == 1:
            nvf_shape.append(l)
        elif isinstance(p_l, NumberProxy):
            nvf_shape.append(-1)
        else:
            nvf_shape.append(l)

    return tuple(nvf_shape)


@lru_cache(maxsize=2048)
def compute_contiguity(
    shape: torch.Size | Sequence[int], stride: Sequence[int]
) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    """
    Computes the contiguity and stride_order of a tensor using nvFuser's notion.

    The contiguity is represented by True, False and None. True represents
    dimensions that are contiguous, and False represents dimensions that are not
    contiguous, and None represents stride-0 or size-1 dimensions.

    The stride_order represents the order of each dimension from innermost to
    outermost.

    For example, a tensor with shape (1, 2, 3) and stride (6, 3, 1):
        contiguity is (None, True, True);
        stride_order is (2, 1, 0);
    For example, a tensor with shape (2, 3, 4) and stride (12, 1, 3):
        contiguity is (True, True, True);
        stride_order is (2, 0, 1);

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.
        stride (Sequence[int]): The stride of the tensor.

    Returns:
        Tuple[Tuple[bool, ...], Tuple[int, ...]]: The contiguity and stride_order
    """
    from nvfuser import compute_tensor_descriptor as nv_compute_td

    return tuple(tuple(x) for x in nv_compute_td(shape, stride))


def make_key_from_dtensor(tensor: torch.Tensor) -> tuple:
    if IS_TORCH_DISTRIBUTED_AVAILABLE and isinstance(tensor, DTensor):
        key = (tensor.device_mesh, tensor.placements)
    else:
        key = ()
    return key


def to_runtime_descriptors(args) -> tuple:
    """
    Converts the arguments to their runtime descriptors.

    Only Tensor objects are converted to runtime descriptors. Non-Tensor objects
    are converted to None.

    Args:
        args: The arguments to convert.

    Returns:
        Tuple: The runtime descriptors of the arguments.
    """
    return tuple(
        compute_contiguity(arg.shape, arg.stride()) + (make_key_from_dtensor(arg),) if isinstance(arg, Tensor) else None
        for arg in args
    )


# TODO Consider making this just a function, because it's faster to call a function than a callable class
@dataclass(slots=True)
class FusionDefinitionWrapper:
    """
    A callable object wrapping a nvFuser fusion definition.
    """

    get_fd: Callable[[tuple[type | tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]], ...]], FusionDefinition]
    to_descriptors: Callable
    name: str
    use_cache: bool
    cache_info: None | Callable = None
    cache_clear: None | Callable = None
    last_used: None | FusionDefinition = None
    last_inputs: None | Sequence[tuple] = None
    store_inputs: bool = False
    save_fake_inputs: bool = False
    enable_options: None | list[str] = None
    disable_options: None | list[str] = None

    @annotate_for_profile("FusionDefinitionWrapper.__call__")
    def __call__(self, *args):
        if self.use_cache or self.last_used is None:
            self.last_used = self.get_fd(self.to_descriptors(args))
        fd = self.last_used

        if self.store_inputs:
            self.last_inputs = args

        if dist.is_available() and any(isinstance(t, torch.distributed.tensor.DTensor) for t in args):
            with annotate_for_profile(self.name):
                output = nvfd.execute_with_dtensors(fd, args)
                return output
        else:
            with annotate_for_profile(self.name):
                return fd.execute(
                    args,
                    device=fd._selected_device,
                    save_repro_inputs=self.save_fake_inputs,
                    _enable_options=self.enable_options,
                    _disable_options=self.disable_options,
                )

    def __repr__(self):
        return f"FusionDefinitionWrapper({self.name})"


def all_tagged(bsym: BoundSymbol, tag: prims.OpTags) -> bool:
    """:obj:`True` if `bsym` and its subsymbols all are tagged with ``tag``."""
    if not has_tags(bsym, {tag}):
        return False

    for sbsym in bsym.subsymbols:
        if not has_tags(sbsym, {tag}):
            return False

    return True


def create_fusion_definition_wrapper(
    bsyms: list[BoundSymbol], name: str, sorted_unique_inputs: list[Proxy], sorted_unique_outputs: list[Proxy]
) -> FusionDefinitionWrapper:
    # NOTE Region Inputs and Outputs
    # The inputs and outputs to a region are represented as sets, which are sorted by name
    #   for determinism. Because they're sets, the inputs and outputs to each region are
    #   unique.
    # It's OK to reorder inputs to regions and outputs from regions, become the dataflow of those
    #   objects is captured by names in the trace.
    # These properties are distinct from the inputs and outputs to the trace itself, which
    #   may contain duplicates and whose order must be preserved.
    store_inputs: None | bool = get_compile_option(
        "nv_store_fusion_inputs", "Allow nvFuser to store fusion inputs for repro."
    )
    save_fake_inputs: None | bool = get_compile_option(
        "nv_save_fake_inputs", "Allow nvFuser to store fake tensor inputs for repro."
    )
    enable_options: list[str] = get_compile_option("nv_enable_options", "List of NVFUSER_ENABLE options to set.") or []
    disable_options: list[str] = (
        get_compile_option("nv_disable_options", "List of NVFUSER_DISABLE options to set.") or []
    )
    skip_cache: bool = get_compile_option("nv_skip_cache", "Skip cache for nvFuser fusions.") or False

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

    fdw = FusionDefinitionWrapper(
        get_fd,
        to_runtime_descriptors,
        name,
        not skip_cache,
        get_fd.cache_info,
        get_fd.cache_clear,
        store_inputs=store_inputs,
        save_fake_inputs=save_fake_inputs,
        enable_options=enable_options,
        disable_options=disable_options,
    )
    return fdw


class nvFuserExecutor(FusionExecutor):
    # Max number of times that this nvFuserExecutor instance can fuse.
    _optimization_fuel: int | FUEL_LEVEL

    def __init__(self):
        super().__init__("nvfuser", version=nvfuser.version())

        # TODO: Replace this with a query to a compile option
        self._use_rematerialization = True

        fuel_str = os.getenv("NVFUSER_OPTIMIZATION_FUEL")
        if fuel_str:
            self.set_fuel(int(fuel_str))
        else:
            self.set_fuel(FUEL_LEVEL.UNLIMITED)

        env_var_save_serde = os.getenv("ENABLE_NVFUSER_SERIALIZATION", None)
        save_serde: bool = env_var_save_serde in ("true", "1")
        self.write_cache_on_exit(save_serde)

    def write_cache_on_exit(self, save_cache: bool = False):
        """
        Selects whether nvFuser writes its cache when the program exits.

        Args:
            save_cache (bool): A flag that enables saving nvFuser cache.
            Defaults to False.

        nvFuser's serialization will save the FusionCache data structure and any
        CUDA cubins into a FlatBuffer binary upon exiting the python program.
        The binary is stored in /tmp/nvfuser_kernel_db/ with the filename
        nvf_serde_[local_rank]_[cuda_major]_[cuda_minor]_[nvrtc_major]_[nvrtc_minor].

        Details:
         * If the common workspace is exists, nvFuser will load it automatically
         when the FusionCache is constructed.
         * When this function is enabled, then when the program exits NvFuser
         will save the FusionCache, overwritting the previous common workspace.
         * If this function is disabled, then when the program exits NvFuser
         does nothing. The previous common workspace is preserved if it exists.
         * If there are any issues when loading the serialized binary, it is
         deleted and the FusionCache is created with its default constructor.
         * When the LOCAL_RANK environment variable is set for ddp or fsdp, a
         separate fusion cache is saved for each device.
        """
        from nvfuser import enable_automatic_serialization, disable_automatic_serialization

        if save_cache:
            enable_automatic_serialization()
        else:
            disable_automatic_serialization()

    def get_fuel(self, amount: int = 1, /) -> bool:
        if self._optimization_fuel is FUEL_LEVEL.UNLIMITED:
            return True

        if self._optimization_fuel < amount:
            return False

        self._optimization_fuel -= amount
        return True

    def set_fuel(self, value: int | FUEL_LEVEL):
        if isinstance(value, FUEL_LEVEL):
            self._optimization_fuel = value
        else:
            assert isinstance(value, int)
            if value < 0:
                raise ValueError(f"optimization_fuel must be non-negative: {value}")
            self._optimization_fuel = value

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

    def _dce_bsyms(self, input_list, output, bsyms: list[BoundSymbol]) -> list[BoundSymbol]:
        trace = TraceCtx(None)
        trace.bound_symbols = bsyms
        bsyms.append(prims.python_return.bind(output, output=None))
        needed_proxies: set[Variable] = set()
        trace = dce(trace, needed_proxies)
        # update the input_list by removing the unused inputs
        input_list[:] = [x for x in input_list if variableify(x) in needed_proxies]
        return list(filter(lambda x: x.sym != prims.python_return, trace.bound_symbols))

    def fuse(self, region: Region, fusion_counter: int) -> BoundSymbol:
        sorted_unique_inputs: list[Proxy] = [unvariableify(x) for x in region.inputs]
        sorted_unique_outputs: list[Proxy] = [unvariableify(x) for x in region.outputs]

        flattened_bsyms: list[BoundSymbol] = []
        for bsym in region.bound_symbols:
            flattened_bsyms.extend(self.flatten(bsym))

        flattened_bsyms = self._dce_bsyms(sorted_unique_inputs, sorted_unique_outputs, flattened_bsyms)

        fusion_name = f"nvFusion{fusion_counter}"
        annotation = f"{fusion_name}: ({', '.join(bsym.sym.name for bsym in flattened_bsyms)})"
        fdw: FusionDefinitionWrapper = create_fusion_definition_wrapper(
            flattened_bsyms, annotation, sorted_unique_inputs, sorted_unique_outputs
        )

        fusion_bsym: BoundSymbol = self.register_temporary_operation(
            fusion_name, fdw, inputs=sorted_unique_inputs, outputs=sorted_unique_outputs, bsyms=flattened_bsyms
        )

        return fusion_bsym

    # TODO Update the replacement of redundant proxies to use a visitor pattern
    #   when that architecture is added in the future
    def cse(self, trace: TraceCtx) -> TraceCtx:
        """Remove bound symbols whose right hand side is common expression.
        Nvfuser specific CSE pass.

        Args:
            trace:

        Returns:
            :class:`TraceCtx` with common subexpression eliminated.
        """

        start_time_ns = time.perf_counter_ns()

        cse_trace = from_trace(trace)

        # The trace_rhs_to_bsym_map is used for CSE on trace outside of nvFusion region.
        # TODO: CSE on overall trace should NOT be inside fusion pass for nvfuser executor.
        trace_rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbolInterface] = {}

        # For bound symbols with redundant rhs expressions, map the output proxies to the output proxies of the common bound symbol.
        redundant_map: dict[Variable, Proxy] = {}
        new_bsyms = {bsym: bsym for bsym in trace.bound_symbols}

        # Updates the trace's proxy
        def map_redundant(x: Any) -> Any:
            if isinstance(x, Proxy):
                return redundant_map.get(Variable(x), x)
            return x

        for bsym in trace.bound_symbols:
            if bsym.sym.is_fusion != True:
                new_bsyms[bsym] = cse_single_bsym(redundant_map, trace_rhs_to_bsym_map, bsym)
                continue

            # The fusion_rhs_to_bsym_map is used only for CSE inside a nvFusion region.
            fusion_rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbolInterface] = {}

            # Rematerialization can replace saved intermediates with extra
            # computation. In this case, replacing a redundant operation would
            # reference an argument that is removed from the fusion's arguments.
            # If the original variable does not exist in this fusion bsym's
            # arguments, then skip this redundant mapping.
            vargs = [variableify(x) for x in bsym.args]
            this_fusion_redundant_map = {k: v for k, v in redundant_map.items() if k in vargs}

            # Apply cse transformation to subsymbols.
            cse_subsymbols = map(
                partial(cse_single_bsym, this_fusion_redundant_map, fusion_rhs_to_bsym_map), bsym.subsymbols
            )
            remove_none_subsymbols = tuple(filterfalse(lambda a: a is None, cse_subsymbols))
            new_subsymbols = replace_redundant_inputs(this_fusion_redundant_map, remove_none_subsymbols)

            # Add any new redundant mappings for this fusion to the main dictionary.
            redundant_map.update(this_fusion_redundant_map)

            # Map redundant args and outputs that have a common subexpression to the same value.
            # Remove identical values from the bsym's arguments and outputs.
            #  * First, variableify the proxies so they are hashable.
            #  * Then, create a dictionary where the keys are variables and the values are their original proxies.
            #  * Lastly, create a new tuple given the dictionary values.
            new_args = tree_map(map_redundant, bsym.args)
            if isinstance(new_args, Sequence):
                new_args = tuple({variableify(x): x for x in new_args}.values())

            new_output = tree_map(map_redundant, bsym.output)
            if isinstance(new_output, Sequence):
                new_output = tuple({variableify(x): x for x in new_output}.values())

            # Create new bsym with updated args, subsymbols and outputs.
            new_bsyms[bsym] = replace(bsym, args=new_args, subsymbols=new_subsymbols, output=new_output)

            # TODO Add (rhs, bsym) key, value pairs for nvfusion outputs to trace_rhs_to_bsym_map

        # New bound symbols are still incorrect. Its _ctx_call dict points to the
        # old nvFuser fusion. We need to update it to use the new definition.
        new_symbols = [new_bsyms.get(bsym, bsym) for bsym in trace.bound_symbols]
        cse_trace.bound_symbols = list(filterfalse(lambda a: a is None, new_symbols))

        return_bsym = cse_trace.bound_symbols[-1]
        assert return_bsym.sym.id == prims.PrimIDs.RETURN
        trace_output = tree_map(map_redundant, return_bsym.args)
        cse_trace.bound_symbols[-1] = prims.python_return.bind(*trace_output, output=None)

        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000

        cse_trace.set_provenance(
            TraceProvenance(f"Nvfuser Common Subexpression Elimination (took {elapsed_time_millis} milliseconds)")
        )
        return cse_trace

    # TODO Restore fusion logic here -- this just replaces supported operations in isolation at the moment
    def fusion_pass(self, trace: TraceCtx) -> TraceCtx:
        start_time_ns: int = time.perf_counter_ns()
        # Replace uniform with uniform_philox and rng state operators for better rematerialization
        from thunder.core.rematerialization import replace_uniform

        trace = replace_uniform(trace)

        fusedtrace: TraceCtx = from_trace(trace)

        producers, consumers = utils.producers_and_consumers(trace)
        from thunder.executors.data_dependent_partition import Node, fuse_bound_symbols

        fused_bsyms = []

        # TODO has_cuda_input_or_output is too restrictive a check on what should be fused
        # TODO check whether a function would output a CPU tensor? -- can nvFuser fuse such operations?
        #   ex. device_put to a CPU device from a CUDA device
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

        bound_symbol_groups = fuse_bound_symbols(trace, _should_fuse)

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

            nv_enable_shape_only_fusion: None | bool = get_compile_option(
                "nv_enable_shape_only_fusion",
                "Allow nvFuser to create Fusion with shape only operations. Defaults to False.",
            )

            if not nv_enable_shape_only_fusion:
                # Don't fuse a region which has only Shape Operations.
                all_shape_ops = all(map(lambda bsym: all_tagged(bsym, prims.OpTags.SHAPE_OP), bsyms))
                if all_shape_ops:
                    fused_bsyms.extend(bsyms)
                    continue

            if len(bsyms) == 1:
                bsym: BoundSymbol = bsyms[0]
                can_fuse: bool = self.can_fuse(bsym)
                cuda_in_or_out: bool = self.has_cuda_input_or_output(bsym)
                if not can_fuse or not cuda_in_or_out:
                    fused_bsyms.append(bsym)
                    continue

            if self.get_fuel():
                fusion_bsym: BoundSymbol = self.fuse(region, fusion_counter)
                fused_bsyms.append(fusion_bsym)
                fusion_counter += 1
            else:
                fused_bsyms.extend(region.bound_symbols)

        fusedtrace.bound_symbols = fused_bsyms

        # Some of the operations might be better placed with its consumers (for
        # example residual connection in transformer block). This pass moves
        # them to the consumer.
        if self._use_rematerialization:
            fusedtrace = rematerialize(fusedtrace)

        fusedtrace = remove_redundant_casts(fusedtrace)
        fusedtrace = self.cse(fusedtrace)
        fusedtrace = dce(fusedtrace)

        fusedtrace = update_fusion_call_ctx(fusedtrace)

        end_time_ns: int = time.perf_counter_ns()
        elapsed_time_ns: int = end_time_ns - start_time_ns
        elapsed_time_millis: int = elapsed_time_ns // 1000000
        fusedtrace.set_provenance(TraceProvenance(f"Fusion (took {elapsed_time_millis} milliseconds)"))

        return fusedtrace


ex = nvFuserExecutor()
register_executor(ex)


def register_supported(sym_or_id: Hashable, translator: Callable, checker: Callable):
    ex.register_supported(sym_or_id, checker)
    id = sym_or_id.id if isinstance(sym_or_id, Symbol) else sym_or_id
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


def _bitcast_check(src: TensorProxy, dtype: dtypes.dtype) -> bool:
    return (
        nvfuser_version() > LooseVersion("0.29.0")
        and _convert_element_type_check(src, dtype)
        and src.dtype.bytes == dtype.bytes
    )


# TODO: Expose bitcast in nvfuser to Python.
# def bitcast(src: TensorProxy, dtype: dtypes.dtype, *, fd: FusionDefinition, lc_to_nv_map: dict):
#     nva = getnv(src, fd, lc_to_nv_map)
#     nvdtype = lcdtype_to_nvdtype(dtype)
#
#     return fd.ops.bitcast(nva, nvdtype)
#
#
# register_supported(PrimIDs.BITCAST, bitcast, _bitcast_check)

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
        fd._selected_device is None or fd._selected_device == device.index,
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
    seed: int | NumberProxy | TensorProxy,
    offset: int | NumberProxy | TensorProxy,
) -> bool:
    return (
        is_supported_device(device)
        and is_supported_dtype(dtype)
        and is_supported_tensor_or_number(seed)
        and is_supported_tensor_or_number(offset)
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
    if any(map(lambda x: isinstance(x, NumberProxy), shape)):
        nv_shape = getnv(shape, fd, lc_to_nv_map)
    else:
        nv_shape = shape

    return fd.ops.broadcast_in_dim(nva, nv_shape, broadcast_dimensions)


register_supported(PrimIDs.BROADCAST_IN_DIM, broadcast_in_dim, _broadcast_in_dim_check)


def _cat_check(tensors: list[TensorProxy], dim: int) -> bool:
    # Validates tensors and concatenated dimension lengths
    for t in tensors:
        if not is_supported_tensor(t):
            return False

    return True


# NOTE nvFuser's cat prim accepts dim as a Python Number, not a constant
def cat(tensors: list[TensorProxy], dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nvtensors = list(getnv(t, fd, lc_to_nv_map) for t in tensors)

    return fd.ops.cat(nvtensors, dim)


register_supported(PrimIDs.CAT, cat, _cat_check)


def _stride_order_check(a: TensorProxy, order: Sequence[int]) -> bool:
    return is_supported_tensor(a)


def stride_order(a: TensorProxy, order: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.stride_order(nva, order)


register_supported(PrimIDs.STRIDE_ORDER, stride_order, _stride_order_check)


# NOTE nvFuser does not support dilation > 0
def _pad_check(a: TensorProxy, padding_value: Number, padding_config: tuple[int, int, int]) -> bool:
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
#   This is in constrast to thunder.jit's pad primitive, which specifies padding
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


def reshape(a: TensorProxy, shape: list[int, NumberProxy, ...], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    if any(map(lambda x: isinstance(x, NumberProxy), shape)):
        nv_shape = getnv(shape, fd, lc_to_nv_map)
    else:
        nv_shape = shape

    return fd.ops.reshape(nv_a, nv_shape)


register_supported(PrimIDs.RESHAPE, reshape, _reshape_check)
register_supported(dtensor_reshape_prim, reshape, _reshape_check)


# NOTE nvFuser's slice operation only supports all strides == 1
def _slice_check(
    a: TensorProxy, start_indices: Sequence[int], end_indices: Sequence[int], strides: Sequence[int] | None = None
) -> bool:
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

    # fd.ops.slice prefers python Number as slice indices, which allows the FusionDefinition print out to inline its
    # indices. fd.ops.slice requires all input sequence to have identical element type, so we can only inline_number
    # when all slice indices are python numbers.
    inline_number = all(map(lambda x: not isinstance(x, Proxy), start_indices + end_indices + strides))

    nv_start_indices = getnv(start_indices, fd, lc_to_nv_map, inline_number=inline_number)
    nv_end_indices = getnv(end_indices, fd, lc_to_nv_map, inline_number=inline_number)
    nv_strides = getnv(strides, fd, lc_to_nv_map, inline_number=inline_number)

    return fd.ops.slice(nva, nv_start_indices, nv_end_indices, nv_strides)


register_supported(PrimIDs.SLICE, nv_slice, _slice_check)


def _squeeze_check(a: TensorProxy, /, dims: Sequence[int]) -> bool:
    return is_supported_tensor(a)


# NOTE nvFuser's squeeze operation requires the shape of the tensor be specified
def squeeze(a: TensorProxy, /, dims: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.squeeze(nva, dims)


register_supported(PrimIDs.SQUEEZE, squeeze, _squeeze_check)


# NOTE: Currently `_advanced_indexing` seems to return a `TensorProxy` of wrong shape
# when input is 0-size tensor, leading to a broken nvfuser definition.
# So for now, it'd be reasonable to disallow 0-size tensors.
# Related: https://github.com/Lightning-AI/lightning-thunder/issues/2068
def _take_check(a: TensorProxy, /, index: TensorProxy, dim: int) -> bool:
    return are_supported_tensors(a, index) and a.numel > 0


def take(a: TensorProxy, /, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_index = getnv(index, fd, lc_to_nv_map)

    return fd.ops.index_select(nv_a, nv_index, dim)


register_supported(PrimIDs.TAKE, take, _take_check)


def take_along_axis(
    a: TensorProxy, /, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_index = getnv(index, fd, lc_to_nv_map)

    return fd.ops.take_along_axis(nv_a, nv_index, dim)


register_supported(PrimIDs.TAKE_ALONG_AXIS, take_along_axis, _take_check)


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
def _elementwise_unary_check(a: Number | TensorProxy) -> bool:
    return is_supported_tensor_or_number(a)


def _elementwise_nnary_check(args: tuple[TensorProxy]) -> bool:
    return are_supported_tensors_or_numbers(*args)


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


def imag(a: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.imag(nva)


register_supported(PrimIDs.IMAG, imag, _elementwise_unary_check)


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


def clone(a: TensorProxy, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.set(nva)


register_supported(PrimIDs.CLONE, clone, _elementwise_unary_check)


def shallow_copy(a: TensorProxy, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return nva


register_supported(PrimIDs.SHALLOW_COPY, shallow_copy, _elementwise_unary_check)


# update_aliases is disabled.  nvfuser does not support it.
# TODO: Enable this once nvfuser supports it.
# def update_aliases(aliases: tuple[TensorProxy], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
#     nvaliases = tuple(getnv(alias, fd, lc_to_nv_map) for alias in aliases)
#     return tuple(fd.ops.set(nvalias) for nvalias in nvaliases)


# register_supported(PrimIDs.UPDATE_ALIASES, update_aliases, _elementwise_nnary_check)


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


def div(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    a_dtype = dtypes.to_dtype(a)
    b_dtype = dtypes.to_dtype(b)

    if dtypes.is_integer_dtype(a_dtype) and dtypes.is_integer_dtype(b_dtype):
        return fd.ops.div(nva, nvb)

    # NOTE It's currently significantly faster for nvFuser to multiply the reciprocal than divide
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


register_supported(PrimIDs.LE, le, _elementwise_binary_check)


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
register_dtensor_supported(dtensor_mul_prim.id, mul, _elementwise_binary_check)


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


def maximum(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.maximum(nva, nvb)


register_supported(PrimIDs.MAXIMUM, maximum, _elementwise_binary_check)


def minimum(a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.minimum(nva, nvb)


register_supported(PrimIDs.MINIMUM, minimum, _elementwise_binary_check)


def bitwise_left_shift(
    a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_left_shift(nva, nvb)


def bitwise_right_shift(
    a: TensorProxy | Number, b: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_right_shift(nva, nvb)


register_supported(PrimIDs.BITWISE_LEFT_SHIFT, bitwise_left_shift, _elementwise_binary_check)
register_supported(PrimIDs.BITWISE_RIGHT_SHIFT, bitwise_right_shift, _elementwise_binary_check)


#
# Elementwise ternary operations
#


def _elementwise_ternary_check(a: Number | TensorProxy, b: Number | TensorProxy, c: Number | TensorProxy) -> bool:
    return are_supported_tensors_or_numbers(a, b, c)


def lerp(
    start: TensorProxy, end: TensorProxy, weight: TensorProxy | Number, *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nv_start = getnv(start, fd, lc_to_nv_map)
    nv_end = getnv(end, fd, lc_to_nv_map)
    nv_weight = getnv(weight, fd, lc_to_nv_map)

    return fd.ops.lerp(nv_start, nv_end, nv_weight)


register_supported(PrimIDs.LERP, lerp, _elementwise_ternary_check)


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

    # explicit type promotion is necessary, since nvfuser can't do this properly with scalar inputs. See
    # issue: https://github.com/NVIDIA/Fuser/issues/3816
    # Determines result dtype
    numbertype, tensordtype = utils.check_same_dtype(a, b)
    dtype = tensordtype if tensordtype is not None else numbertype

    # NOTE: for scalar inputs, dtype mapping is different. e.g. float -> double. We convert dtypes to strong
    # type if the output is supposed to be a tensor proxy
    if any(map(lambda x: isinstance(x, TensorProxy), (pred, a, b))):
        dtype = dtypes.to_strong_dtype(dtype)

    return fd.ops.cast(fd.ops.where(nvpred, nva, nvb), lcdtype_to_nvdtype(dtype))


register_supported(PrimIDs.WHERE, where, _elementwise_ternary_check)

#
# Reduction operations
#


# TODO Checks that the dtype is supported by nvFuser
def _reduction_check(a: TensorProxy, dims: Sequence[int]) -> bool:
    return is_supported_tensor(a, allow_low_precision_floats=False) and not any(
        isinstance(dim, NumberProxy) for dim in dims
    )


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


def _copy__check(
    copy_from: TensorProxy,
    copy_to: TensorProxy,
    *,
    grad_enabled: bool,
) -> bool:
    return are_supported_tensors(copy_from, copy_to)


def copy_(
    copy_from: TensorProxy,
    copy_to: TensorProxy,
    *,
    grad_enabled: bool,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nvcopy_from = getnv(copy_from, fd, lc_to_nv_map)
    nvcopy_to = getnv(copy_to, fd, lc_to_nv_map)
    alias_output = fd.ops.set(nvcopy_from)
    fd.add_output(alias_output, alias_input=nvcopy_to)
    return alias_output


register_supported(PrimIDs.COPY_, copy_, _copy__check)


# Removes excessive float casts, like those that occur when autocasting
# NOTE This passes actually changes a program's semantics, because it will take a sequence like
#   fp32 -> fp16 -> fp32 and remove all the operations, but casting fp32 values to fp16 can
#   changes the values (because most fp32 values are not representable in fp16)
# NOTE This only handles conversions performed by CONVERT_ELEMENT_TYPE, and not conversions caused
#   by other Symbols, like torch.to, which may be unflattened
# TODO This could be extended to non-float conversions, like complex -> complex conversions
def remove_redundant_casts(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    start_time_ns = time.perf_counter_ns()

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
            utils.check(len(bsym.kwargs) == 1, lambda: "Expected two arguments for convert element type")
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
                lambda: "Expected no subsymbols when creating a new BoundSymbol in the remove redundant casts pass",
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

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    rrctrace.set_provenance(TraceProvenance(f"Remove redundant casts (took {elapsed_time_millis} milliseconds)"))
    return rrctrace


def _linear_check(a: TensorProxy, b: TensorProxy, bias: TensorProxy | None) -> bool:
    enable_linear: None | bool = get_compile_option("nv_enable_linear", "Enable nvFuser linear.")
    if not enable_linear:
        return False
    # Verify linear inputs and bias (optional) are supported tensors.
    if not are_supported_tensors(a, b) or (bias is not None and not is_supported_tensor(bias)):
        return False
    return True


def linear(
    a: TensorProxy,
    b: TensorProxy,
    bias: TensorProxy | None,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)
    nvbias = None if bias is None else getnv(bias, fd, lc_to_nv_map)
    return fd.ops.linear(nva, nvb, nvbias)


register_supported(PrimIDs.LINEAR, linear, _linear_check)


def _matmul_check(
    a: TensorProxy,
    b: TensorProxy,
) -> bool:
    enable_matmul: None | bool = get_compile_option("nv_enable_matmul", "Enable nvFuser matmul.")

    if not enable_matmul or not are_supported_tensors(a, b):
        return False
    return True


def matmul(
    a: TensorProxy,
    b: TensorProxy,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)
    return fd.ops.matmul(nva, nvb)


register_supported(PrimIDs.MATMUL, matmul, _matmul_check)


def _shape_check(
    a: TensorProxy,
) -> bool:
    # TODO: currently we cannot support this yet. fusion_pass needs to be
    # updated to ensure that the fused region consumes all NumberProxy within
    # and not leak it out as a fusion output, since nvfuser cannot yet produce
    # scalar outputs.
    return False


def shape(
    a: TensorProxy,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    ret = []
    for i in range(a.ndim):
        ret.append(fd.ops.size(nva, i))
    return ret


register_supported(PrimIDs.SHAPE, shape, _shape_check)


# Registering SDPA operators for nvFuser
# SDPA requires an execution and grad transform since the forward and backward passes are called through different implementations.
# For both execution and grad transform, a new operator is registered with nvfuserex (ex.register_operator) and then added to the translation map (register_supported).
# The operators are tagged with OpTag.RANDOM_OP to prevent rematerialization in backward pass.
# Finally, the complete rule is registered through ex.register_supported, with the execution and grad transform wrapping around these operators.


# SDPA Forward
def _scaled_dot_product_flash_attention_forward_meta(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
) -> tuple[TensorProxy, TensorProxy, int, int]:
    # Reference metadata:
    # * query (batch_size, num_heads, query_seq_len, E)
    # * key (batch_size, num_heads, key_seq_len, E)
    # * value (batch_size, num_heads, key_seq_len, Ev)
    # * output (batch_size, num_heads, query_seq_len, Ev)

    # at::_scaled_dot_product_flash_attention returns {output, log_sumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask}.
    # In nvFuser, we only save {output, log_sumexp, philox_seed/offset} for backward since the other variables are not required for non-nested input tensors.
    # For non-nested tensor, cum_seq_q/k is undefined, max_q/k can be inferred from input size, and we set `return_debug_mask=False`, so `debug_attn_mask` is a 1D zero tensor.

    batch_size, num_heads, query_seq_len, E = query.shape

    UPDATED_SDPA = LooseVersion(torch.__version__) >= LooseVersion("2.7.0")
    philox_shape = (2,) if UPDATED_SDPA else ()
    dtype = dtypes.uint64 if UPDATED_SDPA else dtypes.int64
    device = query.device if UPDATED_SDPA else "cpu"

    return (
        output := TensorProxy(like=query, shape=(batch_size, num_heads, query_seq_len, E)),
        log_sumexp := TensorProxy(
            shape=(batch_size, num_heads, query_seq_len), dtype=dtypes.float32, device=query.device, requires_grad=False
        ),
        philox_seed := TensorProxy(shape=philox_shape, dtype=dtype, device=device, requires_grad=False),
        philox_offset := TensorProxy(shape=(), dtype=dtype, device=device, requires_grad=False),
    )


def _scaled_dot_product_flash_attention_forward(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    inputs = [query, key, value, dropout_p, is_causal, scale]
    nv_inputs = []
    for inp in inputs:
        nv_inp = getnv(inp, fd, lc_to_nv_map) if inp is not None else None
        nv_inputs.append(nv_inp)

    return fd.ops.sdpfa_fwd(*nv_inputs)


nv_sdpfa_fwd = ex.register_operator(
    "nv_sdpfa_fwd",
    meta=_scaled_dot_product_flash_attention_forward_meta,
    fn=_scaled_dot_product_flash_attention_forward,
    tags=[prims.OpTags.RANDOM_OP],
)

register_supported(nv_sdpfa_fwd.id, _scaled_dot_product_flash_attention_forward, None)


# SDPA Backward
def _scaled_dot_product_flash_attention_backward_meta(
    grad_out: TensorLike,
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    out: TensorLike,
    logsumexp: TensorLike,
    dropout_p: float,
    is_causal: bool,
    philox_seed: TensorLike,
    philox_offset: TensorLike,
    *,
    scale: None | float = None,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    batch_size, num_heads, query_seq_len, E = query.shape
    key_seq_len = key.shape[2]

    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/f57b00704e498a676854a02974ca9e0c42188b23/torch/_meta_registrations.py#L5043-L5063
    grad_query = TensorProxy(like=query, shape=(batch_size, num_heads, query_seq_len, E))
    grad_key = TensorProxy(like=key, shape=(batch_size, num_heads, key_seq_len, E))
    grad_value = TensorProxy(like=value, shape=(batch_size, num_heads, key_seq_len, E))
    return (grad_query, grad_key, grad_value)


def _scaled_dot_product_flash_attention_backward(
    grad_out: TensorProxy,
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    out: TensorProxy,
    logsumexp: TensorProxy,
    dropout_p: float,
    is_causal: bool,
    philox_seed: TensorProxy,
    philox_offset: TensorProxy,
    *,
    scale: None | float = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = [grad_out, query, key, value, out, logsumexp, dropout_p, is_causal, philox_seed, philox_offset, scale]
    nv_inputs = []
    for inp in inputs:
        nv_inp = getnv(inp, fd, lc_to_nv_map) if inp is not None else None
        nv_inputs.append(nv_inp)

    return fd.ops.sdpfa_bwd(*nv_inputs)


nv_sdpfa_bwd = ex.register_operator(
    "nv_sdpfa_bwd",
    meta=_scaled_dot_product_flash_attention_backward_meta,
    fn=_scaled_dot_product_flash_attention_backward,
    tags=[prims.OpTags.RANDOM_OP],
)

register_supported(nv_sdpfa_bwd.id, _scaled_dot_product_flash_attention_backward, None)


# Checker for SDPA
def _scaled_dot_product_flash_attention_check(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: Proxy | None,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: None | float = None,
) -> bool:
    # fd.ops.sdpfa_fwd and fd.ops.sdpfa_bwd are adding in versions 0.2.9 and 0.2.10 respectively.
    if nvfuser_version() < LooseVersion("0.2.10"):
        return False

    # SDPA requires nvfuser version 0.2.27 or higher for torch 2.7.0 or higher.
    if LooseVersion(torch.__version__) >= LooseVersion("2.7.0") and nvfuser_version() < LooseVersion("0.2.27"):
        return False

    enable_sdpa: None | bool = get_compile_option("nv_enable_sdpa", "Enable nvFuser flash attention SDPA.")

    if not enable_sdpa:
        return False

    # Flash attn does not support attn_mask currently.
    if attn_mask is not None:
        return False

    if not are_supported_tensors(query, key, value):
        return False

    # FP64 is not supported by flash attention
    supported_dtypes = (dtypes.float16, dtypes.bfloat16)
    _input_dtype_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None, supported_dtypes)
    _input_shape_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None)

    # nvFuser only implements flash attention currently.
    backend = _fused_sdp_choice(query, key, value, None, dropout_p, is_causal, scale)
    return backend == SpdaBackend.FLASH_ATTENTION


# SDPA execution_transform -- calls nv_sdpfa_fwd operator registered above
def scaled_dot_product_flash_attention(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: TensorProxy = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
):
    (attn_output, logsumexp, philox_seed, philox_offset) = nv_sdpfa_fwd(
        query, key, value, dropout_p, is_causal, scale=scale
    )
    return attn_output


# SDPA grad_transform -- calls nv_sdpfa_fwd and nv_sdpfa_bwd registered above
def scaled_dot_product_flash_attention_grad(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
):
    (attn_output, logsumexp, philox_seed, philox_offset) = nv_sdpfa_fwd(
        query, key, value, dropout_p, is_causal, scale=scale
    )
    grad_out = get_grad(attn_output)
    grad_query, grad_key, grad_val = nv_sdpfa_bwd(
        grad_out,
        query,
        key,
        value,
        attn_output,
        logsumexp,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )
    put_grads((query, key, value), (grad_query, grad_key, grad_val))
    return attn_output


# Register the complete rule for SDPA in nvfuser executor
ex.register_supported(
    ltorch.scaled_dot_product_attention,
    checker=_scaled_dot_product_flash_attention_check,
    execution_transform=scaled_dot_product_flash_attention,
    grad_transform=scaled_dot_product_flash_attention_grad,
)


def _embedding_check(
    input: TensorProxy,
    weight: TensorProxy,
    padding_idx: None | int,
    max_norm: None | float,
    norm_type: None | float,
    scale_grad_by_freq: None | bool,
    sparse: None | bool,
) -> bool:
    if nvfuser_version() < LooseVersion("0.2.25"):
        return False
    enable_embedding: None | bool = get_compile_option("nv_enable_embedding", "Enable nvFuser embedding.")
    if enable_embedding is not None:
        warnings.warn(
            "nv_enable_embedding is no longer used. embedding through nvfuserex is enabled by default, option nv_enable_embedding is ignored"
        )
    # Verify input and weight are supported tensors.
    if not are_supported_tensors(input, weight) or (weight.ndim != 2):
        return False
    return True


def embedding(
    input: TensorProxy,
    weight: TensorProxy,
    padding_idx: None | int = None,
    max_norm: None | float = None,
    norm_type: None | float = 2.0,
    scale_grad_by_freq: None | bool = False,
    sparse: None | bool = False,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    # embedding forward without renorm could be implemented as `index_select`, which is supported
    # by nvfuser codegen
    if max_norm is None:
        nv_input = getnv(input, fd, lc_to_nv_map)
        nv_weight = getnv(weight, fd, lc_to_nv_map)
        restore_shape = None
        # high order indices can be reshaped into an array and then restore the shape after
        # index_select
        if nv_input.ndim > 1:
            restore_shape = []
            for i in range(input.ndim):
                restore_shape.append(nv_input.size(i))
            restore_shape.append(nv_weight.size(weight.ndim - 1))
            nv_input = fd.ops.reshape(nv_input, [-1])
        ret = fd.ops.index_select(nv_weight, nv_input, 0)
        if restore_shape is not None:
            ret = fd.ops.reshape(ret, restore_shape)
        return ret
    else:
        inputs = [input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
        nv_inputs = []
        for inp in inputs:
            nv_inp = getnv(inp, fd, lc_to_nv_map) if inp is not None else None
            nv_inputs.append(nv_inp)
        return fd.ops.embedding_fwd(*nv_inputs)


register_supported(PrimIDs.EMBEDDING, embedding, _embedding_check)
register_supported(ltorch.embedding, embedding, _embedding_check)


def _index_put_check(a: TensorProxy, /, indices: Sequence[TensorProxy], values: TensorProxy, accumulate: bool) -> bool:
    # temporary flag to allow scatter-like operations to be consumed by nvfuserex
    enable_scatter: None | bool = get_compile_option("nv_enable_scatter", "Enable nvFuser scatter-like operations.")
    if not enable_scatter:
        return False

    # TODO: limited support inside nvfuser. remove this when codegen support is generalized.
    # see nvfuser issue: https://github.com/NVIDIA/Fuser/issues/4857 tracking indexing operation support.
    if len(indices) != 1 or indices[0].ndim != 1:
        return False

    if accumulate:
        return False

    return True


def index_put(
    a: TensorProxy,
    /,
    indices: Sequence[TensorProxy],
    values: TensorProxy,
    accumulate: bool,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> any:
    utils.check(
        not accumulate, lambda: "Unsupported accumulate in index_put by nvfuserex", exception_type=AssertionError
    )
    nva = getnv(a, fd, lc_to_nv_map)
    nvi = getnv(indices[0], fd, lc_to_nv_map)
    # construct the shape of broadcast indices tensor as
    # [-1, *a.shape[1:]]
    shapes = nva.shape()
    flag = [-1]
    for i in range(1, nva.ndim):
        flag += [shapes[i]]
    # broadcast index tensor nvi to abide to scatter semantics
    nvi_b = fd.ops.broadcast_in_dim(nvi, flag, [0])

    nvs = getnv(values, fd, lc_to_nv_map)

    # index_put is translated to scatter in nvfuser
    return fd.ops.scatter(nva, nvi_b, nvs, 0)


register_supported(PrimIDs.INDEX_PUT, index_put, _index_put_check)


def _scatter_check(a: TensorProxy, /, index: TensorProxy, src: TensorProxy | Number, dim: int) -> bool:
    # temporary flag to allow scatter-like operations to be consumed by nvfuserex
    enable_scatter: None | bool = get_compile_option("nv_enable_scatter", "Enable nvFuser scatter-like operations.")
    if not enable_scatter:
        return False
    return True


def scatter(
    a: TensorProxy,
    /,
    index: TensorProxy,
    src: TensorProxy | Number,
    dim: int,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvi = getnv(index, fd, lc_to_nv_map)
    nvs = getnv(src, fd, lc_to_nv_map)

    # index_put is translated to scatter in nvfuser
    return fd.ops.scatter(nva, nvi, nvs, dim)


register_supported(PrimIDs.SCATTER, scatter, _scatter_check)


def _cross_entropy_check(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike,
    size_average: None | Any,
    ignore_index: int,
    reduce: None | Any,
    reduction: str,
    label_smoothing: float,
    *args,
) -> bool:
    if nvfuser_version() < LooseVersion("0.2.10"):
        return False

    # TODO: support higher dim inputs
    if a.ndim != 2 or a.ndim - 1 != target.ndim:
        return False

    if a.shape[0] != target.shape[0]:
        return False

    # input must be cast to float32
    # since we use fmax which only supports float32
    if dtypes.to_torch_dtype(a.dtype) != torch.float32:
        return False

    # We only optimize for the following cases
    if reduction != "mean":
        return False

    if ignore_index >= 0:
        return False

    if any(x is not None for x in (weight, size_average, reduce)):
        return False

    if label_smoothing != 0.0:
        return False

    return True


def cross_entropy_fwd_meta(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    losses: TensorLike
    check(
        reduction == "mean",
        lambda: f"cross entropy expected reduction to be 'mean' but was given {reduction}",
    )
    losses = TensorLike(like=a, shape=())

    max_log_sum_exp: TensorLike = TensorLike(like=target, dtype=dtypes.float32)
    num_valid_indices: TensorLike = TensorLike(like=losses, shape=(), dtype=dtypes.float32)
    a_max: TensorLike = TensorLike(like=target, dtype=dtypes.float32)
    return losses, a_max, max_log_sum_exp, num_valid_indices


def cross_entropy_fwd(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_target = getnv(target, fd, lc_to_nv_map)
    nv_ignore_index = getnv(ignore_index, fd, lc_to_nv_map)

    zero_scalar = fd.define_scalar(0, dtype=lcdtype_to_nvdtype(a.dtype))

    # modify the labels to account for ignore index and then do a
    # gather/ take_along_axis on the input tensor
    ne = fd.ops.ne(nv_target, nv_ignore_index)
    where_0 = fd.ops.where(ne, nv_target, zero_scalar)
    where = fd.ops.broadcast_in_dim(where_0, shape=[nv_target.shape()[-1], 1], broadcast_dims=[0])
    gather0 = fd.ops.take_along_axis(nv_a, where, dim=1)
    gather = fd.ops.reshape(gather0, new_shape=[nv_target.shape()[-1]])

    # compute log (sum(exp(A - max(A, dim=1), dim=1))
    max = fd.ops.max(nv_a, 1)
    max2 = fd.ops.broadcast_in_dim(max, shape=[nv_target.shape()[-1], 1], broadcast_dims=[0])
    sub = fd.ops.sub(nv_a, max2)
    exp = fd.ops.exp(sub)
    sum_1 = fd.ops.sum(exp, 1)
    log = fd.ops.log(sum_1)

    # compute S = POST_GATHER(A) - max(A, dim=1) - log(sum(exp(A - max(A, dim=1), dim=1))
    gather_sub = fd.ops.sub(gather, max)
    log_softmax_post_gather = fd.ops.sub(gather_sub, log)

    # set the values for ignore index to 0
    neg = fd.ops.neg(log_softmax_post_gather)
    where_1 = fd.ops.where(ne, neg, zero_scalar)

    # sum_2_cvt computes the number of valid indices
    sum_2 = fd.ops.sum(ne)
    sum_2_cvt = fd.ops.cast(sum_2, dtype=DataType.Float)

    # compute the mean
    sum_3 = fd.ops.sum(where_1)
    div = fd.ops.div(sum_3, sum_2_cvt)
    return div, max, log, sum_2_cvt


nv_cross_entropy_fwd = ex.register_operator(
    "nv_cross_entropy_fwd",
    meta=cross_entropy_fwd_meta,
    tags=[prims.OpTags.REDUCTION_OP],
)
register_supported(nv_cross_entropy_fwd.id, cross_entropy_fwd, None)


def cross_entropy_bwd_meta(
    g: TensorLike,
    a: TensorLike,
    *,
    target: TensorLike,
    a_max: TensorLike,
    max_log_sum_exp: TensorLike,
    valid_indices: TensorLike,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Any:
    return TensorProxy(like=a)


def cross_entropy_bwd(
    g: TensorLike,
    a: TensorLike,
    *,
    target: TensorLike,
    a_max: TensorLike,
    max_log_sum_exp: TensorLike,
    valid_indices: TensorLike,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_target = getnv(target, fd, lc_to_nv_map)
    nv_ignore_index = getnv(ignore_index, fd, lc_to_nv_map)
    nv_g = getnv(g, fd, lc_to_nv_map)
    nv_valid_indices = getnv(valid_indices, fd, lc_to_nv_map)
    nv_a_max = getnv(a_max, fd, lc_to_nv_map)
    nv_max_log_sum_exp = getnv(max_log_sum_exp, fd, lc_to_nv_map)

    zero = fd.define_scalar(0, dtype=DataType.Int)
    one = fd.define_scalar(1, dtype=DataType.Int)

    # scatter the gradients (negative) - this is backward of nll loss
    iotas = fd.ops.iota(nv_a.shape()[-1], zero, one, dtype=DataType.Int)
    iotas_bcast = fd.ops.broadcast_in_dim(iotas, shape=nv_a.shape(), broadcast_dims=[nv_a.ndim - 1])
    neg_gradients = fd.ops.neg(nv_g)
    neg_gradients_mod = fd.ops.div(neg_gradients, nv_valid_indices)

    target_bcast = fd.ops.broadcast_in_dim(nv_target, shape=nv_a.shape(), broadcast_dims=[nv_target.ndim - 1])
    mask = fd.ops.eq(iotas_bcast, target_bcast)
    neg_gradients_mod_bcast = fd.ops.broadcast_in_dim(neg_gradients_mod, shape=nv_a.shape(), broadcast_dims=[])
    scattered_vals = fd.ops.where(mask, neg_gradients_mod_bcast, zero)

    ne = fd.ops.ne(target_bcast, nv_ignore_index)
    new_target_bcast = fd.ops.where(ne, neg_gradients_mod_bcast, zero)

    # build the softmax
    nv_a_max_bcast = fd.ops.broadcast_in_dim(nv_a_max, shape=nv_a.shape(), broadcast_dims=[nv_a_max.ndim - 1])
    input_minus_max = fd.ops.sub(nv_a, nv_a_max_bcast)
    nv_max_log_sum_exp_bcast = fd.ops.broadcast_in_dim(
        nv_max_log_sum_exp, shape=nv_a.shape(), broadcast_dims=[nv_max_log_sum_exp.ndim - 1]
    )
    log_softmax = fd.ops.sub(input_minus_max, nv_max_log_sum_exp_bcast)
    recomputed_softmax = fd.ops.exp(log_softmax)

    # this should be gradient - softmax * gradient_sum
    softmax_mul_grad_sum = fd.ops.mul(recomputed_softmax, new_target_bcast)
    difference = fd.ops.sub(scattered_vals, softmax_mul_grad_sum)

    return difference


nv_cross_entropy_bwd = ex.register_operator(
    "nv_cross_entropy_bwd",
    meta=cross_entropy_bwd_meta,
)

register_supported(nv_cross_entropy_bwd.id, cross_entropy_bwd, None)


def cross_entropy_transform(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Any:
    result, _, _, _ = nv_cross_entropy_fwd(
        a, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing
    )
    return result


def cross_entropy_grad(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Any:
    fwd, max, log_sum_exp, valid_indices = nv_cross_entropy_fwd(
        a, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing
    )

    grad_out = get_grad(fwd)

    a_grad = nv_cross_entropy_bwd(
        grad_out,
        a,
        target=target,
        a_max=max,
        max_log_sum_exp=log_sum_exp,
        valid_indices=valid_indices,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    put_grads(a, a_grad)

    return fwd


ex.register_supported(
    ltorch.cross_entropy,
    execution_transform=cross_entropy_transform,
    grad_transform=cross_entropy_grad,
    checker=_cross_entropy_check,
)


def _topk_check(
    a: TensorProxy, /, k: int, dim: int | None = None, largest: Number = 1, sorted: Number = 1, *args
) -> bool:
    if a.ndim <= 0:
        return False
    if dim >= a.ndim or (dim is not None and dim < -a.ndim):
        return False
    return True


def topk_transform(
    a: TensorProxy,
    /,
    k: int,
    dim: int | None = None,
    largest: Number = 1,
    sorted: Number = 1,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvk = getnv(k, fd, lc_to_nv_map)
    return fd.ops.topk(nva, nvk, dim, bool(largest), bool(sorted))


register_supported(prims.topk, topk_transform, _topk_check)


def _argsort_check(a: TensorProxy, /, dim: int | None = None, descending: bool = False, stable: bool = False) -> bool:
    return True


def argsort_transform(
    a: TensorProxy,
    /,
    dim: int | None = None,
    descending: bool = False,
    stable: bool = False,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> TensorProxy:
    """Transform argsort operation for NVFuser execution.

    Args:
        a: Input tensor
        dim: Dimension along which to sort
        descending: Sort in descending order if True
        stable: Whether to use a stable sorting algorithm
        fd: Fusion definition
        lc_to_nv_map: Map from Lightning tensors to NVFuser tensors

    Returns:
        TensorProxy: Tensor of indices that would sort the input tensor
    """
    nva = getnv(a, fd, lc_to_nv_map)
    return fd.ops.argsort(nva, dim, bool(descending), bool(stable))


# Register argsort with NVFuser
register_supported(ltorch.argsort, argsort_transform, _argsort_check)


def _grouped_mm_check(
    a: TensorProxy,
    b: TensorProxy,
    offsets: TensorProxy,
) -> bool:
    if not are_supported_tensors(a, b, offsets):
        return False

    return nvfuser_version() >= LooseVersion("0.2.28")


def _grouped_mm_transform(
    a: TensorProxy,
    b: TensorProxy,
    offsets: TensorProxy,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> list[TensorLike]:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)
    nvoffsets = getnv(offsets, fd, lc_to_nv_map) if offsets is not None else None
    return fd.ops.grouped_mm(nva, nvb, nvoffsets)


register_supported(prims._grouped_mm, _grouped_mm_transform, _grouped_mm_check)


def _cumsum_check(a: TensorProxy, dim: int, /, dtype: dtypes.dtype | None = None) -> bool:
    if a.ndim != 1:
        return False

    return is_supported_tensor(a)


# Emulate cumsum using matmul: cumsum(a) = a @ triu(ones)
#
# This is suboptimal. Revisit this after nvFuser has a scan-based cumsum
# implementation.
def cumsum_transform(
    a: TensorProxy,
    dim: int,
    /,
    # For reasons I don't yet understand, `dtype` is a `torch.dtype` in forward but
    # a `dtypes.dtype` in backprop.
    dtype: torch.dtype | dtypes.dtype | None = None,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> TensorProxy:
    if dtypes.is_integer_dtype(a.dtype):
        # torch.matmul can't do integers on GPU so we convert `a` to
        # float.
        compute_dtype = DataType.Float
    else:
        compute_dtype = lcdtype_to_nvdtype(a.dtype)

    if dtype is None:
        out_dtype = lcdtype_to_nvdtype(a.dtype if a.dtype not in dtypes.integer_dtypes else dtypes.int64)
    else:
        out_dtype = lcdtype_to_nvdtype(dtypes.to_dtype(dtype))

    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_a = fd.ops.cast(nv_a, compute_dtype)

    mask = fd.ops.full((a.numel, a.numel), fd.define_scalar(1), compute_dtype)
    mask = fd.ops.triu(mask)

    out = fd.ops.matmul(nv_a, mask)
    out = fd.ops.cast(out, out_dtype)
    return out


register_supported(ltorch.cumsum, cumsum_transform, _cumsum_check)


# At module/class level
NVFUSER_SUPPORTS_OPTIONS = nvfuser_version() >= LooseVersion("0.2.23")
assert NVFUSER_SUPPORTS_OPTIONS, (
    f"Installed version of nvFuser {nvfuser_version()} is not supported, please upgrade to 0.2.23 or later."
)
