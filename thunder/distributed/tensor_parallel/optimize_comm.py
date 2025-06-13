from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core import utils
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import variableify
from thunder.core.trace import from_trace
from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if TYPE_CHECKING:
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol
    from thunder.core.proxies import ProxyInterface
    from thunder.core.trace import VariableInterface


__all__ = [
    "remove_redundant_comms",
]


def _get_tensor_parallel_layer_type(bsym: BoundSymbol) -> TensorParallelLayerType:
    value = bsym.flat_args[2]
    utils.check_type(value, TensorParallelLayerType)
    return value


def remove_redundant_comms(trace: TraceCtx) -> TraceCtx:
    """Remove redundant paris of column-wise linear postprocessing and row-wise linear preprocessing.

    Args:
        trace: A trace modified by both of :func:`~thunder.distributed.tensor_parallel.column_parallel`
            and :func:`~thunder.distributed.tensor_parallel.row_parallel`.
    """
    from thunder.distributed import prims as dist_prims

    current_column_wise_parallel_linear_bsym: BoundSymbol | None = None

    interesting_pairs: list[tuple[BoundSymbol, BoundSymbol]] = []
    bsym_to_idx: dict[BoundSymbol, int] = {}
    idx_to_bsym: dict[int, BoundSymbol] = {}
    new_bsyms: list[BoundSymbol] = []

    bsym: BoundSymbol
    for idx, bsym in enumerate(trace.bound_symbols):
        bsym_to_idx[bsym] = idx
        idx_to_bsym[idx] = bsym
        new_bsyms.append(bsym)
        match bsym.sym.id:
            case dist_prims.PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT:
                match _get_tensor_parallel_layer_type(bsym):
                    case TensorParallelLayerType.ROW_PARALLEL_LINEAR:
                        if current_column_wise_parallel_linear_bsym is not None:
                            interesting_pairs.append((current_column_wise_parallel_linear_bsym, bsym))
                    case _:
                        if current_column_wise_parallel_linear_bsym is not None:
                            current_column_wise_parallel_linear_bsym = None
            case dist_prims.PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT:
                match _get_tensor_parallel_layer_type(bsym):
                    case TensorParallelLayerType.COLUMN_PARALLEL_LINEAR:
                        current_column_wise_parallel_linear_bsym = bsym
            case _:
                pass

    new_trace = from_trace(trace)
    new_trace.bound_symbols = new_bsyms

    if interesting_pairs:
        indices_to_filter = []
        # For column-parallel linear: postprocessed -> col-parallel linear output
        # For row-parallel linear: preprocessed -> col-parallel linear output
        swap_map: dict[VariableInterface, ProxyInterface] = {}
        for col_postprocess_bsym, row_preprocess_bsym in interesting_pairs:
            col_liinear_output: TensorProxy = col_postprocess_bsym.flat_proxy_args[0]
            utils.check_type(col_liinear_output, TensorProxy)

            row_linear_input: TensorProxy = row_preprocess_bsym.flat_proxy_outs[0]
            utils.check_type(row_linear_input, TensorProxy)

            # TODO(crcrpar): Better to make sure that between column-wise parallel linear row-wise parallel linear,
            # the existing bsyms are elementwise.
            if col_liinear_output.shape == row_linear_input.shape:
                indices_to_filter.extend([bsym_to_idx[col_postprocess_bsym], bsym_to_idx[row_preprocess_bsym]])

                swap_map[variableify(col_postprocess_bsym.flat_proxy_outs[0])] = col_liinear_output

                orig_row_linear_input: TensorProxy = row_preprocess_bsym.flat_proxy_args[0]
                swap_map[variableify(row_linear_input)] = orig_row_linear_input

        indices_to_filter = set(indices_to_filter)
        new_bsyms: list[BoundSymbol] = []
        for idx, bsym in enumerate(trace.bound_symbols):
            if idx in indices_to_filter:
                continue
            new_bsyms.append(bsym.from_bsym_swap_proxies(swap_map=swap_map, skip_output=True))
        new_trace.bound_symbols = new_bsyms

    return new_trace
