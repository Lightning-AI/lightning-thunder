from __future__ import annotations
from typing import TYPE_CHECKING
import inspect

from torch.fx import GraphModule
from torch._logging._internal import trace_structured
from torch._logging._internal import trace_structured_artifact

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
    from torch._guards import CompileId


_SUPPORT_COMPILE_ID_KWARG: bool = "compile_id" in inspect.signature(trace_structured).parameters


def payload_fn_of(fn: GraphModule | Callable[[Any], Any]) -> Callable[[], str]:
    if isinstance(fn, GraphModule):

        def f() -> str:
            return fn.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
            )

        return f

    def f() -> str:
        return f"{fn}\n"

    return f


# TODO: use `trace_structured_artifact` once `compile_id` is merged.
#   https://github.com/pytorch/pytorch/pull/160440.
# note: `compile_id` is a kwarg since v2.7.0.
def _log_to_torch_trace(
    name: str,
    fn: GraphModule | Callable[[Any], Any],
    compile_id: CompileId | None = None,
) -> None:
    payload_fn = payload_fn_of(fn)
    if compile_id is not None and _SUPPORT_COMPILE_ID_KWARG:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": name,
                "encoding": "string",
            },
            payload_fn=payload_fn,
        )
    else:
        trace_structured_artifact(
            name=name,
            encoding="string",
            payload_fn=payload_fn,
        )
