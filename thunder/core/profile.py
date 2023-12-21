import contextlib
import os

import torch


_ENABLED = os.getenv("THUNDER_ANNOTATE_TRACES") in ("1", "y", "Y")


def profiling_enabled() -> bool:
    return _ENABLED


@contextlib.contextmanager
def add_markers(msg: str) -> None:
    if not profiling_enabled():
        yield
        return

    assert "\n" not in msg, msg  # Both NVTX and JSON forbid newlines
    assert '"' not in msg, msg  # The PyTorch profiler does not properly escape quotations

    with torch.profiler.record_function(msg):
        torch.cuda.nvtx.range_push(msg)
        try:
            yield

        finally:
            torch.cuda.nvtx.range_pop()
