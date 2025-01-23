from collections.abc import Callable
import contextlib
import functools
import os
import warnings

import torch


_ENABLED = os.getenv("THUNDER_ANNOTATE_TRACES") in ("1", "y", "Y")

# However, nvtx is incredibly cheap so we no longer bother requiring the
# environment variable.
try:
    import nvtx

    _ENABLED = True
except ImportError:
    if _ENABLED:
        msg = "Requested nvtx but the package is not available."
        msg += "\nUse `pip install -m pip install nvtx`."
        warnings.warn(msg)
        _ENABLED = False
        raise


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


# The main interface to profiling something. Generally used as a decorator:
#   @thunder.core.profile.annotate_for_profile("foo")
#   def foo(...): ...
# but alternatively as a `with` context:
#   with thunder.core.profile.annotate_for_profile("name for a block of code"):
#      # ... code ...
annotate_for_profile: Callable[[str], None] = None


if _ENABLED:
    annotate_for_profile = functools.partial(nvtx.annotate, domain="thunder")
else:

    class _no_annotate(contextlib.nullcontext):
        """
        A profiling decorator that does nothing.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, fqn):
            return fqn

    annotate_for_profile = _no_annotate
