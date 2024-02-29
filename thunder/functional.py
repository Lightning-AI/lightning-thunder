import thunder
from collections.abc import Callable, Sequence
from typing import Any
from thunder.core.langctxs import LanguageContext
from thunder.extend import Executor
from thunder.core.options import (
    INTERPRETATION_OPTIONS,
    CACHE_OPTIONS,
    SHARP_EDGES_OPTIONS,
)


# note: keep this roughly in sync with thunder.jit
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any | LanguageContext = None,
    executors: None | Sequence[Executor] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    interpretation: None | INTERPRETATION_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    disable_torch_autograd: bool = False,  # TODO Revisit this UX for gtc
    **compile_options,  # TODO GTC Make this explicit -- dict of options
) -> Callable:
    if interpretation is None:
        interpretation = INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS
    return thunder.jit(
        fn,
        langctx=langctx,
        executors=executors,
        sharp_edges=sharp_edges,
        interpretation=interpretation,
        cache=cache,
        disable_torch_autograd=disable_torch_autograd,
        **compile_options,
    )
