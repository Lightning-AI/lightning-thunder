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
    disable_torch_autograd: bool = False,  # TODO Revisit this UX for RC1
    **compile_options,  # TODO RC1 Make this explicit -- dict of options
) -> Callable:
    """Just-in-time compile a function.

    Args:
        fn: A function to compile.
    Keyword Args:
        langctx: the language context, which language / library to emulate. default: "torch" for PyTorch compatibility.
        executors: list of executors to use. Defaults to the executors returned by `thunder.get_default_executors()` and always amened by `thunder.get_always_executors()`.
                   You can get a list of all available executors with `thunder.get_all_executors()`.
        sharp_edges: sharp edge detection action. What to do when thunder detects a construct that is likely to lead to errors. Can be ``"allow"``, ``"warn"``, ``"error"``. Defaults to ``"allow"``.
        cache: caching mode. default: ``"constant values"```

               - ``"no caching"`` - disable caching and always recompute,
               - ``"constant values"`` - require Tensors to be of the same shape, device, dtype etc., and integers and strings to match exactly,
               - ``"same input"`` - don't check, but just assume that a cached function works if it exists.
        interpretation: default: ``"translate functions"``

               - ``"python interpreter"`` run in the cpython interpreter, you need to program thunder explicitly,
               - ``"translate functions"`` use the thunder interpreter to translate torch functions to thunder and (optionally) detect sharp edges
    """
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
