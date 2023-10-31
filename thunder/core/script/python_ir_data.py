import functools
import sys
from types import CodeType
from typing import Callable, Union
from collections.abc import Iterable

from thunder.core.script import parse


SUPPORTS_PREPROCESSING = (3, 9) <= sys.version_info < (3, 11)
X_THUNDER_STORE_ATTR = "X_THUNDER_STORE_ATTR"


# TODO(robieta): replace callsites.
get_instruction = functools.partial(parse.ThunderInstruction.make, line_no=-1)


def debug_compare_functions_print(diffs: dict[str, tuple[list, list]]):
    for k, (v1, v2) in diffs.items():
        if not (v1 is None and v2 is None):
            print(f"Differences in: {k}")
            print(f"  CodeObject 1: {v1}")
            print(f"  CodeObject 2: {v2}")


def debug_compare_functions(
    code1: Union[CodeType, Callable], code2: Union[CodeType, Callable], *, show=False
) -> dict[str, tuple[list, list]]:
    if not isinstance(code1, CodeType):
        code1 = code1.__code__
    if not isinstance(code2, CodeType):
        code2 = code2.__code__

    attrs = [
        "co_argcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_freevars",
        "co_cellvars",
    ]

    diffs = {}
    for attr in attrs:
        v1 = getattr(code1, attr)
        v2 = getattr(code2, attr)

        if v1 != v2:
            if isinstance(v1, dict) and isinstance(v2, dict):
                diffs[attr] = (v1 - v2, v2 - v1)
            if isinstance(v1, str) and isinstance(v2, str):
                diffs[attr] = (v1, v2)
            elif isinstance(v1, Iterable) and isinstance(v2, Iterable):
                diffs[attr] = (set(v1) - set(v2), set(v2) - set(v1))
            else:
                diffs[attr] = (v1, v2)

    if show:
        debug_compare_functions_print(diffs)

    return diffs
