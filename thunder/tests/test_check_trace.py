import thunder
from thunder.dev_utils.check_trace import check_trace
import torch
import pytest


def test_missing_symbol():
    def fn(a, b):
        c = a + b
        d = 2 * c
        return d

    jfn = thunder.jit(fn)

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    jfn(a, b)

    tr = thunder.last_traces(jfn)[-1]
    check_trace(tr)

    del tr.bound_symbols[-3]

    with pytest.raises(AssertionError, match="unknown proxy"):
        check_trace(tr)


def test_debug_option():
    class BrokenTransform(thunder.core.transform_common.Transform):
        def transform_traces_pre_prologue(self, pro, comp, epi, **kwargs):
            new_comp = thunder.core.trace.from_trace(comp)
            new_comp.bound_symbols = comp.bound_symbols[:]
            del new_comp.bound_symbols[2]
            return pro, new_comp, epi

    def fn(a, b):
        c = a + b
        d = 2 * c
        return d

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    # works
    jfn = thunder.jit(fn, debug_options=thunder.DebugOptions(check_traces=True))
    jfn(a, b)

    # broken with nice error
    jfn = thunder.jit(fn, transforms=(BrokenTransform(),), debug_options=thunder.DebugOptions(check_traces=True))

    with pytest.raises(AssertionError, match="unknown proxy"):
        jfn(a, b)

    # broken with less nice error
    jfn = thunder.jit(fn, transforms=(BrokenTransform(),), executors=())
    with pytest.raises(UnboundLocalError, match="cannot access local|referenced before assignment"):
        jfn(a, b)
