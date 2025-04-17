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
