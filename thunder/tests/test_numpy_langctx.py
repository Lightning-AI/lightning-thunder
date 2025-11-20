from thunder.numpy import size as np_size
from thunder.core.langctxs import langctx, Languages, resolve_language
from thunder.core.proxies import TensorProxy
from thunder.core.trace import detached_trace
from thunder.core.devices import cpu
from thunder.core.dtypes import float32


def test_numpy_langctx_registration_and_len_size():
    with detached_trace():
        t = TensorProxy(shape=(2, 3), device=cpu, dtype=float32)

    with langctx(Languages.NUMPY):
        assert len(t) == 2  # axis 0 length
        assert t.size() == 6  # total elements
        assert np_size(t) == 6


def test_numpy_langctx_resolve_language():
    numpy_ctx_by_enum = resolve_language(Languages.NUMPY)
    numpy_ctx_by_name = resolve_language("numpy")

    assert numpy_ctx_by_enum is numpy_ctx_by_name
    assert numpy_ctx_by_enum.name == "numpy"
