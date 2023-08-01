"""Suite of Python-isms which will challenge preprocessing."""
import pytest
import torch

import thunder
import thunder.examine

from thunder.core.script.instrumentation import intercept_errors, get_error_ctx
from thunder.core.utils import enable_debug_asserts
from thunder.tests.test_script import skipif_not_python_3_10

enable_debug_asserts()


def loop_relu(x):
    for _ in range(5):
        x = torch.add(x, 1)
    return x


def first_wrapper(x):
    return loop_relu(x)


def second_wrapper(x):
    return first_wrapper(x)


@skipif_not_python_3_10
def test_nested_inline(capfd):
    with intercept_errors() as errors, pytest.raises(RecursionError):
        thunder.compile(second_wrapper)

    assert "first_wrapper" in (msg := "\n".join(errors)), msg
    assert not get_error_ctx()

    thunder.examine.examine(second_wrapper, torch.ones((1,)))
    assert "first_wrapper" in (msg := capfd.readouterr().out), msg
    assert not get_error_ctx()
