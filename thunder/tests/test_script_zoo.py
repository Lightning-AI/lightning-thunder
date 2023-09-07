"""Suite of Python-isms which will challenge preprocessing."""
import pytest
import torch
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten as torch_tree_flatten

import thunder
import thunder.examine

from thunder.core.script.instrumentation import intercept_errors, get_error_ctx
from thunder.core.utils import enable_debug_asserts
from thunder.tests.test_script import skipif_not_python_3_10

enable_debug_asserts()


def mapping_comprehension(func, data):
    _ = {k: func for k in data}
    return func(data)


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(apaz-cli): Comprehensions introduce nonlocal variables.")
def test_comprehension():
    tom = thunder.compile(mapping_comprehension)


def invoke_missing(x):
    if []:  # Empty list is falsy but is not elided by the bytecode parser.
        return foo(x)


def inner_wrapper(x):
    return invoke_missing(x)


def outer_wrapper(x):
    return inner_wrapper(x)


@skipif_not_python_3_10
def test_nested_inline(capfd):
    with intercept_errors() as errors, pytest.raises(ValueError):
        thunder.compile(outer_wrapper)

    assert "inner_wrapper" in (msg := "\n".join(errors)), msg
    assert not get_error_ctx()

    thunder.examine.examine(outer_wrapper, torch.ones((1,)))
    assert "inner_wrapper" in (msg := capfd.readouterr().out), msg
    assert not get_error_ctx()


class StoreTorchOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.relu

    def forward(self, x):
        return self.relu(x)


@pytest.mark.xfail(reason="TODO(robieta, t-vi): fails with `could not eliminate self argument`")
def test_store_fn():
    model = StoreTorchOpModule()
    tom = thunder.compile(model)


class ConditionalAccessModule(torch.nn.Module):
    def __init__(self, has_relu):
        super().__init__()
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.has_relu:
            x = self.relu(x)
        return x


@skipif_not_python_3_10
@pytest.mark.parametrize(
    "branch",
    (
        pytest.param(False, marks=pytest.mark.xfail(reason="`self.relu` is unconditionally accessed.")),
        True,
    ),
)
def test_conditional_access(branch):
    model = ConditionalAccessModule(has_relu=branch)
    tom = thunder.compile(model)


def call_to_flatten(x):
    flat_x, _ = torch_tree_flatten([[[[x]]]])
    return flat_x


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(robieta, t-vi): Handle recursion during inlining.")
def test_tree_flatten():
    thunder.compile(call_to_flatten)


class SubclassLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x)


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(robieta, t-vi): Subclasses fail during `signature.bind`")
def test_subclass():
    model = SubclassLinear(2, 2)
    thunder.compile(model)


class LazyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = None

    def forward(self, x):
        if self.relu is None:
            self.relu = torch.nn.ReLU()

        return self.relu(x)


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(robieta, t-vi): Could not eliminate self")
def test_lazy_init():
    thunder.compile(LazyLayer())


class LazyGetattrLayer(torch.nn.Module):
    def forward(self, x):
        if getattr(self, "relu", None) is None:
            self.relu = torch.nn.ReLU()

        return self.relu(x)


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(robieta, t-vi): `relu` is accessed too early.")
def test_lazy_getattr_init():
    thunder.compile(LazyGetattrLayer())


class CountingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x):
        self.count += 1
        return x * self.count / (self.count + 1)


@skipif_not_python_3_10
@pytest.mark.xfail(reason="TODO(robieta, t-vi): Could not eliminate self")
def test_self_mutation():
    tom = thunder.compile(CountingModule())
    x = torch.ones((1,))
    assert_close(CountingModule()(x), tom(x))


def loop_add(a, b):
    for _ in range(100):
        a = a + b

    return a


def test_loop_add():
    thunder.compile(loop_add)


def transitive_value_alias(z):
    x = None
    y = None
    if z:
        y = 2
    elif True:
        pass
    print(x, y)


def test_transitive_value_alias(capfd):
    # https://github.com/Lightning-AI/lightning-thunder/issues/855
    tom = thunder.compile(transitive_value_alias)
    _ = capfd.readouterr()
    tom(True)
    assert "None 2" in capfd.readouterr().out
    tom(False)
    assert "None None" in capfd.readouterr().out


def apply_context_manager(ctx):
    with ctx:
        pass


def test_apply_context_manager():
    thunder.compile(apply_context_manager)
