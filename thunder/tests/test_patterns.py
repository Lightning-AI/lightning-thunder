from numbers import Number

import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
from thunder.core.patterns import Pattern, bind_names, numbered_ancestors
from thunder.core.proxies import TensorProxy
from thunder.core.symbol import BoundSymbol


def test_simple_matching():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    # One match
    def foo(a, b):
        return a + b

    trc = thunder.trace()(foo, a, b)

    def add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            return True, {}

        return False, None

    p = Pattern()
    p.match(add_matcher)
    matches = p(trc)

    assert len(matches) == 1

    # No matches
    def foo(a, b):
        return a - b

    trc = thunder.trace()(foo, a, b)

    matches = p(trc)
    assert len(matches) == 0

    # Three matches
    def foo(a, b):
        c = a + b
        d = c + b
        e = d + c
        return e

    trc = thunder.trace()(foo, a, b)

    matches = p(trc)
    assert len(matches) == 3


def test_multiop_matching():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        c = a + b
        d = a - 5
        e = a - b
        return c, d, e

    trc = thunder.trace()(foo, a, b)

    def add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            bn = bind_names(bsym)
            return True, {"a": bn.a, "b": bn.b}

        return False, None

    def sub_matcher(bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict) -> tuple[bool, dict]:
        if bsym.sym.name == "sub":
            bn = bind_names(bsym)
            a = match_ctx["a"]
            b = match_ctx["b"]
            if a is bn.a and b is bn.b:
                return True, {}

        return False, None

    # Matches the addition and the second subtraction
    p = Pattern()
    p.match(add_matcher)
    p.match(sub_matcher)
    matches = p(trc)

    assert len(matches) == 1

    match = matches[0]
    (idx0, bsym0), (idx1, bsym1) = match
    assert idx0 == 2
    assert idx1 == 4


def test_match_multiple():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        c = a + b
        d = c * 5
        e = d - 2
        f = e**2
        return f

    trc = thunder.trace()(foo, a, b)

    def elemwise_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        name: str = bsym.sym.name

        if name in {"add", "sub", "mul", "pow"}:
            return True, {}

        return False, None

    # Matches the add, mul, and sub, but not the final pow (because min_times=2)
    p = Pattern()
    p.match(elemwise_matcher, min_times=2, max_times=3)
    matches = p(trc)

    assert len(matches) == 1
    match = matches[0]
    assert len(match) == 3

    # Matches two patterns -- the first pattern is add mul sub, the second pattern is just pow
    p = Pattern()
    p.match(elemwise_matcher, min_times=1, max_times=3)
    matches = p(trc)

    assert len(matches) == 2
    assert len(matches[0]) == 3
    assert len(matches[1]) == 1

    def foo(a, b):
        x = a / b
        c = a + b
        d = c * 5
        e = d - 2
        f = e**2
        g = f + 2
        return x, g

    trc = thunder.trace()(foo, a, b)

    def div_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "true_divide":
            return True, {}

        return False, None

    def add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            return True, {}

        return False, None

    # Fails to match because the second addition cannot be reordered to be adjacent to the rest of the pattern
    p = Pattern()
    p.match(div_matcher)
    p.match(elemwise_matcher, min_times=1, max_times=2)
    p.match(add_matcher)

    matches = p(trc)
    assert len(matches) == 0

    def foo(a, b):
        x = a / b
        c = a + b
        d = c * 5
        e = d - 2
        f = e**2
        g = f + 2
        y = d + 2
        return x, g, y

    trc = thunder.trace()(foo, a, b)

    # Succeeds because the second addition can be reordered adjacent to the rest of the pattern
    matches = p(trc)
    assert len(matches) == 1
    assert len(matches[0]) == 4

    # Fails to match because the elemwise_matcher greedily consumes the addition, preventing the final match
    p = Pattern()
    p.match(div_matcher)
    p.match(elemwise_matcher, min_times=1, max_times=-1)
    p.match(add_matcher)
    matches = p(trc)
    assert len(matches) == 0

    # Successfully matches all operations when the final match is not included
    p = Pattern()
    p.match(div_matcher)
    p.match(elemwise_matcher, min_times=1, max_times=-1)
    matches = p(trc)
    assert len(matches) == 1
    assert len(matches[0]) == 7


def test_dataflow():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    # e cannot be reordered after c
    def foo(a, b):
        c = a + b
        d = c + 1
        e = d + a
        return e

    trc = thunder.trace()(foo, a, b)

    # Matches add operations with two tensor inputs
    def add_tensor_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            bn = bind_names(bsym)
            return (isinstance(bn.a, TensorProxy) and isinstance(bn.b, TensorProxy)), {}

        return False, None

    p = Pattern()
    p.match(add_tensor_matcher)
    p.match(add_tensor_matcher)
    matches = p(trc)
    assert len(matches) == 0

    # e can be reordered after c
    def foo(a, b):
        c = a + b
        d = c + 1
        e = c + a
        return d, e

    trc = thunder.trace()(foo, a, b)

    matches = p(trc)
    assert len(matches) == 1


def test_windowing():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        c = a + b
        d = c + 1
        e = d + 2
        f = e + 9
        g = f + 1
        h = g + 2
        x = h - 2
        i = c - 2
        j = e + 2
        return x, i, j

    trc = thunder.trace()(foo, a, b)

    # Matches add operations with two tensor inputs
    def add_tensor_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            bn = bind_names(bsym)
            return (isinstance(bn.a, TensorProxy) and isinstance(bn.b, TensorProxy)), {}

        return False, None

    def add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            return True, {}

        return False, None

    def sub_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "sub":
            return True, {}

        return False, None

    # The first add and the subtraction that can be reordered next to it (that produces i) are too far apart to match
    p = Pattern()
    p.match(add_tensor_matcher)
    p.match(sub_matcher)
    matches = p(trc)
    assert len(matches) == 0

    # Matching any add and the sub works, in fact two pairs are created, the first are the operations that
    #   create d and i, and the second are the operations that create h and x (earlier additions can't match
    #   with the sub that produces x, because it depends on h, which depends on all earlier additions)
    p = Pattern()
    p.match(add_matcher)
    p.match(sub_matcher)
    matches = p(trc)
    assert len(matches) == 2

    (idx0, bsym0), (idx1, bsym1) = matches[0]

    assert idx0 == 3
    assert idx1 == 9

    (idx0, bsym0), (idx1, bsym1) = matches[1]

    assert idx0 == 7
    assert idx1 == 8


def test_previously_matched():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        c = a + b
        d = c - a

        e = d + b
        return e

    trc = thunder.trace()(foo, a, b)

    def add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            return True, {}

        return False, None

    def assert_sub_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        assert len(previously_matched) == 1
        pbsym: BoundSymbol = previously_matched[0]
        assert pbsym.sym.name == "add"

        if bsym.sym.name == "sub":
            return True, {}

        return False, None

    p = Pattern()
    p.match(add_matcher)
    p.match(assert_sub_matcher)

    matches = p(trc)
    assert len(matches) == 1

    def assert_add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        assert len(previously_matched) == 2
        add_bsym, sub_bsym = previously_matched
        assert add_bsym.sym.name == "add"
        assert sub_bsym.sym.name == "sub"

        if bsym.sym.name == "add":
            return True, {}

        return False, None

    p = Pattern()
    p.match(add_matcher)
    p.match(assert_sub_matcher)
    p.match(assert_add_matcher)

    matches = p(trc)
    assert len(matches) == 1


def test_context():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        c = a + b
        d = c - a

        # This path leads to a pow, which cannot be matched, requiring the pattern "unroll" back
        #   to the subtraction
        e = d + 5
        f = e**2

        # This path will be matched
        g = d + 2
        h = g - 3

        return f, h

    trc = thunder.trace()(foo, a, b)

    def first_add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            bn = bind_names(bsym)
            return True, {"a": bn.a, "b": bn.b, "first_add_result": bsym.output}

        return False, None

    # Matches a sub of the output of the first addition minus the first argument of the first addition
    def sub_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "sub":
            bn = bind_names(bsym)
            a = match_ctx["a"]
            b = match_ctx["b"]
            add_out = match_ctx["first_add_result"]
            if bn.a is add_out and bn.b is a:
                return True, {}

        return False, None

    def second_add_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "add":
            bn = bind_names(bsym)
            return True, {"second_a": bn.a, "second_b": bn.b, "second_add_result": bsym.output}

        return False, None

    # Matches a subtraction whose first argument is the result of the second addition
    def second_sub_matcher(
        bsym: BoundSymbol, *, previously_matched: list[BoundSymbol], match_ctx: dict
    ) -> tuple[bool, None | dict]:
        if bsym.sym.name == "sub":
            bn = bind_names(bsym)
            second_add_result = match_ctx["second_add_result"]
            if bn.a is second_add_result and isinstance(bn.b, Number):
                return True, {}

        return False, None

    p = Pattern()
    p.match(first_add_matcher)
    p.match(sub_matcher)
    p.match(second_add_matcher)
    p.match(second_sub_matcher)

    matches = p(trc)
    assert len(matches) == 1
    match = matches[0]
    assert len(match) == 4
