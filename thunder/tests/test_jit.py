from functools import partial, wraps
from itertools import product

import sys
import dis

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.jit import is_jitting, jit, JITError
from thunder.core.jit_ext import phantom_jit, litjit


def xfailif_python_3_11_plus(f):
    if sys.version_info >= (3, 11):
        return pytest.mark.xfail(f, reason=f"not yet implemented for Python 3.11+, got {sys.version_info=}")
    return f


def test_no_return():
    def foo():
        pass

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_return():
    def foo():
        return 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_addition():
    def foo():
        return 3 + 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_input_number_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

    args = (5, 2)

    assert jfoo(*args) == foo(*args)


def test_input_tensor_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_constant_if():
    def foo(a, b):
        if 3 < 5:
            return a + b
        else:
            assert False

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_if():
    def foo(a, b):
        if a < b:
            return a
        elif b > a:
            return b
        else:
            return 0

    jfoo = jit(foo)

    cases = (
        (5, 3),
        (9, 12),
        (2, 2),
    )

    for case in cases:
        assert jfoo(*case) == foo(*case)


def test_while():
    # produces POP_JUMP_BACKWARD_IF_TRUE/FALSE in 3.11
    def foo(l):
        i = 0
        res = []
        v = l[0]
        while v:
            res.append(i)
            i = i + 1
            v = l[i]
        return res

    def bar(l):
        i = 0
        res = []
        v = l[0]
        while not v:
            res.append(i)
            i = i + 1
            v = l[i]
        return res

    def baz(l):
        i = 0
        res = []
        v = l[0]
        while v is not None:
            res.append(i)
            i = i + 1
            v = l[i]
        return res

    def tom(l):
        i = 0
        res = []
        v = l[0]
        while v is None:
            res.append(i)
            i = i + 1
            v = l[i]
        return res

    l = [True, True, False, True]

    assert foo(l) == jit(foo)(l)

    l = [False, False, True]
    assert bar(l) == jit(bar)(l)

    l = [False, False, None]
    assert baz(l) == jit(baz)(l)

    l = [None, None, False]
    assert tom(l) == jit(tom)(l)


def test_and_or():
    # JUMP_IF_TRUE/FALSE_OR_POP
    def foo(a, b):
        return a and b

    def bar(a, b):
        return a or b

    jfoo = jit(foo)
    jbar = jit(bar)

    cases = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (object(), True),
        (object(), False),
    )
    for case in cases:
        assert jfoo(*case) == foo(*case)
        assert jbar(*case) == bar(*case)


def test_dunder_bool():
    class mycls:
        def __init__(self, value):
            self.value = value

        # True if self.value is even
        def __bool__(self):
            return (self.value % 2) == 0

    def foo(a):
        if a:
            return 1
        return -1

    jfoo = jit(foo)

    cases = (
        (mycls(4),),
        (mycls(5),),
    )

    for case in cases:
        assert jfoo(*case) == foo(*case)


def test_dunder_bool_instance():
    class X:
        def __bool__(self):
            return False

    x = X()
    assert bool(x) == jit(bool)(x) == False

    x.__bool__ = lambda: True  # dunder methods use class attribute, not instance attribute.
    assert bool(x) == jit(bool)(x) == False


def test_function_call():
    def fn(fn):
        return fn

    assert fn(fn) == jit(fn)(fn)
    assert fn(fn=fn) == jit(fn)(fn=fn)


def test_function_call():
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a + 1, b)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_call_function_ex():
    def foo(a, b):
        return a + b

    def argsplat(*args):
        return foo(*args)

    def kwargsplat(**kwargs):
        return foo(**kwargs)

    assert any(i.opname == "CALL_FUNCTION_EX" and not i.arg & 1 for i in dis.get_instructions(argsplat))
    assert any(i.opname == "CALL_FUNCTION_EX" and i.arg & 1 for i in dis.get_instructions(kwargsplat))

    kwargs = {"a": 1, "b": 2}

    res1 = argsplat(*kwargs.values())
    res2 = kwargsplat(**kwargs)
    jres1 = jit(argsplat)(*kwargs.values())
    jres2 = jit(kwargsplat)(**kwargs)

    assert_close(res1, jres1)
    assert_close(res2, jres2)


def test_build_const_key_map():
    def fn1(a, b):
        return {"a": a, "b": b}

    # test order for collisions
    def fn2(a, b):
        return {"a": a, "a": b}

    jfn1 = jit(fn1)
    jfn2 = jit(fn2)

    assert jfn1(1, 2) == fn1(1, 2)
    assert jfn2(1, 2) == fn2(1, 2)


def test_build_map_dict_merge():
    addall = lambda *args, **kwargs: sum(args) + sum(kwargs.values())
    foo = lambda *args, **kwargs: addall(*args, **kwargs)

    assert any(i.opname == "BUILD_MAP" for i in dis.get_instructions(foo))
    assert any(i.opname == "DICT_MERGE" for i in dis.get_instructions(foo))

    jfoo = jit(foo)

    args = (4, 3)
    kwargs = {"a": 1, "b": 2}

    thunder_result = jfoo(*args, **kwargs)
    python_result = foo(*args, **kwargs)

    with pytest.raises(KeyError, match="got multiple values for keyword argument") as excinfo:
        d = {"a": 3, "b": 4}
        mergefail = lambda **kwargs: addall(**kwargs, **d)
        jfail = jit(mergefail)
        jfail(**kwargs)

    assert_close(thunder_result, python_result)


def test_dict_update():
    addall = lambda *args, **kwargs: sum(args) + sum(kwargs.values())
    foo = lambda *args, **kwargs: addall(*args, **{**kwargs, "x": 1})

    assert any(i.opname == "DICT_UPDATE" for i in dis.get_instructions(foo))

    args = (4, 3)
    kwargs = {"a": 1, "b": 2}

    thunder_result = jit(foo)(*args, **kwargs)
    python_result = foo(*args, **kwargs)

    assert_close(thunder_result, python_result)


def test_inner_function_definition():
    def foo(a, b):
        def bar(a, b):
            return a + b

        return bar(a + 1, b)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)

    def foo(a, b):
        def bar(a, b=b):
            return a + b

        return bar(a + 1)

    assert_close(foo(*args), jit(foo)(*args))

    def foo(a, b):
        def bar(a: int, *, b: int = b):
            return a + b

        return bar(a + 1)

    assert_close(foo(*args), jit(foo)(*args))


def test_inner_closure():
    # NOTE The addition of closing over value also tests
    #   the STORE_DEREF opcode
    def foo(a, b):
        value = 5

        def bar(a):
            return a + b + value

        return bar(a + 1)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_delete_deref():
    def foo(a, b):
        value = 5

        def bar(a):
            nonlocal value
            del value
            return a + b + value

        return bar(a + 1)

    jfoo = jit(foo)

    args = (4, 3)

    with pytest.raises(NameError, match="'value'"):
        python_result = foo(*args)
    with pytest.raises(NameError, match="'value'"):
        thunder_result = jfoo(*args)


def test_locals_globals():
    def fn():
        funny_name_nowhere_else = True
        return locals() | globals()

    assert "test_locals_globals" in jit(fn)()
    assert "funny_name_nowhere_else" in jit(fn)()


def test_unpack_sequence():
    def foo(tup):
        a, b = tup
        return a + b

    def bar(tup):
        a, b = map(lambda x: x, tup)  # unpack iterable
        return a + b

    jfoo = jit(foo)
    jbar = jit(bar)

    args = (4, 3)

    thunder_result = jfoo(args)
    python_result = foo(args)

    assert_close(thunder_result, python_result)

    thunder_result = jbar(args)
    python_result = bar(args)

    assert_close(thunder_result, python_result)


def test_exception_traceback():
    def bar(a):
        raise ValueError(f"I don't like {a}")

    def foo(b):
        return bar(b + 1)

    jfoo = jit(foo)

    args = (4,)

    with pytest.raises(ValueError) as excinfo:
        thunder_result = jfoo(*args)

    assert "foo in file" in str(excinfo.value.__cause__)
    assert "bar in file" in str(excinfo.value.__cause__)


def test_finally():
    l = []

    def foo():
        try:
            l.append(1)
            raise ValueError("test")
            l.append(2)
        except KeyError:
            l.append(3)
        except ValueError:
            l.append(4)
            raise
        finally:
            l.append(5)

    with pytest.raises(ValueError):
        foo()

    l_orig = l

    l = []

    with pytest.raises(ValueError):
        jit(foo)()

    assert l_orig == l


def test_raise():
    msg = "lorem ipsum"

    def foo():
        raise ValueError(msg)

    jfoo = jit(foo)

    with pytest.raises(ValueError) as excinfo:
        jfoo()

    assert msg in str(excinfo.value)


def test_bare_except():
    msg = "lorem ipsum"

    def bare_except():
        try:
            raise ValueError(msg)
        except:
            return True

    assert jit(bare_except)() == True


def test_trivial_try_finally():
    def trivial_try_finally():
        try:
            pass
        finally:
            return True

    assert jit(trivial_try_finally)() == True


def test_try_finally():
    def try_finally():
        try:
            var = False
            raise ValueError
        except ValueError:
            var = True
        finally:
            return var

    assert jit(try_finally)() == True


def test_match_exception():
    def match_exception():
        error_set = (ValueError, IndexError)
        try:
            raise ValueError
        except error_set:
            return True

    assert jit(match_exception)() == True


def test_match_as():
    msg = "lorem ipsum"

    def match_as():
        try:
            raise ValueError(msg)
        except ValueError as e:
            return str(e)

    assert msg in jit(match_as)()


def test_list():
    def foo():
        l = [1, 2, 3]
        l[3:] = l[:2]
        l[0] = l[-1]
        del l[2]
        return l

    assert foo() == jit(foo)()


def test_raise_external():
    msg = "lorem ipsum"

    def raise_external():
        raise ValueError(msg)

    with pytest.raises(ValueError) as excinfo:
        jit(raise_external)()

    assert msg in str(excinfo.value)


def test_raise_from():
    msg = "lorem ipsum"

    def raise_from():
        try:
            raise ValueError(msg) from IndexError(msg)
        except ValueError as e:
            return (str(e), str(e.__cause__))

    res = jit(raise_from)()
    assert msg in res[0] and msg in res[1]


def test_raise_from_external():
    msg = "lorem ipsum"

    def raise_from_external():
        raise ValueError(msg) from IndexError(msg)

    with pytest.raises(ValueError) as excinfo:
        jit(raise_from_external)()

    e = excinfo.value
    assert type(e) == ValueError
    # TODO: If we drop the UserException here, update
    # assert type(e.__cause__) == IndexError and msg in str(e.__cause__), excinfo.value
    assert type(e.__cause__.__cause__) == IndexError and msg in str(e.__cause__.__cause__), excinfo.value


def test_nested_try_except():
    def nested_try_except():
        try:
            raise ValueError
        except ValueError as e1:
            try:
                raise IndexError from e1
            except IndexError:
                pass
            return True

    assert jit(nested_try_except)() == True


def test_inner_nested_try_except():
    def inner_nested_try_except():
        try:
            try:
                raise ValueError
            except ValueError:
                pass
        except Exception:
            return False
        return True

    assert jit(inner_nested_try_except)() == True


def test_cross_function_exceptions():
    def foo():
        def bar():
            raise ValueError

        bar()

    def cross_function_exceptions():
        try:
            foo()
        except ValueError:
            return True

    assert jit(cross_function_exceptions)() == True


def test_walrus_operator():
    def foo(a, b):
        c = (a := b)
        return c

    jfoo = jit(foo)

    assert jfoo(3, 8) == foo(3, 8)


def test_build_map():
    def foo(a, b):
        return {0: a, 1: b, 2: 3, "a": 4, a: 5}

    jfoo = jit(foo)

    # a, b
    cases = (
        (-3, 9),
        (1, 1),
        (0, 1),
        (2, 5),
    )

    for a, b in cases:
        assert jfoo(a, b) == foo(a, b)


def test_map_add_set_add():
    def fn():
        d = {i: i * 2 for i in range(10)}
        s = {i * 2 for i in range(10)}
        return d, s

    jfn = jit(fn)
    assert jfn() == fn()


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1543
def test_kwargs():
    def foo(a, b, *, c=2):
        return a + b + c

    jfoo = jit(foo)

    assert jfoo(2, 3) == foo(2, 3)
    assert jfoo(a=2, b=7, c=3) == foo(a=2, b=7, c=3)

    # Same case as above except c can be specified positionally
    def foo(a, b, c=2):
        return a + b + c

    jfoo = jit(foo)

    assert jfoo(2, 3) == foo(2, 3)
    assert jfoo(a=2, b=7, c=3) == foo(a=2, b=7, c=3)


def test_args_kwargs():
    def bar(a, b):
        return a + b

    def foo(a, **kwargs):
        return bar(a, **kwargs)

    jfoo = jit(foo)
    assert jfoo(2, b=3) == foo(2, b=3)
    assert jfoo(a=2, b=3) == foo(a=2, b=3)


def test_partials():
    def foo(a, b, c):
        return a - b * c

    pfoo = partial(foo, 2, c=3)
    jpfoo = jit(pfoo)

    assert jpfoo(4) == pfoo(4)
    assert jpfoo(-9) == pfoo(-9)

    ppfoo = partial(pfoo, -5)
    jppfoo = jit(ppfoo)

    assert jppfoo() == ppfoo()

    # Tests that keywords "stack" as expected (later partials take precedence)
    pfoo = partial(foo, c=2)
    ppfoo = partial(pfoo, c=-2)
    jppfoo = jit(ppfoo)

    assert jppfoo(7, 9) == ppfoo(7, 9)
    assert jppfoo(7, 9, c=4) == ppfoo(7, 9, c=4)

    # Tests that args "stack" as expected
    pfoo = partial(foo, 7)
    ppfoo = partial(pfoo, 9)
    jppfoo = jit(ppfoo)

    assert jppfoo(5) == ppfoo(5)
    assert jppfoo(-3) == ppfoo(-3)


def test_using_imported_modules():
    import operator

    def foo(a, b):
        return operator.add(a, b)

    jfoo = jit(foo)

    assert jfoo(3, 5) == foo(3, 5)


def test_reduce():
    import functools
    import operator

    def foo(a, b):
        return functools.reduce(operator.add, (a, b))

    jfoo = jit(foo)

    assert jfoo(3, 5) == foo(3, 5)
    assert jfoo(-2, 0) == foo(-2, 0)


def test_calling_methods():
    class mycls:
        def __init__(self, v: int):
            self.v = v

        def my_add(self, b):
            return self.v + b

        @classmethod
        def my_add_class(cls, b):
            o = cls(2)
            return o.v + b

        @staticmethod
        def my_add_static(b):
            return 3 + b

    x = mycls(5)

    # these use LOAD_METHOD / CALL_METHOD
    def foo(x, a):
        return x.my_add(a)

    def foo_class(x, a):
        return x.my_add_class(a)

    def foo_static(x, a):
        return x.my_add_static(a)

    # these use LOAD_ATTR / CALL_FUNCTION
    def bar(x, a):
        meth = x.my_add(a)
        return meth

    def bar_class(x, a):
        meth = x.my_add_class(a)
        return meth

    def bar_static(x, a):
        meth = x.my_add_static(a)
        return meth

    jfoo = jit(foo)
    jfoo_class = jit(foo_class)
    jfoo_static = jit(foo_static)
    jbar = jit(bar)
    jbar_class = jit(bar_class)
    jbar_static = jit(bar_static)

    assert jfoo(x, 7) == foo(x, 7)
    assert jfoo_class(x, 7) == foo_class(x, 7)
    assert jfoo_static(x, 7) == foo_static(x, 7)
    assert jbar(x, 7) == bar(x, 7)
    assert jbar_class(x, 7) == bar_class(x, 7)
    assert jbar_static(x, 7) == bar_static(x, 7)


def test_wrapped_functions():
    def wrap(fn):
        @wraps(fn)
        def foo(*args, **kwargs):
            return fn(*args, **kwargs)

        return foo

    @wrap
    def foo(a, b):
        return a + b

    assert jit(foo)(3, 4) == foo(3, 4)


def test_callable_classes():
    class mycls:
        def __init__(self, v: int):
            self.v = v

        def __call__(self, b):
            return self.v + b

    x = mycls(5)

    def foo(x, a):
        return x(a)

    jfoo = jit(foo)

    assert jfoo(x, 7) == foo(x, 7)


def test_build_slice():
    def foo(a, b):
        l = [0, 1, 2, 3, 4, 5, 6]
        return l[a:b], l[a:], l[:b], l[1:2:2], l[0:a:b]

    jfoo = jit(foo)

    assert jfoo(1, 4) == foo(1, 4)
    assert jfoo(0, -1) == foo(0, -1)


def test_format_value():
    # Tests FVS_HAVE_SPEC and FVC_NONE
    def foo(a, b):
        return f"{a:3.2f}, {b:2.1f}"

    jfoo = jit(foo)

    assert jfoo(2.34, 123234.79289) == foo(2.34, 123234.79289)

    class mycls:
        def __repr__(self):
            return "repr"

        def __str__(self):
            return "str"

    # Tests FVC_NONE
    def foo(a, b):
        return f"{a}, {b}"

    jfoo = jit(foo)

    x = mycls()
    assert jfoo(x, "goodbye") == foo(x, "goodbye")

    # Tests FVC_STR
    def foo(a):
        return f"{a!s}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)

    # Tests FVC_REPR
    def foo(a):
        return f"{a!r}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)

    # Tests FVC_ASCII
    def foo(a):
        return f"{a!a}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)


def test_import():
    def foo(a, b):
        import operator

        return operator.add(a, b)

    jfoo = jit(foo)

    assert jfoo(-1, 3) == foo(-1, 3)

    def foo(a, b):
        from operator import add

        return add(a, b)

    jfoo = jit(foo)

    assert jfoo(2, 7) == foo(2, 7)

    def foo(a):
        import torch.nn as nn
        from torch.nn.functional import relu

        return relu(a)

    jfoo = jit(foo)

    a = torch.randn((2, 2))

    assert_close(jfoo(a), foo(a))

    def foo():
        # test relative import
        from .lit_gpt_model import Config

        return Config

    assert jit(foo)() is foo()

    def foo():
        # test relative import
        from . import lit_gpt_model

        return lit_gpt_model.Config

    assert jit(foo)() is foo()


def test_unhashable_lookaside():
    def fn():
        import weakref

        ws = weakref.WeakSet()
        wr = weakref.ref(ws)
        wr()

    jit(fn)()


def test_generator():
    def my_generator_1():
        yield from range(5)

    def my_generator_2():
        yield 1
        val = 1
        while True:
            val = yield 2 * val

    jgen_1 = jit(my_generator_1)
    jgen_2 = jit(my_generator_2)

    actual = list(jgen_1())
    expected = list(my_generator_1())
    assert actual == expected

    run_gen = my_generator_2()
    j_run_gen = jgen_2()
    actual = [j_run_gen.send(x) for x in [None, 1, 2, 3]]
    expected = [run_gen.send(x) for x in [None, 1, 2, 3]]
    assert actual == expected


def test_binary_operations():
    ops = ["+", "&", "//", "<<", "@", "*", "%", "|", "**", ">>", "-", "/", "^"]

    # NOTE Not all of the number ops support floats (for example lshift)
    number_inps = (
        (5, 9),
        (2, 8),
        (8, 2),
    )

    bool_inps = (
        (True, True),
        (False, True),
    )

    tensor_a = torch.randint(1, 10, (2, 2))
    tensor_b = torch.randint(1, 10, (2, 2))

    for op in ops:
        foo = eval(f"lambda a, b: a {op} b")
        jfoo = jit(foo)
        d = {}
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = jit(bar)

        if op != "@":
            for a, b in number_inps:
                assert jfoo(a, b) == foo(a, b)
                assert jbar(a, b) == bar(a, b)

        if op in {"&", "|", "^"}:
            for a, b in bool_inps:
                assert jfoo(a, b) == foo(a, b)
                assert jbar(a, b) == bar(a, b)

        a1 = tensor_a.clone()
        b1 = tensor_b.clone()
        expected = foo(a1, b1)
        a2 = tensor_a.clone()
        b2 = tensor_b.clone()
        actual = jfoo(a2, b2)
        assert_close(a1, a2)
        assert_close(b1, b2)
        assert_close(actual, expected)

        a1 = tensor_a.clone()
        b1 = tensor_b.clone()
        a2 = tensor_a.clone()
        b2 = tensor_b.clone()
        if op == "/":
            a1 = a1.float()
            a2 = a2.float()
        actual = jbar(a2, b2)
        expected = bar(a1, b1)
        assert_close(a1, a2)
        assert_close(b1, b2)
        assert_close(actual, expected)


def test_get_and_for_iter():
    def foo(a):
        for x in (1, 2, 3):
            a = a + x
        return a

    jfoo = jit(foo)

    assert jfoo(5) == foo(5)

    def foo(d):
        for k, v in d.items():
            if k == "stop":
                return v

    jfoo = jit(foo)

    d = {"start": 5, "stop": 9}

    assert jfoo(d) == foo(d)


def test_unary_not():
    def foo(a):
        return not a

    jfoo = jit(foo)

    assert jfoo(False) == foo(False)
    assert jfoo(0) == foo(0)
    assert jfoo(3.14) == foo(3.14)
    assert jfoo(1j) == foo(1j)

    class mycls(int):
        def __init__(self, v):
            self.v = v

        def __bool__(self) -> bool:
            return self.v % 2 == 0

    cases = (
        (mycls(1), 1),
        (mycls(2), 2),
    )

    for o, case in cases:
        assert jfoo(o) == foo(case % 2 == 0)

    assert jfoo([]) == foo([])
    assert jfoo([1, 2]) == foo([2, 3])


def test_unary_neg_invert():
    def invert(x):
        return ~x

    def neg(x):
        return -x

    def pos(x):
        return +x

    for fn in (invert, neg, pos):
        jfn = jit(fn)
        for v in [1, 2, torch.tensor(3)]:
            assert fn(v) == jfn(v)

    for fn in (invert, neg, pos):
        jfn = jit(fn)
        with pytest.raises(TypeError) as exc_expected:
            fn(object())
        with pytest.raises(TypeError) as exc_actual:
            jfn(object())
        assert str(exc_expected.value) == str(exc_actual.value)


def test_unpack_ex():
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    def foo(a):
        a, b, *l = a
        return a, b, l

    jfoo = jit(foo)

    assert jfoo(alphabet) == foo(alphabet)

    def foo(a):
        *l, x, y, z = a
        return l, x, y, z

    jfoo = jit(foo)

    assert jfoo(alphabet) == foo(alphabet)

    def foo(a):
        a, b, c, d, *l, z = a
        return a, b, c, d, l, z

    jfoo = jit(foo)

    assert jfoo(alphabet) == foo(alphabet)

    def foo(a):
        (*l,) = a
        return l

    jfoo = jit(foo)

    assert jfoo(alphabet) == foo(alphabet)


def test_list_to_tuple():
    def ltt():
        return (*[1, 2, 3],)

    assert any(i.opname == "LIST_TO_TUPLE" for i in dis.get_instructions(ltt))
    assert jit(ltt)() == ltt()


def test_use_of_deleted_raises_correctly():
    def foo(a):
        b = a
        del b
        c = b + a
        return a

    jfoo = jit(foo)

    with pytest.raises(UnboundLocalError, match=r".*local variable 'b' referenced before assignment.*"):
        jfoo(5)


def test_delete_fast():
    def foo(a):
        b = a
        del b
        c = b + a
        return a

    jfoo = jit(foo)
    with pytest.raises(UnboundLocalError, match="'b'"):
        jfoo(5)


def test_delete_global():
    global x
    x = 5

    def foo(a):
        global x
        del x
        return a + x

    jfoo = jit(foo)

    with pytest.raises(NameError):
        jfoo(5)


x = 7


def test_store_global():
    def foo(a):
        global x
        x = a

    jfoo = jit(foo)

    jfoo(6)
    assert x == 6


def test_bool_conversion():
    def foo(a):
        return bool(a)

    jfoo = jit(foo)

    literal_cases = (0, 1, False, True, -0.5, 73, complex(1, 2), None, [], (), "", "abc", [1, 3], (5, 6), {}, {1: 3})

    for x in literal_cases:
        assert jfoo(x) == foo(x)

    # Checks default class behavior
    class mycls:
        pass

    x = mycls()

    assert jfoo(x) == foo(x)
    assert jfoo(mycls) == foo(mycls)

    # Checks dunder bool handling (by default classes are true)
    class mycls:
        def __bool__(self):
            return False

    x = mycls()

    assert jfoo(x) == foo(x)

    # Classes that define dunder len and not dunder bool use dunder len for their bool() conversion
    class mycls:
        def __len__(self):
            return 0

    x = mycls()

    assert jfoo(x) == foo(x)

    class mycls:
        def __len__(self):
            return 1

    x = mycls()

    assert jfoo(x) == foo(x)


def test_store_attr():
    def foo(a, v):
        a.foo = v

    def bar(a):
        del a.foo

    jfoo = jit(foo)
    jbar = jit(bar)

    class mycls:
        pass

    x = mycls()

    jfoo(x, 5)
    assert x.foo == 5
    jbar(x)
    assert not hasattr(x, "foo")

    # Checks that dunder setattr is called
    class mycls:
        def __setattr__(self, name, value):
            # NOTE This can't call __setattr__ again (not even indirectly, like through self.bar = value)
            #   because that would cause infinite recursion
            # This avoids the infinite recursion by calling objec'ts dunder setattr, which isn't hooked
            super().__setattr__("bar", value)

    x = mycls()

    jfoo(x, 5)
    assert x.bar == 5


def test_builtin_getattr():
    x = 5
    assert x.__add__ == jit(getattr)(x, "__add__")


def test_simple_attribute():
    class SimpleNamespace:
        x: int
        y: int

    obj = SimpleNamespace()
    obj.x = 1
    obj.y = 2

    def foo():
        return obj.x + obj.y

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_dunder_getattr():
    history = []

    class X:
        def __getattr__(self, name):
            history.append(f"X.__getattr__ {is_jitting()}")
            return 1

    def foo():
        return X().a

    assert foo() == jit(foo)() == 1
    assert tuple(history) == ("X.__getattr__ False", "X.__getattr__ True")


@pytest.mark.xfail(reason="__getattribute__ support is not yet implemented.")
def test_dunder_getattribute():
    history = []

    class MyClass:
        def __getattribute__(self, name):
            history.append(f"__getattr__ {is_jitting()}")
            return 1

    def foo():
        x = MyClass()
        x.a = 2  # __getattribute__ will take precedence
        return x.a

    assert foo() == jit(foo)() == 1
    assert tuple(history) == ("__getattr__ False", "__getattr__ True")


def test_property():
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting()}")
            return 1

        @property
        def lazy_x(self):
            history.append(f"lazy_x {is_jitting()}")
            result = getattr(self, "_x", None)
            if result is None:
                self._x = result = 2
            return result

    def foo():
        return MyClass().x

    assert foo() == jit(foo)() == 1
    assert tuple(history) == ("x False", "x True")
    history.clear()

    def foo():
        return MyClass().lazy_x

    assert foo() == jit(foo)() == 2
    assert tuple(history) == ("lazy_x False", "lazy_x True")
    history.clear()


def test_property_with_setter():
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting()}")
            return self._x

        @x.setter
        def x(self, value) -> None:
            self._x = value

    my_class = MyClass()
    my_class.x = 5
    my_class.__dict__["x"] = 8  # Make sure property takes precedence

    def foo():
        return my_class.x

    assert foo() == jit(foo)() == 5
    assert tuple(history) == ("x False", "x True")


@pytest.mark.xfail(reason=".setter support is not yet implemented.")
def test_property_with_instrumented_setter():
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting()}")
            return self._x

        @x.setter
        def x(self, value) -> None:
            history.append(f"x.setter {is_jitting()}")
            self._x = value

    my_class = MyClass()
    my_class.__dict__["x"] = 8  # Make sure property takes precedence
    history.clear()

    def foo():
        my_class.x = 5
        return my_class.x

    assert foo() == jit(foo)() == 5
    assert tuple(history) == ("x False", "x.setter False", "x True", "x.setter True")


def test_compare():
    # uses ROT_THREE in Python 3.10
    def fn(a):
        return 2 <= a <= 4

    jfn = jit(fn)
    for a in (1, 3, 5):
        assert fn(a) == jfn(a)


def test_comprehension():
    def foo():
        return tuple([i for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_nested_comprehension():
    def foo():
        return tuple([[j for j in enumerate(range(i))] for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_comprehension_nonlocal():
    def foo():
        counter = 0

        def increment():
            nonlocal counter
            counter = counter + 1
            return counter

        return tuple([i + increment() for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_comprehension_nonlocal_inplace():
    def foo():
        counter = 0

        def increment():
            nonlocal counter
            counter += 1
            return counter

        return tuple([i + increment() for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_set_creation():
    def foo():
        return {1, *[2, 3]}

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_contains():
    def foo(a, seq):
        return a in seq

    def nfoo(a, seq):
        return a not in seq

    jfoo = jit(foo)
    jnfoo = jit(nfoo)

    assert jfoo(2, (1, 2, 3)) == foo(2, (1, 2, 3))
    assert jnfoo(2, (1, 2, 3)) == nfoo(2, (1, 2, 3))
    assert jfoo(5, (1, 2, 3)) == foo(5, (1, 2, 3))
    assert jnfoo(5, (1, 2, 3)) == nfoo(5, (1, 2, 3))

    assert jfoo(2, [1, 2, 3]) == foo(2, [1, 2, 3])
    assert jnfoo(2, [1, 2, 3]) == nfoo(2, [1, 2, 3])
    assert jfoo(5, [1, 2, 3]) == foo(5, [1, 2, 3])
    assert jnfoo(5, [1, 2, 3]) == nfoo(5, [1, 2, 3])

    assert jfoo("a", {"a": 1, "b": 2, "c": 3}) == foo("a", {"a": 1, "b": 2, "c": 3})
    assert jnfoo("a", {"a": 1, "b": 2, "c": 3}) == nfoo("a", {"a": 1, "b": 2, "c": 3})
    assert jfoo("f", {"a": 1, "b": 2, "c": 3}) == foo("f", {"a": 1, "b": 2, "c": 3})
    assert jnfoo("f", {"a": 1, "b": 2, "c": 3}) == nfoo("f", {"a": 1, "b": 2, "c": 3})

    assert jfoo("a", {"a", "b", "c"}) == foo("a", {"a", "b", "c"})
    assert jnfoo("a", {"a", "b", "c"}) == nfoo("a", {"a", "b", "c"})
    assert jfoo("f", {"a", "b", "c"}) == foo("f", {"a", "b", "c"})
    assert jnfoo("f", {"a", "b", "c"}) == nfoo("f", {"a", "b", "c"})


def test_contains_custom_containers():
    def foo(a, seq):
        return a in seq

    def nfoo(a, seq):
        return a not in seq

    jfoo = jit(foo)
    jnfoo = jit(nfoo)

    class mycontainscontainer:
        def __init__(self, len):
            self.list = [*range(len)]

        def __contains__(self, v):
            for a in self.list:
                if v is a or v == a:
                    return True
            return False

    o = mycontainscontainer(3)

    assert jfoo(2, o) == foo(2, o)
    assert jnfoo(2, o) == nfoo(2, o)
    assert jfoo(5, o) == foo(5, o)
    assert jnfoo(5, o) == nfoo(5, o)

    class myitercontainer:
        def __init__(self, len):
            self.list = [*range(len)]

        def __iter__(self):
            return self.list.__iter__()

    o = myitercontainer(3)

    assert jfoo(2, o) == foo(2, o)
    assert jnfoo(2, o) == nfoo(2, o)
    assert jfoo(5, o) == foo(5, o)
    assert jnfoo(5, o) == nfoo(5, o)

    class mygetitemcontainer:
        def __init__(self, len):
            self.list = [*range(len)]
            self.len = len

        def __getitem__(self, i):
            return self.list[i]

        def __len__(self):
            return self.len

    o = mygetitemcontainer(3)

    assert jfoo(2, o) == foo(2, o)
    assert jnfoo(2, o) == nfoo(2, o)
    assert jfoo(5, o) == foo(5, o)
    assert jnfoo(5, o) == nfoo(5, o)


def test_name_opcodes_and_print_expr():
    from types import FunctionType
    from contextlib import redirect_stdout
    import io

    co = compile("x = 5; print(x); del x;", "<string>", "single")
    fn = FunctionType(co, globals())
    jfn = jit(fn)

    py_redirect = io.StringIO()
    with redirect_stdout(py_redirect):
        fn()
    py_out: str = py_redirect.getvalue()

    jit_redirect = io.StringIO()
    with redirect_stdout(jit_redirect):
        jfn()
    jit_out: str = jit_redirect.getvalue()

    assert py_out == jit_out

    # Checks display hook setting
    saved = sys.displayhook
    try:

        def foo(x):
            print("display hook!")
            print(x)

        sys.displayhook = foo
        co = compile("x = 5; print(x); del x;", "<string>", "single")
        fn = FunctionType(co, globals())
        jfn = jit(fn)

        py_redirect = io.StringIO()
        with redirect_stdout(py_redirect):
            fn()
        py_out: str = py_redirect.getvalue()

        jit_redirect = io.StringIO()
        with redirect_stdout(jit_redirect):
            jfn()
        jit_out: str = jit_redirect.getvalue()

        assert "display hook!" in py_out
        assert py_out == jit_out
    finally:
        sys.displayhook = saved

    co = compile("x = 5; del x; print(x)", "<string>", "single")
    fn = FunctionType(co, globals())
    jfn = jit(fn)

    with pytest.raises(NameError, match="'x' is not defined"):
        jfn()


def test_load_build_class():
    def foo():
        class C:
            def __init__(self):
                self.bar = 5

        return C, C().bar

    jfoo = jit(foo)

    cp, cb = foo()
    jp, jb = jfoo()
    assert cb == jb
    assert cp().bar == jp().bar

    assert any(i.opname == "LOAD_BUILD_CLASS" for i in dis.get_instructions(foo))
    assert any(i.opname == "LOAD_BUILD_CLASS" for i in jfoo._last_interpreted_instructions)


def test_with():
    class CtxMgr:
        def __init__(self, l):
            self.l = l

        def __enter__(self):
            self.l.append("enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.l.append((str(exc_type), str(exc_val)))

    def fn(should_raise: bool = False):
        l = []
        with CtxMgr(l) as ctx:
            ctx.l.append("within")
            if should_raise:
                raise RuntimeError("test", l)
            return l

    jfn = jit(fn)

    assert fn() == jfn()

    with pytest.raises(RuntimeError) as exc_expected:
        fn(should_raise=True)
    with pytest.raises(RuntimeError) as exc_actual:
        jfn(should_raise=True)

    assert exc_expected.value.args[1] == exc_actual.value.args[1]


def test_super():
    class A:
        def foo(self):
            return f"Hello {type(self)} {__class__}"

        @classmethod
        def bar(self):
            return f"Hello {type(self)} {__class__}"

    class B(A):
        def foo(self):
            return super().foo()

        @classmethod
        def bar(self):
            return super().bar()

    class C(A):
        def foo(self):
            return super(C, self).foo()

    def foo():
        b = B()
        c = C()
        return (b.foo(), c.foo())

    assert jit(foo)() == foo()

    def bar():
        b = B()
        c = C()
        super(b, C)

    with pytest.raises(TypeError) as exc_expected:
        bar()
    with pytest.raises(TypeError) as exc_actual:
        jit(bar)()
    # Python 3.11 improved the grammar, so do we
    assert str(exc_expected.value).replace("be type", "be a type") == str(exc_actual.value)

    def baz():
        b = B()
        return b.bar()

    assert jit(baz)() == baz()


def test_is_jitting():
    def foo():
        return is_jitting()

    assert not foo()
    assert jit(foo)()


def test_autograd_function():
    class DoubleFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a * 2

        @staticmethod
        def backward(ctx, grad_out):
            return 2 * grad_out

    a = torch.randn(5, 5, requires_grad=True)

    def fn():
        b = DoubleFn.apply(a)
        return b

    assert_close(fn(), jit(fn)())


def test_torch_autocast_nograd():
    def fn(a, b):
        with torch.autocast("cpu"):
            return a @ b

    def fn2(a, b):
        with torch.no_grad():
            return a @ b

    a = torch.randn(5, 5)
    b = torch.randn(5, 5)
    expected = fn(a, b)
    actual = jit(fn)(a, b)

    assert_close(actual, expected)  # also check dtype (should be bf16)

    a = torch.randn(5, 5, requires_grad=True)
    b = torch.randn(5, 5, requires_grad=True)
    expected = fn2(a, b)
    actual = jit(fn2)(a, b)

    assert_close(actual, expected)  # also check dtype (should be bf16)


def test_is_jitting_opaque():
    def foo():
        return tuple(map(lambda _: is_jitting(), range(3)))

    assert foo() == (False, False, False)
    with pytest.raises(JITError):
        jit(foo)()


#
# "Phantom" / No-side-effect tests
#


def test_phantom_simple_list():
    def foo(a):
        a.append(4)
        return a

    pfoo = phantom_jit(foo)

    l = [1, 2, 3]
    result = pfoo(l)

    assert result == [1, 2, 3, 4]
    assert l == [1, 2, 3]


def test_phantom_aliasing_lists():
    def foo(a, b):
        a.append(7)
        return a, b

    pfoo = phantom_jit(foo)

    l = [3, 5]
    a, b = pfoo(l, l)

    assert l == [3, 5]
    assert a == [3, 5, 7]
    assert a == b


def test_phantom_deep_aliasing_lists():
    def foo(a, b):
        a.append(7)
        b[0].append(99)
        return a, b

    pfoo = phantom_jit(foo)

    l = [3, 5]
    a = [[1, 3, 5], l]
    b = [l, [11, 13, 15]]
    ra, rb = pfoo(a, b)

    assert l == [3, 5]
    assert a == [[1, 3, 5], l]
    assert b == [l, [11, 13, 15]]

    assert ra == [[1, 3, 5], [3, 5, 99], 7]
    assert rb == [[3, 5, 99], [11, 13, 15]]


def test_phantom_aliasing_dicts():
    def foo(d):
        d["hi"] = "bye"
        d["also"] = "and then"
        return d

    pfoo = phantom_jit(foo)

    d = {"hi": "hello", "x": "y"}

    result = pfoo(d)

    assert d == {"hi": "hello", "x": "y"}
    assert result == {"hi": "bye", "x": "y", "also": "and then"}


def test_phantom_is():
    def foo(a, b):
        return a is b

    pfoo = phantom_jit(foo)

    l = [1, 3]
    assert pfoo(l, l) is True

    l0 = [5, 7]
    assert pfoo(l, l0) is False


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1744")
def test_phantom_nonlocal():
    y = 5

    def foo(x):
        nonlocal y
        y += x
        return y

    pfoo = phantom_jit(foo)

    assert y == 5
    result = pfoo(3)
    assert result == 8
    assert y == 5


# TODO This should probably throw an error that we attempted to modify the non-copyable operator module
@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1745")
def test_phantom_import():
    def foo(a, b):
        import operator

        operator.hi = "bye"
        return operator.add(a, b)

    pfoo = phantom_jit(foo)

    import operator

    pfoo(1, 3)
    assert not hasattr(operator, "hi")


def test_phantom_uncopyable_in_collection():
    def foo(a):
        a.append(7)
        return a

    pfoo = phantom_jit(foo)

    # NOTE Modules cannot be deep-copied (which causes a warning to be thrown)
    import operator

    l = [operator, 3, 5]

    with pytest.warns(UserWarning):
        result = pfoo(l)

    assert l == [operator, 3, 5]
    assert result == [operator, 3, 5, 7]

    def foo(a):
        a[0].append(7)
        return a

    pfoo = phantom_jit(foo)

    l = [[1, 3], operator]
    result = pfoo(l)

    assert l == [[1, 3], operator]
    assert result == [[1, 3, 7], operator]


def test_phantom_object_aliasing():
    def foo(a, b, c):
        a.one = 2
        b.append(a)
        c["hi"] = "bye"

        assert a.z is c

        return a, b, c

    pfoo = phantom_jit(foo)

    class mycls:
        pass

    x = mycls()
    y = [x, 3]
    z = {0: x, 1: y}
    x.z = z

    a, b, c = pfoo(x, y, z)

    assert not hasattr(x, "one")
    assert a.one == 2
    assert y == [x, 3]
    # NOTE b, an output, does not contain x like y does, because x was replaced with a in the computation
    assert b == [a, 3, a]
    assert z == {0: x, 1: y}
    # NOTE c, an output, does not contain keys x and y like z does, because x was replaced with a, and y replaced with b
    assert c == {0: a, 1: b, "hi": "bye"}


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1747")
def test_phantom_modification_warning():
    def foo(a):
        a.one = 2
        return a

    pfoo = phantom_jit(foo)

    class mycls:
        pass

    with pytest.warns(UserWarning):
        pfoo(mycls)


_test_phantom_globals_global = 5


def test_phantom_globals():
    # Reads a global
    def foo(a):
        return a + _test_phantom_globals_global

    pfoo = phantom_jit(foo)

    result = pfoo(5)

    assert result == 10

    # Writes to the global value don't affect the global value, but appear correctly
    def foo(a):
        global _test_phantom_globals_global
        _test_phantom_globals_global = a
        return _test_phantom_globals_global

    pfoo = phantom_jit(foo)

    result = pfoo(10)
    assert result == 10
    assert _test_phantom_globals_global == 5

    # Deletes don't affect the global value
    def foo():
        global _test_phantom_globals_global
        del _test_phantom_globals_global

    pfoo = phantom_jit(foo)

    pfoo()
    assert _test_phantom_globals_global == 5

    # Deletes are effected locally
    def foo():
        global _test_phantom_globals_global
        del _test_phantom_globals_global
        return _test_phantom_globals_global

    pfoo = phantom_jit(foo)

    with pytest.raises(NameError):
        pfoo()

    # Multiple loads of the same global work as expected
    def bar(a):
        global _test_phantom_globals_global
        _test_phantom_globals_global = a + 1

    def foo(a):
        global _test_phantom_globals_global
        bar(a)
        return _test_phantom_globals_global + 1

    pfoo = phantom_jit(foo)

    result = pfoo(10)
    assert result == 12
    assert _test_phantom_globals_global == 5


# Tests that directly assigning to the globals() dict does not modify the actual globals dict
@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1760")
def test_phantom_globals_fn():
    def foo(a):
        g = globals()
        g["_test_phantom_globals_fn_global"] = a

    pfoo = phantom_jit(foo)

    pfoo(5)

    with pytest.raises(NameError):
        assert "_test_phantom_globals_fn_global" not in globals()


# Tests that the random state is preserved even when making random calls
def test_phantom_randint():
    import random

    def foo():
        return random.randint(0, 5)

    pfoo = phantom_jit(foo)

    s0 = random.getstate()

    result0 = pfoo()
    result1 = pfoo()

    s1 = random.getstate()

    assert s0 == s1
    assert result0 == result1


# Tests that the random state is preserved even when directly setting the random seed
@pytest.mark.skipif(sys.version_info > (3, 11), reason="https://github.com/Lightning-AI/lightning-thunder/issues/1762")
def test_phantom_seed():
    import random

    def foo():
        random.seed(1234)
        return random.randint(0, 5)

    pfoo = phantom_jit(foo)

    s0 = random.getstate()

    result0 = pfoo()
    result1 = pfoo()

    s1 = random.getstate()

    assert result0 == result1
    assert s0 == s1

    random_result: int
    try:
        random.seed(1234)
        random_result = random.randint(0, 5)
    finally:
        random.setstate(s0)

    assert random_result == result0


#
# "Thunder" tests
#


@pytest.mark.skipif(sys.version_info > (3, 11), reason="Only supports Python 3.10 at the moment")
def test_interpreter_stats():
    def foo(a, b):
        return a + b

    ljfoo = litjit(foo)

    result = ljfoo(5, 7)

    assert result == 12

    history: list = thunder.last_interpreted_history(ljfoo)

    # Looks for the int instruction (there's almost certainly a nicer way to do this)
    found: bool = False
    for inst in history:
        if isinstance(inst, str):
            if inst.startswith("Opaque call to <method-wrapper '__add__' of int object"):
                found = True
                break

    assert found


#
# Network tests
#


def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTMLPBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTMLPBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt_csa():
    from thunder.benchmarks import NanoGPTCSABenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTCSABenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt_block():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt():
    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["test"])
    bench = NanoGPTBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_litgpt():
    from thunder.benchmarks import LitGPTBenchmark
    from thunder.tests.lit_gpt_model import Config

    cfg: Config = Config.from_name("gpt-neox-like")
    bench = LitGPTBenchmark(config=cfg, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))
