from collections.abc import Iterable, Iterator, Sequence
from contextlib import redirect_stdout
from functools import partial, wraps
from itertools import product

import io
import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.interpreter import (
    is_jitting_with_raise,
    is_jitting,
    make_opaque,
    interpret,
    InterpreterError,
    print_last_interpreted_history,
)

#
# Test suite for core Python interpreter functionality
#
interpret_no_tracking = interpret


# This wraps the jit call into a tracking one (using a wrapper function
# rather than partial to get a nice test name).
def interpret_tracking(*args, **kwargs):
    return interpret(
        *args, with_provenance_tracking=True, uncacheable_classes=(torch.Tensor, int, float, str, type(None)), **kwargs
    )


# This will be called by PyTest and parametrize each test that has
# a jit attribute (all?) with versions that use jit and jit_tracking.
def pytest_generate_tests(metafunc):
    if "jit" in metafunc.fixturenames:
        metafunc.parametrize("jit", [interpret, interpret_tracking])


def skipif_python_3_11_plus(f):
    if sys.version_info >= (3, 11):
        return pytest.mark.skip(f, reason=f"not yet implemented for Python 3.11+, got {sys.version_info=}")
    return f


def test_no_return(jit):
    def foo():
        pass

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_return(jit):
    def foo():
        return 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_addition(jit):
    def foo():
        return 3 + 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_input_number_addition(jit):
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

    args = (5, 2)

    assert jfoo(*args) == foo(*args)


def test_input_tensor_addition(jit):
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_dup_top_two(jit):
    def foo(a):
        a[-1] += a.pop()
        return a

    if "DUP_TOP_TWO" in dis.opmap.keys():
        assert any(i.opname == "DUP_TOP_TWO" for i in dis.get_instructions(foo))
    assert jit(foo)([1, 2, 3]) == foo([1, 2, 3])


def test_constant_if(jit):
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


def test_if(jit):
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


def test_while(jit):
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


def test_and_or(jit):
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


def test_dunder_bool(jit):
    jitting = False

    class mycls:
        def __init__(self, value):
            self.value = value

        # True if self.value is even
        def __bool__(self):
            assert is_jitting_with_raise() == jitting
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
        jitting = False
        r = foo(*case)
        jitting = True
        jr = jfoo(*case)
        assert r == jr


def test_dunder_bool_instance(jit):
    jitting = False

    class X:
        def __bool__(self):
            assert is_jitting_with_raise() == jitting
            return False

    x = X()
    jitting = False
    bx = bool(x)
    jitting = True
    jbx = jit(bool)(x)
    assert bx == jbx == False

    x.__bool__ = lambda: True  # dunder methods use class attribute, not instance attribute.
    jitting = False
    bx = bool(x)
    jitting = True
    jbx = jit(bool)(x)
    assert bx == jbx == False


def test_function_call(jit):
    jitting = False

    def fn(fn):
        assert is_jitting_with_raise() == jitting
        return fn

    jitting = False
    r = fn(fn)
    jitting = True
    jr = jit(fn)(fn)
    assert r == jr

    jitting = False
    r = fn(fn=fn)
    jitting = True
    jr = jit(fn)(fn=fn)
    assert r == jr


def test_nested_function_call(jit):
    jitting = False

    def bar(a, b):
        assert is_jitting_with_raise() == jitting
        return a + b

    def foo(a, b):
        assert is_jitting_with_raise() == jitting
        return bar(a + 1, b)

    jfoo = jit(foo)
    args = (4, 3)

    python_result = foo(*args)
    jitting = True
    thunder_result = jfoo(*args)

    assert_close(thunder_result, python_result)


def test_call_function_ex(jit):
    jitting = False

    def foo(a, b):
        assert is_jitting_with_raise() == jitting
        return a + b

    def argsplat(*args):
        assert is_jitting_with_raise() == jitting
        return foo(*args)

    def kwargsplat(**kwargs):
        assert is_jitting_with_raise() == jitting
        return foo(**kwargs)

    assert any(i.opname == "CALL_FUNCTION_EX" and not i.arg & 1 for i in dis.get_instructions(argsplat))
    assert any(i.opname == "CALL_FUNCTION_EX" and i.arg & 1 for i in dis.get_instructions(kwargsplat))

    kwargs = {"a": 1, "b": 2}

    jitting = False
    res1 = argsplat(*kwargs.values())
    res2 = kwargsplat(**kwargs)

    jitting = True
    jres1 = jit(argsplat)(*kwargs.values())
    jres2 = jit(kwargsplat)(**kwargs)

    assert_close(res1, jres1)
    assert_close(res2, jres2)


def test_build_const_key_map(jit):
    def fn1(a, b):
        return {"a": a, "b": b}

    # test order for collisions
    def fn2(a, b):
        return {"a": a, "a": b}

    assert any(i.opname == "BUILD_CONST_KEY_MAP" for i in dis.get_instructions(fn1))
    assert any(i.opname == "BUILD_CONST_KEY_MAP" for i in dis.get_instructions(fn2))

    jfn1 = jit(fn1)
    jfn2 = jit(fn2)

    assert jfn1(1, 2) == fn1(1, 2)
    assert jfn2(1, 2) == fn2(1, 2)


def test_build_map_dict_merge(jit):
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


def test_dict_update(jit):
    jitting = False

    def addall(*args, **kwargs):
        assert is_jitting_with_raise() == jitting
        return sum(args) + sum(kwargs.values())

    def foo(*args, **kwargs):
        assert is_jitting_with_raise() == jitting
        return addall(*args, **{**kwargs, "x": 1})

    assert any(i.opname == "DICT_UPDATE" for i in dis.get_instructions(foo))

    args = (4, 3)
    kwargs = {"a": 1, "b": 2}

    jitting = False
    python_result = foo(*args, **kwargs)
    jitting = True
    thunder_result = jit(foo)(*args, **kwargs)

    assert_close(thunder_result, python_result)


def test_inner_function_definition(jit):
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


def test_inner_closure(jit):
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


def test_delete_deref(jit):
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


def test_locals_globals(jit):
    def fn():
        funny_name_nowhere_else = True
        return locals() | globals()

    assert "test_locals_globals" in jit(fn)()
    assert "funny_name_nowhere_else" in jit(fn)()


def test_unpack_sequence(jit):
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


def test_exception_traceback(jit):
    def bar(a):
        raise ValueError(f"I don't like {a}")

    def foo(b):
        return bar(b + 1)

    jfoo = jit(foo)

    args = (4,)

    with pytest.raises(ValueError) as excinfo:
        thunder_result = jfoo(*args)

    tb_string = "".join(str(tbe) for tbe in excinfo.traceback)
    assert "in foo\n" in tb_string
    assert "in bar\n" in tb_string


def test_finally(jit):
    jitting = False
    l = []

    def foo():
        try:
            assert is_jitting_with_raise() == jitting
            l.append(1)
            raise ValueError("test")
            l.append(2)
        except KeyError:
            assert is_jitting_with_raise() == jitting
            l.append(3)
        except ValueError:
            assert is_jitting_with_raise() == jitting
            l.append(4)
            raise
        finally:
            assert is_jitting_with_raise() == jitting
            l.append(5)

    with pytest.raises(ValueError):
        jitting = False
        foo()

    l_orig = l

    l = []

    with pytest.raises(ValueError):
        jitting = True
        jit(foo)()

    assert l_orig == l


def test_raise(jit):
    msg = "lorem ipsum"
    jitting = False

    class ExampleException(ValueError):
        def __init__(self):
            assert is_jitting_with_raise() == jitting
            super().__init__(msg)

    def foo():
        raise ExampleException  # Constructed implicitly

    jfoo = jit(foo)

    with pytest.raises(ExampleException) as excinfo:
        jitting = False
        foo()

    with pytest.raises(ExampleException) as excinfo:
        jitting = True
        jfoo()

    assert msg in str(excinfo.value)


def test_bare_except(jit):
    msg = "lorem ipsum"
    jitting = False

    def bare_except():
        try:
            assert is_jitting_with_raise() == jitting
            raise ValueError(msg)
        except:
            assert is_jitting_with_raise() == jitting
            return True

    assert bare_except() == True
    jitting = True
    assert jit(bare_except)() == True


def test_trivial_try_finally(jit):
    def trivial_try_finally():
        try:
            pass
        finally:
            return True

    assert jit(trivial_try_finally)() == True


def test_try_finally(jit):
    def try_finally():
        try:
            var = False
            raise ValueError
        except ValueError:
            var = True
        finally:
            return var

    assert jit(try_finally)() == True


def test_match_exception(jit):
    def match_exception():
        error_set = (ValueError, IndexError)
        try:
            raise ValueError
        except error_set:
            return True

    assert jit(match_exception)() == True


def test_match_as(jit):
    msg = "lorem ipsum"

    def match_as():
        try:
            raise ValueError(msg)
        except ValueError as e:
            return str(e)

    assert msg in jit(match_as)()


def test_list(jit):
    def foo():
        l = [1, 2, 3]
        l[3:] = l[:2]
        l[0] = l[-1]
        del l[2]
        return l

    assert foo() == jit(foo)()


def test_raise_external(jit):
    msg = "lorem ipsum"

    def raise_external():
        raise ValueError(msg)

    with pytest.raises(ValueError) as excinfo:
        jit(raise_external)()

    assert msg in str(excinfo.value)


def test_raise_from(jit):
    msg = "lorem ipsum"

    def raise_from():
        try:
            raise ValueError(msg) from IndexError(msg)
        except ValueError as e:
            return (str(e), str(e.__cause__))

    res = jit(raise_from)()
    assert msg in res[0] and msg in res[1]


def test_raise_from_external(jit):
    msg = "lorem ipsum"

    def raise_from_external():
        raise ValueError(msg) from IndexError(msg)

    with pytest.raises(ValueError) as excinfo:
        jit(raise_from_external)()

    e = excinfo.value
    assert type(e) == ValueError
    assert type(e.__cause__) == IndexError and msg in str(e.__cause__), excinfo.value


def test_nested_try_except(jit):
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


def test_inner_nested_try_except(jit):
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


def test_cross_function_exceptions(jit):
    jitting = False

    def foo():
        assert is_jitting_with_raise() == jitting

        def bar():
            assert is_jitting_with_raise() == jitting
            raise ValueError

        bar()

    def cross_function_exceptions():
        try:
            assert is_jitting_with_raise() == jitting
            foo()
        except ValueError:
            assert is_jitting_with_raise() == jitting
            return True

    jitting = False
    assert cross_function_exceptions() == True
    jitting = True
    assert jit(cross_function_exceptions)() == True


def test_walrus_operator(jit):
    def foo(a, b):
        c = (a := b)
        return c

    if "DUP_TOP" in dis.opmap.keys():
        assert any(i.opname == "DUP_TOP" for i in dis.get_instructions(foo))
    jfoo = jit(foo)

    assert jfoo(3, 8) == foo(3, 8)


def test_build_map(jit):
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


def test_map_add_set_add(jit):
    def fn():
        d = {i: i * 2 for i in range(10)}
        s = {i * 2 for i in range(10)}
        return d, s

    jfn = jit(fn)
    assert jfn() == fn()


def test_kwargs(jit):
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


def test_args_kwargs(jit):
    def bar(a, b):
        return a + b

    def foo(a, **kwargs):
        return bar(a, **kwargs)

    jfoo = jit(foo)
    assert jfoo(2, b=3) == foo(2, b=3)
    assert jfoo(a=2, b=3) == foo(a=2, b=3)


def test_partials(jit):
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


def test_using_imported_modules(jit):
    import operator

    def foo(a, b):
        return operator.add(a, b)

    jfoo = jit(foo)

    assert jfoo(3, 5) == foo(3, 5)


def test_reduce(jit):
    import functools
    import operator

    # Trivial reduce over native types {
    def foo(a):
        return functools.reduce(operator.add, a, 0)

    jfoo = jit(foo)

    assert jfoo((1, 2, 3)) == foo((1, 2, 3)) == 6
    # }

    # Reduce over Tensor.shape {
    def foo(a):
        return functools.reduce(operator.add, a.shape, 0)

    jfoo = jit(foo)

    assert jfoo(torch.rand(1, 2, 3)) == foo(torch.rand(1, 2, 3)) == 6
    # }

    # Custom Iterable over Tensor.shape {
    class mycls(object):
        def __init__(self, t):
            self.t = t

        def __iter__(self):
            assert is_jitting_with_raise() == jitting
            self.it = iter(self.t.shape)
            return self

        def __next__(self):
            assert is_jitting_with_raise() == jitting
            return next(self.it)

        @property
        def shape(self):
            assert is_jitting_with_raise() == jitting
            return self.t.shape

    def foo(a):
        return functools.reduce(operator.add, a, 0)

    jitting = False
    assert foo(mycls(torch.rand(1, 2, 3))) == 6

    jitting = True
    jfoo = jit(foo)
    assert jfoo(mycls(torch.rand(1, 2, 3))) == 6
    # }

    # Test reduce function is being jitted {
    jitting = True

    def add(x, y):
        assert is_jitting_with_raise() == jitting
        return x + y

    def foo(a, fn):
        return functools.reduce(fn, a.shape, 0)

    jfoo = jit(foo)

    assert jfoo(torch.rand(1, 2, 3), add) == 6
    assert jfoo(mycls(torch.rand(1, 2, 3)), add) == 6
    # }

    # Check we error when init value is not provided with empty Iterable {
    jitting = True

    def foo(a):
        return functools.reduce(operator.add, a)

    jfoo = jit(foo)

    with pytest.raises(TypeError, match=r"reduce\(\) of empty iterable with no initial value"):
        jfoo(mycls(torch.rand(1).squeeze()))
    # }

    # Test provided init values vs not provided {
    def foo_no_init(a):
        return functools.reduce(operator.add, a)

    def foo_with_init(a):
        return functools.reduce(operator.add, a, 0)

    jfoo_no_init = jit(foo_no_init)
    jfoo_with_init = jit(foo_with_init)

    assert jfoo_no_init((1, 2, 3)) == jfoo_with_init((1, 2, 3)) == 6
    # }

    # Test providing init as None that should trigger type mismatch {
    def foo(a):
        return functools.reduce(operator.add, a, None)

    jfoo = jit(foo)

    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \+: 'NoneType' and 'int'"):
        foo((1, 2, 3))

    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for \+: 'NoneType' and 'int'"):
        jfoo((1, 2, 3))
    # }

    # Check no kwargs are allowed {
    def foo(x):
        return functools.reduce(function=operator.add, iterable=x)

    jfoo = jit(foo)

    with pytest.raises(Exception, match=r"reduce\(\) takes no keyword arguments"):
        foo((1, 2, 3))

    with pytest.raises(Exception, match=r"got some positional-only arguments passed as keyword arguments"):
        jfoo((1, 2, 3))
    # }


# Test for issue "jit: passing jitted functions as arguments to jitted
# functions fails."
def test_reduce_jitted_reduce_fn(jit):
    import functools

    def foo(a, fn):
        return functools.reduce(fn, a, 0)

    jitting = True

    def add(x, y):
        assert is_jitting_with_raise() == jitting
        return x + y

    jfoo = jit(foo)
    jadd = jit(add)

    assert jfoo((1, 2, 3), jadd) == 6


def test_namedtuple_lookaside(jit):
    from collections import namedtuple

    typename = "MyNamedTuple"
    field_names = ("a", "b", "c")

    # Test returnign just the type {
    def f():
        return namedtuple(typename, field_names)

    jf = jit(f)

    jtype = jf()
    assert isinstance(jtype, type)
    assert jtype.__name__ == typename
    assert all(hasattr(jtype, field) for field in field_names)

    # Check module name
    import inspect

    assert jtype.__module__ == inspect.currentframe().f_globals["__name__"]
    # }

    # Test accessing elements {
    a = torch.rand(1)
    b = torch.rand(1)
    c = torch.rand(1)

    def f(a, b, c):
        nt = namedtuple(typename, field_names)
        obj = nt(a, b, c)
        return obj[0]

    jf = jit(f)

    assert f(a, b, c) is a
    assert jf(a, b, c) is a

    def f(a, b, c):
        nt = namedtuple(typename, field_names)
        obj = nt(a, b, c)
        return obj.a

    jf = jit(f)

    assert f(a, b, c) is a
    assert jf(a, b, c) is a
    # }


def test_calling_methods(jit):
    jitting = False

    class mycls:
        def __init__(self, v: int):
            assert is_jitting_with_raise() == jitting
            self.v = v

        def my_add(self, b):
            assert is_jitting_with_raise() == jitting
            return self.v + b

        @classmethod
        def my_add_class(cls, b):
            assert is_jitting_with_raise() == jitting
            o = cls(2)
            return o.v + b

        @staticmethod
        def my_add_static(b):
            assert is_jitting_with_raise() == jitting
            return 3 + b

    x = mycls(5)

    # these use LOAD_METHOD / CALL_METHOD
    def foo(x, a):
        assert is_jitting_with_raise() == jitting
        return x.my_add(a)

    def foo_class(x, a):
        assert is_jitting_with_raise() == jitting
        return x.my_add_class(a)

    def foo_static(x, a):
        assert is_jitting_with_raise() == jitting
        return x.my_add_static(a)

    # these use LOAD_ATTR / CALL_FUNCTION
    def bar(x, a):
        assert is_jitting_with_raise() == jitting
        meth = x.my_add(a)
        return meth

    def bar_class(x, a):
        assert is_jitting_with_raise() == jitting
        meth = x.my_add_class(a)
        return meth

    def bar_static(x, a):
        assert is_jitting_with_raise() == jitting
        meth = x.my_add_static(a)
        return meth

    jfoo = jit(foo)
    jfoo_class = jit(foo_class)
    jfoo_static = jit(foo_static)
    jbar = jit(bar)
    jbar_class = jit(bar_class)
    jbar_static = jit(bar_static)

    jitting = False
    fres = foo(x, 7)
    fres_class = foo_class(x, 7)
    fres_static = foo_static(x, 7)
    bres = bar(x, 7)
    bres_class = bar_class(x, 7)
    bres_static = bar_static(x, 7)

    jitting = True
    assert jfoo(x, 7) == fres
    assert jfoo_class(x, 7) == fres_class
    assert jfoo_static(x, 7) == fres_static
    assert jbar(x, 7) == bres
    assert jbar_class(x, 7) == bres_class
    assert jbar_static(x, 7) == bres_static


def test_wrapped_functions(jit):
    jitting = False

    def wrap(fn):
        assert is_jitting_with_raise() == jitting

        @wraps(fn)
        def inner(*args, **kwargs):
            assert is_jitting_with_raise() == jitting
            return fn(*args, **kwargs)

        return inner

    @wrap
    def foo(a, b):
        assert is_jitting_with_raise() == jitting
        return a + b

    jitting = False
    res = foo(3, 4)
    jitting = True
    jres = jit(foo)(3, 4)
    assert res == jres


def test_callable_classes(jit):
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


@pytest.mark.xfail(reason="Lookaside not triggered, requires further investigation. Delete the test above when fixed.")
def test_callable_classes_jitting(jit):
    jitting = False

    def foo(x, a):
        class mycls:
            assert is_jitting_with_raise() == jitting

            def __init__(self, v: int):
                assert is_jitting_with_raise() == jitting
                self.v = v

            def __call__(self, b):
                assert is_jitting_with_raise() == jitting
                return self.v + b

        x = mycls(5)

        return x(a)

    jitting = False
    res = foo(x, 7)

    jitting = True
    jfoo = jit(foo)
    jres = jfoo(x, 7)

    assert res == jres


def test_build_slice(jit):
    def foo(a, b):
        l = [0, 1, 2, 3, 4, 5, 6]
        return l[a:b], l[a:], l[:b], l[1:2:2], l[0:a:b]

    jfoo = jit(foo)

    assert jfoo(1, 4) == foo(1, 4)
    assert jfoo(0, -1) == foo(0, -1)


def test_format_value(jit):
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


@pytest.mark.xfail(
    reason="Need to implement builtin format, repr, and ascii lookasides. See help('FORMATTING'). When fixed, delete the test above."
)
def test_format_value_jitting(jit):
    jitting = False

    # Tests FVS_HAVE_SPEC and FVC_NONE
    def foo(a, b):
        return f"{a:3.2f}, {b:2.1f}"

    jfoo = jit(foo)

    assert jfoo(2.34, 123234.79289) == foo(2.34, 123234.79289)

    class mycls:
        def __repr__(self):
            assert is_jitting_with_raise() == jitting
            return "repr"

        def __str__(self):
            assert is_jitting_with_raise() == jitting
            return "str"

    # Tests FVC_NONE
    def foo(a, b):
        return f"{a}, {b}"

    x = mycls()

    jitting = False
    res = foo(x, "goodbye")
    jitting = True
    jfoo = jit(foo)
    jres = jfoo(x, "goodbye")

    assert res == jres

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


def test_import(jit):
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
        from .litgpt_model import Config

        return Config

    assert jit(foo)() is foo()

    def foo():
        # test relative import
        from . import litgpt_model

        return litgpt_model.Config

    assert jit(foo)() is foo()

    # reload is implemented using exec of the module
    from . import litgpt_model
    import importlib

    importlib.reload(litgpt_model)
    assert hasattr(litgpt_model, "GPT")


def test_locals_lookaside(jit):
    def foo():
        try:
            # Locals starts empty
            assert locals() == {}

            # Modifications to locals are preserved
            l = locals()
            assert locals()["l"] is not None, locals()
            l["a"] = 5

            # The identity of locals() is the same across calls
            assert l is locals(), (l, locals())

            # Deletions in localsplus are deleted in locals
            del l
            assert not "l" in locals().keys(), locals()

            # The objects stored in variables are the same as those in locals
            b = object()
            assert b is locals()["b"]

            # Modifying locals does not modify localsplus
            assert locals()["a"] == 5, locals()
            name_err = a == 5  # type: ignore (intentional)
            raise Exception("Unreachable.")
        except NameError as e:
            assert "not defined" in str(e)

    foo()
    jit(foo)()


def test_match_statement(jit):
    def foo():
        dct = {"a": 1, "b": 2, "z": 3}
        match dct:
            case {"a": a, "b": b, **rest}:
                assert rest == {"z": 3}
                return a + b
            case _:
                assert False

    jfoo = jit(foo)
    assert foo() == 3
    assert jfoo() == 3
    assert any(i.opname == "MATCH_KEYS" for i in jfoo._last_interpreted_instructions)
    assert any(i.opname == "MATCH_MAPPING" for i in jfoo._last_interpreted_instructions)
    if "COPY_DICT_WITHOUT_KEYS" in dis.opmap.keys():
        assert any(i.opname == "COPY_DICT_WITHOUT_KEYS" for i in jfoo._last_interpreted_instructions)

    # Test MATCH_SEQUENCE
    def bar():
        lst = [1, 2, 3]
        match lst:
            case [a, b, *rest]:
                assert rest == [3]
                return a + b
            case _:
                assert False

    jbar = jit(bar)
    assert bar() == 3
    assert jbar() == 3
    assert any(i.opname == "MATCH_SEQUENCE" for i in jbar._last_interpreted_instructions)
    assert any(i.opname == "GET_LEN" for i in jbar._last_interpreted_instructions)
    assert any(i.opname == "UNPACK_EX" for i in jbar._last_interpreted_instructions)


def test_class_match_statement(jit):
    class Cls:
        __match_args__ = ("a", "b")

        def __init__(self, a, b):
            self.a = a
            self.b = b

    def foo():
        c = Cls(1, 2)
        match c:
            case Cls(a, b):
                assert a == 1
                assert b == 2
                return a + b
            case _:
                assert False

    jfoo = jit(foo)
    assert foo() == 3
    assert jfoo() == 3
    assert any(i.opname == "MATCH_CLASS" for i in jfoo._last_interpreted_instructions)


def test_match_fallthrough(jit):
    def foo():
        dct = {"a": 1, "b": 2}
        match dct:
            case 1:
                assert False
            case "str":
                assert False
            case [a, b]:
                assert False
            case {"y": y, "z": z}:
                assert False

        match dct:
            case 1:
                assert False
            case [a, b]:
                assert False
            case "str":
                assert False
            case {"y": y, "z": z}:
                assert False
            case _:
                return True
        assert False

    jfoo = jit(foo)
    assert foo() is True
    assert jfoo() is True


@pytest.mark.xfail(reason='"exec() and eval() lookaside ignores locals()"')
def test_exec_import_star(jit):
    # Assert that we can actually generate the instruction
    to_exec = "from itertools import *"
    compiled = compile(to_exec, "<string>", "exec")
    assert any(i.opname == "IMPORT_STAR" for i in dis.get_instructions(compiled))

    # Run with globals dict of current frame,
    def foo():
        exec_globals = globals().copy()
        exec(to_exec, exec_globals)
        assert list(exec_globals["repeat"](None, 2)) == [None, None]  # type: ignore

    jfoo = jit(foo)
    jfoo()


def test_import_star_module(jit):
    def foo():
        c = compile("from thunder.tests.module_example import *", "<string>", "exec")
        exec(c, globals())
        assert "_returns_three" not in globals().keys()
        assert "returns_two" in globals().keys()
        return globals()["returns_five"]()

    assert foo() == 5
    assert foo() == jit(foo)()


def test_unhashable_lookaside(jit):
    def fn():
        import weakref

        ws = weakref.WeakSet()
        wr = weakref.ref(ws)
        wr()

    jit(fn)()


def test_enumerate_lookaside(jit):
    jitting = False

    class mycls:
        def __init__(self, val):
            self.list = [val] * 3

        def __iter__(self):
            assert is_jitting_with_raise() == jitting
            return self.list.__iter__()

    def foo(a, start=0):
        return list(enumerate(a, start))

    o = mycls(2)
    jfoo = jit(foo)

    jitting = False
    res1 = foo(o)
    res2 = foo([1, 2, 3], 8)
    res3 = foo("mystr", True)
    res4 = foo(o, -3)

    jitting = True
    jres1 = jfoo(o)
    jres2 = jfoo([1, 2, 3], 8)
    jres3 = jfoo("mystr", True)
    jres4 = jfoo(o, -3)
    assert res1 == jres1
    assert res2 == jres2
    assert res3 == jres3
    assert res4 == jres4

    with pytest.raises(TypeError, match="object is not iterable$"):
        jfoo(12)

    class myclsnotiterable:
        def __init__(self):
            pass

    with pytest.raises(TypeError, match="object is not iterable$"):
        jfoo(myclsnotiterable())
    with pytest.raises(TypeError, match="object cannot be interpreted as an integer$"):
        jfoo(o, -2.5)


def test_len_lookaside(jit):
    jitting = False

    class mycls:
        def __init__(self, v=5):
            self.v = v

        def __len__(self):
            assert is_jitting_with_raise() == jitting
            return self.v

    def foo(a):
        return len(a)

    jfoo = jit(foo)

    o = mycls()

    jitting = False
    res1 = foo(o)
    res2 = foo([1, 2, 3])
    res3 = foo("mystr")

    jitting = True
    jres1 = jfoo(o)
    jres2 = jfoo([1, 2, 3])
    jres3 = jfoo("mystr")
    assert res1 == jres1
    assert res2 == jres2
    assert res3 == jres3

    o = mycls(-1)

    with pytest.raises(ValueError, match="__len__\(\) should return >= 0"):
        jfoo(o)

    o = mycls(0.42)

    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        jfoo(o)

    class myclswithoutlen:
        def __init__(self):
            pass

    o = myclswithoutlen()

    with pytest.raises(TypeError, match="object of type 'myclswithoutlen' has no len()"):
        jfoo(o)

    class myclsnegfloat:
        def __len__(self):
            assert is_jitting_with_raise() == jitting
            return -0.6

    o = myclsnegfloat()
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        jfoo(o)


def test_any_lookaside(jit):
    jitting = False

    def foo(a):
        return any(a)

    jfoo = jit(foo)

    assert jfoo([1, 2, 3]) == foo([1, 2, 3])

    with pytest.raises(TypeError):
        jfoo(True)

    class myitercontainer:
        def __init__(self, val):
            self.list = [val] * 3

        def __iter__(self):
            assert is_jitting_with_raise() == jitting
            return self.list.__iter__()

    o = myitercontainer(True)
    jitting = False
    res = foo(o)
    jitting = True
    jres = jfoo(o)
    assert res == jres

    o = myitercontainer(False)
    jitting = False
    res = foo(o)
    jitting = True
    jres = jfoo(o)
    assert res == jres


def test_generator(jit):
    jitting = False

    def my_generator_1():
        assert is_jitting_with_raise() == jitting
        yield from range(5)
        assert is_jitting_with_raise() == jitting

    def my_generator_2():
        assert is_jitting_with_raise() == jitting
        yield 1
        assert is_jitting_with_raise() == jitting
        val = 1
        while True:
            assert is_jitting_with_raise() == jitting
            val = yield 2 * val

    jgen_1 = jit(my_generator_1)
    jgen_2 = jit(my_generator_2)

    jitting = True
    actual = list(jgen_1())
    jitting = False
    expected = list(my_generator_1())
    assert actual == expected

    jitting = True
    j_run_gen = jgen_2()
    actual = [j_run_gen.send(x) for x in [None, 1, 2, 3]]
    jitting = False
    run_gen = my_generator_2()
    expected = [run_gen.send(x) for x in [None, 1, 2, 3]]
    assert actual == expected


def test_binary_operations(jit):
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


def test_binary_op_on_types(jit):
    def fn():
        return int | None

    assert jit(fn)() == fn()


def test_get_and_for_iter(jit):
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


def test_iter_lookaside_and_sentinel(jit):
    def foo():
        gen = iter([1, 2, 3, None, "Unreachable"])
        for x in iter(lambda: next(gen), None):
            assert x != "Unreachable"

        sentinel = object()
        l = [1, 2, 3, "Unreachable"]
        for x in iter(lambda: l.pop(0) if len(l) != 1 else sentinel, sentinel):
            assert x != "Unreachable"
        assert l == ["Unreachable"]

    foo()
    jit(foo)()


@pytest.mark.xfail(reason="We don't currently return the exact iterator types as Python does.")
def test_iter_lookaside_types(jit):
    class IterableExample(Iterable):
        def __iter__(self):
            return iter([1, 2, 3])

    class NonIterableExample:
        def __iter__(self):
            return iter([1, 2, 3])

    class SequenceExample(Sequence):
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            if idx >= len(self):
                raise IndexError
            return idx + 1

    class NonSequenceExample:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            if idx >= len(self):
                raise IndexError
            return idx + 1

    def foo():
        example_classes = (IterableExample, NonIterableExample, SequenceExample, NonSequenceExample)
        for C in example_classes:
            it = iter(C())
            li = list(it)
            assert li == [1, 2, 3]

    foo()
    jit(foo)()

    def seqiter():
        return iter(NonSequenceExample())

    def calliter():
        sentinel = object()
        return iter(lambda: sentinel, sentinel)

    assert type(seqiter()) is type(jit(seqiter)())
    assert type(calliter()) is type(jit(calliter)())


def test_iter_lookaside_types_jitting(jit):
    jitting = False

    class IterableExample(Iterable):
        def __iter__(self):
            assert is_jitting_with_raise() == jitting
            return iter([1, 2, 3])

    class NonIterableExample:
        def __iter__(self):
            assert is_jitting_with_raise() == jitting
            return iter([1, 2, 3])

    class SequenceExample(Sequence):
        def __len__(self):
            assert is_jitting_with_raise() == jitting
            return 3

        def __getitem__(self, idx):
            assert is_jitting_with_raise() == jitting
            if idx >= len(self):
                raise IndexError
            return idx + 1

    class NonSequenceExample:
        def __len__(self):
            # TODO: list init is not looked aside in non-tracking
            assert jit is interpret_no_tracking or is_jitting_with_raise() == jitting
            return 3

        def __getitem__(self, idx):
            # TODO: list init is not looked aside in non-tracking
            assert jit is interpret_no_tracking or is_jitting_with_raise() == jitting
            if idx >= len(self):
                raise IndexError
            return idx + 1

    def foo():
        example_classes = (IterableExample, NonIterableExample, SequenceExample, NonSequenceExample)
        for C in example_classes:
            it = iter(C())
            li = list(it)
            assert li == [1, 2, 3]

    jitting = False
    foo()
    jitting = True
    jit(foo)()

    def seqiter():
        return iter(NonSequenceExample())

    def calliter():
        sentinel = object()
        return iter(lambda: sentinel, sentinel)

    jitting = False
    sres = type(seqiter())
    cres = type(calliter())

    jitting = True
    jsres = type(jit(seqiter)())
    jcres = type(jit(calliter)())

    # assert sres is jsres
    # assert cres is jcres


def test_unary_not(jit):
    def foo(a):
        return not a

    jfoo = jit(foo)

    assert jfoo(False) == foo(False)
    assert jfoo(0) == foo(0)
    assert jfoo(3.14) == foo(3.14)
    assert jfoo(1j) == foo(1j)

    jitting = False

    class mycls(int):
        def __init__(self, v):
            self.v = v

        def __bool__(self) -> bool:
            assert is_jitting_with_raise() == jitting
            return self.v % 2 == 0

    cases = (
        (mycls(1), 1),
        (mycls(2), 2),
    )

    for o, case in cases:
        jitting = False
        res = foo(case % 2 == 0)
        jitting = True
        jres = jfoo(o)
        assert res == jres

    assert jfoo([]) == foo([])
    assert jfoo([1, 2]) == foo([2, 3])


def test_unary_neg_invert(jit):
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


def test_unpack_ex(jit):
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


def test_list_to_tuple(jit):
    def ltt():
        return (*[1, 2, 3],)

    assert any(i.opname == "LIST_TO_TUPLE" for i in dis.get_instructions(ltt))
    assert jit(ltt)() == ltt()


def test_annotations(jit):
    annotation_executed = inner_executed = False
    jitting = False

    def annotation(fn: Callable[[], int]) -> Callable[[], Callable[[], int]]:
        assert is_jitting_with_raise() == jitting
        nonlocal annotation_executed
        annotation_executed = True
        return lambda: fn

    def foo():
        assert is_jitting_with_raise() == jitting

        @annotation
        def inner() -> int:
            assert is_jitting_with_raise() == jitting

            nonlocal inner_executed
            inner_executed = True
            return 5

        return inner()()

    assert foo() == 5
    assert annotation_executed and inner_executed, (annotation_executed, inner_executed)

    annotation_executed = inner_executed = False
    jitting = True
    assert jit(foo)() == 5
    assert annotation_executed and inner_executed, (annotation_executed, inner_executed)


def test_use_of_deleted_raises_correctly(jit):
    def foo(a):
        b = a
        del b
        assert a == 5
        c = b + a
        return a

    jfoo = jit(foo)

    with pytest.raises(UnboundLocalError, match=r".*local variable 'b' referenced before assignment.*"):
        jfoo(5)


def test_delete_fast(jit):
    def foo(a):
        b = a
        del b
        assert a == 5
        c = b + a
        return a

    jfoo = jit(foo)
    with pytest.raises(UnboundLocalError, match="'b'"):
        jfoo(5)


def test_delete_global(jit):
    global x
    x = 5

    def foo(a):
        global x
        y = x
        del x
        assert y == 5
        return a + x

    jfoo = jit(foo)

    with pytest.raises(NameError):
        jfoo(5)


x = 7


def test_store_global(jit):
    def foo(a):
        global x
        x = a
        assert x == a

    jfoo = jit(foo)

    jfoo(6)
    assert x == 6


def test_bool_conversion(jit):
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

    jitting = False

    # Checks dunder bool handling (by default classes are true)
    class mycls:
        def __bool__(self):
            assert is_jitting_with_raise() == jitting
            return False

    x = mycls()

    jitting = False
    res = foo(x)
    jitting = True
    jres = jfoo(x)
    assert res == jres

    # Classes that define dunder len and not dunder bool use dunder len for their bool() conversion
    class mycls:
        def __len__(self):
            assert is_jitting_with_raise() == jitting
            return 0

    x = mycls()

    jitting = False
    res = foo(x)
    jitting = True
    jres = jfoo(x)
    assert res == jres

    class mycls:
        def __len__(self):
            assert is_jitting_with_raise() == jitting
            return 1

    x = mycls()

    jitting = False
    res = foo(x)
    jitting = True
    jres = jfoo(x)
    assert res == jres


def test_store_attr(jit):
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


@pytest.mark.xfail(
    reason="Lookaside not triggered, need to implement setattr lookaside (Objects/object.c:1029). "
    "Also implement and test delattr (PyObject_SetAttr(obj, NULL)). Delete the test above when fixed."
)
def test_store_attr_jit(jit):
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

    jitting = True

    # Checks that dunder setattr is called
    class mycls:
        def __setattr__(self, name, value):
            # NOTE This can't call __setattr__ again (not even indirectly, like through self.bar = value)
            #   because that would cause infinite recursion
            # This avoids the infinite recursion by calling object's dunder setattr, which isn't hooked

            assert is_jitting_with_raise() == jitting
            super().__setattr__("bar", value)

    x = mycls()

    jfoo(x, 5)
    assert x.bar == 5


def test_builtin_getattr(jit):
    x = 5
    assert x.__add__ == jit(getattr)(x, "__add__")


def test_builtin_getattr_str_subclass(jit):
    x = 5

    class S(str):
        pass

    s = S("__add__")
    add1 = getattr(x, s)
    add2 = jit(getattr)(x, s)
    assert add1 == add2


def test_simple_attribute(jit):
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


def test_dunder_getattr(jit):
    history = []

    class X:
        def __getattr__(self, name):
            history.append(f"X.__getattr__ {is_jitting_with_raise()}")
            return 1

    def foo():
        return X().a

    assert foo() == jit(foo)() == 1
    assert tuple(history) == ("X.__getattr__ False", "X.__getattr__ True")


@pytest.mark.skip(reason="__getattribute__ support is not yet implemented.")
def test_dunder_getattribute(jit):
    history = []

    class MyClass:
        def __getattribute__(self, name):
            history.append(f"__getattr__ {is_jitting_with_raise()}")
            return 1

    def foo():
        x = MyClass()
        x.a = 2  # __getattribute__ will take precedence
        return x.a

    assert foo() == jit(foo)() == 1
    assert tuple(history) == ("__getattr__ False", "__getattr__ True")


def test_property(jit):
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting_with_raise()}")
            return 1

        @property
        def lazy_x(self):
            history.append(f"lazy_x {is_jitting_with_raise()}")
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


def test_property_with_setter(jit):
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting_with_raise()}")
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


@pytest.mark.skip(reason=".setter support is not yet implemented.")
def test_property_with_instrumented_setter(jit):
    history = []

    class MyClass:
        @property
        def x(self):
            history.append(f"x {is_jitting_with_raise()}")
            return self._x

        @x.setter
        def x(self, value) -> None:
            history.append(f"x.setter {is_jitting_with_raise()}")
            self._x = value

    my_class = MyClass()
    my_class.__dict__["x"] = 8  # Make sure property takes precedence
    history.clear()

    def foo():
        my_class.x = 5
        return my_class.x

    assert foo() == jit(foo)() == 5
    assert tuple(history) == ("x False", "x.setter False", "x True", "x.setter True")


def test_compare(jit):
    # uses ROT_THREE in Python 3.10
    def fn(a):
        return 2 <= a <= 4

    jfn = jit(fn)
    for a in (1, 3, 5):
        assert fn(a) == jfn(a)


def test_comprehension(jit):
    def foo():
        return tuple([i for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_nested_comprehension(jit):
    def foo():
        return tuple([[j for j in enumerate(range(i))] for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_comprehension_nonlocal(jit):
    def foo():
        counter = 0

        def increment():
            nonlocal counter
            counter = counter + 1
            return counter

        return tuple([i + increment() for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_comprehension_nonlocal_inplace(jit):
    def foo():
        counter = 0

        def increment():
            nonlocal counter
            counter += 1
            return counter

        return tuple([i + increment() for i in range(10)])

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_unimpl_inplace(jit):
    class C:
        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            return C(self.value + other.value)

    def foo():
        a = C(3)
        b = C(5)
        ida1 = id(a)
        a += b
        ida2 = id(a)
        assert ida1 != ida2
        assert a.value == 8

    foo()
    jit(foo)()


def test_unsupported_operator(jit):
    def foo():
        return 2 @ 3  # type: ignore

    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for @:"):
        foo()
    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for @:"):
        jit(foo)()


def test_set_creation(jit):
    def foo():
        return {1, *[2, 3]}

    jfoo = jit(foo)
    assert foo() == jfoo()


def test_contains(jit):
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


def test_contains_custom_containers(jit):
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


def test_name_opcodes_and_print_expr(jit):
    from types import FunctionType

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


def test_displayhook(jit):
    from contextlib import redirect_stdout
    import io
    import code

    # TODO: Implement the lookaside for exec(). Under the hood, `code.InteractiveInterpreter().runsource('5;6;7')``
    # just compiles the string and calls exec(), plus a little bit of error handling.
    # I'm not entirely convinced that the PRINT_EVAL is going through our system at the moment, but
    # it for sure would with an exec() lookaside. I'm also not sure what makes InteractiveInterpreter
    # interactive. It isn't *actually* in interactive mode. So, why is PRINT_EXPR in the interpreted
    # instructions? A mystery.

    py_redirect = io.StringIO()
    with redirect_stdout(py_redirect):
        # Avoid clobbering this interpreter's display hook, and ensure it's interactive.
        # Why is this necessary?
        interpreter = code.InteractiveInterpreter()

        def smt(s):
            interpreter.runsource(s)

        smt("from thunder.core.interpreter import interpret")
        smt(
            """
def foo():
    import sys
    import code
    sys.displayhook = lambda x: print('redirected', x) if x is not None else None
    # Create a new interpreter that's in interactive mode inside the jit
    code.InteractiveInterpreter().runsource('5;6;7')
    sys.displayhook = sys.__displayhook__
    print('Reset.')
"""
        )
        smt("jfoo = interpret(foo)")
        smt("jfoo()")
        smt("assert any(i.opname == 'PRINT_EXPR' for i in jfoo._last_interpreted_instructions)")

    py_out: str = py_redirect.getvalue()
    assert py_out == "redirected 5\nredirected 6\nredirected 7\nReset.\n", py_out


def test_load_build_class(jit):
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


def test_with(jit):
    jitting = False

    class CtxMgr:
        def __init__(self, l):
            assert is_jitting_with_raise() == jitting
            self.l = l

        def __enter__(self):
            assert is_jitting_with_raise() == jitting
            self.l.append("enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            assert is_jitting_with_raise() == jitting
            self.l.append((str(exc_type), str(exc_val)))

    def fn(should_raise: bool = False):
        l = []
        with CtxMgr(l) as ctx:
            assert is_jitting_with_raise() == jitting
            ctx.l.append("within")
            if should_raise:
                raise RuntimeError("test", l)
            return l

    jitting = False
    res = fn()
    jitting = True
    jfn = jit(fn)
    jres = jfn()
    assert res == jres

    with pytest.raises(RuntimeError) as exc_expected:
        jitting = False
        fn(should_raise=True)
    with pytest.raises(RuntimeError) as exc_actual:
        jitting = True
        jfn(should_raise=True)

    assert exc_expected.value.args[1] == exc_actual.value.args[1]


def test_async_with(jit):
    jitting = False

    class ACtxMgr:
        def __init__(self, l):
            assert is_jitting_with_raise() == jitting
            self.l = l

        async def __aenter__(self):
            assert is_jitting_with_raise() == jitting
            self.l.append("enter")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            assert is_jitting_with_raise() == jitting
            self.l.append((str(exc_type), str(exc_val)))

    async def fn(should_raise: bool = False):
        l = []
        async with ACtxMgr(l) as ctx:
            assert is_jitting_with_raise() == jitting
            ctx.l.append("within")
            if should_raise:
                raise RuntimeError("test", l)
            return l

    jfn = jit(fn)

    import asyncio

    jitting = False
    res = asyncio.run(fn())
    jitting = True
    jres = asyncio.run(jfn())
    assert res == jres

    with pytest.raises(RuntimeError) as exc_expected:
        jitting = False
        asyncio.run(fn(should_raise=True))
    with pytest.raises(RuntimeError) as exc_actual:
        jitting = True
        asyncio.run(jfn(should_raise=True))

    assert exc_expected.value.args[1] == exc_actual.value.args[1]


def test_async_for(jit):
    jitting = False

    async def async_gen():
        assert is_jitting_with_raise() == jitting
        for i in range(5):
            assert is_jitting_with_raise() == jitting
            yield i

    async def fn():
        def it2(i):
            assert is_jitting_with_raise() == jitting
            return i * 2

        assert is_jitting_with_raise() == jitting
        l = [it2(i) async for i in async_gen()]
        async for i in async_gen():
            assert is_jitting_with_raise() == jitting
            l.append(i * 3)
        return l

    import asyncio

    jitting = False
    res = asyncio.run(fn())
    jitting = True
    jfn = jit(fn)
    jres = asyncio.run(jfn())
    assert res == jres


def test_super(jit):
    jitting = False

    class A:
        def foo(self):
            assert is_jitting_with_raise() == jitting
            return f"Hello {type(self)} {__class__}"

        @classmethod
        def bar(self):
            assert is_jitting_with_raise() == jitting
            return f"Hello {type(self)} {__class__}"

    class B(A):
        def foo(self):
            assert is_jitting_with_raise() == jitting
            return super().foo()

        @classmethod
        def bar(self):
            assert is_jitting_with_raise() == jitting
            return super().bar()

    class C(A):
        def foo(self):
            assert is_jitting_with_raise() == jitting
            return super().foo()

    def foo():
        b = B()
        c = C()
        return (b.foo(), c.foo())

    jitting = False
    res = foo()
    jitting = True
    jres = jit(foo)()
    assert res == jres

    def bar():
        b = B()
        c = C()
        super(b, C)

    with pytest.raises(TypeError) as exc_expected:
        jitting = False
        bar()
    with pytest.raises(TypeError) as exc_actual:
        jitting = True
        jit(bar)()
    # Python 3.11 improved the grammar, so do we
    assert str(exc_expected.value).replace("be type", "be a type") == str(exc_actual.value)

    def baz():
        b = B()
        return b.bar()

    jitting = False
    res = baz()
    jitting = True
    jres = jit(baz)()
    assert res == jres


def test_is_jitting_with_raise(jit):
    def foo():
        return is_jitting_with_raise()

    assert not foo()
    assert jit(foo)()


def test_autograd_function(jit):
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


def test_torch_autocast_nograd(jit):
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


def test_module_hooks(jit):
    def cook_hook(l, name):
        def fn(*args):
            l.append((name, thunder.core.interpreter.is_jitting_with_raise()))

        return fn

    m = torch.nn.Linear(4, 4)
    l = []
    handles = []

    try:
        handles.append(torch.nn.modules.module.register_module_forward_hook(cook_hook(l, "global forward")))
        handles.append(torch.nn.modules.module.register_module_forward_pre_hook(cook_hook(l, "global forward pre")))
        handles.append(torch.nn.modules.module.register_module_full_backward_hook(cook_hook(l, "global full backward")))
        handles.append(
            torch.nn.modules.module.register_module_full_backward_pre_hook(cook_hook(l, "global full backward pre"))
        )
        handles.append(m.register_forward_hook(cook_hook(l, "module forward")))
        handles.append(m.register_forward_pre_hook(cook_hook(l, "module forward pre")))
        handles.append(m.register_full_backward_hook(cook_hook(l, "module full backward")))
        handles.append(m.register_full_backward_pre_hook(cook_hook(l, "module full backward pre")))

        x = torch.randn(3, 4)

        jm = jit(m)
        y = jm(x)
        y.sum().backward()

        # Find the hook registration in the history the normal way
        found = False
        for item in jm._last_interpreted_history:
            if (not isinstance(item, dict)) or (item["kind"] != "Opaque"):
                continue
            _fn = item["fn"]
            if _fn == torch._C._FunctionBase.register_hook:  # type: ignore
                found = True
                break

        assert found

        # Redirect print_last_interpreted_history from stdout to a string, and assert that it's in there.
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_last_interpreted_history(jm, use_colors=False, indent=False)

        match_against = "Opaque call to <method 'register_hook' of 'torch._C._FunctionBase' objects> with name _FunctionBase.register_hook"
        assert match_against in buf.getvalue()
        buf.close()

        jit_l = l[:]
        l.clear()
        y = m(x)
        y.sum().backward()

        assert len(jit_l) == len(l)
        for (jn, jj), (pn, pj) in zip(jit_l, l):
            assert jn == pn
            # we expect forward to be execute via the jit, backward
            # assert bool(jj), f"{jn} {jj=}"
    finally:
        for h in handles:
            h.remove()


def test_eval_exec_exception(jit):
    def fn_eval():
        return eval("1/0")

    def fn_exec():
        exec("a = 1/0")

    with pytest.raises(ZeroDivisionError):
        jit(fn_eval)()
    with pytest.raises(ZeroDivisionError):
        jit(fn_exec)()

    # This should raise inside opaque `compile`
    def fn_eval():
        return eval("x = lambda a: a + 1")

    with pytest.raises(SyntaxError):
        jit(fn_eval)()


def test_is_jitting_with_raise_opaque(jit):
    def foo():
        return tuple(map(lambda _: is_jitting_with_raise(), range(3)))

    assert foo() == (False, False, False)
    with pytest.raises(InterpreterError):
        jit(foo)()


def test_is_jitting_opaque(jit):
    @make_opaque
    def foo():
        assert not is_jitting()
        return 1

    assert foo() == jit(foo)()


def test_exception_in_list_init(jit):
    def foo(l):
        for i in l:
            yield i

    def bar():
        return list(foo(2))

    with pytest.raises(TypeError):
        bar()

    with pytest.raises(TypeError):
        jit(bar)()


#
# Network tests
#


def test_nanogpt_mlp(jit):
    from thunder.benchmarks import NanoGPTMLPBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTMLPBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt_csa(jit):
    from thunder.benchmarks import NanoGPTCSABenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTCSABenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt_block(jit):
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_nanogpt(jit):
    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["test"])
    bench = NanoGPTBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()
    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


def test_litgpt(jit):
    from thunder.benchmarks import LitGPTBenchmark
    from thunder.tests.litgpt_model import Config

    cfg: Config = Config.from_name("gpt-neox-like")
    bench = LitGPTBenchmark(config=cfg, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))
