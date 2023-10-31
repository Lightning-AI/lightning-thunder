import copy
import dis
import itertools
import re
import sys
import textwrap

from thunder.core.script import parse
import thunder.core.script.protograph as protograph
from thunder.core.script.protograph_passes import apply_protograph_passes
import thunder.core.script.python_ir_data as python_ir_data
from thunder.core.utils import enable_debug_asserts

import pytest

enable_debug_asserts()
TEST_CASES = []


def add_parse_test(spec: str | None = None):
    def wrapper(f):
        TEST_CASES.append((f, spec))
        return f

    return wrapper


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[x, 1: CONST]
  BINARY_ADD . . . . . .  (x, 1) -> v0
  RETURN_VALUE . . . . .  (v0) ->
"""
)
def simple_fn(x):
    return x + 1


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[x]
  POP_JUMP_IF_FALSE . . . (x) ->
      -> 1, 2(Jump)

Block 1:  [] => []
  LOAD[x, 2: CONST]
  INPLACE_ADD . . . . . . (x, 2) -> v0
  STORE[x]
  JUMP_ABSOLUTE*
      -> 2(Jump)

Block 2:  [] => []
  LOAD[x, 1: CONST]
  BINARY_ADD . . . . . .  (x, 1) -> v0
  RETURN_VALUE . . . . .  (v0) ->             
"""
)
def simple_if_fn(x):
    if x:
        x += 2
    return x + 1


@add_parse_test(
    r"""
Block 0:  [] => [layer_0, x]
  LOAD[mask]
  LOAD_METHOD . . . . . . . . . . . . . . . .  (mask) -> v0, v1
  CALL_METHOD . . . . . . . . . . . . . . . .  (v0, v1) -> v2
  STORE[has_mask]
  LOAD[layer_0, x, has_mask]
  POP_JUMP_IF_FALSE . . . . . . . . . . . . .  (v2) ->
      -> 1, 2(Jump)

Block 1:  [⓵ , ⓶ ] => [⓵ , ⓶ , mask]
  LOAD[mask]
  JUMP_FORWARD
      -> 3(Jump)

Block 2:  [⓵ , ⓶ ] => [⓵ , ⓶ , None]
  LOAD[None: CONST]
  JUMP_ABSOLUTE*
      -> 3(Jump)

Block 3:  [⓵ , ⓶ , ⓷ ] => [layer_1, v0]
  CALL_FUNCTION . . . . . . . . . . . . . . .  (⓵ , ⓶ , ⓷ ) -> v0
  STORE[x]
  LOAD[layer_1, x, has_mask]
  POP_JUMP_IF_FALSE . . . . . . . . . . . . .  (has_mask) ->
      -> 4, 5(Jump)

Block 4:  [⓵ , ⓶ ] => [v0]
  LOAD[mask]
  CALL_FUNCTION . . . . . . . . . . . . . . .  (⓵ , ⓶ , mask) -> v0
  STORE[x]
  LOAD[x]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 5:  [⓵ , ⓶ ] => [v0]
  LOAD[None: CONST]
  CALL_FUNCTION . . . . . . . . . . . . . . .  (⓵ , ⓶ , None) -> v0
  STORE[x]
  LOAD[x]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 6:  [⓵ ] => []
  RETURN_VALUE* . . . . . . . . . . . . . . .  (⓵ ) ->
"""
)
def cse_candidate(x, mask, layer_0, layer_1):
    has_mask = mask.any()
    x = layer_0(x, mask if has_mask else None)
    x = layer_1(x, mask if has_mask else None)
    return x


@add_parse_test(
    r"""
Block 0:  [] => [v1]
  LOAD[range: GLOBAL, 4: CONST]
  CALL_FUNCTION . . . . . . . . . .  (range, 4) -> v0
  GET_ITER . . . . . . . . . . . . . (v0) -> v1
  JUMP_ABSOLUTE*
      -> 1(Jump)

Block 1:  [⓵ ] => [v0, v1]
  FOR_ITER . . . . . . . . . . . . . (⓵ ) -> v0, v1
      -> 2, 4(Jump)

Block 2:  [⓵ , ⓶ ] => [⓵ ]
  STORE[_]
  LOAD[x, y]
  INPLACE_ADD . . . . . . . . . . .  (x, y) -> v0
  STORE[x]
  JUMP_ABSOLUTE
      -> 1(Jump)

Block 3:  [] => []
  LOAD[x]
  RETURN_VALUE . . . . . . . . . . . (x) ->

Block 4:  [⓵ , ⓶ ] => []
  POP_TOP* . . . . . . . . . . . . . (⓶ ) ->
  POP_TOP* . . . . . . . . . . . . . (⓵ ) ->
  JUMP_ABSOLUTE*
      -> 3(Jump)
"""
)
def simple_loop_fn(x, y):
    # NOTE:
    #   preprocessing doesn't understand that `range(4)` guarantees at least
    #   one pass through the loop which is why the return is `U[x_1, x]`
    #   instead of `x_1`.
    for _ in range(4):
        x += y
    return x


@add_parse_test(
    r"""
Block 0:  [] => [v1]
  LOAD[range: GLOBAL, 10: CONST]
  CALL_FUNCTION . . . . . . . . . . . (range, 10) -> v0
  GET_ITER . . . . . . . . . . . . .  (v0) -> v1
  JUMP_ABSOLUTE*
      -> 1(Jump)

Block 1:  [⓵ ] => [v0, v1]
  FOR_ITER . . . . . . . . . . . . .  (⓵ ) -> v0, v1
      -> 2, 7(Jump)

Block 2:  [⓵ , ⓶ ] => [⓵ ]
  STORE[i]
  LOAD[i, x]
  COMPARE_OP . . . . . . . . . . . .  (⓶ , x) -> v0
  POP_JUMP_IF_FALSE . . . . . . . . . (v0) ->
      -> 3, 4(Jump)

Block 3:  [⓵ ] => [i]
  POP_TOP . . . . . . . . . . . . . . (⓵ ) ->
  LOAD[i]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 4:  [⓵ ] => [⓵ ]
  JUMP_ABSOLUTE
      -> 1(Jump)

Block 5:  [] => [i]
  LOAD[i]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 6:  [⓵ ] => []
  RETURN_VALUE* . . . . . . . . . . . (⓵ ) ->

Block 7:  [⓵ , ⓶ ] => []
  POP_TOP* . . . . . . . . . . . . .  (⓶ ) ->
  POP_TOP* . . . . . . . . . . . . .  (⓵ ) ->
  JUMP_ABSOLUTE*
      -> 5(Jump)
"""
)
def loop_with_break(x):
    for i in range(10):
        if i > x:
            break
    return i


@add_parse_test(
    r"""
Block 0:  [] => [v0]
  LOAD[k]
  GET_ITER . . . . . . . . . . .  (k) -> v0
  JUMP_ABSOLUTE*
      -> 1(Jump)

Block 1:  [⓵ ] => [v0, v1]
  FOR_ITER . . . . . . . . . . .  (⓵ ) -> v0, v1
      -> 2, 7(Jump)

Block 2:  [⓵ , ⓶ ] => [⓵ ]
  STORE[_]
  LOAD[x, 1: CONST]
  INPLACE_ADD . . . . . . . . . . (x, 1) -> v0
  STORE[x]
  LOAD[done_fn: GLOBAL, k]
  CALL_FUNCTION . . . . . . . . . (done_fn, k) -> v1
  POP_JUMP_IF_FALSE . . . . . . . (v1) ->
      -> 3, 4(Jump)

Block 3:  [⓵ ] => [v0]
  LOAD[x, 2: CONST]
  INPLACE_MULTIPLY . . . . . . .  (x, 2) -> v0
  STORE[x]
  POP_TOP . . . . . . . . . . . . (⓵ ) ->
  LOAD[x]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 4:  [⓵ ] => [⓵ ]
  JUMP_ABSOLUTE
      -> 1(Jump)

Block 5:  [] => [v0]
  LOAD[x, 1: CONST]
  INPLACE_SUBTRACT . . . . . . .  (x, 1) -> v0
  STORE[x]
  LOAD[x]
  JUMP_ABSOLUTE*
      -> 6(Jump)

Block 6:  [⓵ ] => []
  RETURN_VALUE* . . . . . . . . . (⓵ ) ->

Block 7:  [⓵ , ⓶ ] => []
  POP_TOP* . . . . . . . . . . .  (⓶ ) ->
  POP_TOP* . . . . . . . . . . .  (⓵ ) ->
  JUMP_ABSOLUTE*
      -> 5(Jump)
"""
)
def loop_with_else(x, k):
    for _ in k:
        x += 1

        # Without this break we will always execute the `else` branch, so Python
        # will optimize out the jump and inline the subtraction after the loop.
        #
        # And if we don't gate it behind a conditional it the parser is smart
        # enough to elide the jump to the start of the loop.
        if done_fn(k):
            x *= 2
            break
    else:
        x -= 1

    return x


def done_fn(_):
    return True


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[x, 2: CONST]
  BUILD_TUPLE . . . . . . (x, 2) -> v0
  STORE[t]
  LOAD[t]
  UNPACK_SEQUENCE . . . . (v0) -> v1, v2
  STORE[a, _]
  LOAD[a]
  RETURN_VALUE . . . . .  (v2) ->
"""
)
def tuple_fold(x):
    t = (x, 2)
    a, _ = t
    return a


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[x, 2: CONST]
  BUILD_TUPLE . . . . . . (x, 2) -> v0
  STORE[t]
  LOAD[t]
  UNPACK_EX . . . . . . . (v0) -> v1, v2
  STORE[a, _]
  LOAD[a]
  RETURN_VALUE . . . . .  (v2) ->
"""
)
def tuple_fold_ex(x):
    t = (x, 2)
    a, *_ = t
    return a


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[aaa: CONST, x, 1: CONST]
  BUILD_TUPLE . . . . . . . . . . .  (x, 1) -> v0
  LOAD[2: CONST]
  BUILD_TUPLE . . . . . . . . . . .  (v0, 2) -> v1
  BUILD_TUPLE . . . . . . . . . . .  (aaa, v1) -> v2
  STORE[t]
  LOAD[t, 1: CONST]
  BINARY_SUBSCR . . . . . . . . . .  (v2, 1) -> v3
  UNPACK_SEQUENCE . . . . . . . . .  (v3) -> v4, v5
  UNPACK_SEQUENCE . . . . . . . . .  (v5) -> v6, v7
  STORE[y, a, b]
  LOAD[y]
  RETURN_VALUE . . . . . . . . . . . (v7) ->
"""
)
def nested_tuple_fold(x):
    t = ("aaa", ((x, 1), 2))
    (y, a), b = t[1]
    return y


@add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[x]
  POP_JUMP_IF_FALSE . . . . . . . (x) ->
      -> 1, 5(Jump)

Block 1:  [] => [v0]
  LOAD[inner]
  GET_ITER . . . . . . . . . . .  (inner) -> v0
  JUMP_ABSOLUTE*
      -> 2(Jump)

Block 2:  [⓵ ] => [v0, v1]
  FOR_ITER . . . . . . . . . . .  (⓵ ) -> v0, v1
      -> 3, 6(Jump)

Block 3:  [⓵ , ⓶ ] => [⓵ ]
  STORE[_]
  LOAD[x, 2: CONST]
  INPLACE_TRUE_DIVIDE . . . . . . (x, 2) -> v0
  STORE[x]
  JUMP_ABSOLUTE
      -> 2(Jump)

Block 4:  [] => []
  LOAD[x, 1: CONST]
  INPLACE_FLOOR_DIVIDE . . . . .  (x, 1) -> v0
  STORE[x]
  LOAD[x]
  POP_JUMP_IF_TRUE . . . . . . .  (v0) ->
      -> 5, 1(Jump)

Block 5:  [] => []
  LOAD[inner]
  LOAD_ATTR . . . . . . . . . . . (inner) -> v0
  RETURN_VALUE . . . . . . . . .  (v0) ->

Block 6:  [⓵ , ⓶ ] => []
  POP_TOP* . . . . . . . . . . .  (⓶ ) ->
  POP_TOP* . . . . . . . . . . .  (⓵ ) ->
  JUMP_ABSOLUTE*
      -> 4(Jump)
"""
)
def nested_loop_fn(x, inner):
    while x:
        for _ in inner:
            x /= 2
        x //= 1
    return inner.count


# @add_parse_test()
def context_manager(x, ctx):
    with ctx() as c:
        x += 1


# TODO(robieta): re-enable when we can support generators
# @add_parse_test()
def simple_generator(k, suffix):
    if k < 0:
        yield None

    yield from range(k)

    yield from suffix


class TestClass:
    def f(self, x):
        return self.__name__ + x


add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[self]
  LOAD_ATTR . . . . . . . (self) -> v0
  LOAD[x]
  BINARY_ADD . . . . . .  (v0, x) -> v1
  RETURN_VALUE . . . . .  (v1) ->
"""
)(TestClass().f)


class TestClassWithSuper(TestClass):
    def f(self, x):
        return super().f(x) + 1


add_parse_test(
    r"""
Block 0:  [] => []
  LOAD[super: GLOBAL]
  CALL_FUNCTION . . . . .  (super) -> v0
  LOAD_METHOD . . . . . .  (v0) -> v1, v2
  LOAD[x]
  CALL_METHOD . . . . . .  (v1, v2, x) -> v3
  LOAD[1: CONST]
  BINARY_ADD . . . . . . . (v3, 1) -> v4
  RETURN_VALUE . . . . . . (v4) ->
"""
)(TestClassWithSuper().f)


def make_nonlocal_test():
    x: int

    @add_parse_test()
    def access_nonlocal():
        nonlocal x
        y, x = x, 1
        return y


# Re-enable when nonlocal variables are supported.
# make_nonlocal_test()


# @add_parse_test()
def try_finally(f):
    try:
        f.write("Test")
    finally:
        f.close()


# @add_parse_test()
def try_except_finally(f, log):
    try:
        f.write("Test")
    except OSError:
        log("Fail")
    finally:
        f.close()


# =============================================================================
# == Parametrized tests ========================================================
# =============================================================================
@pytest.mark.skipif(
    not python_ir_data.SUPPORTS_PREPROCESSING,
    reason=f"Python version {sys.version_info=} does not support preprocessing",
)
@pytest.mark.parametrize(
    ("fn", "spec"),
    TEST_CASES,
    ids=[
        f"test_parse_{fn.__self__.__class__.__name__ + '().' if hasattr(fn, '__self__') else ''}{fn.__name__}"
        for fn, _ in TEST_CASES
    ],
)
def test_parse(fn, spec: str | None):
    _, _, summary = parse.functionalize_blocks(fn.__code__)
    if spec is None:
        print(f"\n\n{summary}")
        pytest.skip("No parse spec provided.")

    # Check if the raw bytecode matches. (If it doesn't there's no point in testing anything else.)
    pattern = re.compile(r"^  ([A-Z_]+\*?)(.*)$")
    recovered_bytecode: list[str] = []
    for line in spec.splitlines():
        if match := pattern.match(line):
            opname, remainder = match.groups()
            if remainder and remainder.startswith("["):
                assert opname in ("LOAD", "STORE", "DELETE"), line
                assert remainder.endswith("]"), line
                for i in remainder[1:-1].split(", "):
                    recovered_bytecode.append(f"{opname}_{(i.split(': ')[1:] or ['FAST'])[0]}")
            else:
                recovered_bytecode.append(opname)

    bytecode = {idx: i.opname for idx, i in enumerate(dis.get_instructions(fn))}
    adjusted: list[str] = []
    for recovered in recovered_bytecode:
        corrected: str | None = (
            {
                "JUMP_ABSOLUTE*": {"RETURN_VALUE": "RETURN_VALUE"},
                "RETURN_VALUE*": {},
            }
            .get(recovered, {recovered: recovered})
            .get(bytecode.get(len(adjusted), ""))
        )
        if corrected is not None:
            adjusted.append(corrected)

    if "\n".join(adjusted) != "\n".join(bytecode.values()):
        zipped = itertools.zip_longest(adjusted, bytecode.values(), fillvalue="")
        delta = "\n".join(f"  {i:<25} {'' if j == i else j}" for i, j in zipped)
        msg = (
            f"Disassembled input does not match:\n{delta}\n"
            f"Cannot test using this Python version: {sys.version_info}"
        )

        # For some unfathomable reason, pytest will muck with spaces in error messages.
        # (Sometimes deleting them, sometimes adding newlines???) Thus, whenever we want
        # to align we need to use a non-breaking space.
        pytest.skip(msg.replace(" ", "\u00A0"))

    def clean(s: str):
        return "\n".join(l.rstrip() for l in textwrap.dedent(s).strip().splitlines(False))

    assert clean(summary) == clean(spec)


def test_debug_print_protoflows(capfd):
    proto_graph = apply_protograph_passes(protograph.ProtoGraph.from_code(tuple_fold.__code__))
    _ = capfd.readouterr()
    proto_graph.debug_print_protoflows()
    msg = textwrap.dedent(capfd.readouterr().out).strip()
    expected = textwrap.dedent(
        """
        Protoblock 0:
                              Inputs, Outputs
                  BUILD_TUPLE, (0, 1) -> (2)
              UNPACK_SEQUENCE, (2) -> (1, 0)
                 RETURN_VALUE, (0) -> ()
    """
    ).strip()
    assert msg == expected, msg


def test_abstract_value():
    x = protograph.AbstractValue()
    assert x == x
    assert x in {x}
    with pytest.raises(NotImplementedError):
        copy.copy(x)

    y = protograph.AbstractValue()
    assert x != y
    assert x not in {y}
    assert x in {x, y}
    assert len({x, y}) == 2

    assert x.substitute({x: y}) == y


def test_value_missing():
    x = protograph.NonPyObject(protograph.NonPyObject.Tag.MISSING)
    assert x == protograph.NonPyObject(protograph.NonPyObject.Tag.MISSING)
    assert x in {protograph.NonPyObject(protograph.NonPyObject.Tag.MISSING)}

    # Sanity check that it doesn't always compare equal
    assert x != protograph.AbstractValue()


def test_external_ref():
    key = parse.VariableKey("self", parse.VariableScope.LOCAL)
    x = protograph.ExternalRef(key)
    y = protograph.ExternalRef(key)

    assert x == y
    assert x in {y}


def test_abstract_phivalue():
    x = protograph.IntermediateValue()
    y = protograph.IntermediateValue()
    xy = protograph.AbstractPhiValue((x, y))

    # Deduplicate.
    assert len(xy.constituents) == 2
    assert protograph.AbstractPhiValue((x, x, y)) == xy

    # Flatten.
    assert xy == protograph.AbstractPhiValue((x, protograph.AbstractPhiValue((x, y))))

    # Replace constituents.
    x_prime = protograph.IntermediateValue()
    y_prime = protograph.IntermediateValue()
    xy_prime = xy.substitute({x: x_prime, y: y_prime})
    assert xy_prime == protograph.AbstractPhiValue((x_prime, y_prime))

    # Direct replacement takes precidence.
    z = protograph.IntermediateValue()
    assert xy.substitute({x: x_prime, y: y_prime, xy: z}) is z

    # Direct replacements still need to propagate.
    a = protograph.IntermediateValue()
    b = protograph.IntermediateValue()
    ab = protograph.AbstractPhiValue((a, b))
    assert xy.substitute({xy: xy_prime, x_prime: a, y_prime: b}) == ab
