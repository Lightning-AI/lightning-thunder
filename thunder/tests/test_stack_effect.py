import re
import textwrap

from thunder.core.script.protograph import ALIAS_OPCODES

import pytest


# Populate with the contents of `https://github.com/python/cpython/blob/74a2b79c6265c92ef381b5ff0dc63903bf0178ac/Python/bytecodes.c#L2090`
BYTECODES_C = """
...
            res = import_from(tstate, from, name);
            ERROR_IF(res == NULL, error);
        }

        inst(JUMP_FORWARD, (--)) {
            JUMPBY(oparg);
        }

        inst(JUMP_BACKWARD, (--)) {
            assert(oparg < INSTR_OFFSET());
            JUMPBY(-oparg);
            CHECK_EVAL_BREAKER();
        }

        inst(POP_JUMP_IF_FALSE, (cond -- )) {
            if (Py_IsTrue(cond)) {
                _Py_DECREF_NO_DEALLOC(cond);
            }
            else if (Py_IsFalse(cond)) {
                _Py_DECREF_NO_DEALLOC(cond);
                JUMPBY(oparg);
            }
...
"""


def generate():
    reference_effects = []
    inst_pattern = re.compile(r"^\s*inst\(([A-Z_]+),\s*\((.*)\)\)\s*\{\s*$")
    for line in BYTECODES_C.splitlines(keepends=False):
        if match := inst_pattern.search(line):
            opname, effect = match.groups()
            reference_effects.append((opname, effect.strip()))

    for opname, effect in sorted(reference_effects):
        print(f"    {opname:<30} {effect}")


RAW_EFFECTS = """
    BINARY_SUBSCR                  unused/1, container, sub -- res
    BUILD_CONST_KEY_MAP            values[oparg], keys -- map
    BUILD_LIST                     values[oparg] -- list
    BUILD_MAP                      values[oparg*2] -- map
    BUILD_SET                      values[oparg] -- set
    BUILD_SLICE                    start, stop, step if (oparg == 3) -- slice
    BUILD_STRING                   pieces[oparg] -- str
    BUILD_TUPLE                    values[oparg] -- tup
    CALL_FUNCTION_EX               unused, func, callargs, kwargs if (oparg & 1) -- result
    COMPARE_OP                     unused/1, left, right -- res
    CONTAINS_OP                    left, right -- b
    DELETE_ATTR                    owner --
    DELETE_DEREF                   --
    DELETE_FAST                    --
    DELETE_GLOBAL                  --
    DELETE_NAME                    --
    DELETE_SUBSCR                  container, sub --
    DICT_MERGE                     update --
    DICT_UPDATE                    update --
    EXTENDED_ARG                   --
    FORMAT_VALUE                   value, fmt_spec if ((oparg & FVS_MASK) == FVS_HAVE_SPEC) -- result
    FOR_ITER                       unused/1, iter -- iter, next
    GET_ITER                       iterable -- iter
    GET_LEN                        obj -- obj, len_o
    GET_YIELD_FROM_ITER            iterable -- iter
    IS_OP                          left, right -- b
    JUMP_FORWARD                   --
    LIST_APPEND                    list, unused[oparg-1], v -- list, unused[oparg-1]
    LIST_EXTEND                    list, unused[oparg-1], iterable -- list, unused[oparg-1]
    LOAD_ASSERTION_ERROR           -- value
    LOAD_ATTR                      unused/9, owner -- res2 if (oparg & 1), res
    LOAD_BUILD_CLASS               -- bc
    LOAD_CLASSDEREF                -- value
    LOAD_CLOSURE                   -- value
    LOAD_CONST                     -- value
    LOAD_DEREF                     -- value
    LOAD_FAST                      -- value
    LOAD_GLOBAL                    unused/1, unused/1, unused/1, unused/1 -- null if (oparg & 1), v
    LOAD_NAME                      -- v
    MAP_ADD                        key, value --
    MATCH_CLASS                    subject, type, names -- attrs
    MATCH_KEYS                     subject, keys -- subject, keys, values_or_none
    MATCH_MAPPING                  subject -- subject, res
    MATCH_SEQUENCE                 subject -- subject, res
    NOP                            --
    POP_EXCEPT                     exc_value --
    POP_JUMP_IF_FALSE              cond --
    POP_JUMP_IF_TRUE               cond --
    POP_TOP                        value --
    RAISE_VARARGS                  args[oparg] --
    RERAISE                        values[oparg], exc -- values[oparg]
    RETURN_VALUE                   retval --
    SETUP_ANNOTATIONS              --
    SET_ADD                        set, unused[oparg-1], v -- set, unused[oparg-1]
    SET_UPDATE                     set, unused[oparg-1], iterable -- set, unused[oparg-1]
    STORE_ATTR                     counter/1, unused/3, v, owner --
    STORE_DEREF                    v --
    STORE_FAST                     value --
    STORE_GLOBAL                   v --
    STORE_NAME                     v --
    STORE_SUBSCR                   counter/1, v, container, sub --
    UNARY_INVERT                   value -- res
    UNARY_NEGATIVE                 value -- res
    UNARY_NOT                      value -- res
    UNPACK_EX                      seq -- unused[oparg & 0xFF], unused, unused[oparg >> 8]
    UNPACK_SEQUENCE                unused/1, seq -- unused[oparg]
    WITH_EXCEPT_START              exit_func, lasti, unused, val -- exit_func, lasti, unused, val, res
"""


EXPECTED = {}
effect_pattern = re.compile(r"^([A-Z_]+)\s*(.*)--(.*)$")
for line in textwrap.dedent(RAW_EFFECTS).strip().splitlines(keepends=False):
    opname, pop, push = effect_pattern.search(line).groups()

    # We cannot handle repeated inputs (`unused/1, unused/1, unused/1, unused/1`)
    if opname == "LOAD_GLOBAL":
        continue

    inputs = tuple(j for i in pop.split(",") if (j := i.strip()))
    idx_map = {name: idx - len(inputs) for idx, name in enumerate(inputs)}
    assert len(inputs) == len(idx_map), (opname, inputs)
    push_values = tuple(idx_map.setdefault(j, len(idx_map) - len(inputs)) for i in push.split(",") if (j := i.strip()))
    if any(i < 0 for i in push_values) or isinstance(ALIAS_OPCODES.get(opname), tuple):
        EXPECTED[opname] = push_values


# TODO(robieta): Investigate
KNOWN_FAILURES = {
    "DICT_MERGE",
    "DICT_UPDATE",
    "LIST_APPEND",
    "LIST_EXTEND",
    "FOR_ITER",
    "GET_LEN",
    "MAP_ADD",
    "MATCH_KEYS",
    "RERAISE",
    "SET_ADD",
    "SET_UPDATE",
    "WITH_EXCEPT_START",
}


@pytest.mark.parametrize("opname", EXPECTED)
def test_stack_effects(opname):
    if opname in KNOWN_FAILURES:
        pytest.xfail()

    expected = tuple(None if i >= 0 else i for i in EXPECTED[opname])
    assert expected == ALIAS_OPCODES.get(opname)


if __name__ == "__main__":
    generate()
