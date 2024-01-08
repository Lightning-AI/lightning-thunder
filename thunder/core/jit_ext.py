from typing import Any
from collections.abc import ValuesView, Iterable, Iterator
from types import CellType, ModuleType, CodeType, BuiltinFunctionType, FunctionType, MethodType
from collections.abc import Callable, Sequence
import weakref
import random
from functools import partial, wraps
import copy
import contextvars
import warnings
from enum import Enum, auto
from io import StringIO

import torch
from thunder.core.proxies import proxy, Proxy, TensorProxy, make_proxy_name, variableify, unvariableify
from thunder.core.trace import set_tracectx, reset_tracectx, tracectx
from thunder.core.jit import (
    jit,
    _jit,
    default_callbacks,
    JIT_CALLBACKS,
    default_opcode_interpreter,
    _default_lookaside_map,
    default_lookaside,
    JITFrame,
    do_raise,
    get_jitcompilectx,
    JitCompileCtx,
    is_opaque,
)
from thunder.core.langctx import set_langctx, reset_langctx, get_default_langctx
from thunder.core.codeutils import get_siginfo, SigInfo
import thunder.core.prims as prims
from thunder.common import transform_for_execution
from thunder.core.symbol import Symbol

#
# jit_ext.py implements extensions of thunder's interpreter
#

#
# Helpers
#

# Objects and funtions related to creating proxies
# TODO Should these be version with the Python version?

_relaxed_deepcopy_dispatch = {}

_atomic_copy_types = {
    type(None),
    type(Ellipsis),
    type(NotImplemented),
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    CodeType,
    type,
    range,
    BuiltinFunctionType,
    weakref.ref,
    property,
}

for typ in _atomic_copy_types:
    _relaxed_deepcopy_dispatch[typ] = copy._deepcopy_atomic


# Returns (None, None) or (copier, dont_memoize), where if dont_memoize is true then the copy
#   will not be recorded in the relaxed_deepcopies memo dictionary
def relaxed_deepcopy_dispatch(cls: type, /) -> None | Callable:
    return _relaxed_deepcopy_dispatch.get(cls, None)


_immutable_types = {
    type(None),
    type(Ellipsis),
    type(NotImplemented),
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    type,
    range,
    BuiltinFunctionType,
    weakref.ref,
    property,
    FunctionType,
    tuple,
    frozenset,
    slice,
}


def is_immutable(val: Any, /) -> bool:
    return type(val) in _immutable_types


_uncopyable_types = {
    ModuleType,
    contextvars.ContextVar,
}


def is_uncopyable(val: Any, /) -> bool:
    return type(val) in _uncopyable_types


# Modifies the deepcopy() function defined here:
#   https://github.com/python/cpython/blob/3.10/Lib/copy.py#L128
#   to...
#   ... be "relaxed" and not fail if any part of the object cannot be deepcopied
#   ... have an extended dispatch that can optionally skip memoization
#       (useful for associating multiple proxies with a single input tensor
# NOTE If an object skips memoization then it will have a different proxy for everytime it appears
#   in the input

from copyreg import dispatch_table


def relaxed_deepcopy(x: Any, /, memo: dict = None, _nil=[]) -> Any:
    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)
    copier = relaxed_deepcopy_dispatch(cls)

    y = _nil
    if copier is not None:
        y = copier(x, memo)
    elif is_uncopyable(x):
        # NOTE This addition is to handle attempts to deepcopy things like modules, which will otherwise fail
        #   below because they define __reduce_ex__
        # warnings.warn(f"Couldn't proxy object {x} of type {type(x)}; modifications to it will not be prevented")
        y = copy._deepcopy_atomic(x, memo)
    elif issubclass(cls, type):
        y = copy._deepcopy_atomic(x, memo)
    elif (copier := getattr(x, "__deepcopy__", None)) is not None:
        y = copier(memo)

    # NOTE Copying with a reductor is so exception-prone that
    #   we wrap it in a try/except block
    if y is _nil:
        rv = None
        try:
            if reductor := dispatch_table.get(cls):
                rv = reductor(x)
            elif reductor := getattr(x, "__reduce_ex__", None):
                rv = reductor(4)
            elif reductor := getattr(x, "__reduce__", None):
                rv = reductor()

            if rv is not None:
                if isinstance(rv, str):
                    y = x
                else:
                    y = copy._reconstruct(x, memo, *rv, deepcopy=relaxed_deepcopy)
        except Exception as ex:
            # TODO Think about refining this to just catch type errors
            rv = None

        if rv is None:
            # warnings.warn(f"Couldn't proxy object {x} of type {type(x)}; modifications to it will not be prevented")
            y = copy._deepcopy_atomic(x, memo)

    # Skips memoization if x is its own copy (which implies y is x)
    if y is not x:
        memo[d] = y
        # Make sure x lives at least as long as d
        copy._keep_alive(x, memo)

    return y


def _deepcopy_list(x, memo, deepcopy=relaxed_deepcopy):
    y = []
    memo[id(x)] = y
    append = y.append
    for a in x:
        append(deepcopy(a, memo))
    return y


_relaxed_deepcopy_dispatch[list] = _deepcopy_list


def _deepcopy_tuple(x, memo, deepcopy=relaxed_deepcopy):
    y = [deepcopy(a, memo) for a in x]
    # We're not going to put the tuple in the memo, but it's still important we
    # check for it, in case the tuple contains recursive mutable structures.
    try:
        return memo[id(x)]
    except KeyError:
        pass
    for k, j in zip(x, y):
        if k is not j:
            y = tuple(y)
            break
    else:
        y = x
    return y


_relaxed_deepcopy_dispatch[tuple] = _deepcopy_tuple


def _deepcopy_dict(x, memo, deepcopy=relaxed_deepcopy):
    y = {}
    memo[id(x)] = y
    for key, value in x.items():
        y[deepcopy(key, memo)] = deepcopy(value, memo)
    return y


_relaxed_deepcopy_dispatch[dict] = _deepcopy_dict


def _deepcopy_tensor(x, memo, deepcopy=relaxed_deepcopy):
    y = x.detach()
    memo[id(x)] = y
    return y


_relaxed_deepcopy_dispatch[torch.Tensor] = _deepcopy_tensor

try:
    from org.python.core import PyStringMap
except ImportError:
    PyStringMap = None

if PyStringMap is not None:
    _relaxed_deepcopy_dispatch[PyStringMap] = _deepcopy_dict


# Copy instance methods
def _deepcopy_method(x, memo, deepcopy=relaxed_deepcopy):
    return type(x)(x.__func__, deepcopy(x.__self__, memo))


_relaxed_deepcopy_dispatch[MethodType] = _deepcopy_method

#
# phantom mode (no side effects)
#


class PhantomInterpreterCtxInterface:
    def __init__(self):
        # We get input nonlocals as elements of func.__closure__
        self.nonlocals_inputs: dict[tuple[Callable, int], CellType] = {}

        # We store here changes to nonlocals. Deletions are represented
        # as clearing the cell.
        # For cells from input_nonlocals, we expect to have a mapping from
        # the input cell to another cell here.
        # For nonlocals we created ourselves (and only those), we have
        # key is value .
        # There are two key ways that nonlocals can be outputs:
        # - changing a value in an input cell,
        # - being attached to a func.__closure__ for some func we hand out
        # Cells identified by the id are either values in nonlocals_inputs
        # or nonlocals_map itself, so it is guaranteed that they are alive
        # and the id is not reused.
        self.nonlocals_map: dict[int, CellType] = {}

    def proxify(self, val: Any, /, *, name: None | str = None, **kwargs) -> Any:
        raise NotImplementedError("Abstract method!")


# TODO Track deleted to deal with duplicate names
# TODO Make proxy construction extensible
# TODO Probably don't want to track what's stored or proxied based on name alone
# TODO The current stored/proxify relationship is far from correct
class PhantomInterpreterCtx(PhantomInterpreterCtxInterface):
    def __init__(self):
        super().__init__()

        # Maps from ids to input objects
        # NOTE This extends the lifetime of all inputs to be at least the lifetime of the interpreter,
        #   this prevents Python from reusing the id of one of the inputs, which is how we track them
        # NOTE This means that the proxies of inputs have lifetimes that are at least the
        #   lifetime of the interpreter, too
        self.input_map: dict[int, Any] = {}

        # For input objects that map to have a proxy based on id,
        #   this maps from their ids to their unique proxy objects
        # NOTE This dict is compatible with copy.deepcopy()'s memo dict
        # NOTE Not all inputs map to proxies by id
        #   tensor inputs, for example, map to proxies based on how they're unpacked
        self.input_id_to_proxy_map: dict[int, Any] = {}

        # ids of all proxies
        self.proxy_id_set: set[int] = set()

        # Maps from lookups into global dicts to proxied values
        # NOTE Handling global dicts are probably a good use case for a dict proxy object
        # NOTE This assumes that global dicts persist for the lifetime our interpreter
        #   (in fact, we're going to assume that they persist for the lifetime of the
        #   Python interpreter)
        self.global_lookups: dict[tuple[int, str], Any] = {}

        # Records deleted values
        self.global_deletions: set[tuple[int, str]] = set()

    @property
    def inputs(self) -> ValuesView[tuple[str, Any]]:
        return self.input_map.values()

    # Returns the object's proxy (itself, if it is a proxy)
    # NOTE This must be called for each "unique deriviation" or "unique history" of each object
    #   The same object might be acquired in multiple distinct ways -- accessed as a global,
    #   an input in a list, an input in a dict... and each acquisition should call proxify
    #   Whether proxify actually creates a proxy for each unique derivation or returns
    #   a common proxy for each is dependent on how it's extended -- by default
    #   the same proxy is returned for each derivation, but this behavior can
    #   be overridden
    def proxify(self, val: Any, /, *, name: None | str = None, **kwargs) -> Any:
        val_id = id(val)

        # Checks if val itself is a proxy
        if val_id in self.proxy_id_set:
            return val

        # Checks to see if val is associated with an existing proxy
        p: None | Any = self.input_id_to_proxy_map.get(val_id, None)
        if p is not None:
            return p

        # Adds the object to the input map to ensure it lives as long as the interpreter does
        self.input_map[val_id] = (name, val)

        # Immutable objects are their own proxies
        if is_immutable(val) or is_uncopyable(val):
            return val

        # Copies the input_proxy_map because relaxed_deepcopy might mutate it but then fail, and
        #   if the deepcopy fails then we don't want to modify the input_proxy_map
        memo = self.input_id_to_proxy_map
        p = relaxed_deepcopy(val, memo=memo)

        # Updates the proxy id set
        self.proxy_id_set.add(id(p))
        self.input_id_to_proxy_map[val_id] = p

        # Some objects, like types, return themselves when deepcopied
        if p is val:
            # warnings.warn(f"Couldn't proxy {name} of type {type(val)}; modifications to it will not be prevented")
            return val

        # We used to removes the id(memo) entry, however this runs into problems
        # when the next deepcopy finds copies of old objects under the id.
        # We thus keep everything, even if this means that we waste memory.

        # Is this OKish? (it duplicates things because it isn't only the new.)
        # or is self.proxy_id_set = {id(v) for id in memo.values()} ?
        self.proxy_id_set.update(id(v) for v in memo.values())

        return p


_phantomctx = contextvars.ContextVar("phantomctx")


# Sets the phantom ctx
def set_phantomctx(ctx: PhantomInterpreterCtxInterface) -> Any:
    return _phantomctx.set(ctx)


# Returns the current phantom ctx
def get_phantomctx() -> PhantomInterpreterCtxInterface:
    return _phantomctx.get()


# Resets the phantom ctx
def reset_phantomctx(token) -> None:
    _phantomctx.reset(token)


phantom_callbacks: dict[JIT_CALLBACKS, Callable] = {}


def register_phantom_callback(key: JIT_CALLBACKS) -> Callable:
    assert key not in phantom_callbacks

    def deco(fn: Callable):
        assert str(key).split(".")[-1].lower() in fn.__name__, f"{fn} as hook for {key}"
        phantom_callbacks[key] = fn
        return fn

    return deco


# TODO Handle deleting globals
# TODO Use something like inspect.getmodule() and vars() to acquire the module of these globals, and then
#   to acquire its globals in the future
# Handles global loads and stores, essentially keeping an additional dictionary
#   that tracks modifications to all the global dictionaries
@register_phantom_callback(JIT_CALLBACKS.LOAD_GLOBAL_CALLBACK)
def _load_global_callback(globals_dict: dict, name: str, /) -> Any:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    ctx.proxify(globals_dict)

    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)

    if key in ctx.global_deletions:
        return do_raise(NameError(f"name '{name}' is not defined"))

    p: None | Any = ctx.global_lookups.get(key, None)

    if p is not None:
        return p

    val: Any = globals_dict[name]
    p = ctx.proxify(val, name=name)
    ctx.global_lookups[key] = p

    return p


# TODO Consider if this should suppress the population of frame.globals[name] completely
@register_phantom_callback(JIT_CALLBACKS.STORE_GLOBAL_CALLBACK)
def _store_global_callback(globals_dict: dict, name: str, val: Any, /) -> Any:
    ctx: PhantomInterpreterCtx = get_phantomctx()

    # Records the store
    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)
    ctx.global_lookups[key] = val

    # Records that this object is no longer deleted (if it was)
    ctx.global_deletions.discard(key)

    # Returns the existing value (so it's unmodified)
    return globals_dict[name]


@register_phantom_callback(JIT_CALLBACKS.DELETE_GLOBAL_CALLBACK)
def _delete_global_callback(globals_dict: dict, name: str, /) -> None:
    ctx: PhantomInterpreterCtx = get_phantomctx()

    assert name in globals_dict

    # Records the deletion
    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)
    ctx.global_deletions.add(key)


@register_phantom_callback(JIT_CALLBACKS.LOAD_DEREF_CALLBACK)
def _load_deref_callback(cell: CellType, /) -> Any:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    assert id(cell) in ctx.nonlocals_map
    return ctx.nonlocals_map[id(cell)].cell_contents


# TODO: in principle this should also be done for LOAD_FAST if Python devs
#       ever decide they want to merge the opcodes
@register_phantom_callback(JIT_CALLBACKS.LOAD_CLOSURE_CALLBACK)
def _load_closure_callback(cell: CellType, /) -> CellType:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    assert id(cell) in ctx.nonlocals_map
    return ctx.nonlocals_map[id(cell)]


@register_phantom_callback(JIT_CALLBACKS.STORE_DEREF_CALLBACK)
def _store_deref_callback(cell: CellType, value: Any, /) -> None:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    assert id(cell) in ctx.nonlocals_map
    ctx.nonlocals_map[id(cell)].cell_contents = value
    return cell


@register_phantom_callback(JIT_CALLBACKS.DELETE_DEREF_CALLBACK)
def _delete_deref_callback(cell: CellType, /) -> None:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    assert id(cell) in ctx.nonlocals_map
    del ctx.nonlocals_map[id(cell)].cell_contents


@register_phantom_callback(JIT_CALLBACKS.MAKE_CELL_CALLBACK)
def _make_cell_callback(cell: CellType, /) -> CellType:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    ctx.nonlocals_map[id(cell)] = cell
    return cell


@register_phantom_callback(JIT_CALLBACKS.FUNCTION_START_CALLBACK)
def _function_start_callback(fn: Callable, frame: JITFrame, /) -> None:
    ctx: PhantomInterpreterCtx = get_phantomctx()
    closure_rec = getattr(fn, "__closure__", None)
    if closure_rec is None:
        closure_rec = ()
    for i, c in enumerate(closure_rec):
        assert isinstance(c, CellType)
        if id(c) not in ctx.nonlocals_map:
            # unseen cell is an input
            ctx.nonlocals_inputs[(fn, i)] = c
            # copy cell
            if c == CellType():
                cc = CellType()
            else:
                val = c.cell_contents
                name: str = fn.__code__.co_freevars[i]
                p = ctx.proxify(val, name=name, history=())
                cc = CellType(p)
            ctx.nonlocals_map[id(c)] = cc
        elif ctx.nonlocals_map[id(c)] is not c:
            # it's an input that has been previously seen, but we see it again
            ctx.nonlocals_inputs[(fn, i)] = c
        else:
            # it is a cell that we generated
            pass

    # strictly speaking, we would expect this to only happen in Python <= 3.10
    for i, c in enumerate(frame.localsplus):
        if isinstance(c, CellType) and frame.get_localsplus_name(i) in frame.code.co_cellvars:
            # a cell that we create on call
            ctx.nonlocals_map[id(c)] = c


def phantom_jit(
    fn: Callable,
    *,
    opcode_interpreter: Callable = default_opcode_interpreter,
    fn_lookaside: Callable = default_lookaside,
    ctx: None | PhantomInterpreterCtxInterface = None,
    callbacks: dict[JIT_CALLBACKS, Callable] = phantom_callbacks,
    debug_log: None | StringIO = None,
) -> Callable:
    jit_ = partial(
        jit, opcode_interpreter=opcode_interpreter, fn_lookaside=fn_lookaside, callbacks=callbacks, debug_log=debug_log
    )
    jfn = jit_(fn)

    @wraps(jfn)
    def fn(*args, **kwargs) -> Callable:
        # Acquires the random state, which may be implicitly modified from calls to Python's
        #   random module
        # TODO Consider extending this block to be extensible and handle other
        #   commonly used implicit states
        # TODO Also consider modeling Python's random module explicitly so that
        #   the random state appears to be an input

        initial_random_state = random.getstate()
        try:
            ctx_ = PhantomInterpreterCtx() if ctx is None else ctx

            tok: Any = set_phantomctx(ctx_)

            si = get_siginfo(fn, args, kwargs)

            pargs = []
            for name, x in si.args:
                p = hsig(x, name)
                pargs.append(p)

            if si.varargs is not None:
                varargs_name, varargs = si.varargs
                p = hsig(varargs, varargs_name)
                pargs.extend(p)

                # Eagerly unpacks varargs
                def _varargs_unpack(varargs):
                    for x in varargs:
                        pass

                jit_(_varargs_unpack)(varargs)

            pkwargs = {}
            for name, x in si.kwargs.items():
                p = hsig(x, name)
                pkwargs[name] = p

            if si.varkwargs is not None:
                varkwargs_name, x = si.varkwargs
                p = hsig(x, varkwargs_name)
                pkwargs.update(p)

            try:
                result = jfn(*pargs, **pkwargs)
            finally:
                # Resets the random state
                random.setstate(initial_random_state)

            # Propagates metadata
            # TODO Find a better way to do this? -- why doesn't wraps propagate these?
            fn._last_interpreted_instructions = jfn._last_interpreted_instructions
            fn._last_interpreted_history = jfn._last_interpreted_history
            return result
        finally:
            reset_phantomctx(tok)

    return fn


#
# thunder mode (no side effects + creates a thunder program to execute)
#


import time

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx
from thunder.torch import _torch_to_thunder_function_map
from thunder.clang import _clang_fn_set

#
# Thunder lookasides
#

# NOTE Calls into symbols MUST use this lookaside -- we don't want to jit into them
# TODO Add all thunder operations (see https://github.com/Lightning-AI/lightning-thunder/issues/1804)
_thunder_symbol_lookaside_map = {}
_thunder_symbol_lookaside_map.update(_torch_to_thunder_function_map)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1817
#   This currently just calls getattr, assuming that call has no side effects
def _thunder_getattr_lookaside(origin: Any, key: str, *maybe_default: Any) -> Any:
    val = getattr(origin, key, *maybe_default)

    ctx: ThunderInterpreterCtx = get_phantomctx()
    ii: None | InterpreterInfo = ctx.get_info(origin)

    # Short-circuits if the origin is not tracked (indicating the origin is an intermediate)
    if ii is None:
        return val

    val = hgetattr(ii.history, val, origin, key)
    return val


# https://docs.python.org/3.13/library/functions.html#func-list
def _thunder_list_lookaside(*args) -> list:
    if len(args) == 0:
        return []

    (iterable,) = args

    def impl():
        l = []
        for x in iterable:
            l.append(x)
        return l

    return _jit(impl)


# https://docs.python.org/3.13/library/functions.html#func-tuple
def _thunder_tuple_lookaside(*args) -> tuple:
    if len(args) == 0:
        return ()

    (iterable,) = args

    # NOTE It's odd that this returns a list, but it's necessary to show
    #   how the items are extracted from the iterable so they can be proxied
    #   correctly
    def impl():
        l = []
        for x in iterable:
            l.append(x)
        return l

    return tuple(_jit(impl))


# https://docs.python.org/3.13/library/functions.html#zip
# Records the provenance of the top-level items in the iterables being zipped
def _thunder_zip_lookaside(*iterables, strict: bool = False) -> tuple:
    seqs = []
    for seq in iterables:
        # NOTE It's odd that this returns a list, but it's necessary to show
        #   how the items are extracted from the iterable so they can be
        #   proxied correctly
        # NOTE We can't just capture seq in a closure here, because
        #   that would bind to the last seq enumerated
        def impl(seq):
            l = []
            for x in seq:
                l.append(x)
            return l

        seqs.append(_jit(impl, seq))

    return zip(*seqs, strict=strict)


_thunder_symbol_lookaside_map_update = {
    getattr: _thunder_getattr_lookaside,
    list: _thunder_list_lookaside,
    tuple: _thunder_tuple_lookaside,
    zip: _thunder_zip_lookaside,
}
_thunder_symbol_lookaside_map.update(_thunder_symbol_lookaside_map_update)


# TODO Currently this has to capture fn to get __self__, we should really revise the way
#   lookasides are called from _jit so that self is passed as the first argument here
def _thunder_getitem_lookaside(fn, *args):
    assert len(args) == 2 or (len(args) == 1 and hasattr(fn, "__self__"))

    slf: Any
    key: Any
    if len(args) == 2:
        slf, key = args
    else:
        slf = fn.__self__
        (key,) = args

    # TODO Probably want to make a simpler pattern for this
    ctx: ThunderInterpreterCtx = get_phantomctx()
    ii: None | InterpreterInfo = ctx.get_info(slf)

    # Short-circuits if we were not tracking the object we're performing the getitem on (indicating that the
    #   item is an intermediate, and does not need its provenance tracked)
    if ii is None:
        result = slf.__getitem__(key)
        return result

    result = ii.obj.__getitem__(key)
    return hgetitem(ii.history, result, slf, key)


def _thunder_iter_lookaside(fn, *args):
    if hasattr(fn, "__self__"):
        # TODO: can this happen after we changed unpacking?
        assert len(args) == 0
        origin = fn.__self__
    else:
        assert len(args) == 1
        (origin,) = args

    ctx: ThunderInterpreterCtx = get_phantomctx()
    ii: None | InterpreterInfo = ctx.get_info(origin)

    # Short-circuits if we were not tracking the object we're performing the getitem on (indicating that the
    #   item is an intermediate, and does not need its provenance tracked)
    if ii is None:
        return origin.__iter__()

    result = ii.obj.__iter__()
    return hiter(ii.history, result, origin)


def _thunder_next_lookaside(fn, *args):
    if hasattr(fn, "__self__"):
        # TODO: can this happen after we changed unpacking?
        assert len(args) == 0
        origin = fn.__self__
    else:
        assert len(args) == 1
        (origin,) = args

    ctx: ThunderInterpreterCtx = get_phantomctx()
    ii: None | InterpreterInfo = ctx.get_info(origin)

    # Short-circuits if we were not tracking the object we're performing the getitem on (indicating that the
    #   item is an intermediate, and does not need its provenance tracked)
    if ii is None:
        val = origin.__next__()
        return val

    result = ii.obj.__next__()
    ii.iterator_counter += 1
    return hnext(ii.history, result, origin, ii.iterator_counter)


# String name to function
_thunder_opaque_dunder_lookaside_map = {
    "__getitem__": _thunder_getitem_lookaside,
    "__iter__": _thunder_iter_lookaside,
    "__next__": _thunder_next_lookaside,
}


# Looks for an interpreter lookaside first, then for a thunder lookaside, and finally
#   looks for the default lookaside
def thunder_lookaside(fn, *args, **kwargs) -> None | Callable:
    ctx: ThunderInterpreterCtx = get_phantomctx()

    # Performs dunder lookasides
    # TODO Acquire names more consistently
    if is_opaque(fn) and hasattr(fn, "__name__"):
        thunder_interpreter_lookaside: None | Callable = _thunder_opaque_dunder_lookaside_map.get(fn.__name__, None)
        if thunder_interpreter_lookaside:
            return partial(thunder_interpreter_lookaside, fn)

    # Performs symbol lookasides
    thunder_lookaside: None | Callable = _thunder_symbol_lookaside_map.get(fn, None)

    if thunder_lookaside is not None:
        return thunder_lookaside

    # NOTE Symbols "lookaside" to themselves; this just prevents their internals from being jitted
    if isinstance(fn, Symbol):
        return fn

    # NOTE clang operations are not symbols, but we still prevent their internals from being jitted
    # TODO In the future we probably shouldn't NEED to do this, but it will likely make for an easier UX
    if fn in _clang_fn_set:
        return fn

    return default_lookaside(fn, *args, **kwargs)


#
# Thunder interpreter proxies
#


# History-related objects and helpers
# TODO Extend with additional ways of acquiring tensors
class UNPACK_ACTION(Enum):
    FROM_SIGNATURE = auto()
    GETATTR = auto()
    GETITEM = auto()
    GLOBALS_DICT = auto()
    ITER = auto()
    NEXT = auto()


def hsig(obj: Any, name: str, /) -> Any:
    ctx = get_phantomctx()
    history = ((UNPACK_ACTION.FROM_SIGNATURE, obj, name),)
    return ctx.proxify(obj, name=name, history=history)


def hgetitem(prior_history, obj: Any, origin: Any, key: Any, /) -> Any:
    ctx = get_phantomctx()
    history = prior_history + ((UNPACK_ACTION.GETITEM, obj, origin, key),)
    return ctx.proxify(obj, history=history)


def hglobalsdict(globals_dict: dict, /) -> Any:
    ctx = get_phantomctx()
    history = ((UNPACK_ACTION.GLOBALS_DICT, globals_dict, globals_dict["__name__"]),)
    return ctx.proxify(globals_dict, history=history)


def hgetattr(prior_history, obj: Any, origin: Any, key: Any, /) -> Any:
    ctx = get_phantomctx()
    history = prior_history + ((UNPACK_ACTION.GETATTR, obj, origin, key),)
    return ctx.proxify(obj, history=history)


def hiter(prior_history, obj: Any, origin: Any, /) -> Any:
    ctx = get_phantomctx()
    history = prior_history + ((UNPACK_ACTION.ITER, obj, origin),)
    return ctx.proxify(obj, history=history)


def hnext(prior_history, obj: Any, origin: Any, iter_counter, /) -> Any:
    ctx = get_phantomctx()
    history = prior_history + ((UNPACK_ACTION.NEXT, obj, origin, iter_counter),)
    return ctx.proxify(obj, history=history)


#
# Thunder entrypoints
#
from thunder.core.proxies import proxy


# NOTE Historical names are only derived from the signature parameters directly
# TODO We could think about renaming tensors without names from parameters based on how they're
#   named in the computation
def get_name(history) -> None | str:
    last = history[-1]
    ua, *_ = last

    if ua is UNPACK_ACTION.FROM_SIGNATURE:
        ua, obj, name = last
        return name

    return None


# TODO Handle number proxies, too
def _tensor_copier(t: torch.Tensor, /, *, history: tuple):
    name: None | str = get_name(history)
    p: Proxy = proxy(t, name=name)

    return p


# Tracks the provenance of objects, as well as how inputs are proxied and what objects they are "replaced" with
#   The "replacement" is the object we provide to our interpreter, it could be:
#       1) The original object (if the object is immutable or we are assuming it is for the moment)
#       2) A shallow copy of the object (to avoid modifying the original)
#       3) A Proxy instead of the original object (like TensorProxy objects for torch.Tensor objects)
class InterpreterInfo:
    def __init__(self, obj: Any, *, replacement: Any, proxy: Proxy, history: None | tuple):
        self.obj = obj
        self.replacement = replacement

        assert isinstance(proxy, Proxy)
        self.proxy = proxy

        self.history = history

        # NOTE The following is just to properly record iterator state
        self.iterator_counter: int = 0


# Tracks a set of objects
# NOTE Ideally we would just be able to use a class like weakref.WeakValueDictionary to track
#   objects without holding a reference to them, but not every object can have a weak reference.
#   This deals with the issue by holding an actual reference to objects that cannot have a weak
#   reference.
# TODO Maybe there's a better name for this class?
# TODO Handle tracking weakrefs better, https://github.com/Lightning-AI/lightning-thunder/issues/1816
class Tracker:
    def __init__(self):
        self.d = {}

    def add(self, obj: Any):
        obj_id: int = id(obj)

        if obj_id in self.d:
            # Asserts that the weak reference is still alive
            #   (if it's not, it should have been deleted from the
            #    dictionary because of the callback created below)
            val = self.d[obj_id]
            if isinstance(val, weakref.ref):
                assert val() is not None

            return

        try:

            def cb(x, d=self.d, key=obj_id):
                del d[key]

            w = weakref.ref(obj, cb)
            self.d[obj_id] = w
        except:
            self.d[obj_id] = obj

    def __contains__(self, obj: Any) -> bool:
        obj_id: int = id(obj)
        return obj_id in self.d


class ThunderInterpreterCtx(PhantomInterpreterCtxInterface):
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()

        self.fn = fn

        self._prologue_trc: TraceCtx = TraceCtx(fn)
        self._prologue_trc.args = args
        self._prologue_trc.kwargs = kwargs

        self._computation_trc: TraceCtx = TraceCtx()

        self._input_ids_to_info_map: dict[int, InterpreterInfo] = {}
        self._replacement_ids_to_info_map: dict[int, InterpreterInfo] = {}

        self._intermediates: Tracker = Tracker()

    @property
    def prologue_trace(self) -> TraceCtx:
        return self._prologue_trc

    @property
    def computation_trace(self) -> TraceCtx:
        return self._computation_trc

    @property
    def inputs(self) -> ValuesView[InterpreterInfo]:
        return self._input_ids_to_info_map.values()

    def get_info(self, val: Any, /) -> None | InterpreterInfo:
        ii: None | InterpreterInfo = self._input_ids_to_info_map.get(id(val), None)

        if ii:
            return ii

        ii = self._replacement_ids_to_info_map.get(id(val), None)
        return ii

    def get_replacement(self, val: Any, /) -> None | Any:
        if (info := self.get_info(val)) is not None:
            return info.replacement
        return None

    # NOTE All proxies are constructed in the context of the computation trace, and their
    #   names must be added to the prologue trace (this is done when constructing the prologue trace)
    def proxify(self, val: Any, /, *, name: None | str = None, history: tuple, **kwargs) -> Any:
        # Checks if we've already seen this object (currently not supported)
        ii: None | InterpreterInfo = self.get_info(val)

        # TODO Resolve multiple histories -- sometimes proxify is called on an object
        #   which is actually not introduced through a novel mechanism
        if ii is not None:
            return ii.replacement

        # Intermediate values are not proxied
        if val in self._intermediates:
            return val

        # TODO FIXME This is a temporary hack to allow loading objects like modules and functions whose
        #   provenance we don't yet understand
        if len(history) == 0 and is_immutable(val):
            return val

        assert len(history) > 0, f"Attempting to proxy {val=}, but it has no history"

        p: Any
        if isinstance(val, torch.Tensor):
            p = _tensor_copier(val, history=history)
            ii = InterpreterInfo(val, replacement=p, proxy=p, history=history)
            self._input_ids_to_info_map[id(val)] = ii
            self._replacement_ids_to_info_map[id(p)] = ii
        elif is_uncopyable(val) or is_immutable(val):
            # TODO We should probably expand our understanding
            #   of uncopyable objects. For the phantom jit
            #   we should prevent their modification. For
            #   the thunder jit it's OK to modify them
            #   in the computation.
            ii = InterpreterInfo(val, replacement=val, proxy=Proxy(name=name), history=history)
            self._input_ids_to_info_map[id(val)] = ii
            self._replacement_ids_to_info_map[id(val)] = ii
            return val
        else:
            p = copy.copy(val)
            ii = InterpreterInfo(val, replacement=p, proxy=Proxy(name=name), history=history)
            self._input_ids_to_info_map[id(val)] = ii
            self._replacement_ids_to_info_map[id(p)] = ii

        return p


#
# Thuder interpreter callbacks
#


# Called whenever an object is pushed onto the stack
#   This does two things:
#       1) If the value to be pushed has a replacement value, the replacement value is pushed
#           instead. A common example of this is pushing a TensorProxy instead of a torch.Tensor object.
#           This assumes that when objects are acquired from other places -- like the globals dict, or
#           or closures -- that they are also converted to replacement values OR simply pushed on the
#           stack without inspection.
#       2) Adds the values to the context's "intermediates." This will even add input objects to
#           "intemediates," and we should update this to only push objects that are known to be
#           intermediates or change the name "intermediates." Currently, objects which
#           are actually intermediates can be discovered by diffing "intermediates" with the context's
#           input tracking datastructures.
def _thunder_push_stack_callback(val: Any, /, *, source: None | str = None) -> Any:
    ctx: ThunderInterpreterCtx = get_phantomctx()

    if (replacement := ctx.get_replacement(val)) is not None:
        return replacement

    if isinstance(val, torch.Tensor):
        source_str: str = "" if source is None else f"Its source was {source}"
        raise AssertionError(
            f"The thunder interpreter attempted to push an actual torch.tensor object onto the stack. The tensor had shape {val.shape} and dtype {val.dtype}. {source_str}"
        )

    ctx._intermediates.add(val)
    return val


def _thunder_load_global_callback(globals_dict: dict, key: str, /) -> Any:
    hglobalsdict(globals_dict)

    def impl():
        return globals_dict[key]

    return _jit(impl)


def _thunder_store_global_callback(globals_dict: dict, key: str, val: Any, /) -> Any:
    hglobalsdict(globals_dict)

    def impl():
        globals_dict[key] = val

    return _jit(impl)


def _thunder_delete_global_callback(globals_dict: dict, key: str, /) -> None:
    hglobalsdict(globals_dict)

    def impl():
        del globals_dict[key]

    return _jit(impl)


_thunder_callbacks = {
    JIT_CALLBACKS.PUSH_STACK_CALLBACK: _thunder_push_stack_callback,
    JIT_CALLBACKS.LOAD_GLOBAL_CALLBACK: _thunder_load_global_callback,
    JIT_CALLBACKS.STORE_GLOBAL_CALLBACK: _thunder_store_global_callback,
    JIT_CALLBACKS.DELETE_GLOBAL_CALLBACK: _thunder_delete_global_callback,
}

thunder_callbacks: dict[JIT_CALLBACKS, Callable] = phantom_callbacks | _thunder_callbacks


def thunder_jit(fn: Callable, ctx: ThunderInterpreterCtx, *, debug_log: None | StringIO = None) -> Callable:
    return phantom_jit(fn, fn_lookaside=thunder_lookaside, ctx=ctx, callbacks=thunder_callbacks, debug_log=debug_log)


# TODO Add support for transforms
# TODO Introduce caching
# TODO Support other langctx
def _create_callable(cd: CompileData, cs: CompileStats) -> Callable:
    @wraps(cd.fn)
    def fn_(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # TODO Caching goes here

        # Currently executes the program eagerly as a placeholder
        jfn: Callable
        interpreter_ctx = ThunderInterpreterCtx(cd.fn, *args, **kwargs)
        lang = get_default_langctx()
        try:
            lang_tok = set_langctx(lang)
            trace_tok = set_tracectx(interpreter_ctx.computation_trace)
            cs.last_trace_tracing_start = time.time_ns()
            jfn = thunder_jit(cd.fn, ctx=interpreter_ctx, debug_log=cd.debug_log)
            result = jfn(*args, **kwargs)
            prims.python_return(result)
            cs.last_trace_tracing_stop = time.time_ns()
        finally:
            reset_tracectx(trace_tok)
            reset_langctx(lang_tok)

        # Constructs the prologue
        #   The prologue ...
        #   - Accepts the original function's parameters
        #   - Acquires all inputs to the computation, including closures and globals
        #   - Unpacks all inputs
        #   - Validates that the input is valid for the computational trace it's associated with
        #   - Returns the flattened inputs
        # TODO Validate the inputs in the prologue, currently it just unpacks
        # TODO Only unpack values actually used in the computation (and the intermediates necessary to acquire them)
        prologue_trc = interpreter_ctx.prologue_trace
        computation_trc = interpreter_ctx.computation_trace
        prologue_rvals: set = set()
        already_unpacked: set[int] = set()
        global_dicts: dict = {}

        def unpack(ii: InterpreterInfo):
            if id(ii.obj) in already_unpacked:
                return ii.proxy

            def from_signature_action(_, obj: Any, name: str):
                bsym = prims.unpack_trivial.bind(ii.proxy, output=ii.proxy)
                prologue_trc.bound_symbols.append(bsym)
                prologue_rvals.add(variableify(ii.proxy))

            def getattr_action(_, obj: Any, origin: Any, key: str):
                unpacked = unpack(interpreter_ctx.get_info(origin))
                bsym = prims.unpack_attr.bind(unpacked, key, output=ii.proxy)
                prologue_trc.bound_symbols.append(bsym)
                prologue_rvals.add(variableify(ii.proxy))

            def getitem_action(_, obj: Any, origin: Any, key: Any):
                unpacked = unpack(interpreter_ctx.get_info(origin))
                bsym = prims.unpack_key.bind(unpacked, key, output=ii.proxy)
                prologue_trc.bound_symbols.append(bsym)
                prologue_rvals.add(variableify(ii.proxy))

            def globals_dict_action(_, globals_dict: dict, module_name: str):
                bsym = prims.python_vars.bind(module_name, output=ii.proxy)
                prologue_trc.bound_symbols.append(bsym)
                prologue_rvals.add(variableify(ii.proxy))
                global_dicts[module_name] = globals_dict

            def iter_action(_, obj: Any, origin: Any):
                unpacked = unpack(interpreter_ctx.get_info(origin))
                bsym = prims.unpack_iter.bind(unpacked, output=ii.proxy)
                prologue_trc.bound_symbols.append(bsym)
                prologue_rvals.add(variableify(ii.proxy))

            # NOTE The next action is special in that it deals with a stateful iterator,
            #   so it handles calling next() however many times the next() is called on the iterator
            # TODO Consider modeling next as returning the iterator, so that the
            #   state effect on the iterator appears functional
            def next_action(_, obj: Any, origin: Any, iter_counter: int):
                unpacked = unpack(interpreter_ctx.get_info(origin))

                # Finds all other elements unpacked from this iterator
                unpacks = []
                for ii in interpreter_ctx.inputs:
                    h = ii.history
                    last = h[-1]
                    ua, *_ = action

                    if ua is UNPACK_ACTION.NEXT:
                        _, other, other_origin, other_iter_counter = action
                        if other_origin is origin:
                            unpacks.append(action)

                for (
                    _,
                    other,
                    _,
                    _,
                ) in sorted(unpacks, key=lambda x: x[3]):
                    other_ii = interpreter_ctx.get_info(other)

                    if id(other_ii.obj) in already_unpacked:
                        continue

                    other_p = other_ii.proxy
                    bsym = prims.unpack_next.bind(unpacked, output=other_p)
                    prologue_trc.bound_symbols.append(bsym)
                    prologue_rvals.add(variableify(other_p))
                    already_unpacked.add(id(other_ii.obj))
                    if not prologue_trc.has_name(other_p.name):
                        prologue_trc.add_name(other_p.name)

            d = {
                UNPACK_ACTION.FROM_SIGNATURE: from_signature_action,
                UNPACK_ACTION.GETATTR: getattr_action,
                UNPACK_ACTION.GETITEM: getitem_action,
                UNPACK_ACTION.GLOBALS_DICT: globals_dict_action,
                UNPACK_ACTION.ITER: iter_action,
                UNPACK_ACTION.NEXT: next_action,
            }

            assert len(ii.history) > 0, f"Trying to unpack {ii.obj} without a history"
            action = ii.history[-1]
            ua, *_ = action
            d[ua](*action)

            already_unpacked.add(id(ii.obj))
            if not prologue_trc.has_name(ii.proxy.name):
                prologue_trc.add_name(ii.proxy.name)
            return ii.proxy

        rvals: tuple
        with tracectx(prologue_trc):
            ii: InterpreterInfo
            for ii in interpreter_ctx.inputs:
                unpack(ii)

            # Returns all inputs
            # TODO We should review the computation trace and only unpack and return the inputs that
            #   actually get used in the computation
            rvals = tuple(unvariableify(x) for x in prologue_rvals)
            prims.python_return(rvals)

        # Constructs the computation trace's signature
        si = SigInfo("computation")
        si.args = list((p.name, None) for p in rvals)
        computation_trc._siginfo = si
        computation_trc.args = rvals

        # Unpacks inputs
        # TODO This currently does the unpacks at the end of he trace, then moves them to the beginning, there's
        #   almost certainly a more elegant way to do this
        with tracectx(computation_trc):
            for p in rvals:
                prims.unpack_trivial(p)

        bsyms = computation_trc.bound_symbols
        computation_trc.bound_symbols = bsyms[-len(rvals) :] + bsyms[: -len(rvals)]

        # TODO Apply transforms like grad

        extraces = transform_for_execution(
            computation_trc,
            executors_list=cd.executors_list,
        )

        # TODO Apply post-optimiation transforms

        extrace = extraces[-1]

        pro = prologue_trc.python_callable(global_dicts=global_dicts)
        c = extrace.python_callable()

        # Executes the traced program
        cs.last_trace_host_execution_start = time.time_ns()
        computation_result = c(*pro(*args, **kwargs))
        cs.last_trace_host_execution_stop = time.time_ns()

        # TODO Update cache

        # Updates metadata
        # TODO What should the last_traces be in this case?
        cs.last_traces = extraces
        # TODO What should the last executed be in this case?
        cs.last_executed = c
        cs.last_interpreted_instructions = jfn._last_interpreted_instructions
        cs.last_interpreted_history = jfn._last_interpreted_history
        cs.last_prologue = prologue_trc

        cs.last_trace_host_stop = time.time_ns()
        return computation_result

    fn_._lc_cd = cd
    fn_._lc_cs = cs
    return fn_


# TODO Support recursive litjiting
# NOTE This is an analogue to thunder.compile, because how it handles trace generation
#   is sufficiently distinct that merging the two would be quite tricky
def litjit(
    fn: Callable, executors_list: None | Sequence[Executor] = None, debug_log: None | StringIO = None
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=None,
        executors_list=executors_list,
        cache_mode=None,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=True,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=True,
        debug_log=debug_log,
    )

    cs = CompileStats()
    fn_ = _create_callable(cd, cs)
    return fn_
