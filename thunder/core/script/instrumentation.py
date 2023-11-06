import contextlib
import functools
import inspect
import logging
import threading
import typing

from thunder.core.utils import debug_asserts_enabled


T = typing.TypeVar("T")
_STORAGE = threading.local()


def _lookup_state(name: str, factory: typing.Callable[[], T]) -> T:
    if not hasattr(_STORAGE, name):
        setattr(_STORAGE, name, factory())
    return getattr(_STORAGE, name)


get_stack = functools.partial(_lookup_state, "stack", list)
get_init_ctx = functools.partial(_lookup_state, "init_ctx", dict)
get_error_ctx = functools.partial(_lookup_state, "error_ctx", list)
get_logger = functools.partial(_lookup_state, "logger", lambda: logging.error)


class InstrumentingBase:
    def __new__(cls, *_, **__) -> "InstrumentingBase":
        self = super().__new__(cls)
        if stack := get_stack():
            get_init_ctx()[id(self)] = (self, tuple(stack))

        return self

    def _concise_repr(self) -> str:
        return f"<{self.__class__.__name__} object at {hex(id(self))}>"


def emit_ctx(v, follow_delegates: bool):
    for f, args, kwargs, delegate_to in reversed(get_init_ctx()[id(v)][1]):
        signature = inspect.signature(f)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        def fmt_arg(k, v):
            if v is signature.parameters[k].default:
                return "..."

            if isinstance(v, InstrumentingBase):
                v_repr = v._concise_repr()
            elif callable(v) and hasattr(v, "__name__"):
                v_repr = v.__name__
            else:
                v_repr = repr(v)

            return v_repr

        if delegate_to is None or not follow_delegates:
            arg_str = ", ".join(fmt_arg(k, v) for k, v in bound.arguments.items())
            yield f"  {f.__name__:<30} {arg_str}"

        else:
            x = bound.arguments[delegate_to]
            yield f"  {f.__name__:<30} {fmt_arg(delegate_to, x)}"
            yield from emit_ctx(x, follow_delegates)
            break


def maybe_flush_errors():
    if not get_stack() and (error_ctx := get_error_ctx()):
        get_logger()("\n".join(reversed(error_ctx)) + "\n")
        error_ctx.clear()


@contextlib.contextmanager
def intercept_errors():
    prior_logger = get_logger()
    errors = []
    try:
        _STORAGE.logger = lambda s: errors.append(s)
        yield errors
    finally:
        _STORAGE.logger = prior_logger


def verbose_error(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not debug_asserts_enabled():
            return f(*args, **kwargs)

        try:
            return f(*args, **kwargs)
        except BaseException as e:
            bound = inspect.signature(f).bind(*args, **kwargs)
            bound.apply_defaults()

            f_name = f"| f.__name__ = {f.__name__} |"
            lines = [f"\n{'-' * len(f_name)}\n{f_name}\n{'-' * len(f_name)}\n"]
            for k, v in bound.arguments.items():
                lines.append(f"Argument(`{k}`):\n  {v}\n")
                if id(v) in get_init_ctx():
                    lines.extend(
                        [
                            "Context (raw):",
                            *reversed(tuple(emit_ctx(v, follow_delegates=False))),
                            "\nContext (augmented):",
                            *reversed(tuple(emit_ctx(v, follow_delegates=True))),
                        ]
                    )

            get_error_ctx().append("\n".join(lines))
            maybe_flush_errors()
            raise

    return wrapped


def record(delegate_to: typing.Optional[str] | typing.Callable = None):
    # Hack to allow you to to decorate with `@record` instead of `@record()`.
    if callable(delegate_to):
        return record()(delegate_to)

    def wrapper(f):
        f_verbose = verbose_error(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if not debug_asserts_enabled():
                return f(*args, **kwargs)

            stack = get_stack()
            try:
                stack.append((f, args, kwargs, delegate_to))
                return f_verbose(*args, **kwargs)

            finally:
                _ = stack.pop()
                maybe_flush_errors()
                if not stack:
                    get_init_ctx().clear()

        return wrapped

    return wrapper
