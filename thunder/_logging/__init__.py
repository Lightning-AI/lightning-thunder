from __future__ import annotations
from typing import TYPE_CHECKING
import functools
import logging
import os
import sys
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "get_logger",
]


_THUNDER_LOGGER_SETUP: bool = False
_THUNDER_LOGS: str = "THUNDER_LOGS"
_LOG_NAME_TO_LEVEL: dict[str, int] = {}
_TRACE_LOG_LEVEL: int | None = None


_HELP_MSG = (
    "THUNDER_LOGS help\n"
    "Supported values of THUNDER_LOGS are:\n"
    "\texecutors: Allow all executors to log\n"
    "\t<executor name>: Allow specified executor to log\n"
    'By prefixing an executor\'s name or "executors" with `+` or `-`, logging level is set to logging.DEBUG or logging.WARNING, otherwise logging.INFO'
    "Note that multiple executor names can be specified\n"
    'Example command: `THUNDER_LOGS="+nvfuser,sdpa" python example.py` would show logging.DEBUG and higher messages of nvfuser and logging.WARNING and higher of sdpa executor'
)


@functools.cache
def _get_thunder_logs_value() -> tuple[str] | None:
    thunder_log_configs: str | None = os.environ.get(_THUNDER_LOGS)
    if thunder_log_configs is not None:
        return tuple(a.lower() for a in thunder_log_configs.split(","))
    return None


def _get_log_level(thunder_log_value: str) -> int:
    if thunder_log_value.startswith("+"):
        return logging.DEBUG
    elif thunder_log_value.startswith("-"):
        return logging.WARNING
    else:
        return logging.INFO


def _setup_logging():
    thunder_log_configs: str | None = _get_thunder_logs_value()

    if thunder_log_configs is not None and (len(thunder_log_configs) == 1 and thunder_log_configs[0].lower() == "help"):
        warnings.warn(_HELP_MSG)
        sys.exit(0)

    is_logging_active: bool = False
    if thunder_log_configs is not None:
        global _TRACE_LOG_LEVEL, _LOG_NAME_TO_LEVEL
        for logger_name in thunder_log_configs:
            if logger_name.lower() in ("debug", "info", "warning", "error", "critical"):
                msg = f"Logging level {logger_name} is not supported and is ignored. \n{_HELP_MSG}"
                warnings.warn(msg)
                continue
            log_level = _get_log_level(logger_name)
            if logger_name.endswith("traces"):
                _TRACE_LOG_LEVEL = log_level
            else:
                _LOG_NAME_TO_LEVEL[logger_name] = log_level
    if not is_logging_active:
        logging.basicConfig(level=logging.CRITICAL + 100)


def extract_executor_names_from_thunder_logs() -> set[str] | None:
    thunder_log_configs = _get_thunder_logs_value()

    if thunder_log_configs is None:
        return None

    thunder_log_configs = [e if not e.startswith("+") else e[1:] for e in thunder_log_configs]
    return set(thunder_log_configs)


def define_filter_for_extend_executor() -> Callable[[logging.LogRecord], bool]:
    """Defines filter for `thunder.extend.Executor.can_execute`.

    For example, `THUNDER_LOGS=nvfuser`, messages from `can_execute` will be filtered
    out if the executor object's name is not nvfuser.
    When `THUNDER_LOGS=executors`, then no messages are filtered.
    """
    from thunder.core.prims import PrimIDs

    always_true: bool = "executors" in _LOG_NAME_TO_LEVEL
    names: set[str] = set(_LOG_NAME_TO_LEVEL.keys()) if not always_true else set()

    def filter(record: logging.LogRecord) -> bool:
        if always_true:
            return True

        return record.executor_name in names and (
            not isinstance(record.symbol_id, PrimIDs) or record.symbol_id.value > PrimIDs.PUT_GRAD.value
        )

    return filter


def get_logger(name: str) -> logging.Logger:
    """Get logger for a given name.

    Name usually is a module name such as `thunder.extend` or `thunder.executors.nvfuserex_impl`.
    """
    if not _THUNDER_LOGGER_SETUP:
        _setup_logging()
    logger = logging.getLogger(name)

    level: int | None = None
    filter: logging.Filter | Callable[[logging.LogRecord], bool] | None = None
    if name == "thunder" and _TRACE_LOG_LEVEL is not None:
        level = _TRACE_LOG_LEVEL
    elif "extend" in name:
        if _LOG_NAME_TO_LEVEL:
            if "executors" in _LOG_NAME_TO_LEVEL:
                level = _LOG_NAME_TO_LEVEL["executors"]
            else:
                level = min(_LOG_NAME_TO_LEVEL.values())
            filter = define_filter_for_extend_executor()
    elif "executors" in name:
        for log_name, log_level in _LOG_NAME_TO_LEVEL.items():
            if log_name in name:
                level = log_level
                break
    else:
        pass
    if level is None:
        level = logging.CRITICAL + 100

    # logging.handlers is a list of handlers as per https://docs.python.org/3/library/logging.html#logging.Logger.handlers
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(fmt=formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(level)
        logger.propagate = False

        if filter is not None:
            logger.addFilter(filter)
    return logger
