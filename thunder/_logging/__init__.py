from __future__ import annotations
import functools
import logging
import os
import sys
import warnings


__all__ = [
    "get_logger",
]


_THUNDER_LOGGER_SETUP: bool = False
_THUNDER_LOGS: str = "THUNDER_LOGS"


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

    thunder_log_configs = [e for e in thunder_log_configs if e.upper() not in _STR_TO_LOGGING_LEVEL]
    thunder_log_configs = [e if not e.startswith("+") else e[1:] for e in thunder_log_configs]
    return set(thunder_log_configs)


class ExecutorLogFilter:
    def __init__(self, name):
        super().__init__()
        self._is_logger_for_executor = "executor" in name
        self.names_not_to_filter = extract_executor_names_from_thunder_logs()

    @functools.cached_property
    def always_true(self) -> bool:
        return self.names_not_to_filter is None or not self.names_not_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        if self.always_true:
            return True

        if (maybe_executor_name := getattr(record, "executor_name", None)) is None:
            if self._is_logger_for_executor:
                return any(name in record.name for name in self.names_not_to_filter)
            else:
                return True
        else:
            return maybe_executor_name in self.names_not_to_filter


def get_logger(name: str) -> logging.Logger:
    if not _THUNDER_LOGGER_SETUP:
        _setup_logging()
    logger = logging.getLogger(name)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(fmt=formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    executor_log_filter = ExecutorLogFilter(name)
    logger.addFilter(executor_log_filter)
    return logger
