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
_STR_TO_LOGGING_LEVEL: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


@functools.cache
def _get_thunder_logs_value() -> tuple[str] | None:
    thunder_log_configs: str | None = os.environ.get(_THUNDER_LOGS)
    if thunder_log_configs is not None:
        return tuple(a.lower() for a in thunder_log_configs.split(","))
    return None


@functools.cache
def available_executor_names() -> set[str]:
    from thunder.extend import get_all_executors

    return {e.name for e in get_all_executors()}


def _setup_logging():
    thunder_log_configs: str | None = _get_thunder_logs_value()

    if thunder_log_configs is not None and (len(thunder_log_configs) == 1 and thunder_log_configs[0].lower() == "help"):
        levels = "".join(
            [f"\t{key.lower()}: set logging level to {value}\n" for key, value in _STR_TO_LOGGING_LEVEL.items()]
        )
        msg = (
            "THUNDER_LOGS help\n"
            "Supported values of THUNDER_LOGS are:\n"
            "\texecutors: Allow all executors to log\n"
            "\t<executor name>: Allow specified executor to log\n"
            f"{levels}"
            'By giving `+` to each executor name or "executors", logging level is set to logging.DEBUG, otherwise logging.WARNING'
            "Note that multiple executor names can be specified\n"
            'Example command: `THUNDER_LOGS="+nvfuser,sdpa" python example.py` would show logging.DEBUG and higher messages of nvfuser and logging.WARNING and higher of sdpa executor'
        )
        warnings.warn(msg)
        sys.exit(0)

    is_logging_active: bool = False
    if thunder_log_configs is not None:
        for maybe_log_level in thunder_log_configs:
            if (logging_level := _STR_TO_LOGGING_LEVEL.get(maybe_log_level.upper())) is not None:
                logging.basicConfig(level=logging_level)
                is_logging_active = True
            else:
                if maybe_log_level.startswith("+"):
                    logging.basicConfig(level=logging.DEBUG)
                else:
                    logging.basicConfig(level=logging.WARNING)
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
    def __init__(self):
        super().__init__()
        self.names_not_to_filter = extract_executor_names_from_thunder_logs()

    @functools.cached_property
    def always_true(self) -> bool:
        return self.names_not_to_filter is None or not self.names_not_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        if self.always_true:
            return True

        if (maybe_executor_name := getattr(record, "executor_name", None)) is None:
            return True
        else:
            return maybe_executor_name in self.names_not_to_filter


def _create_filter() -> ExecutorLogFilter:
    return ExecutorLogFilter()


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
    executor_log_filter = _create_filter()
    logger.addFilter(executor_log_filter)
    return logger
