from contextlib import contextmanager
from enum import Enum, auto
from typing import Any
import warnings

import torch

from thunder.core.transform_common import Transform
from thunder.extend import Executor, get_default_executors


@contextmanager
def pretty_warnings():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        yield
        for warning in caught_warnings:
            if issubclass(warning.category, UserWarning):
                print(f"{warning.category.__name__}: {warning.message}")


class Lookaside:
    def __init__(self, fn, replace_with):
        self._fn = fn
        self._replace_with = replace_with


class PluginPolicy(Enum):
    PRE = auto()
    POST = auto()


class Plugin:
    policy: PluginPolicy = PluginPolicy.PRE

    def setup_lookasides(self) -> list[Lookaside] | None:
        return None

    def setup_transforms(self) -> list[Transform] | None:
        return None

    def setup_executors(self) -> list[Executor] | None:
        return None


class Interpreter(Enum):
    THUNDER_JIT = auto()
    THUNDER_FX = auto()


class Recipe:
    def __init__(self, plugins: Plugin, interpreter: Interpreter = Interpreter.THUNDER_JIT):
        self.lookasides = []
        self.transforms = []
        self.executors = []
        self.plugins = plugins if plugins is not None else []
        if isinstance(interpreter, str):
            if interpreter == "thunder.jit":
                interpreter = Interpreter.THUNDER_JIT
            elif interpreter == "thunder.fx":
                interpreter = Interpreter.THUNDER_FX
            else:
                raise ValueError(
                    f"Invalid interpreter {interpreter}. Supported interpreters: ['thunder.jit', 'thunder.fx']."
                )
        self.interpreter = interpreter

    def add_plugins(self, plugins: list[Plugin]):
        self.plugins.extend(plugins)

    @classmethod
    def validate(cls, model):
        # this is expected to raise if validation fails
        pass

    def setup_lookasides(self) -> list[Lookaside] | None:
        return None

    def setup_transforms(self) -> list[Transform] | None:
        return None

    def setup_executors(self) -> list[Executor]:
        return list(get_default_executors())

    def setup_config(self) -> dict[str, Any]:
        return {}

    def apply(self, model):
        with pretty_warnings():
            self.validate(model)

        self.config = self.setup_config()

        pre_plugins = [el for el in self.plugins if el.policy == PluginPolicy.PRE]
        post_plugins = [el for el in self.plugins if el.policy == PluginPolicy.POST]

        lookasides = []

        for plugin in pre_plugins:
            lookasides.extend(plugin.setup_lookasides() or [])

        lookasides.extend(self.setup_lookasides() or [])

        for plugin in post_plugins:
            lookasides.extend(plugin.setup_lookasides() or [])

        from thunder.core import jit_ext, interpreter

        if lookasides is not None:
            for lookaside in lookasides:
                wrapped_replacement_fn = interpreter.interpreter_needs_wrap(lookaside._replace_with)
                jit_ext._general_jit_lookaside_map[lookaside._fn] = wrapped_replacement_fn

        self.lookasides = lookasides

        transforms = []
        for plugin in pre_plugins:
            transforms.extend(plugin.setup_transforms() or [])

        transforms.extend(self.setup_transforms() or [])

        for plugin in post_plugins:
            transforms.extend(plugin.setup_transforms() or [])

        self.transforms = transforms

        executors = []
        for plugin in pre_plugins:
            executors.extend(plugin.setup_executors() or [])

        executors.extend(self.setup_executors() or [])

        for plugin in post_plugins:
            executors.extend(plugin.setup_executors() or [])

        self.executors = executors

        if self.interpreter == Interpreter.THUNDER_JIT:
            from thunder import jit

            thunder_model = jit(model, transforms=self.transforms, executors=self.executors, **self.config)

        elif self.interpreter == Interpreter.THUNDER_FX:
            from thunder.dynamo import ThunderCompiler

            thunder_backend = ThunderCompiler(transforms=self.transforms, executors=self.executors, **self.config)
            thunder_model = torch.compile(model, backend=thunder_backend)

        else:
            raise AttributeError(
                f"Interpreter must be one of 'Interpreter.THUNDER_JIT', 'Interpreter.THUNDER_FX'. Found: {self.interpreter}."
            )

        return thunder_model
