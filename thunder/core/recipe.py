from contextlib import contextmanager
from enum import Enum, auto
from typing import Any
import warnings

import torch

from thunder.core.transform_common import Transform
from thunder.extend import Executor, TemporaryExecutor

_RECIPE_REGISTRY: dict[str, type["Recipe"]] = {}


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
        self._lookaside_executor = None

    def add_plugins(self, plugins: list[Plugin]):
        self.plugins.extend(plugins)

    @classmethod
    def validate(cls, model):
        return True

    def setup_lookasides(self) -> list[Lookaside] | None:
        return None

    def setup_transforms(self) -> list[Transform] | None:
        return None

    def setup_executors(self) -> list[Executor]:
        return []

    def setup_config(self) -> dict[str, Any]:
        return {}

    _registry = _RECIPE_REGISTRY  # optional alias for clarity

    @classmethod
    def register(cls, key: str):
        def decorator(subcls):
            cls._registry[key] = subcls
            return subcls

        return decorator

    @classmethod
    def get_for_model(cls, model):
        module_path = f"{model.__class__.__module__}.{model.__class__.__name__}"
        parts = module_path.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            recipe_cls = cls._registry.get(key)
            if recipe_cls and recipe_cls.validate(model):
                return recipe_cls()

        default_recipe_cls = cls._registry.get("")
        if default_recipe_cls:
            return default_recipe_cls()
        raise RuntimeError("No applicable recipe found and no default registered.")

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
            self._lookaside_executor = TemporaryExecutor()
            for lookaside in lookasides:
                self._lookaside_executor._lookasides[lookaside._fn] = lookaside._replace_with

        self.lookasides = lookasides

        transforms = []
        for plugin in pre_plugins:
            transforms.extend(plugin.setup_transforms() or [])

        transforms.extend(self.setup_transforms() or [])

        for plugin in post_plugins:
            transforms.extend(plugin.setup_transforms() or [])

        self.transforms = transforms

        executors = []

        if self._lookaside_executor is not None:
            executors.append(self._lookaside_executor)

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
