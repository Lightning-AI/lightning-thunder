from contextlib import contextmanager
from typing import Any
import warnings

from thunder.core.transform_common import Transform
from thunder.extend import Executor, get_default_executors

import torch


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


class Recipe:
    # thunder.jit | torch.compile
    compiler = "thunder.jit"

    def __init__(self):
        pass

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
        lookasides = self.setup_lookasides()

        from thunder.core import jit_ext, interpreter

        if lookasides is not None:
            for lookaside in lookasides:
                wrapped_replacement_fn = interpreter.interpreter_needs_wrap(lookaside._replace_with)
                jit_ext._general_jit_lookaside_map[lookaside._fn] = wrapped_replacement_fn

        self.lookasides = lookasides
        self.executors = self.setup_executors()
        self.transforms = self.setup_transforms()

        if self.compiler == "thunder.jit":
            from thunder import jit

            thunder_model = jit(model, transforms=self.transforms, executors=self.executors, **self.config)

        elif self.compiler == "torch.compile":
            from thunder.dynamo import ThunderCompiler

            thunder_backend = ThunderCompiler(transforms=self.transforms, executors=self.executors, **self.config)
            thunder_model = torch.compile(model, backend=thunder_backend)

        else:
            raise AttributeError(f"Compiler must be one of 'thunder.jit', 'torch.compile'. Found: {self.compiler}.")

        return thunder_model


class DynamoRecipe(Recipe):
    compiler = "torch.compile"
