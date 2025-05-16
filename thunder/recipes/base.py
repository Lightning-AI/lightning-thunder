from thunder import Recipe, Plugin, DebugOptions, Transform, Executor
from thunder.core.recipe import Interpreter
from thunder.executors import nvfuser_available
from thunder.executors.torch_compile import torch_compile_ex
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks

from typing import Any


class BaseRecipe(Recipe):
    def __init__(
        self,
        executors,
        show_progress=False,
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        if not isinstance(executors, list) or not all(isinstance(x, str) for x in executors):
            raise TypeError("executors must be a list of strings")
        self.executors = executors
        self.show_progress = show_progress

    def setup_config(self) -> dict[str, Any]:
        if not self.show_progress:
            return {}
        return dict(debug_options=DebugOptions(show_interpreter_progress=True))

    def setup_transforms(self) -> list[Transform]:
        transforms = [PrunePrologueChecks()]

        return transforms

    def setup_executors(self) -> list[Executor]:
        available_ex = super().setup_executors()
        available_ex_map = {ex.name: ex for ex in available_ex}

        selected_ex = []
        for name in self.executors:
            if name == "nvfuser" and not nvfuser_available:
                raise RuntimeError("NVFuser was specified as an executor but is not available.")

            if name not in available_ex_map:
                raise ValueError(f"Executor {name} is not supported.")

            selected_ex.append(available_ex_map[name])

        if not selected_ex:
            raise ValueError(f"No matching executors found for: {self.executors}")

        return selected_ex
