from thunder import Recipe, Plugin, DebugOptions, Transform, Executor
from thunder.core.recipe import Interpreter
from thunder.executors import nvfuser_available
from thunder.executors.torch_compile import torch_compile_ex
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks

from typing import Any


class BaseRecipe(Recipe):
    def __init__(
        self,
        show_progress=False,
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        self.executors = ["cudnn", "sdpa", "torchcompile_xentropy", "nvfuser"]
        self.show_progress = show_progress

    def setup_config(self) -> dict[str, Any]:
        if not self.show_progress:
            return {}
        return dict(debug_options=DebugOptions(show_interpreter_progress=True))

    def setup_transforms(self) -> list[Transform]:
        transforms = [PrunePrologueChecks()]

        return transforms

    def setup_executors(self) -> list[Executor]:
        if not isinstance(self.executors, list):
            raise TypeError(f"self.executors must be a list of executor names, got {type(self.executors).__name__}")

        available_ex = super().setup_executors()
        available_ex_map = {ex.name: ex for ex in available_ex}
        selected_ex = []
        for name in self.executors:
            if name not in available_ex_map:
                if name == "nvfuser":
                    raise ValueError(
                        """Executor nvfuser was specified in the recipe but is not available in the current environment.
                    See https://github.com/Lightning-AI/lightning-thunder/?tab=readme-ov-file#quick-start for install instructions."""
                    )
                else:
                    raise ValueError(
                        f"Executor '{name}' was specified in the recipe but is not available in the current environment."
                    )

            selected_ex.append(available_ex_map[name])

        return selected_ex
