from thunder import Recipe, Plugin, DebugOptions, Transform, Executor
from thunder.core.recipe import Interpreter
from thunder.executors.torch_compile import torch_compile_ex
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks

from typing import Any


class BaseRecipe(Recipe):
    def __init__(
        self,
        show_progress=False,
        fuser="nvfuser",
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        self.fuser = fuser
        self.show_progress = show_progress

    def setup_config(self) -> dict[str, Any]:
        if not self.show_progress:
            return {}
        return dict(debug_options=DebugOptions(show_interpreter_progress=True))

    def setup_transforms(self) -> list[Transform]:
        transforms = [PrunePrologueChecks()]

        return transforms

    def setup_executors(self) -> list[Executor]:
        executors = super().setup_executors()

        if self.fuser == "nvfuser":
            return executors
        elif self.fuser == "torch.compile":
            executors = [el for el in executors if el.name not in ["torchcompile_cat", "nvfuser"]]
            executors.append(torch_compile_ex)
            return executors

        raise ValueError(f"Invalid fuser {self.fuser}. Allowed fusers: 'nvfuser', 'torch.compile'.")
