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
        fuser="nvfuser",
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        self.executors = ["torch", "python", "cudnn", "sdpa"]
        self.setup_fuser()
        self.show_progress = show_progress

    def setup_config(self) -> dict[str, Any]:
        if not self.show_progress:
            return {}
        return dict(debug_options=DebugOptions(show_interpreter_progress=True))

    def setup_transforms(self) -> list[Transform]:
        transforms = [PrunePrologueChecks()]

        return transforms

    def setup_fuser(self) -> None:
        if self.fuser == "nvfuser":
            if "nvfuser" not in self.executors:
                self.executors.append("nvfuser")
        elif self.fuser == "torch.compile":
            if "torchcompile_xentropy" in self.executors:
                self.executors.remove("torchcompile_xentropy")
            if "torchcompile" not in self.executors:
                self.executors.append("torchcompile")
        else:
            raise NotImplementedError(
                f"Unknown fuser '{self.fuser}'. Supported options are 'nvfuser' and 'torch.compile'."
            )

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
                    Please ensure it is installed by running:
                    ```
                    pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
                    ```
                    Note that nvfuser needs to be installed for the exact pytorch version in use.
                    Alternatively you can switch from nvFuser to torch.compile as the fusion executor by specifying `fuser="torch.compile"`.
                    """
                    )
                else:
                    raise ValueError(
                        f"Executor '{name}' was specified in the recipe but is not available in the current environment."
                    )

            selected_ex.append(available_ex_map[name])

        return selected_ex
