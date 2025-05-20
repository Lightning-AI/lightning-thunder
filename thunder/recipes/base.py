import torch
from thunder import Recipe, Plugin, DebugOptions, Transform, Executor
from thunder.core.recipe import Interpreter
from thunder.executors import nvfuser_available
from thunder.executors.torch_compile import torch_compile_ex
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks
from thunder.extend import get_all_executors
from typing import Any


def get_nvfuser_package_for_torch(torch_version) -> str:
    torch_major_minor = torch_version[:4].replace(".", "")

    nvfuser_map = {
        "21": "nvfuser-cu118-torch21",
        "22": "nvfuser-cu118-torch22",
        "23": "nvfuser-cu118-torch23",
        "24": "nvfuser-cu118-torch24",
        "25": "nvfuser-cu124-torch25",  # prefer cu124 over cu121
        "26": "nvfuser-cu126-torch26",  # prefer cu126 over cu124
        "27": "nvfuser-cu128-torch27",  # prefer cu128 over cu126
    }

    if torch_major_minor not in nvfuser_map:
        raise RuntimeError(
            f"No known nvFuser wheel for PyTorch {torch_version}. "
            f"Supported versions are: {', '.join(sorted(nvfuser_map.keys()))}"
        )

    return nvfuser_map[torch_major_minor]


class BaseRecipe(Recipe):
    def __init__(
        self,
        show_progress=False,
        fuser="nvfuser",
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        self.executors = ["cudnn", "sdpa", "torchcompile_xentropy"]
        self.fuser = fuser
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

        available_ex = list(get_all_executors())
        available_ex_map = {ex.name: ex for ex in available_ex}
        selected_ex = []
        for name in self.executors:
            if name not in available_ex_map:
                if name == "nvfuser":
                    torch_version = torch.__version__.split("+")[0]
                    nvfuser_package = get_nvfuser_package_for_torch(torch_version)

                    raise ValueError(
                        f"""Executor 'nvfuser' was specified in the recipe but is not available in the current environment.
Please ensure it is installed by running:
    pip install torch=={torch_version} {nvfuser_package}

Note that nvfuser needs to be installed for the exact PyTorch version in use.
Alternatively, switch to torch.compile by setting `fuser="torch.compile"`.
"""
                    )

                else:
                    raise ValueError(
                        f"Executor '{name}' was specified in the recipe but is not available in the current environment."
                    )

            selected_ex.append(available_ex_map[name])

        return selected_ex
