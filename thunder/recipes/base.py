import torch
from thunder import Recipe, Plugin, DebugOptions, Transform, Executor
from thunder.core.recipe import Interpreter
from thunder.executors import nvfuser_available
from thunder.executors.torch_compile import torch_compile_ex
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks
from thunder.extend import get_executor
from typing import Any


def get_nvfuser_package_hint() -> str:

    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda or "unknown"

    known_versions = {
        "2.5": "nvfuser-cu124-torch25",
        "2.6": "nvfuser-cu126-torch26",
        "2.7": "nvfuser-cu128-torch27",
    }

    torch_key = ".".join(torch_version.split(".")[:2])
    package = known_versions.get(torch_key)

    if package:
        return f"""nvFuser was specified but not found in your environment.
You are running torch {torch_version} and CUDA {cuda_version}.

Try installing the matching nvFuser package:
  pip install {package}

For more options, see:
  https://github.com/NVIDIA/Fuser

Alternatively, switch to the torch.compile fuser with `fuser="torch.compile"`.
"""
    else:
        return f"""nvFuser was specified but we don't currently support torch {torch_version}.

Please upgrade to torch 2.6 or 2.7 and then run
```
pip install nvfuser-cu126-torch26 # for torch 2.6
pip install nvfuser-cu128-torch27 # for torch 2.7
```

For compatibility options, see:
  https://github.com/NVIDIA/Fuser

Alternatively, switch to the torch.compile fuser with `fuser="torch.compile"`.
"""


class BaseRecipe(Recipe):
    def __init__(
        self,
        show_progress=False,
        fuser="nvfuser",
        interpreter="thunder.jit",
        plugins=None,
    ):
        super().__init__(interpreter=interpreter, plugins=plugins)
        self.executor_names = ["cudnn", "sdpa", "torchcompile_xentropy"]
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
            if "nvfuser" not in self.executor_names:
                self.executor_names.append("nvfuser")
        elif self.fuser == "torch.compile":
            if "torchcompile_xentropy" in self.executor_names:
                self.executor_names.remove("torchcompile_xentropy")
            if "torchcompile" not in self.executor_names:
                self.executor_names.append("torchcompile")
        else:
            raise NotImplementedError(
                f"Unknown fuser '{self.fuser}'. Supported options are 'nvfuser' and 'torch.compile'."
            )

    def setup_executors(self) -> list[Executor]:
        if not isinstance(self.executor_names, list):
            raise TypeError(
                f"self.executor_names must be a list of executor names, got {type(self.executor_names).__name__}"
            )

        executors = []

        for name in self.executor_names:
            executor = get_executor(name)
            if executor is None:

                if name == "nvfuser":
                    hint = get_nvfuser_package_hint()
                    print(hint)
                else:
                    raise ValueError(
                        f"Executor '{name}' was specified in the recipe but is not available in the current environment."
                    )

            executors.append(executor)

        return executors
