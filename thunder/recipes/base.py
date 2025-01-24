import thunder
from thunder.transforms.cudagraph import CUDAGraphTransform
from thunder.transforms.prune_prologue_checks import PrunePrologueChecks

from typing import Any

class BaseRecipe(thunder.Recipe):
    def __init__(self, reduce_overhead=True, fuser="nvfuser", show_progress=False):
        super().__init__()
        self.reduce_overhead = reduce_overhead
        self.fuser = fuser
        self.show_progress = show_progress

    def setup_config(self) -> dict[str, Any]:
        if not self.show_progress:
            return {}
        return dict(debug_options=thunder.DebugOptions(show_interpreter_progress=True))

    def setup_transforms(self) -> list[thunder.Transform]:
        transforms: list[thunder.Transform] = [PrunePrologueChecks()]

        if self.reduce_overhead:
            return transforms + [CUDAGraphTransform()]

        return transforms

    def setup_executors(self) -> list[thunder.Executor]:
        executors = super().setup_executors()

        if self.fuser == "nvfuser":
            return executors
        elif self.fuser == "torch.compile":
            executors = [el for el in executors if el.name not in ["torchcompile_cat", "nvfuser"]]
            executors.append(thunder.executors.torch_compile.torch_compile_ex)
            return executors

        raise ValueError(f"Invalid fuser {self.fuser}. Allowed fusers: 'nvfuser', 'torch.compile'.")
