import torch

from collections import OrderedDict


class ThunderCompiler:
    def __init__(self, **thunder_options):
        """
        A class that compiles a `fx.GraphModule` to a `thunder.ThunderModule`.
        This class is meant to be used as a backend for the `torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to `thunder.jit`.

        Example:
            >>> import torch
            >>> from thunder.dynamo import ThunderCompiler
            >>> backend = ThunderCompiler()
            >>> x = torch.ones(2, requires_grad=True)
            >>> @torch.compile(backend=backend)
            ... def func(x):
            ...     x = torch.sin(x)
            ...     if x.sum() > 0:
            ...         return x + 1
            ...     else:
            ...         return x - 1
            >>> out = func(x)
        """
        from thunder import ThunderModule

        # The dictionary is ordered so that the last compiled GraphModule is the
        # most recent one.
        self.gm_to_thunder: OrderedDict[torch.fx.GraphModule, ThunderModule] = OrderedDict()
        self.thunder_options = thunder_options

        # There will be pieces of Dynamo IR that Thunder cannot compile, so we
        # will need to build a fallback mechanism to handle those cases.
        # Possible stages of the compilation that need to be saved for inspection:
        # 1. The GraphModule as it was passed to ThunderCompiler.
        # 2. The GraphModule after split for Thunder/PyTorch.
        # 3. If the whole GraphModule is not supported, record the reasons why.

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        from thunder import jit

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        gm.real_recompile()

        # Here in the future we could add some logic to check if the GraphModule
        # is executable by Thunder, but for now we simply compile it and return
        jitted_gm = jit(gm, **self.thunder_options)
        self.gm_to_thunder[gm] = jitted_gm
        return jitted_gm
