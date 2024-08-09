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

        # Thunder-compiled functions should be readily available for inspection
        # and testing, so we will store them in a list. The order of the
        # functions in the list will be the same as the order in which they were
        # compiled. In addition, we will store a mapping from the ThunderModule
        # to the GraphModule that was passed to ThunderCompiler. This will allow
        # us to inspect the GraphModule that was compiled by Thunder.
        self.thunder_fns: list[ThunderModule] = []
        self.thunder_to_gm: OrderedDict[ThunderModule, torch.fx.GraphModule] = OrderedDict()

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
        self.thunder_fns.append(jitted_gm)
        self.thunder_to_gm[jitted_gm] = gm
        return jitted_gm
