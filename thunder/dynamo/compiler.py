from __future__ import annotations
from functools import partial
from looseversion import LooseVersion
from typing import TYPE_CHECKING
import warnings

import torch

from thunder.core.baseutils import run_once
from thunder.dynamo.utils import recompile_graph, remove_empty_autocast
from thunder.dynamo.splitter import _splitter

if TYPE_CHECKING:
    from thunder.dynamo.utils import SubgraphInfo


@run_once
def _warn_thunder_compiler():
    warnings.warn(
        "The ThunderCompiler is in active development and may not work as expected."
        + " Please report any issues you encounter to the Lightning Thunder team."
    )


class ThunderCompiler:
    def __init__(self, **thunder_options):
        """
        A class that compiles a :class:`torch.fx.GraphModule` to a :class:`thunder.ThunderModule`.
        This class is meant to be used as a backend for the :func:`torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to :func:`thunder.jit`. Besides all the arguments to :func:`thunder.jit`,
                             it accepts `torch_inductor_options` which are passed to `torch.compile` if part of the graph
                             is not supported by thunder.

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
        from thunder import jit

        _warn_thunder_compiler()

        if LooseVersion(torch.__version__) < LooseVersion("2.4.0"):
            # NOTE: PyTorch 2.3 or lower has bug in `split_module` function used in splitter.
            # See https://github.com/Lightning-AI/lightning-thunder/pull/1075#issuecomment-2324918409
            err_msg = f"thunder.jit as torch.compile backend is only supported with PyTorch version 2.4 or later, found version {torch.__version__}"
            raise RuntimeError(err_msg)

        # Thunder-compiled functions should be readily available for inspection
        # and testing, so we will store them in a list[SubgraphInfo]. The order of the
        # functions in the list will be the same as the order in which they were
        # compiled.
        # Ref to the documentation of `SubgraphInfo` to know more about the information it contains.
        self.subgraph_infos: list[SubgraphInfo] = []

        torch_inductor_options = thunder_options.pop("torch_inductor_options", {})

        self.thunder_options = thunder_options
        self._thunder_jit = partial(jit, **thunder_options)
        self._torch_compile = partial(torch.compile, **torch_inductor_options)

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        gm = remove_empty_autocast(gm)

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        recompile_graph(gm)

        # The whole graph may not be supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        split_module, subgraph_info = _splitter(gm, self._thunder_jit, self._torch_compile, sample_args)
        self.subgraph_infos.append(subgraph_info)
        return split_module
