from typing import Any
from collections.abc import Callable
import collections
import traceback

import thunder
from thunder.core.trace import TraceCtx
from thunder.core.transforms import bsym_list_to_dag, Node
from thunder.core.proxies import TensorProxy, CollectionProxy
from thunder.core.symbol import BoundSymbol
from thunder.torch import _torch_to_thunder_function_map
from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.core.langctxs import resolve_language, LanguageContext, Languages
import torch
from warnings import warn
from itertools import chain
import importlib


# TODO Maybe make collect_into a set?
class CollectFunctionsUsed(torch.overrides.TorchFunctionMode):
    def __init__(self, collect_into: dict):
        self.functions_call_sites = collections.defaultdict(list)
        self.collect_into = collect_into

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        qn = getattr(func, "__qualname__", None)
        if qn.startswith("getset_descriptor."):
            qn = getattr(func.__self__, "__qualname__", qn)
        mod = getattr(func, "__module__", None)
        if mod is not None and qn is not None and not qn.startswith(mod):
            modstr = " of " + mod
        else:
            modstr = ""
        self.functions_call_sites[(f"{qn or func}{modstr}", func)].append(traceback.format_stack())
        return func(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.collect_into.update(sorted(self.functions_call_sites.items()))
        super().__exit__(exc_type, exc_value, traceback)


_method_name_remap_map = {
    "div": "true_divide",
}


# TODO Maybe have this print additional information and return more metadata?
# TODO Accept kwargs for jit (like langctx)
# TODO Add profiling (or profiling option) to determine if we have a slowdown
# TODO If an error occurs, try to minify the program to produce a smaller sample to reproduce the error
def examine(fn: Callable, *args, show_call_stack: bool | int = False, **kwargs):
    """
    show_call_stack: bool | int=False:  if you pass True, a call stack will be printed for each invocation of an unknown function. If you pass a number, the stack will be limited to a depth of that value.
    """
    # Step 0, runs the operation with our torch function mode to collection information
    #   and ensure the operation itself is working correctly
    collected_ops = {}
    torch_result: Any

    if not callable(fn):
        # `examine` doesn't throw error and doesn't crash the user program.
        # Hence, using print and return.
        print(
            f"examine: expected `fn` to be a callable instead received {type(fn)}. Use `examine(fn, *args, **kwargs)` to test `fn(*args, **kwargs)`"
        )
        return

    with CollectFunctionsUsed(collected_ops):
        try:
            torch_result = fn(*args, **kwargs)
        except Exception as e:
            print("Failed to run the unmodified function. Please verify that your code runs without thunder")
            print(f"The code failed with exception - {e}")
            return

    # Step 1 Identifies supported (and unsupported) operations
    supported_ops = set()
    all_auto_registered_ops = list(chain(*torch_auto_registered_ops.values()))
    auto_registered_ops = set()
    for name, op in collected_ops.keys():
        if op in _torch_to_thunder_function_map:
            supported_ops.add((name, op))
            if op in all_auto_registered_ops:
                auto_registered_ops.add(name)
        elif name.startswith("_TensorBase.") or name.startswith("TensorBase.") or name.startswith("Tensor."):
            # Identifies properties and methods
            # NOTE The approach of testing if the name starts with "_TensorBase." or "Tensor." seems a little hacky

            # Checks if the name is a property
            attr: str
            if name.startswith("_TensorBase."):
                _, attr = name.split(".")
            elif name.startswith("Tensor.") or name.startswith("TensorBase."):
                # Ex name 'Tensor.__rpow__ of torch._tensor'
                _, attr = name.split(" ")[0].split(".")

            # # torch.Tensor still has `__rdiv__` and sets `__rtruediv__=__rdiv__`
            # # Ref: https://github.com/pytorch/pytorch/blob/1deb75b5846c6bb39773a4f210379f983250d802/torch/_tensor.py#L935-L939
            attr = "__rtruediv__" if attr == "__rdiv__" else attr
            if hasattr(TensorProxy, attr):
                supported_ops.add((name, op))

            # Checks if the name is a method
            # Remaps method names as appropriate
            method_name = _method_name_remap_map.get(attr, attr)

            torchlang: LanguageContext = resolve_language(Languages.TORCH)
            if torchlang.has_method(method_name):
                supported_ops.add((name, op))

    unsupported_ops = set(collected_ops) - supported_ops

    if len(collected_ops) == 0:
        # NOTE This case avoids a division by zero error below
        print("Found no operations")
    else:
        print(
            f"Found {len(collected_ops)} distinct operations, of which {len(supported_ops)} ({len(supported_ops) / len(collected_ops) * 100:.1f}%) are supported"
        )
        if len(auto_registered_ops) != 0:
            print(f"Note {len(auto_registered_ops)} operators are automatically registered: ")
            for n in auto_registered_ops:
                print(n)

    # Terminates early if there are unsupported operations or there was a preprocessing exception
    if len(unsupported_ops) > 0:
        print(
            "Please file an issue requesting the following operators here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )

        for name, op in unsupported_ops:
            print(f"{name}")
            if show_call_stack is not False:
                call_sites = collected_ops[(name, op)]
                for i, cs in enumerate(call_sites[:5]):
                    if i == 0:
                        print("  used in")
                    else:
                        print("  and in")
                    if show_call_stack is True:
                        start_idx = 0
                        while start_idx < len(cs) and "thunder/examine/__init__.py" not in cs[start_idx]:
                            start_idx += 1
                        start_idx += 1
                    else:
                        start_idx = -1 - show_call_stack

                    print("  " + "  ".join(cs[start_idx:-1]))  # stop before -1 to split off the collector
                if len(call_sites) > i + 1:
                    print(f"  ...and {len(call_sites) - i - 1} more")

        return

    # Step 3 Attempts to compile the function using thunder.jit
    try:
        cfn = thunder.jit(fn)
    except Exception as e:
        print("Encountered an error while compiling the function")
        print(
            "Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )
        raise e

    # Step 4 Attempt to execute the function using thunder.jit
    lc_result: Any
    try:
        lc_result = cfn(*args, **kwargs)
    except Exception as e:
        print("Encountered an error while running the compiled function")
        print(
            "Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )
        # TODO On failure, try to identify where the failure occurred and produce a constructive error
        #   message -- did it happen during caching, unpacking, transformation, callable construction,
        #   or executing the callable?
        raise e

    # TODO Consider comparing the torch_result and lc_result -- they might reasonably be different but we
    #   warn about this

    # TODO Consider returning additional information
    print(f"The function appears to be working as expected")


def warn_fusions() -> bool:
    if not torch.cuda.is_available():
        warn("CUDA is not available, so no fusions will be created.")
        return True

    from thunder.executors.nvfuserex import nvfuser_available

    if not nvfuser_available():
        warn("nvFuser is not available, so no fusions will be created.")
        return True
    return False


# Acquires all fusions in the given trace, returning them as a tuple of
#   (name, fusion) pairs
def get_fusions(trace: TraceCtx, warn_if_fusions_unavailable: bool = True) -> list[tuple[str, Callable]]:
    if warn_if_fusions_unavailable and warn_fusions():
        return []

    fusions = []

    ctx = trace.python_ctx()

    for bsym in trace.bound_symbols:
        sym = bsym.sym
        if sym.is_fusion:
            fusions.append((sym.name, ctx[sym.name]))

    return fusions


# Acquires all the fusion BoundSymbols in the trace
def get_fusion_symbols(trace: TraceCtx, warn_if_fusions_unavailable: bool = True) -> list[BoundSymbol]:
    if warn_if_fusions_unavailable and warn_fusions():
        return []

    fusions = []

    for bsym in trace.bound_symbols:
        sym = bsym.sym
        if sym.is_fusion:
            fusions.append(bsym)

    return fusions


def get_nvfuser_fusion_definition(trace: TraceCtx, name: str, warn_if_fusion_unavailable: bool = True):
    """
    Return the fusion definition for the symbol with the provided name if found.
    """
    if warn_if_fusion_unavailable and warn_fusions():
        return None

    for bsym in trace.bound_symbols:
        if bsym.sym.is_fusion and bsym.sym.name == name:
            _, fusion_ctx, _ = bsym.gather_ctxs()
            if (fusion_definition := fusion_ctx.get(name, None)) is not None:
                return fusion_definition

    return None


def get_nvfuser_repro(trace: TraceCtx, fusion_name: str, /) -> str:
    """
    Helper function to get the repro of a specific nvFusion segment.
    """
    fusion = get_nvfuser_fusion_definition(trace, fusion_name)
    if fusion is None:
        raise RuntimeError(f"Unable to find fusion '{fusion_name}' in trace.")

    if fusion.last_used is None:
        raise RuntimeError(
            "Fusion definition needs to be executed to record the inputs. You must execute the fusion first before you can query the repro."
        )

    if fusion.last_inputs is None:
        raise RuntimeError(
            "Fusion definition inputs need to be recorded. Use compile option 'nv_store_fusion_inputs=True' while tracing."
        )

    fd = fusion.last_used
    # The API for nvFuser version >=2.14
    get_repro = getattr(fd, "repro_script_for", None)
    # The legacy nvFuser API
    if get_repro is None:
        get_repro = getattr(fd, "getReproString", None)
    if get_repro is None:
        raise RuntimeError("The installed version of nvFuser does not support repro generation unless on crash.")

    return get_repro(fusion.last_inputs)


# Copied from `pytorchviz` which has MIT License (Thank you!)
# See https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py#L180
def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def _repr_proxy(t_proxy, show_metadata=False):
    if isinstance(t_proxy, TensorProxy):
        # Should we just delegate to TensorProxy.__repr__ ?
        extra_meta = f"\n shape:{t_proxy.shape} \n dtype:{t_proxy.dtype}" if show_metadata else ""
        return f"name:{t_proxy.name}" + extra_meta

    # For any other proxy, we just print the name.
    return f"name:{t_proxy.name}"


def make_trace_dot(trace: TraceCtx, show_metadata=False):
    """
    Creates a directed graph of the given trace.

    This function is intended to be used to use graphviz to visualize the computation graph of a trace.
    Beware, rendering out a graph for large traces might take a while.

    Roots nodes are colored "green", intermediates are colored "lightblue" and leaves are colored "orange"

    Requires graphviz to be installed, for more information check out -> https://graphviz.readthedocs.io/en/stable/index.html

    .. note::
        To improve the rendering time, one can update `nslimit` and `nslimit1` graph attributes on the returned Digraph before
        calling the `render`.
        Eg. dot_graph.graph_attr["nslimit"] = 5

        Refer the following links for more details:
        [1] https://graphviz.org/docs/attrs/nslimit/
        [2] https://graphviz.org/docs/attrs/nslimit1/


    Args:
        trace (TraceCtx): The Thunder trace to be made into a graph.
        show_metadata (bool): Add more meta-data (like shape, dtype) to the nodes representing the Tensor. Defaults to False.

    Returns:
        graphviz.Digraph: A graphviz directed graph.
    """
    if not importlib.util.find_spec("graphviz"):
        warn("graphviz is not available. Graph cannot be created.")
        return

    import graphviz

    node_attr = dict(
        style="filled", shape="box", align="left", fontsize="10", ranksep="0.1", height="0.2", fontname="monospace"
    )
    dot = graphviz.Digraph(
        node_attr=node_attr,
        graph_attr=dict(size="10,10"),
    )
    dot.strict = True

    roots, leaves = bsym_list_to_dag(trace.bound_symbols)
    leaves_id = {id(leaf) for leaf in leaves}
    roots_id = {id(root) for root in roots}

    def _get_color(node_id):
        if node_id in roots_id:
            return "green"
        if node_id in leaves_id:
            return "orange"
        return "lightblue"

    # All roots will be positioned at same level due to `rank`:`same`.
    roots_g = graphviz.Digraph(graph_attr={"rank": "same"})
    for root in roots:
        roots_g.node(str(id(root)), root.bsym.python(indent=0, print_depth=1)[0], fillcolor=_get_color(str(id(root))))
    dot.subgraph(roots_g)  # Add roots_g to main graph.

    stack = [*roots]
    visited = set()

    # Breadth first
    while stack:
        node: Node = stack.pop(0)
        node_id = id(node)
        visited.add(node_id)
        color = _get_color(node_id)

        # Unpacking collection might be a multi-line.
        node_repr = "\n".join(node.bsym.python(indent=0, print_depth=1))
        node_repr = node_repr.replace("\\", "")
        dot.node(str(node_id), node_repr, fillcolor=color)

        # Add node for args and connect args
        for arg in node.bsym.flat_args:
            # We have collection proxies in backward
            if isinstance(arg, (TensorProxy, CollectionProxy)):
                arg_id = arg.name
                dot.node(arg_id, _repr_proxy(arg, show_metadata))
                dot.edge(arg_id, str(node_id))

        # Connect outputs
        for out in node.bsym.flat_outs:
            # We have collection proxies in backward
            if isinstance(out, (TensorProxy, CollectionProxy)):
                out_id = out.name
                dot.edge(str(node_id), out_id)

        # Add children for exploration
        for child in node.children:
            child_id = id(child)
            if child_id not in visited and not str(child.bsym).startswith("#"):
                stack.append(child)

    # Resize graph based on number of nodes
    resize_graph(dot)
    return dot
