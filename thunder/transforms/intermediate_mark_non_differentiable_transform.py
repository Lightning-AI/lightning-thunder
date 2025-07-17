from thunder.core.transforms import Transform
from thunder.core.trace import TraceCtx as Trace, TraceTag
from thunder.core.proxies import TensorProxy
from thunder.core.prims import PrimIDs


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []

    def add_parent(self, parent):
        self.parents.append(parent)

    def __repr__(self):
        return f"Node(name={self.name}, parents={self.parents}, children={self.children})"


def is_input_node(node, graph_nodes, input_nodes):
    if node.name in input_nodes:
        return True

    for parent in node.parents:
        if is_input_node(graph_nodes[parent], graph_nodes, input_nodes):
            return True

    return False


class IntermediateMarkNonDifferentiableTransform(Transform):
    """
    This transform is used to mark the intermediate output of the computation trace as non-differentiable.
    This is a workaround for the fact currently all output from thunderfx function is marked as differentiable (even if it is not).
    """

    def transform_trace_post_optimization(self, computation_trace: Trace, **kwargs):
        # We only need to do this for augmented forward traces
        # Any output which has interacted with the input Tensors is marked as differentiable.
        # TODO: Also consider the requires_grad flag of the input Tensors.
        if TraceTag.AUGMENTED_FORWARD in computation_trace.tags:

            def create_graph():
                graph_nodes = {}
                input_nodes = []

                def process_bsym(bsym):
                    if bsym.sym.id == PrimIDs.UNPACK_TRIVIAL:
                        # Output of unpack_trivial is treated as input_nodes.
                        for arg in bsym.flat_outs:
                            input_nodes.append(arg.name)

                    for output in bsym.flat_outs:
                        if isinstance(output, TensorProxy):
                            if output.name not in graph_nodes:
                                output_node = Node(output.name)
                                graph_nodes[output.name] = output_node
                            else:
                                output_node = graph_nodes[output.name]

                            for arg in bsym.flat_proxy_args:
                                output_node.add_parent(arg.name)

                for bsym in computation_trace.bound_symbols:
                    if bsym.sym.is_fusion:
                        for sub_bsym in bsym.subsymbols:
                            process_bsym(sub_bsym)
                    else:
                        process_bsym(bsym)

                return graph_nodes, input_nodes

            graph_nodes, input_nodes = create_graph()

            non_differentiable_output = []
            return_bsym = computation_trace.bound_symbols[-1]
            data_for_autograd, (_, _) = return_bsym.args

            for arg in data_for_autograd["flat_output"]:
                if not is_input_node(graph_nodes[arg.name], graph_nodes, input_nodes):
                    non_differentiable_output.append(True)
                else:
                    non_differentiable_output.append(False)

            data_for_autograd["non_differentiable_output"] = tuple(non_differentiable_output)

        return computation_trace
