import torch.cuda
from thunder.dev_utils.debug_transform import DebugTransform
from thunder.core.prims import PrimIDs


class DebugMemoryTransform(DebugTransform):
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.memory_events = []

        def collect_memory_events(bsym, output, *args, **kwargs):
            torch.cuda.synchronize()
            snapshot = torch.cuda.memory._snapshot(device=self.device_index)
            events = snapshot["device_traces"][self.device_index]
            new_events = [event for event in events if event["size"] > 4]
            alloc = torch.cuda.memory_allocated(device=self.device_index)
            self.memory_events[-1].append((bsym, alloc, new_events))
            torch.cuda.memory._record_memory_history(clear_history=True, device=self.device_index)

            return (
                "\n".join(f"{event['action']} - {event['size']} bytes" for event in new_events)
                + f"\ntotal {alloc} bytes allocated"
            )

        super().__init__(post_callback=collect_memory_events)

    def is_applicable(self, bsym) -> bool:
        return bsym.sym.id == PrimIDs.DEL or super().is_applicable(bsym)

    def create_debug_boundsymbol(self, name, bsym, call_ctx, pass_result=False):
        debug_bsym = super().create_debug_boundsymbol(name, bsym, call_ctx, pass_result)
        if bsym.sym.id == PrimIDs.DEL and pass_result:
            # Arguments are deleted; pass only the output
            debug_bsym.args = (debug_bsym.args[0],)
        return debug_bsym

    def transform_trace_post_optimization(self, trace, **kwargs):
        torch.cuda.memory._record_memory_history(clear_history=True, device=self.device_index)
        self.memory_events.append([])
        return super().transform_trace_post_optimization(trace, **kwargs)


class DebugMemoryFXTransform:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.memory_events = []
        self.augumented_graph_modules = []

    def _collect_memory_events(self, node: torch.fx.Node):
        def inner_fn():
            torch.cuda.synchronize()
            snapshot = torch.cuda.memory._snapshot(device=self.device_index)
            events = snapshot["device_traces"][self.device_index]
            new_events = [event for event in events if event["size"] > 4]
            alloc = torch.cuda.memory_allocated(device=self.device_index)
            self.memory_events[-1].append((node, alloc, new_events))
            node.next.stack_trace = (
                'File "<debug>", line 0, in <debug>\nmemory_events = ['
                + ", ".join(f"{event['action']} - {event['size']} bytes" for event in new_events)
                + f"], total {alloc} bytes allocated"
            )
            torch.cuda.memory._record_memory_history(clear_history=True, device=self.device_index)

        inner_fn.__name__ = f"memory_events_{node.name}"

        return inner_fn

    def __call__(self, graph_module: torch.fx.GraphModule):
        torch.cuda.memory._record_memory_history(clear_history=True, device=self.device_index)
        self.memory_events.append([])

        def transform_module(gm: torch.fx.GraphModule):
            nodes = list(gm.graph.nodes)
            for node in nodes:
                if node.op == "call_function" and node.target in (
                    torch.ops.higher_order.tag_activation_checkpoint,
                    torch.utils.checkpoint.checkpoint,
                ):
                    m = node.graph.owning_module
                    for arg_node in node.args:
                        if arg_node.op == "get_attr":
                            called_module = getattr(m, arg_node.target)
                            transform_module(called_module)
                            break
                    else:
                        raise RuntimeError(f"Unexpected call_function node: {node}, target: {node.target}")
                    continue

                node.append(gm.graph.create_node("call_function", self._collect_memory_events(node)))
            return gm

        transform_module(graph_module)
        self.augumented_graph_modules.append(graph_module)
        return graph_module
