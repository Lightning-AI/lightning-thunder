import torch.cuda
from thunder.dev_utils.debug_transform import DebugTransform
from thunder.core.prims import PrimIDs


class DebugMemoryTransform(DebugTransform):
    def __init__(self):
        self.memory_events = []

        def collect_memory_events(bsym, output, *args, **kwargs):
            torch.cuda.synchronize()
            snapshot = torch.cuda.memory._snapshot()
            events, = snapshot["device_traces"]
            new_events = [event for event in events if event["size"] > 4]
            alloc = torch.cuda.memory_allocated()
            self.memory_events[-1].append((bsym, alloc, new_events))
            torch.cuda.memory._record_memory_history(clear_history=True)

            return "\n".join(f"{event["action"]} - {event["size"]} bytes" for event in new_events) + f"\ntotal {alloc} bytes allocated"

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
        torch.cuda.memory._record_memory_history(clear_history=True)
        self.memory_events.append([])
        return super().transform_trace_post_optimization(trace, **kwargs)


class DebugMemoryFXTransform:
    def __init__(self):
        self.memory_events = []
        self.augumented_graph_modules = []

    def _collect_memory_events(self, node: torch.fx.Node):
        def inner_fn():
            torch.cuda.synchronize()
            snapshot = torch.cuda.memory._snapshot()
            events, = snapshot["device_traces"]
            new_events = [event for event in events if event["size"] > 4]
            alloc = torch.cuda.memory_allocated()
            self.memory_events[-1].append((node, alloc, new_events))
            node.next.stack_trace = f"File \"<debug>\", line 0, in <debug>\nmemory_events = [" + ", ".join(f"{event["action"]} - {event["size"]} bytes" for event in new_events) + f"], total {alloc} bytes allocated"
            torch.cuda.memory._record_memory_history(clear_history=True)

        inner_fn.__name__ = f"memory_events_{node.name}"

        return inner_fn

    def __call__(self, graph_module: torch.fx.GraphModule):
        torch.cuda.memory._record_memory_history(clear_history=True)
        self.memory_events.append([])
        nodes = list(graph_module.graph.nodes)
        for node in nodes:
            node.append(graph_module.graph.create_node("call_function", self._collect_memory_events(node)))
        self.augumented_graph_modules.append(graph_module)
        return graph_module
