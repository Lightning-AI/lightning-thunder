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
