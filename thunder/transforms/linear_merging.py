import thunder

__all__ = [
    "LinearMerging",
]


class LinearMerging(thunder.Transform):
    def transform_traces_pre_prologue(self, prologue_trc, computation_trc, epilogue_trc, **kwargs):
        return prologue_trc, computation_trc, epilogue_trc
