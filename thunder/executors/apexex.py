from thunder.extend import OperatorExecutor, register_executor


# TODO Does apex have a version this should use?
apex_ex = OperatorExecutor("apex", version="0.1")
register_executor(apex_ex)


from thunder.executors.apex_entropyex_impl import apex_entropy_available
from thunder.executors.apex_fused_rms_norm_impl import apex_fused_norms_available

__all__ = ["apex_ex", "apex_entropy_available", "apex_fused_norms_available"]
