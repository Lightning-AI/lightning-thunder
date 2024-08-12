from thunder.extend import OperatorExecutor, register_executor


# TODO Does apex have a version this should use?
apex_ex = OperatorExecutor("apex", version="0.1")
register_executor(apex_ex)

__all__ = [
    "apex_ex",
]
