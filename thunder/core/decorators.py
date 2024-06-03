import os


def avoid_torch_nccl_record_streams(func):
    """
    Avoids the allocator thrashing issue in PyTorch NCCL backend.
    """

    env_var = "TORCH_NCCL_AVOID_RECORD_STREAMS"
    value = os.environ.get(env_var, None)

    def wrapper(*args, **kwargs):
        try:
            os.environ[env_var] = "1"
            return func(*args, **kwargs)
        finally:
            if value is not None:
                os.environ[env_var] = value
            else:
                del os.environ[env_var]

    return wrapper
