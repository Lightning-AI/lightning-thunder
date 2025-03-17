from functools import partial

import torch


def benchmark_n(n, model_or_fn, /, *args, **kwargs):
    for _ in range(n):
        _ = model_or_fn(*args, **kwargs)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(n):
        _ = model_or_fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / n


benchmark = partial(benchmark_n, 10)
