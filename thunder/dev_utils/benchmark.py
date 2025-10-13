from functools import partial
import time

import torch


def benchmark_n(n, model_or_fn, /, *args, device: str = "cuda:0", **kwargs):
    for _ in range(n):
        _ = model_or_fn(*args, **kwargs)

    use_cuda_events = device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(n):
            _ = model_or_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / n
    else:
        start = time.perf_counter()
        for _ in range(n):
            _ = model_or_fn(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000.0 / n


benchmark = partial(benchmark_n, 10)
