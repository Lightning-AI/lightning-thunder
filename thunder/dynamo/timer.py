import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer, timer
from torch.utils.benchmark.utils.common import Measurement


class TorchProfileTimer:
    def __init__(self):
        self.prof = profile(activities=[ProfilerActivity.CUDA])
        self.current_time = 0.0

    def _get_kernel_time(self, prof_averages: torch.autograd.profiler_util.EventList) -> float:
        """
        Arguments:
            prof_averages: Output of self.prof.key_averages()
        Returns:
            time_value: Elapsed CUDA time in seconds.
        """
        from torch.autograd import DeviceType

        elapsed_cuda_time = 0
        has_cuda_event = False
        for event in prof_averages:
            if event.device_type != DeviceType.CUDA:
                continue
            has_cuda_event = True
            # Re: torch profiler API changes in https://github.com/pytorch/pytorch/pull/123247
            elapsed_cuda_time = (
                elapsed_cuda_time + event.self_device_time_total
                if hasattr(event, "self_device_time_total")
                else event.self_cuda_time_total
            )
        # assert has_cuda_event, "No CUDA events found"
        # print(has_cuda_event)
        # print(elapsed_cuda_time)
        return elapsed_cuda_time / 1e6

    def _increment_global_time(self, elapsed_time: float) -> None:
        self.current_time += elapsed_time

    def __call__(self):
        try:
            self.prof.stop()
        except AssertionError:
            self.prof.start()
            return self.current_time

        prof_averages = self.prof.key_averages()
        elapsed_cuda_time = self._get_kernel_time(prof_averages)
        self._increment_global_time(elapsed_cuda_time)
        # Clear the internal profiler object to avoid accumulating function events and then restart the profiler
        # See PR: https://github.com/pytorch/pytorch/pull/125510
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
        self.prof.profiler = None

        return self.current_time

    def reset(self):
        self.current_time = 0.0
        try:
            self.prof.stop()
        except AssertionError:
            pass


def kernel_time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
    timer = TorchProfileTimer()
    t = Timer(stmt=stmt, setup=setup, timer=timer, globals=globals)
    return t.blocked_autorange(min_run_time=min_run_time)


def wall_time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
    t = Timer(stmt=stmt, setup=setup, globals=globals)
    return t.blocked_autorange(min_run_time=min_run_time)


# tmp tests
if __name__ == "__main__":
    from nvfuser import FusionDefinition, DataType

    def nvfuser_fusion_id2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
        )
        T1 = fd.define_tensor(
            shape=[2, 2], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0]
        )
        T2 = fd.ops.cos(T0)
        T3 = fd.ops.add(T1, T2)
        S4 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T5 = fd.ops.add(T3, S4)
        fd.add_output(T5)

    with FusionDefinition() as fd:
        nvfuser_fusion_id2(fd)

    inputs = [
        torch.testing.make_tensor((2, 2), dtype=torch.float32, device="cuda:0"),
        torch.testing.make_tensor((2, 2), dtype=torch.float32, device="cuda:0"),
    ]

    # print(kernel_time1(fd.execute, inputs))
    xx = kernel_time(stmt="fd.execute(inputs)", globals={"fd": fd, "inputs": inputs})
    yy = wall_time(stmt="fd.execute(inputs)", globals={"fd": fd, "inputs": inputs})
    print(xx)
    print(yy)
