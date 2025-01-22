import time

import torch
import torchvision.models as models

import thunder

torch.set_float32_matmul_precision('high')

with torch.device("cuda:0"):
    model = models.vit_b_16().requires_grad_(False).eval()
    x = torch.randn(128, 3, 224, 224)

print(model)

thunder_model = thunder.compile(model)

y = thunder_model(x)

torch.cuda.synchronize()

start = time.perf_counter_ns()
for i in range(10):
    y = model(x)
torch.cuda.synchronize()
end = time.perf_counter_ns()

print(end - start)

start = time.perf_counter_ns()
for i in range(10):
    y = thunder_model(x)
torch.cuda.synchronize()
end = time.perf_counter_ns()

print(end - start)


# torch_compiled_model = torch.compile(model)
# y = torch_compiled_model(x)
# 
# start = time.perf_counter_ns()
# for i in range(10):
#     y = torch_compiled_model(x)
# torch.cuda.synchronize()
# end = time.perf_counter_ns()

print(end - start)

# print(thunder.last_traces(compiled_model)[-1])


# print(prof.key_averages().table())
# print(prof.key_averages().total_average().self_device_time_total_str)
# print(prof.key_averages().total_average().device_time_total_str)
# print(dir(prof.key_averages().total_average()))
