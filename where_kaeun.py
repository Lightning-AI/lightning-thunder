import thunder
import torch

# def foo(x):
#     x = torch.Tensor.where(x)
#     return x

# x = torch.randn(3, device='cuda')
x = torch.tensor([1, 1, 1, 0, 1])
# print(x)
# jit_foo = thunder.jit(foo)
# o = jit_foo(x)

# print(thunder.last_traces(jit_foo)[-1])
# print(f"output: {o}")

x = torch.randn(3, 2)
y = torch.ones(3, 2)

def foo(x):
    return torch.where(x)

def bar(x, y):
    return torch.where(x > 0, x, y)

jit_foo = thunder.jit(foo)
o = jit_foo(x)