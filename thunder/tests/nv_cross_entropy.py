import transformers.loss.loss_utils
import thunder
import torch
import transformers.loss.loss_utils

from thunder.executors.nvfuserex import nvfuserex


def fn(input, target):
    a = torch.amax(input, dim=1)
    return torch.nn.functional.cross_entropy(input, target), a


def fn2(input, target):
    a = torch.amax(input, dim=1)
    return transformers.loss.loss_utils.ForCausalLMLoss(input, target, vocab_size=3024), a


input = torch.randn(8192, 32064, requires_grad=True, device="cuda", dtype=torch.float32)
input2 = input.detach().clone().requires_grad_(True)
target = torch.randint(0, 5000, (8192,), requires_grad=False, device="cuda")
o2, b2 = torch.compile(fn)(input2, target)
o, b = thunder.jit(fn, executors=[nvfuserex])(input, target)
print(o)
print(o2)

o.backward()
o2.backward()
are_close = torch.allclose(input.grad, input2.grad, atol=1e-5, rtol=1e-5)
print("are close", are_close)
# print(input.grad)
# print(input2.grad)
