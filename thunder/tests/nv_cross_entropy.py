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


input = torch.randn(8192, 3024, requires_grad=False, device="cuda", dtype=torch.float32)
target = torch.randint(0, 128, (8192,), requires_grad=False, device="cuda")
o2, b2 = torch.compile(fn2)(input, target)
o, b = thunder.jit(fn2, executors=[nvfuserex])(input, target)
print(o)
print(o2)
