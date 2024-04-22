import torch
import thunder
import numpy as np

reduction = "mean"
def max(input):
    output = torch.finfo(input.dtype).max
    return output

cfn = thunder.jit(max)
input =torch.randn(3, 5).type(torch.float64)
output = cfn(input)


def mse(input, target):
    output = torch.nn.functional.mse_loss(input, target, reduction=reduction)
    return output

def mse_thunder(input, target):
    output = thunder.torch.mse_loss(input, target, reduction=reduction)
    return output

input = torch.randn(3, 5, requires_grad=True).type(torch.float64)
target = torch.randn(3, 5).type(torch.float64)

cfn = thunder.jit(mse)
actual_loss = cfn(input, target)


# actual_loss = cfn(input, target)
actual_loss.sum().backward()
thunder_grad = input.retain_grad()
input.grad = None

expected_loss = mse(input, target)
input_grad = torch.ones_like(expected_loss)
answer_grad = torch.ops.aten.mse_loss_backward(actual_loss, input, target, 1)
expected_loss.sum().backward()
pytorch_grad = input.grad

torch.testing.assert_close(thunder_grad, pytorch_grad)

traces = thunder.last_traces(cfn)

grad_jfn = thunder.core.transforms.grad(cfn)
actual_grad, = grad_jfn(input, target)

expected_loss = torch.nn.functional.mse_loss(input, target, reduction = reduction)
go = torch.ones_like(expected_loss)
expected_grad, = torch.autograd.grad(torch.nn.functional.mse_loss(input, target, reduction=reduction), input, go)

print("Max error in loss:", (actual_loss - expected_loss).abs().max().item())
print("Max error in logits grad:", (actual_grad - expected_grad).abs().max().item())