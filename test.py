import torch
import thunder
import thunder.examine


def fn(x):
    x[0].zero_()


jfn = thunder.jit(fn)
x = torch.ones(2, 2)
jfn(x)
# thunder.examine.examine(jfn, x)
