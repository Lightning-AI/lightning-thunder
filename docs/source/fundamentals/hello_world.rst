Hello World
###########

Here is a simple example of how Thunder lets you compile and run PyTorch modules and functions::

  import torch
  import thunder

  def foo(a, b):
    return a + b

  jitted_foo = thunder.jit(foo)

  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = jitted_foo(a, b)

  print(result)

  # prints
  # tensor(
  #  [[4, 4],
  #   [4, 4]])

The compiled function ``jitted_foo`` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by Thunder can be used as part of bigger PyTorch programs.

Thunder currently understands a subset of PyTorch operations, and a subset of Python. It's adding support quickly, however. Reach out on the Thunder repo and open an issue there to easily get help if an operator is currently not supported.
