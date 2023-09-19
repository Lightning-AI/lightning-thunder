Hello World
###########

Here is a simple example of how *thunder* lets you compile and run PyTorch modules and functions::

  import torch
  import thunder

  def foo(a, b):
    return a + b

  cfoo = thunder.compile(foo)

  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = cfoo(a, b)

  print(f"{result=}")

  # prints
  # result=tensor(
  #  [[4, 4],
  #   [4, 4]])

The compiled function ``cfoo`` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by *thunder* can be used as part of bigger PyTorch programs.

*thunder* currently understands a subset of PyTorch operations, and a subset of Python. It's adding support quickly, however, see the :doc:`Get involved <get_involved>` section to easily get help if an operator is currently not supported.
