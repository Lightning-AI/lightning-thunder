Using Examine
#############

We recommend using Thunder's ``examine()`` before compiling a function or a module.

Thunder cannot run every PyTorch module, but you can quickly find out what is missing using ``examine()``.

``examine()`` not only determines if Thunder can compile the module, but provides a helpful report to use when filing an issue requesting support.

You can run examine like this::

  from thunder.examine import examine

  model = MyModel(...)
  examine(model, *args, **kwargs)

Where ``*args and **kwargs`` are valid inputs to the model. If examine determines that Thunder can run the module or function as expected, it will print::

  The function appears to be working as expected

When ``examine`` encounters a module or function with one or more operators it doesn't support, it will specify the operators, like this::

  def foo(a):
    return torch.triu(a)

  import torch
  import thunder
  from thunder.examine import examine

  a = torch.full((2, 2), 1., device='cuda')
  examine(foo, a)

Running the above will print::

  Found 1 distinct operations, of which 0 (0.0%) are supported
  Please file an issue requesting the following operators here: https://github.com/Lightning-AI/lightning-thunder/issues/new
  _VariableFunctionsClass.triu of torch

To recap, ``examine()`` lets you know if Thunder can run a module, and if it can't it will provide a report to file an issue asking for support.
