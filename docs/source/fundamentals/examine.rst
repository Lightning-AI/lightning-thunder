Using Examine
#############

We recommend using *thunder*'s ``examine()`` before compiling a function or a module. *thunder* is in developer preview, and it cannot run every PyTorch module. ``examine()`` not only determines if *thunder* can compile the module, but provides a helpful report to use when filing an issue requesting support.

You can run examine like this::

  from thunder.examine import examine

  model = MyModel(...)
  examine(model, *args, **kwargs)

Where ``*args and **kwargs`` are valid inputs to the model. If examine determines that *thunder* can run the module or function as expected, it will print::

  The function appears to be working as expected

If *thunder* cannot run the module or function it will explain why. There are two principal reasons *thunder* cannot run PyTorch modules:

- *thunder* cannot properly preprocess the module into a traceable function
- the module or function calls one or more PyTorch operators that *thunder* does not yet support

Let's look at examples of both these cases, first, a function that *thunder* cannot yet compile::

  def foo(a):
    result = None

    def bar():
      nonlocal result
      result = a + a

    bar()
    return result

  import torch
  import thunder
  from thunder.examine import examine

  a = torch.full((2, 2), 1., device='cuda')
  examine(foo, a)


In this example, ``foo`` calls ``bar`` which assigns to a nonlocal variable, but *thunder* does not currently support compiling modules or functions that use nonlocals, so examine will print the following::

  Found 1 distinct operations, of which 1 (100.0%) are supported
  Encountered an error while preprocessing the function
  Please file an issue with your function and this error here: https://github.com/Lightning-AI/lightning-thunder/issues/new

  ---------------------------
  | f.__name__ = preprocess |
  ---------------------------

  Argument(`fn`):
    <function foo at 0x7f33e0763d90>

  Argument(`is_module`):
    False


  -------------------------------
  | f.__name__ = acquire_method |
  -------------------------------

  Argument(`method`):
    <function foo at 0x7f33e0763d90>

  Argument(`module`):
    None

  Argument(`mro_klass`):
    None


  nonlocal variables are not supported but instruction = ThunderInstruction(opname='STORE_DEREF', opcode=137, arg=1, argval='result', argrepr='result', offset=2, starts_line=None, is_jump_target=False) found
                foo defined in PATH.py:60
                line 61:     result = None

which can be used to file an issue.

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

To recap, ``examine()`` lets you know if *thunder* can run a module, and if it can't it will provide a report to file an issue asking for support.
