The sharp edges
###############

This section describes features in the language that are not (yet) supported by *thunder*, along with their workarounds. You might encounter these when compiling a module or function with *thunder*. The examples should give you a good idea of how to change your program so that it works.

Note that the fact that something is not supported today doesn't mean it won't be supported at some point in the future. Feel free to reach out to help us prioritize.

Complex control flow
--------------------

Control flow is supported in *thunder*, but certain constructs might still be unsupported.

In particular, attributes need to be resolved at compile time for control flow to work. Here are a few examples of constructs that won't work, followed by their workarounds.

.. code-block::

  # NO
  # `x` uninitialized
  for _ in range(5):
      x = ...

  # YES
  x = ...
  for _ in range(5):
      x = ...

.. code-block::

  # NO
  for i in range(x):
      yield i + 1

  # YES
  [i + 1 for i in range(x)]

.. code-block::

  # NO
  x = (self.foo if x else self.bar)()

  # YES
  x = self.foo() if x else self.bar()

.. code-block::

  # NO
  if getattr(self, "_x", None) is None:
      self._x = torch.ones((1,))
  return x + self._x

  # YES
  # `self._x = None` in __init__
  if self._x is None:
      self._x = torch.ones((1,))
  return x + self.x

.. code-block::

  # NO
  setattr(self, "x", ...)

  # YES
  self.x = ...


Exceptions and context managers
-------------------------------

Try-catch blocks and context managers are currently not supported. You can spot use of context managers in PyTorch programs related to ``no_grad``, automatic mixed precision (AMP), ``record_function``, etc.

As a note, all context managers are try-catch blocks (which is why they are currently unsupported). This means that using ``__enter__``/``__exit__`` directly could leak resources which would otherwise be guaranteed safe using ``with``.

.. code-block::

  # NO
  try:
      return x["foo"]
  except KeyError:
      return 0

  # YES
  return x.get("foo", 0)
  # OR
  return x["foo"] if "foo" in x else 0

.. code-block::

  # NO
  with record_function("demo"):
      ...

  # YES
  # If you absolutely must...
  ctx = record_function("demo")
  ctx.__enter__()
  ...
  ctx.__exit__()


.. code-block::

  # NO
  with torch.no_grad():
      ...

  # YES
  # Set .requires_grad_(False) on all parameters.


Nonlocal variables
------------------

Nonlocal variables are fairly common due to the fact that Python functions will implicitly close over outer scope variables. Less commonly known is the fact that comprehensions can create nonlocal variables if the comprehension references variables other than the loop variables.

.. code-block::

  # No
  x = 0
  return [x + i for i in range(10)]

  # Yes
  x = 0
  result = []
  for i in range(10):
      result.append(x + i)
  return result

.. code-block::

  # No
  def outer(x):
      def inner(y):
          return x + y

      return inner(1), inner(2)

  # Yes
  def outer(x):
      def inner(x, y):
          return x + y

      return inner(x, 1), inner(x, 2)
      # OR:
      f = functools.partial(inner, x=x)
      return f(1), f(2)

.. code-block::

  # No
  x = 0
  def counter():
      nonlocal x
      x += 1
      return x

  # Yes
  class Counter:
      def __init__(self):
          self.x = 1

      def __call__(self):
          self.x += 1
          return self.x


Tensor subclasses
-----------------

*thunder* currently supports Python data types and PyTorch tensors as inputs of functions and models. Subclasses of these types, e.g. lazy tensors or sparse tensors, are not supported today.

Tracing Python builtins, standard library operations and functions that call other languages
--------------------------------------------------------------------------------------------

Calling a Python builtin, standard library operation, or a function that calls into another language is safe to trace, so long as the following rules are observed:

1. The function must not have side effects. For example, calling ``print()`` will execute the ``print()`` function while tracing, but since it's not a *thunder* operation it will not appear in a trace, and so future cached executions will not execute the ``print()`` statement.
2. The function must not manipulate tensor metadata or data. Since the operation won't appear in a trace, these manipulations won't be repeated by *thunder*, and may even cause a crash while tracing.
3. The function must not produce different results across invocations. Again, since the operation won't appear in traces, *thunder* cannot replicate an operation that produces different results when it's invoked, like ``random.random()`` will.

..
  Certain op-level behavior
  -------------------------
  1. Ops which have not yet been added to *thunder*. Please let us know if there’s missing operator support you would like to see and we will be happy to help.
  2. Data dependent control flow (e.g. ``if x.any()``). Since *thunder* generates traces of programs ahead of the actual execution, control flow depending on the values of tensors as opposed to their metadata cannot be handled by *thunder*.


Using Thunder Optimized Modules
-------------------------------

Compiling a module produces a “Thunder Optimized Module” - TOM for short. The TOM is less dynamic than the original module, which facilitates tracing and optimization. The TOM has a reference to the original module, and the TOM and the original module share their parameters.

While modifying the original model's parameters will reflect in the TOM, other changes to the original module will not. In particular:

- Whether model is in ``train`` or ``eval`` mode is captured at compilation time and constant
- The structure of the module is captured at compilation time, and changing the original module's structure will likely break the TOM
- Non-parameter attributes of the module may or may not be captured at compile time and treated as constants

Not all features of PyTorch modules are currently supported, either. Module hooks are not supported, and adding new module attributes in a module's ``forward()`` method is only partially supported.

When a TOM is called, all of the original module's parameter metadata must be the same as when the TOM was compiled, or the call may be silently incorrect. For example, whether a parameter requires grad or not must be the same at compile time and *call time*.

Several TOMs can be compiled from the same module, so you can call ``.train()`` and ``.requires_grad_(True)`` on a module and compile it, then call ``.eval()`` and ``.requires_grad_(False)`` and compile it again, to get compiled modules for train and eval. Before calling the train TOM the module's parameters must require grad, and before calling the eval TOM the module's parameters must not require grad.
