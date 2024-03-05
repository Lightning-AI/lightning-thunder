The sharp edges
###############

This section describes features in the language that are not (yet) supported by Thunder, along with their workarounds. You might encounter these when compiling a module or function with Thunder. The examples should give you a good idea of how to change your program so that it works.

Note that the fact that something is not supported today doesn't mean it won't be supported at some point in the future. Feel free to reach out to help us prioritize.

Inplace operations
------------------

Inplace PyTorch operations like `t.add_(1.0)` are not supported in Thunder yet. Support for inplace operations is coming soon.

Complex control flow
--------------------

Control flow is supported in Thunder, but certain constructs might still be unsupported.

In particular, attributes need to be resolved at tracing time for control flow to work. Data-dependent control flow, that is, when a condition depends on the value of tensors rather than its meta-data like shape or type, is currently not supported.

Tensor subclasses
-----------------

Thunder currently supports Python data types and PyTorch tensors as inputs of functions and models.

Subclasses of these types, e.g. lazy tensors, nested tensors, or sparse tensors are not supported today.

Tracing Python builtins, standard library operations and functions that call other languages
--------------------------------------------------------------------------------------------

Calling a Python builtin, standard library operation, or a function that calls into another language is safe to trace, so long as the following rules are observed:

1. The function must not have side effects. For example, calling ``print()`` will execute the ``print()`` function while tracing, but since it's not a Thunder operation it will not appear in a trace, and so future cached executions will not execute the ``print()`` statement.
2. The function must not manipulate tensor metadata or data. Since the operation won't appear in a trace, these manipulations won't be repeated by Thunder, and may even cause a crash while tracing.
3. The function must not produce different results across invocations. Again, since the operation won't appear in traces, Thunder cannot replicate an operation that produces different results when it's invoked, like ``random.random()`` will.

..
  Certain op-level behavior
  -------------------------
  1. Ops which have not yet been added to *thunder*. Please let us know if there’s missing operator support you would like to see and we will be happy to help.
  2. Data dependent control flow (e.g. ``if x.any()``). Since *thunder* generates traces of programs ahead of the actual execution, control flow depending on the values of tensors as opposed to their metadata cannot be handled by *thunder*.


Using Thunder-optimized Modules
-------------------------------

Compiling a module produces a Thunder-optimized module”. A Thunder-optimized module is less dynamic than the original module, which facilitates tracing and optimization. It has a reference to the original module, and it shares its parameters with it.

While modifying the original model's parameters will reflect in the Thunder-optimized module, other changes to the original module will not. In particular:

- Whether model is in ``train`` or ``eval`` mode is captured at compilation time and constant
- The structure of the module is captured at compilation time, and changing the original module's structure will likely break the Thunder-optimized module
- Non-parameter attributes of the module may or may not be captured at compile time and treated as constants

Not all features of PyTorch modules are currently supported, either. Module hooks are not supported, and adding new module attributes in a module's ``forward()`` method is only partially supported.
