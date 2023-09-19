Thunder Overview
################

This section introduces *thunder*'s core concepts and architecture. For more details, see :doc:`Inside thunder <../advanced/inside_thunder>`.

*thunder* is a deep learning compiler for PyTorch, which means it translates calls to PyTorch modules into a format that is easy to transform and that executors like nvFuser can consume to produce fast executables. This translation must be “valid” - it must produce a simple representation focusing on tensor operations. The format we've chosen, like other deep learning compilers, is a sequence of operations called a program *trace*.

This translation begins with::

  cmodel = thunder.compile(my_module)

or::

  cfn = thunder.compile(my_function)

When given a module, the call to ``thunder.compile()`` returns a *Thunder Optimized Module* (TOM) that shares parameters with the original module (as demonstrated in the :doc:`Train a MLP on MNIST <mlp_mnist>` example), and when given a function it returns a compiled function.

When the TOM or compiled function is called::

  cmodel(*args, **kwargs)

or::

  cfn(*args, **kwargs)

*thunder* begins reviewing the module's or function's Python bytecode and the input. It may be surprising that *thunder* considers the inputs at all, but this is actually required to produce a trace. Different inputs can produce different traces, since the operations called may different based on the properties of the input.

From the Python bytecode and the inputs, *thunder* “preprocesses” the original Python into a “traceable function” that — as its name suggests — is safe for *thunder* to trace. These functions don't attempt to read or write global or nonlocal values, are explicit about their tensor inputs, and have their calls to PyTorch operations translated to the corresponding ``thunder.torch`` operations, among other things.

To trace, some inputs, like PyTorch tensors, are swapped with *proxies* that only have metadata like shape, device, dtype, and whether the tensor requires grad or not. The traceable function is then executed with these proxies. This execution doesn't perform any computation on accelerators, but it records the operators along one path of the traceable function into a trace.

Traces can be transformed (like for backward) and optimized (like by replacing calls to PyTorch operations with calls to faster executors), and the final result of this process is an *execution trace*. *thunder* executes the original call by converting the execution trace into a Python function and calling that function with the actual inputs. For details about this optimization process see the :doc:`thunder step by step <inspecting_traces>` section.

To recap, the complete translation process is:

- For PyTorch modules, a Thunder Optimized Module (TOM) is created from the module
- For PyTorch functions, compilation produces a compiled function
- When the TOM or function is called, preprocessing produces a traceable function
- The traceable function is traced, swapping some inputs with “proxies”, to create a trace
- The trace is transformed and optimized to produce an execution trace
- The execution trace is converted into a Python function and called

This translation process is often slow - it takes tens of seconds for nanoGPT's (https://github.com/karpathy/nanoGPT) largest configuration - so *thunder*'s performance model expects relatively few of these translations and then a lot of uses of the result. This corresponds with many training and inference patterns, where the same program is executed many times.
