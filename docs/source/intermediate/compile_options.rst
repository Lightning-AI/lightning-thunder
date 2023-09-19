Compile options
###############

``thunder.compile()`` supports a variety of options to specify executors, the caching mode, and whether to use CUDA graphs or not. In this section we'll look at each.

*thunder* runs PyTorch operations using executors. To determine which executor executes which operation, the executors are queried, in priority order, for whether they can execute the operation or not. The default executor priority ordering is ``[executors.NVFUSER, executors.TORCH]``, meaning that only if nvFuser declines to execute an operation will it be given to PyTorch. (Behind the scenes there's always a Python executor, too, which handles non-PyTorch operations like unpacking input tuples and adding two Python numbers together).

The priority order can be specified when calling ``thunder.compile()`` using the ``executors_list`` parameter. Additional executors must be registered before being used (see the following examples for how to do so).

*thunder* is intended to compile a few times and run many times. To avoid compiling every time a compiled module is called, *thunder* has a cache mapping from inputs to execution traces. It currently supports three cache modes, which can setting one of the following to ``True`` when calling ``thunder.compile()``:

- ``use_static_caching``: the default
- ``always_trace``: disables caching
- ``use_last_executed``: (unsafe!) always executes the first execution trace constructed, regardless of the inputs

The always_trace and use_last_executed caching modes are mostly interesting to *thunder* developers.

*thunder*'s static cache, used by default, constructs a cache key by looking at PyTorch tensor metadata like ``shape``, ``device``, ``dtype``, and ``requires_grad``. The cache assumes the metadata of module parameters is unchanging. This means, for example, that calling the compiled module with a ``float32`` tensor, then a ``float16`` tensor, causes the compilation process to occur twice.

*thunder* can run its execution trace using CUDA graphs by passing ``use_cudagraphs=True`` to ``thunder.compile()``. This may improve performance when there are only a few distinct inputs (in terms of metadata) to the compiled module.
