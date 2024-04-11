Thunder step by step
####################

As mentioned in the :doc:`Overview <overview>`, Thunder produces a series of traces, culminating in an “execution trace” that Thunder converts to a Python function and calls. In this example we'll see how to inspect this series of traces. Let's start with this very simple program::


  def foo(a, b):
      return a + b

  import thunder
  import torch

  jitted_foo = thunder.jit(foo)
  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = jitted_foo(a, b)
  print(result)

  traces = thunder.last_traces(jitted_foo)
  print(traces[0])

This prints::

  import thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  @torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
  @torch.autocast(device_type="cpu", enabled=False, cache_enabled=False)
  def computation(t0, t1):
    # t0
    # t1
    t2 = ltorch.add(t0, t1, alpha=None)  # t2
      # t2 = prims.add(t0, t1)  # t2
    return (t2, ())

The first trace is a record of the Pytorch operations called while tracing the function. It's printed as a valid Python program, with comments adding additional information, like the device, datatype, and shape of the inputs, and the decomposition of torch.add into “primitive” operations (see :doc:`Inside thunder <../advanced/inside_thunder>` for more on primitive operations). Printing the trace as a valid Python program comes with several advantages:

- Python programs are easy to read, and don't require learning another language
- Python programs can be directly executed, which can facilitate debugging and profiling
- The objects referenced by the program are live Python objects, which can be directly inspected
- Portions of the program are easy to extract and analyze separately

Now let's look at a slightly more complicated function that has more opportunities for optimization and will highlight what *thunder*'s optimization passes do::

  import thunder
  import torch

  def foo(a, b):
      c = a + b
      d = a + b
      e = a * a
      f = d * b
      return c, f

  jitted_foo = thunder.jit(foo)
  a = torch.full((2, 2), 1., device='cuda')
  b = torch.full((2, 2), 3., device='cuda')

  result = jitted_foo(a, b)
  print(result)

  traces = thunder.last_traces(jitted_foo)
  print(traces[0])


The first trace constructed is, again, a record of the PyTorch operations observed while tracing the function::

  import thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  @torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
  @torch.autocast(device_type="cpu", enabled=False, cache_enabled=False)
  def computation(t0, t1):
    # t0
    # t1
    t2 = ltorch.add(t0, t1, alpha=None)  # t2
      # t2 = prims.add(t0, t1)  # t2
    t3 = ltorch.add(t0, t1, alpha=None)  # t3
      # t3 = prims.add(t0, t1)  # t3
    t4 = ltorch.mul(t0, t0)  # t4
      # t4 = prims.mul(t0, t0)  # t4
    t5 = ltorch.mul(t3, t1)  # t5
      # t5 = prims.mul(t3, t1)  # t5
    return ((t2, t5), ())

We see the same additions and multiplications as in the original.

Now let's look at the second trace by printing ``traces[1]``::

  # Constructed by Dead Code Elimination (took 0 milliseconds)
  import thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  @torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
  @torch.autocast(device_type="cpu", enabled=False, cache_enabled=False)
  def computation(t0, t1):
    # t0
    # t1
    t2 = ltorch.add(t0, t1, alpha=None)  # t2
      # t2 = prims.add(t0, t1)  # t2
    t3 = ltorch.add(t0, t1, alpha=None)  # t3
      # t3 = prims.add(t0, t1)  # t3
    t5 = ltorch.mul(t3, t1)  # t5
      # t5 = prims.mul(t3, t1)  # t5
    return ((t2, t5), ())

This next trace in the series has a comment “Constructed by Dead Code Elimination”, letting us know it was constructed by performing dead code elimination on the previous trace. Each trace in the series is the result of a “transform” or “optimization pass” performed on the previous trace. This trace's function no longer has the first multiplication in the original program, because the result of that multiplication is never used - it's “dead code.” Removing it preserves the original computation while doing less work.

``traces[2]`` sets the program up for execution::

  # Constructed by Transform for execution (took 2 milliseconds)
  import torch

  @torch.no_grad()
  @torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
  @torch.autocast(device_type="cpu", enabled=False, cache_enabled=False)
  def computation(t0, t1):
    # t0
    # t1
    [t2, t5] = nvFusion0(t0, t1)
      # t2 = prims.add(t0, t1)  # t2
      # t5 = prims.mul(t2, t1)  # t5
    return ((t2, t5), ())

The transform creates a custom operator that replaces sequences of operators, and these custom operations can be much faster than executing each operation independently. In this case two element-wise operations are fused by the nvFuser executor into the new ``nvFusion0`` operation.

What's interesting about ``traces[2]`` is that its code is not enough to define a valid Python program, because the name nvFusion0 is not defined by the program. Python programs can be represented as code plus a “context,” a dictionary mapping names to Python objects, and we can find ``nvFusion0`` defined in the traces Python context::

  print(traces[6].python_ctx())

  # Prints
  # {'nvFusion0': FusionDefinitionWrapper(nvFusion0: (add, mul))}

We can acquire and print the fusion object and fusion representation from the ``ctx``::

  ctx = traces[2].python_ctx()
  print(nvFusion0 := ctx['nvFusion0'].last_used)

Prints::

  def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
      T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
      T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False, stride_order=[1, 0])
      T2 = fd.ops.add(T0, T1)
      T3 = fd.ops.mul(T2, T1)
      fd.add_output(T2)
      fd.add_output(T3)

which is nvFuser's own description of the operation. We can even see the CUDA code executed by printing::

  print(ctx['nvFusion0'].last_used.last_cuda_code())

which prints::

  __global__ void nvfuser_pointwise_f0_c1_r0_g0(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T1, Tensor<float, 2, 2> T6, Tensor<float, 2, 2> T3) {
    nvfuser_index_t i0;
    i0 = ((nvfuser_index_t)threadIdx.x) + (128LL * ((nvfuser_index_t)blockIdx.x));
    if ((i0 < (T0.logical_size[0LL] * T0.logical_size[1LL]))) {
      float T5[1];
      T5[0] = 0;
      T5[0]
        = T1[i0];
      float T4[1];
      T4[0] = 0;
      T4[0]
        = T0[i0];
      float T2[1];
      T2[0]
        = T4[0]
        + T5[0];
      float T7[1];
      T7[0]
        = T2[0];
      T6[i0]
        = T7[0];
      float T8[1];
      T8[0]
        = T2[0]
        * T5[0];
      T3[i0]
        = T8[0];
    }
  }

Finally, ``traces[3]`` is the result of a lifetime analysis pass, which deletes tensor intermediates when they're no longer needed, freeing memory::

  # Constructed by Delete Last Used (took 0 milliseconds)
  import torch

  @torch.no_grad()
  @torch.autocast(device_type="cuda", enabled=False, cache_enabled=False)
  @torch.autocast(device_type="cpu", enabled=False, cache_enabled=False)
  def computation(t0, t1):
    # t0
    # t1
    [t2, t5] = nvFusion0(t0, t1)
      # t2 = prims.add(t0, t1)  # t2
      # t5 = prims.mul(t2, t1)  # t5
    del t0, t1
    return ((t2, t5), ())

To recap, Thunder can optimize PyTorch modules and functions, and we can see its optimizations by looking at the series of traces it produces when a compiled function is called. The last trace is called the *execution trace*, and Thunder converts it into a Python function and calls it. Traces not only have Python code, but a Python context, too, that can be used to acquire and inspect fusions.
