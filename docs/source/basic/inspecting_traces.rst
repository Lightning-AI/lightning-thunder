Thunder step by step
####################

As mentioned in the :doc:`Overview <overview>`, *thunder* produces a series of traces, culminating in an “execution trace” that *thunder* converts to a Python function and calls. In this example we'll see how to inspect this series of traces. Let's start with this very simple program::

  def foo(a, b):
    return a + b

  import thunder
  import torch

  cfoo = thunder.compile(foo)
  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = cfoo(a, b)
  print(f"{result=}")

  traces = thunder.last_traces(cfoo)
  print(traces[0])

This prints::

  import thunder as thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  def thunder_139803254115824(a, b):
    # a: "cpu i64[2, 2]"
    # b: "cpu i64[2, 2]"
    t0 = ltorch.add(a, b, alpha=None)  # t0: "cpu i64[2, 2]"
      # t0 = prims.add(a, b)  # t0: "cpu i64[2, 2]"
    return t0

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

  import thunder
  import torch

  cfoo = thunder.compile(foo)
  a = torch.full((2, 2), 1., device='cuda')
  b = torch.full((2, 2), 3., device='cuda')

  result = cfoo(a, b)
  print(f"{result=}")

  traces = thunder.last_traces(cfoo)
  print(traces[0])

The first trace constructed is, again, a record of the PyTorch operations observed while tracing the function::

  import thunder as thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  def thunder_140674105445632(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    t0 = ltorch.add(a, b, alpha=None)  # t0: "cuda:0 f32[2, 2]"
      # t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
    t1 = ltorch.add(a, b, alpha=None)  # t1: "cuda:0 f32[2, 2]"
      # t1 = prims.add(a, b)  # t1: "cuda:0 f32[2, 2]"
    t2 = ltorch.mul(a, a)  # t2: "cuda:0 f32[2, 2]"
      # t2 = prims.mul(a, a)  # t2: "cuda:0 f32[2, 2]"
    t3 = ltorch.mul(t1, b)  # t3: "cuda:0 f32[2, 2]"
      # t3 = prims.mul(t1, b)  # t3: "cuda:0 f32[2, 2]"
    return (t0, t3)

We see the same additions and multiplications as in the original.

Now let's look at the second trace by printing ``traces[1]``::

  # Constructed by Dead Code Elimination
  import thunder as thunder
  import thunder.torch as ltorch
  import torch

  @torch.no_grad()
  def thunder_140658837294336(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    t0 = ltorch.add(a, b, alpha=None)  # t0: "cuda:0 f32[2, 2]"
      # t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
    t1 = ltorch.add(a, b, alpha=None)  # t1: "cuda:0 f32[2, 2]"
      # t1 = prims.add(a, b)  # t1: "cuda:0 f32[2, 2]"
    t3 = ltorch.mul(t1, b)  # t3: "cuda:0 f32[2, 2]"
      # t3 = prims.mul(t1, b)  # t3: "cuda:0 f32[2, 2]"
    return (t0, t3)

This next trace in the series has a comment “Constructed by Dead Code Elimination”, letting us know it was constructed by performing dead code elimination on the previous trace. Each trace in the series is the result of a “transform” or “optimization pass” performed on the previous trace. This trace's function no longer has the first multiplication in the original program, because the result of that multiplication is never used - it's “dead code.” Removing it preserves the original computation while doing less work.

``traces[2]`` “flattens” the program::

  # Constructed by Flatten
  import thunder as thunder
  import thunder.core.prims as prims
  import torch

  @torch.no_grad()
  def thunder_140284045757696(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
    t1 = prims.add(a, b)  # t1: "cuda:0 f32[2, 2]"
    t3 = prims.mul(t1, b)  # t3: "cuda:0 f32[2, 2]"
    return (t0, t3)

“Flattening” prepares the trace for executors. Up until this point, the trace contained each of the original operations along with a tree of its decompositions into other lower-level operations, down to primitives (such decompositions are what is shown in comments below each operation). Flattening looks at each operation, chooses one of the decompositions and produces a sequential trace in output. In this case, the ltorch.add and ltorch.mul operations are replaced with their prims.add and prims.mul components, which will be consumed by the nvFuser executor (as we'll see in a moment).

``traces[3]`` is an optimization pass for working with “mixed-precision” modules and functions that use bfloat16 or float16 tensors with float32 tensors, so it doesn't do anything for this program. ``traces[4]`` runs another dead code elimination pass, which can be an important optimization but in this case is also a no-op. Let’s skip ahead to ``traces[5]``, which performs common subexpression elimination::

  # Constructed by Common Subexpression Elimination
  import thunder as thunder
  import thunder.core.prims as prims
  import torch

  @torch.no_grad()
  def thunder_140245185498368(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
    t3 = prims.mul(t0, b)  # t3: "cuda:0 f32[2, 2]"
    return (t0, t3)

The original program had two additions and two multiplications. One of the multiplications was dead code (i.e. it didn't have any effect on the outputs) and was removed by the initial dead code elimination pass. The common subexpression elimination pass recognizes that the two additions are computing the same result, so it replaces the result of the second addition with the result of the first addition. This lets the second addition be removed while still preserving the original computation.

``traces[6]`` performs fusion::

  # Constructed by Fusion
  import torch
  @torch.no_grad()
  def thunder_139850838166784(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    (t0, t3) = nvFusion0(a, b)
      # t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
      # t3 = prims.mul(t0, b)  # t3: "cuda:0 f32[2, 2]"
    return (t0, t3)

Fusion creates a custom operator that replaces sequences of operators, and these custom operations can be much faster than executing each operation independently. In this case two element-wise operations are fused by the nvFuser executor into the new ``nvFusion0`` operation.

What's interesting about ``traces[6]`` is that its code is not enough to define a valid Python program, because the name nvFusion0 is not defined by the program. Python programs can be represented as code plus a “context,” a dictionary mapping names to Python objects, and we can find ``nvFusion0`` defined in the traces Python context::

  print(traces[6].python_ctx())

  # Prints
  # {'nvFusion0': <function fuse.<locals>.fn_ at 0x7faf8f3d5ea0>}

We can acquire and print the fusion object and fusion representation from the ``ctx``::

  ctx = traces[6].python_ctx()
  print(nvFusion0 := ctx['nvFusion0'].last_used)

Prints::

  def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
      T0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False)
      T1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False)
      T2 = fd.ops.add(T0, T1)
      T3 = fd.ops.mul(T2, T1)
      fd.add_output(T2)
      fd.add_output(T3)

which is nvFuser's own description of the operation. We can even see the CUDA code executed by printing::

  print(ctx['nvFusion0'].last_used.last_cuda_code())

which prints::

  __global__ void kernel1(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T1, Tensor<float, 2, 2> T6, Tensor<float, 2, 2> T3) {
    nvfuser_index_t i0;
    i0 = ((nvfuser_index_t)threadIdx.x) + (128 * ((nvfuser_index_t)blockIdx.x));
    if ((i0 < (T0.logical_size[0] * T0.logical_size[1]))) {
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

Finally, ``traces[7]`` is the result of a lifetime analysis pass, which deletes tensor intermediates when they're no longer needed, freeing memory::

  # Constructed by Delete Last Used
  import torch

  @torch.no_grad()
  def foo(a, b):
    # a: "cuda:0 f32[2, 2]"
    # b: "cuda:0 f32[2, 2]"
    (t0, t3) = nvFusion0(a, b)
      # t0 = prims.add(a, b)  # t0: "cuda:0 f32[2, 2]"
      # t3 = prims.mul(t0, b)  # t3: "cuda:0 f32[2, 2]"
  del [a, b]
  return (t0, t3)

To recap, *thunder* can optimize PyTorch modules and functions, and we can see its optimizations by looking at the series of traces it produces when a compiled function is called. The last trace is called the *execution trace*, and *thunder* converts it into a Python function and calls it. Traces not only have Python code, but a Python context, too, that can be used to acquire and inspect fusions.
