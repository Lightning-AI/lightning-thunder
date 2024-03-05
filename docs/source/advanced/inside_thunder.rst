Inside Thunder
##############

This section elaborates on the design of some of *thunder*'s internals.

Bytecode interpretation
=======================

Thunder's interpreter works by:

1. disassembling the PyTorch module or function into CPython bytecode
2. interpreting the bytecode using an extended Python interpreter
3. generating a sequential trace of operations on tensors and numbers

Representing Operations
=======================

Thunder supports a subset of PyTorch's operators (see ``thunder.torch.__init__.py`` for which operators are supported).
Thunder has to define its own versions of PyTorch operators so it knows how to compile and trace them properly. This section details how Thunder represents operations.

Let's start by looking at how ``torch.nn.functional.softmax`` appears in a trace of the nanoGPT (https://github.com/karpathy/nanoGPT) model::

  t63 = ltorch.softmax(t52, dim=-1)  # t63: "cuda:0 f16[8, 12, 64, 64]"
    # t53 = prims.convert_element_type(t52, dtypes.float32)  # t53: "cuda:0 f32[8, 12, 64, 64]"
    # t55 = ltorch.amax(t53, -1, keepdim=True)  # t55: "cuda:0 f32[8, 12, 64, 1]"
      # t54 = prims.amax(t53, (3,))  # t54: "cuda:0 f32[8, 12, 64]"
      # t55 = prims.broadcast_in_dim(t54, [8, 12, 64, 1], [0, 1, 2])  # t55: "cuda:0 f32[8, 12, 64, 1]"
    # t57 = ltorch.sub(t53, t55)  # t57: "cuda:0 f32[8, 12, 64, 64]"
      # t56 = prims.broadcast_in_dim(t55, [8, 12, 64, 64], (0, 1, 2, 3))  # t56: "cuda:0 f32[8, 12, 64, 64]"
      # t57 = prims.sub(t53, t56)  # t57: "cuda:0 f32[8, 12, 64, 64]"
    # t58 = ltorch.exp(t57)  # t58: "cuda:0 f32[8, 12, 64, 64]"
      # t58 = prims.exp(t57)  # t58: "cuda:0 f32[8, 12, 64, 64]"
    # t60 = ltorch.sum(t58, -1, keepdim=True)  # t60: "cuda:0 f32[8, 12, 64, 1]"
      # t59 = prims.sum(t58, (3,))  # t59: "cuda:0 f32[8, 12, 64]"
      # t60 = prims.broadcast_in_dim(t59, [8, 12, 64, 1], [0, 1, 2])  # t60: "cuda:0 f32[8, 12, 64, 1]"
    # t62 = ltorch.true_divide(t58, t60)  # t62: "cuda:0 f32[8, 12, 64, 64]"
      # t61 = prims.broadcast_in_dim(t60, [8, 12, 64, 64], (0, 1, 2, 3))  # t61: "cuda:0 f32[8, 12, 64, 64]"
      # t62 = prims.div(t58, t61)  # t62: "cuda:0 f32[8, 12, 64, 64]"
    # t63 = prims.convert_element_type(t62, dtypes.float16)  # t63: "cuda:0 f16[8, 12, 64, 64]"

Instead of the original operation we see a call to the corresponding ``thunder.torch`` operation, then a comment that describes the decomposition of this operation into other ``thunder.torch`` calls (identified by ``ltorch`` in the snippet above) and ``thunder.core.prims`` calls, with the calls to the *primitives* defined in ``thunder.core.prims`` being “terminal” — they decompose into nothing.

Every Thunder operation can be decomposed into one or more primitives, and these decompositions are essential to trace, transform, and optimize them. For example, every primitive operation defines a “meta function” that maps proxy inputs to proxy outputs. When tracing, some inputs, like PyTorch tensors, are replaced with proxies. We know what the proxy output of operations like ``torch.softmax`` would be by decomposing it into primitives, essentially giving it an implicit meta function. If operations weren't defined in terms of primitives, then each operation would require its own meta function, and its own rule for transforms like autograd, and executors would have to reason about every Pytorch operator.

Primitives are what let Thunder define operations without dramatically increasing its complexity, but the set of primitives must also be carefully chosen. Primitives serve two purposes:

- They must be as simple and few in number as possible. A small set of simple operations is easier to analyze, transform, optimize, and execute than a large set of complicated operations.
- They must be expressive enough to describe and facilitate the execution of deep learning and scientific computations.

Since primitives are as simple as possible, they do not broadcast or type promote, and they typically do not have default arguments. For example, in the above trace the call to ``ltorch.sub`` decomposes into a broadcast and then the primitive for subtraction because broadcast is its own primitive.

Thunder primitives are similar to the operations in JAX's ``jax.lax`` module, which is a wrapper around XLA's HLO operations.

Because the prims are so simple and few, writing the decompositions of PyTorch operations directly in terms of primitives would be painstaking. Instead, Thunder has a “core language” of common deep learning operations, and Thunder's PyTorch decompositions typically call these core language operations or other PyTorch operations. Note that core language operations don't appear in traces for simplicity (they have no other use except producing decompositions and can't be executed directly).
