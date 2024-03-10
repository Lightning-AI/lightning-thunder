.. Lightning-AI-Sandbox documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ⚡ Lightning Thunder
###############################

Lightning Thunder is a deep learning compiler for PyTorch. It makes PyTorch programs faster both on single accelerators or in distributed settings.

The main goal for Lightning Thunder is to allow optimizing user programs in the most extensible and expressive way possible.

**NOTE: Lightning Thunder is alpha and not ready for production runs.** Feel free to get involved, expect a few bumps along the way.

What's in the box
-----------------

Given a program, Thunder can generate an optimized program that:

- computes its forward and backward passes
- coalesces operations into efficient fusion regions
- dispatches computations to optimized kernels
- distributes computations optimally across machines

To do so, Thunder ships with:

- a JIT for acquiring Python programs targeting PyTorch and custom operations
- a multi-level IR to represent them as a trace of a reduced op-set
- an extensible set of transformations on the trace, such as `grad`, fusions, distributed (like `ddp`, `fsdp`), functional (like `vmap`, `vjp`, `jvp`)
- a way to dispatch operations to an extensible collection of executors

Thunder is written entirely in Python. Even its trace is represented as valid Python at all stages of transformation. This allows unprecedented levels of introspection and extensibility.

Thunder doesn't generate device code. It acquires and transforms user programs so that it's possible to optimally select or generate device code using fast executors like:

- `torch.compile <https://pytorch.org/get-started/pytorch-2.0/>`_
- `nvFuser <https://github.com/NVIDIA/Fuser>`_
- `cuDNN <https://developer.nvidia.com/cudnn>`_
- `Apex <https://github.com/NVIDIA/apex>`_
- `PyTorch eager <https://github.com/pytorch/pytorch>`_ operations
- custom kernels, including those written with `OpenAI Triton <https://github.com/openai/triton>`_

Modules and functions compiled with Thunder fully interoperate with vanilla PyTorch and support PyTorch's autograd. Also, Thunder works alongside torch.compile to leverage its state-of-the-art optimizations.

Hello World
-----------

Here is a simple example of how *thunder* lets you compile and run PyTorch modules and functions::

  import torch
  import thunder

  def foo(a, b):
    return a + b

  jitted_foo = thunder.jit(foo)

  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = jitted_foo(a, b)

  print(result)

  # prints
  # tensor(
  #  [[4, 4],
  #   [4, 4]])

The compiled function ``jitted_foo`` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by Thunder can be used as part of bigger PyTorch programs.

.. toctree::
   :maxdepth: 1
   :name: home
   :caption: Home

   self
   Install <fundamentals/installation>
   Hello World <fundamentals/hello_world>
   Using examine <fundamentals/examine>

.. toctree::
   :maxdepth: 1
   :name: basic
   :caption: Basic

   Overview <basic/overview>
   Zero to Thunder <notebooks/zero_to_thunder>
   Thunder step by step <basic/inspecting_traces>
   The sharp edges <basic/sharp_edges>
   Train a MLP on MNIST <basic/mlp_mnist>
   Functional jit <notebooks/functional_jit>

.. toctree::
   :maxdepth: 1
   :name: intermediate
   :caption: Intermediate

   Additional executors <intermediate/additional_executors>
   Distributed Data Parallel <intermediate/ddp>
   What's next <intermediate/whats_next>
   FSDP Tutorial <notebooks/fsdp_tutorial>

.. toctree::
   :maxdepth: 1
   :name: advanced
   :caption: Advanced

   Inside thunder <advanced/inside_thunder>
   Extending thunder <advanced/extending>
   notebooks/adding_custom_operator
   notebooks/adding_custom_operator_backward
   notebooks/adding_operator_executor

.. toctree::
   :maxdepth: 1
   :name: experimental_dev_tutorials
   :caption: Experimental dev tutorials

   notebooks/dev_tutorials/extend
   notebooks/dev_tutorials/patterns

..
   TODO RC1: update notebooks

API reference
=============

.. toctree::
   :maxdepth: 1
   :name: API reference
   :caption: API reference

   reference/thunder
   reference/common/index
   reference/core/index
   reference/clang/index
   reference/examine/index
   reference/distributed/index
   reference/executors/index
   reference/torch/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
