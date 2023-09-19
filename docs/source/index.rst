.. Lightning-AI-Sandbox documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ⚡ Lightning Thunder
###############################

``lightning.compile``, codenamed *thunder*, is a deep learning compiler for PyTorch. It makes PyTorch programs faster on single accelerators or in distributed settings.

..
   which intercepts natively written PyTorch programs, translates them into a program representation, applies optimizations, and executes the program. Intercepting the PyTorch program allows thunder to apply optimizations that the PyTorch otherwise can’t apply and would be difficult for users and developers to implement in a network by hand resulting in faster training.

*thunder* is intended to be fast, expressive, extensible, and easy to inspect.
It's written entirely in Python, and it runs PyTorch modules and functions using PyTorch plus fast executors like nvFuser, cuDNN Fusion Python API, Apex, and custom kernels (including those written with OpenAI Triton).

Modules and functions compiled with *thunder* also interoperate with vanilla PyTorch and support PyTorch's autograd.

*thunder* will be released publicly as a part of Lightning when it is in Beta.


Install Thunder
---------------

If you qualify as an NVIDIA employee, hit the ground running through the :doc:`pre-built container <fundamentals/container>`.

If not, follow the :doc:`install instructions <fundamentals/installation>`.


Hello World
-----------

Here is a simple example of how *thunder* lets you compile and run PyTorch modules and functions::

  import torch
  import thunder

  def foo(a, b):
    return a + b

  cfoo = thunder.compile(foo)

  a = torch.full((2, 2), 1)
  b = torch.full((2, 2), 3)

  result = cfoo(a, b)

  print(f"{result=}")

  # prints
  # result=tensor(
  #  [[4, 4],
  #   [4, 4]])

The compiled function ``cfoo`` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by *thunder* can be used as part of bigger PyTorch programs.


FAQ
---

**Does *thunder* have any benchmarks?**

Yes! See ``thunder/benchmarks`` and the ``litgpt-targets.py`` and ``nanogpt-targets.py`` files for how to import and run them.

**How is *thunder* different from ``torch.compile``?**

``torch.compile`` is PyTorch’s native deep learning compiler. There are two principal differences
between ``torch.compile`` and *thunder*:

1. ``thunder`` is written entirely in Python
2. ``thunder`` has a multi-level IR that can reduce every operation to HLO-like primitives

These traits make ``thunder`` easy to debug, optimize and extend, enabling developers to quickly target optimized kernels or distributed strategies.

**How is thunder different from JAX?**

JAX is a deep learning compiler designed to work with XLA. There are two principal differences between JAX and thunder:

1. JAX exclusively traces functions, and does not have the equivalent of thunder's preprocessing to convert from PyTorch modules to traceable functions
2. *thunder* has a multi-level IR that makes it easy to reason about PyTorch operations, their decompositions and execution


.. toctree::
   :maxdepth: 1
   :name: home
   :caption: Home

   self
   Get thunder (NVIDIA only) <fundamentals/container>
   Install <fundamentals/installation>
   Hello World <fundamentals/hello_world>
   Using examine <fundamentals/examine>
   Get Involved <fundamentals/get_involved>

.. toctree::
   :maxdepth: 1
   :name: basic
   :caption: Basic

   Overview <basic/overview>
   Thunder step by step <basic/inspecting_traces>
   The sharp edges <basic/sharp_edges>
   Train a MLP on MNIST <basic/mlp_mnist>

.. toctree::
   :maxdepth: 1
   :name: intermediate
   :caption: Intermediate

   Compile options <intermediate/compile_options>
   Additional executors <intermediate/additional_executors>
   Distributed Data Parallel <intermediate/ddp>
   What's next <intermediate/whats_next>

.. toctree::
   :maxdepth: 1
   :name: advanced
   :caption: Advanced

   Inside thunder <advanced/inside_thunder>
   Extending thunder <advanced/extending>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
