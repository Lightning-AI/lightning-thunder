Thunder FAQ
################

=================================================
1. How does Thunder compare to ``torch.compile``?
=================================================

Both `Thunder <https://github.com/Lightning-AI/lightning-thunder>`_ and `torch.compile <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler-overview>`_ are both deep learning compilers that take a pytorch module or callable, and return a callable. It seems reasonable to compare them that way. With that said, the focus of the projects are completely different.

Torch compile is a framework for generating optimized kernels. Its focus is on making pytorch code run faster by generating optimized kernels, with minimal code changes. Thunder is a framework for layering optimizations. It generates none of these optimizations itself, instead delegating to other libraries, including ``torch.compile`` and `nvfuser <https://github.com/NVIDIA/Fuser>`_. Our focus is on understandability, extendability, and usability.

Modern deep learning optimization often involves stitching various kernels together, often from different sources. Thunder is designed to make it easy to use these tools together, and easy to add new tools to the mix.

As such, the two are not necessarily comparable. Or, they are, but you would be comparing against a default configuration, which we expect you to extend and change anyway.



============================================
2. How can I use torch.compile with Thunder?
============================================

The correct way to use ``torch.compile`` with Thunder is to use the executor. This way, you get finer grained control over which parts of the model are handled by which executor.

Calling ``torch.compile()`` and then ``thunder.jit()`` is not what you want. It doesn't give a good error message right now, but it should not work.

Instead, register the executor like so::

    import thunder
    from thunder.executors.torch_compile import torch_compile_ex

    def model(x, y):
        return x + y

    jmodel = thunder.jit(model, executors=[torch_compile_ex])


This will pass to ``torch.compile`` all the torch operators and is meant to be used without the nvfuser executor
since they would be competing over fusion opportunities. The advantage over simply doing ``torch.compile`` is that you
still get all of Thunder's advantages, like enabling custom executors (e.g. with custom triton kernels) before it.

You can also use it for a smaller subset of operators where it shines the most. This variant is meant to be used
together with the nvfuser executor. Its current goal is only to fuse RoPE but the set of ops fused will change as each
of the fusion backends evolve::

    import thunder
    from thunder.executors.torch_compile import torch_compile_cat_ex

    def model(x, y):
        return x + y

    executors = [torch_compile_cat_ex, *thunder.get_always_executors()]
    jmodel = thunder.jit(model, executors=executors)


====================================================================================
3. I have a CUDA, Triton, CUDNN, or other gpu kernel. How can I use it with thunder?
====================================================================================

Why, yes! You can register it as an operator for a custom executor. See :doc:`extending thunder <../intermediate/additional_executors>` for more information.


========================================================================
3. Do you support custom hardware, or accelerators that aren't Nvidia's?
========================================================================

Yes, executors are device-agnostic. The python executor for example runs the operation with cpython on the cpu. We've been focusing on the Nvidia stack, but Thunder is designed to be extensible so you can write your own executors for any backend. Just make an executor and register the operators you need. We welcome contributions for executors for other accelerator backends.


=================================================================
4. I ran ``thunder.jit(model)(*args)``, and my model didn't work!
=================================================================

Thunder is in alpha. There will be bugs, and many torch operations are not supported. Try to run ``from thunder.examine import examine; examine(model, *args)``. This will list the operations which are not supported, and if they are all supported, test the model for consistency against torch eager.

If you need certain operations supported for your model, please let us know by creating an issue. We plan to get to all of them (with the exception of any :doc:`sharp edges <sharp_edges>`), but your issues help us prioritize which to do first.

There are potentially any number of other problems which could arise. Some of the problems are known, some may not be. Check out the :doc:`sharp edges <sharp_edges>` page. If what you're seeing still doesn't make sense, let us know by creating an issue.


=======================================
5. Does Thunder support dynamic shapes?
=======================================

No, not at the moment. However, we're actively working on it.

Meta functions operate on the exact shapes of the tensor proxies that pass through them. This is a limitation of the current implementation, and we plan to incorporate dynamic shapes in the future. If you have relevant experience experience with this problem in pytorch, or you are interested in helping us implement it, please let us know by creating an issue or reaching out.


================================================================
6. Does Thunder support inplace operations?
================================================================

Not at the moment. Implementing inplace operations would require tracking which tensors in a trace have been modified by operations in our optimization passes, which currently we represent as purely functional. All deep learning compiler frameworks have to deal with the problem of tensor aliasing in some way. The way we've chosen for now is to pretend that the problem doesn't exist.

The common solution is to represent programs in `SSA form <https://en.wikipedia.org/wiki/Static_single-assignment_form>`_, or do some form of SSA-inspired variable renaming, but SSA is a much less understandable representation than a list of symbols in a trace. Switching to SSA would also complicate optimization passes, and require rewriting many of them to handle these aliasing rules.

There also exists the problem that some backend executors do not support in-place operations. We have some ideas on how to functionalize ops for these executors, but some api issues are unresolved.

We want to support inplace operations eventually, but we are attached to traces as our program representation of choice for optimization passes. Much like with dynamic shapes, if you have relevant experience on how to best incorporate inplace operations without complicating optimization passes, come talk to us about it.
