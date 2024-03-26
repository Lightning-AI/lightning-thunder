Thunder FAQ
################

=================================================
1. How does Thunder compare to ``torch.compile``?
=================================================

Both `Thunder <https://github.com/Lightning-AI/lightning-thunder>`_ and `torch.compile <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler-overview>`_ are both deep learning compilers that take a pytorch module or callable, and return a callable. As such, it seems reasonable to compare them that way. With that said, the focus of the projects are completely different.

Torch compile is a framework for generating optimized kernels. Thunder is a framework for layering optimizations. It generates none of these optimizations itself, instead delegating to other libraries, including ``torch.compile`` and `nvfuser <https://github.com/NVIDIA/Fuser>`_.

As such, the two are not necessarily comparable. Or, they are, but you would be comparing against a default configuration, which we expect you to change.



============================================
2. How can I use torch.compile with Thunder?
============================================

The correct way to use ``torch.compile`` with Thunder is as an executor. This gives you finer grained control over which parts of your model are handled by which

Calling ``torch.compile()`` and then ``thunder.jit()`` is not what you want. It doesn't give a good error message right now, but it should not work.

Instead, register the executor like so::

    import torch
    import thunder

    def model(x, y):
        return x + y

    exc_list = [thunder.extend.get_executor('torchcompile'), \*thunder.get_always_executors()]
    jmodel = thunder.jit(model, executors=exc_list)


====================================================================================
3. I have a CUDA, Triton, CUDNN, or other gpu kernel. How can I use it with thunder?
====================================================================================

Why, yes! You can register it as an operator for a custom executor. See :doc:`extending thunder <../intermediate/additional_executors>` for more information.


========================================================================
3. Do you support custom hardware, or accelerators that aren't Nvidia's?
========================================================================

Yes, executors are device-agnostic. The python executor, for example, runs the operation with cpython on the cpu. We've been focusing on Nvidia, but we welcome contributions for executors for other accelerator backends.


=================================================================
4. I ran ``thunder.jit(model)(*args)``, and my model didn't work! 
=================================================================

Thunder is in alpha. There will be bugs, and many torch operations are not supported. Try to run ``thunder.examine(model)(*args)``. This will list the operations which are not supported, and if they are all supported, test the model for consistency against torch eager.

If you need certain operations supported for your model, please let us know by creating an issue. We plan to get to all of them (with the exception of any :doc:`sharp edges <sharp_edges>`), but your issues help us prioritize which to do first.

There are potentially any number of other problems which could arise. Some of the problems are known, some may not be. Check out the :doc:`sharp edges <sharp_edges>` page. If what you're seeing still doesn't make sense, let us know by creating an issue.



