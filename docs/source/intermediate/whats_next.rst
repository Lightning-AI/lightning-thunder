What's Next
###########

*thunder* is developing rapidly, and this section mentions some of what's happening. Please reach out (see Get Involved) if you're interested in one of these topics.

Compiling the Training Loop
===========================

*thunder* currently supports compiling PyTorch modules - forward computation, loss calculation, backpropagation -, but in the future we plan to support compiling the entire training loop - forward computation, loss calculation, backpropagation, and the optimizer step - for maximum performance.

Fully Shared Data Parallel (FSDP)
=================================

*thunder* currently supports distributed data parallel (DDP), and it has a great architecture for expressing more advanced distributed strategies like FSDP. We expect to extend our core architecture with a sharded tensor-like object, then use it to build a variety of advanced distributed strategies for practitioners, letting them easily access off-the-shelf strategies while giving developers access to powerful tools for creating new ones.

Dynamic Caching
===============

*thunder* currently supports either no caching or static caching, and static caching requires recompiling whenever a module is called with inputs with metadata different than past inputs. This can be overly strict. For example, adding two tensors with shape ``(5, 5)`` is essentially the same as adding two tensors with shape ``(10, 10)``. Dynamic caching will determine if the new metadata would result in a new trace or not, significantly reducing compilation time when training some models.

Memory Layouts and Strides
==========================

*thunder* currently does not model any stride information on tensor proxies. In the future we will likely model some stride information, like memory layout (e.g. channels-last), to support integration with PyTorch programs that use memory layout, and to let executors use memory layout to inform kernel selection.

Functional transforms: vmap and AMP
===================================

*thunder* already has early implementations of JAX's vmap transform and PyTorch's Automatic Mixed Precision (AMP) autocasting, and we're extending our support for these transforms so practitioners can easily apply a variety of composable transforms to PyTorch modules.
