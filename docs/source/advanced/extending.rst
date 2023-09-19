Extending Thunder
#################

This section describes how to add an executor to *thunder* for a PyTorch operation.

First, define a Python function with the same signature as the targeted operation, and have it call your implementation. For example, the Apex executor for ``torch.nn.functional.cross_entropy`` might define its implementation like::

  import torch
  import xentropy_cuda

  def apex_xentropy(
      a: torch.Tensor,  # a is an actual PyTorch tensor
      target,
      weight=None,
      size_average=None,
      ignore_index=-100,
      reduce=None,
      reduction="mean",
      label_smoothing=0.0,
  ):
      losses, max_log_sum_exp = xentropy_cuda.forward(a, target, label_smoothing, half_to_float)

When this implementation is used it will be called with actual PyTorch tensors, and not with proxies.

Next, define a “checker” function with the same signature as the targeted operation that returns True if your operation can execute the targeted operation and False otherwise. Checkers, unlike the implementations, are called with proxies, and not actual PyTorch tensors, because they're called at optimization time. The purpose of a checker function is to let executors target only specific inputs to an operation, and defer to another executor on other inputs.

A checker function for the Apex executor might look like::

  from thunder.core.proxies import TensorProxy

  def apex_xentropy_checker(
      a: TensorProxy,  # a is a proxy
      target,
      weight=None,
      size_average=None,
      ignore_index=-100,
      reduce=None,
      reduction="mean",
      label_smoothing=0.0,
  ):
    # Apex's xentropy only supports "sum", "mean" or "none" reductions
    if reduction not in ["sum", "mean", "none"]:
      return False

    return True

Create a mapping from the name of the PyTorch operation to your replacement implementation's name, its checker, and its implementation::

  _op_to_xentropy = {
      "torch.nn.functional.cross_entropy": ("apex_xentropy", apex_xentropy_checker, apex_xentropy),
  }

Then define a registration function that practitioners can call to access your executor::

  def register_apex_xentropyex(*, add_to_default_executors: bool = True) -> None:
      from thunder.executors import add_operator_executor

      return add_operator_executor("apex_xentropy", _op_to_xentropy, add_to_default_executors=add_to_default_executors)

You can test your executor by registering it, compiling a function that calls the targeted operator, and then verifying that your operation is called (by inspecting the execution trace) and producing the correct output. A good example of this is the tests for the Apex executor.
