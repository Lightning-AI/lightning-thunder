Additional executors
####################

nvFuser and Pytorch are not the only executors available in Thunder today. Additional executors can be added to *thunder* prior to compilation through a registration mechanism, which makes it easy to have specialized executors perform certain operations more efficiently.

This section contains a list of all executors supported by PyTorch beyond nvFuser and PyTorch.

Triton CrossEntropy Executor
============================

The Triton CrossEntropy executor can execute ``torch.cross_entropy()`` using an optimized kernel written in OpenAI Triton (https://github.com/openai/triton). It can be used like in the following example::

  import thunder
  from thunder.executors import nvfuserex, torchex
  from thunder.executors.triton_crossentropy import deregister_triton_entropyex, register_triton_entropyex

  register_triton_entropyex(add_to_default_executors=False)

  def xentropy(logits, labels, weight, reduction, ignore_index):
      return thunder.torch.cross_entropy(
          logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
      )

  jitted_xentropy = thunder.jit(
    xentropy,
    executors_list=['triton_crossentropy', nvfuserex, torchex]
  )

  device = 'cuda'
  dtype = torch.float32

  logits = torch.randn([2048, 50257], device=device, dtype=dtype)
  labels = torch.randint(0, 50257, [2048], device=device)
  weight = torch.rand(50257, device=device, dtype=dtype, requires_grad=False)
  reduction = "sum"
  ignore_index = labels[5].item()

  jitted_xentropy(logits, labels, weight, reduction, ignore_index)
  traces = thunder.last_traces(jitted_xentropy)
  print(traces[-1])

This prints::

  # Constructed by Delete Last Used
  import torch
  @torch.no_grad()
  def xentropy(logits, labels, weight, reduction, ignore_index):
    # logits: "cuda:0 f32[2048, 50257]"
    # labels: "cuda:0 i64[2048]"
    # weight: "cuda:0 f32[50257]"
    # "sum"
    # ignore_index: "int 10106"
    t22 = triton_cross_entropy(logits, labels, weight, None, ignore_index, None, "sum", 0.0)  # t22: "cuda:0 f32[]"
    del [logits, labels, weight, ignore_index]
    return t22

As shown in the above trace, ``triton_cross_entropy()`` is the one running the operation.

Apex CrossEntropy Executor
==========================

The Apex CrossEntropy executor can execute ``torch.cross_entropy()`` through an optimized kernel, like this::

  import thunder
  from thunder.executors import nvfuserex, torchex
  from thunder.executors.apex_entropyex import deregister_apex_entropyex, register_apex_entropyex

  register_apex_entropyex(add_to_default_executors=False)

  def xentropy(logits, labels):
      return thunder.torch.cross_entropy(
          logits, labels, reduction='mean', ignore_index=-1
      )

  jitted_xentropy = thunder.jit(xentropy, executors_list=['apex_xentropy', nvfuserex, torchex])

  device = 'cuda'
  dtype = torch.float32

  logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype))
  labels = torch.randint(0, 50257, [2048], device=device)

  jitted_xentropy(logits, labels)
  traces = thunder.last_traces(jitted_xentropy)
  print(traces[-1])

This prints::

  # Constructed by Delete Last Used
  import torch
  @torch.no_grad()
  def xentropy(logits, labels):
    # logits: "cuda:0 f32[2048, 50257]"
    # labels: "cuda:0 i64[2048]"
    t18 = apex_cross_entropy(logits, labels, None, None, -1, None, "mean", 0.0)  # t18: "cuda:0 f32[]"
    del [logits, labels]
    return t18

showing that Apex is running the operation.

cuDNN SDPA Executor
===================

TODO RC1

TransformerEngine Executor
==========================

TODO RC1
