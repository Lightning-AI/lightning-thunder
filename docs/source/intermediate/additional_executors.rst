Additional executors
####################

nvFuser and Pytorch are not the only executors available in Thunder today. Additional executors can be added to *thunder* prior to compilation through a registration mechanism, which makes it easy to have specialized executors perform certain operations more efficiently.

This section contains a list of all executors supported by PyTorch beyond nvFuser and PyTorch.

Triton CrossEntropy Executor
============================

The Triton CrossEntropy executor can execute ``torch.cross_entropy()`` using an optimized kernel written in OpenAI Triton (https://github.com/openai/triton). It can be used like in the following example::

  import torch
  import thunder
  from thunder.executors.triton_crossentropy import triton_ex as triton_cross_entropy_ex

  def xentropy(logits, labels, weight, reduction, ignore_index):
      return thunder.torch.cross_entropy(
          logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
      )

  jitted_xentropy = thunder.jit(
    xentropy,
    executors=[triton_cross_entropy_ex,]
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

  # Constructed by Delete Last Used (took 0 milliseconds)
  import torch
  from thunder.executors.torchex import no_autocast

  @torch.no_grad()
  @no_autocast
  def computation(logits, labels, weight):
    # logits: "cuda:0 f32[2048, 50257]"
    # labels: "cuda:0 i64[2048]"
    # weight: "cuda:0 f32[50257]"
    t23 = triton_crossentropy(logits, labels, weight, None, 45279, None, 'sum', 0.0)  # t23: "cuda:0 f32[]"
    del logits, labels, weight
    return t23

As shown in the above trace, ``triton_crossentropy()`` is the one running the operation.

Apex CrossEntropy Executor
==========================

The Apex CrossEntropy executor can execute ``torch.cross_entropy()`` through an optimized kernel, like this::

  import torch
  import thunder
  from thunder.executors.apex_entropyex import apex_ex

  def xentropy(logits, labels):
      return thunder.torch.cross_entropy(
          logits, labels, reduction='mean', ignore_index=-1
      )

  jitted_xentropy = thunder.jit(xentropy, executors=[apex_ex,])

  device = 'cuda'
  dtype = torch.float32

  logits = torch.randn([2048, 50257], device=device, dtype=dtype)
  labels = torch.randint(0, 50257, [2048], device=device)

  jitted_xentropy(logits, labels)
  traces = thunder.last_traces(jitted_xentropy)
  print(traces[-1])

This prints::

  # Constructed by Delete Last Used (took 0 milliseconds)
  import torch
  from thunder.executors.torchex import no_autocast

  @torch.no_grad()
  @no_autocast
  def computation(logits, labels):
    # logits: "cuda:0 f32[2048, 50257]"
    # labels: "cuda:0 i64[2048]"
    (t18, _) = apex_cross_entropy(logits, labels, 'mean', 0.0)
    del logits, labels
    return t18

showing that Apex is running the operation.

cuDNN SDPA Executor
===================

TODO RC1

TransformerEngine Executor
==========================

TODO RC1
