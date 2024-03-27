Distributed Data Parallel (DDP)
###############################

Thunder has its own Distributed Data Parallel (DDP) transform that we recommend using, although compiled modules also work with PyTorch's DDP transform.

You can wrap a model in Thunder's ddp like this::

  from thunder.distributed import ddp

  model = MyModel()
  ddp_model = ddp(model)
  cmodel = thunder.jit(ddp_model)

Specifying which rank to broadcast from is optional. ``ddp()`` will broadcast from the lowest rank in that group if ``broadcast_from`` is not specified.

Thunder's ddp is compatible with PyTorch distributed runners like ``torchrun`` (https://pytorch.org/docs/stable/elastic/run.html).

When using PyTorch's DDP, call DDP on the jitted module::

  from torch.nn.parallel import DistributedDataParallel as DDP

  model = MyModel()
  jitted_model = thunder.jit(model)
  ddp_model = DDP(jitted_model)

The ability of Thunder to express distributed algorithms like DDP as a simple transform on the trace is one of Thunder's strengths and is being leveraged to quickly implement more elaborate distributed strategies, like Fully Sharded Data Parallel (FSDP).
