Distributed Data Parallel (DDP)
###############################

*thunder* has its own Distributed Data Parallel (DDP) transform that we recommend using, although compiled modules also work with PyTorch's DDP transform.

You can wrap a model in *thunder*'s ddp like this::

  from thunder.distributed import ddp

  model = MyModel()
  ddp_model = ddp(
    model,
    rank=rank,
    broadcast_from=0,
    process_group=pg,
  )

  cmodel = thunder.compile(ddp_model)

Specifying the process group and which rank to broadcast from is optional. ``ddp()`` will use PyTorch's default process group if none is specified, and will broadcast from the lowest rank in that group if broadcast_from is not specified.

*thunder*'s ddp is compatible with PyTorch distributed runners like torchrun (https://pytorch.org/docs/stable/elastic/run.html).

When using PyTorch's DDP, call DDP on the compiled module::

  from torch.nn.parallel import DistributedDataParallel as DDP

  model = MyModel()
  cmodel = DDP(thunder.compile(model)

The ability of *thunder* to express distributed algorithms like DDP as a simple transform on the trace is one of *thunder*'s strengths and is being leveraged to quickly implement more elaborate distributed strategies.
