import os
import typing
import thunder
import thunder.core.trace
import thunder.executors.torch_compile
import torch
from thunder.dev_utils.utils import _benchmark_fusion_region_with_nvfuser_and_torch_compile
from typing import Any, Iterable, Sequence

def arg_like_tensor_integer(arg: torch.Tensor, f: typing.TextIO):
  """Creates a new argument like the given tensor, which must be an integer
  type. This is separated out because randn() does not work for integer
  types."""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  assert arg.dtype in itypes
  minmax: tuple[torch.Tensor,torch.Tensor] = torch.aminmax(arg)
  # Sometimes the tensor can just be all the same value, in which case
  # randint() will complain that there's no "range" to the low/high params.
  # So we use a completely different method for tensor generation, then.
  if minmax[0].cpu().item() == minmax[1].cpu().item():
    meta = f"data={minmax[0].cpu().item()}, dtype={arg.dtype},"
    meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
    print(f"  torch.tensor({meta}).broadcast_to({arg.shape}),", file=f)
    return
  meta = f"low={minmax[0].cpu().item()}, high={minmax[1].cpu().item()},"
  meta = f"{meta} size={arg.shape},"
  meta = f"{meta} dtype={arg.dtype}, layout={arg.layout},"
  meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
  print(f"  torch.randint({meta}),", file=f)

def arg_like_tensor(arg: torch.Tensor, f: typing.TextIO):
  """Creates a new argument like the given tensor"""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  assert arg.dtype not in itypes
  minmax: tuple[torch.Tensor,torch.Tensor] = torch.aminmax(arg)
  meta = f"size={arg.shape},"
  meta = f"{meta} dtype={arg.dtype}, layout={arg.layout},"
  meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
  print(f"  torch.randn({meta}),", file=f)

def _torch_device(d: thunder.core.devices.Device) -> str:
  """Translates a thunder device back to a torch one."""
  devtype: str = "cuda"
  match d.devicetype:
    case thunder.core.devices.DeviceType.CPU: dev: str = "cpu"
    case thunder.core.devices.DeviceType.CUDA: dev: str = "cuda"
    case thunder.core.devices.DeviceType.META: dev: str = "meta"
  if d.index != 0:
    dev: str = "".join([dev, ":", str(d.index)])
  return dev

def _torch_dtype(dt: thunder.dtypes.dtype) -> str:
  match dt:
    case thunder.dtypes.float8_e5m2 | thunder.dtypes.float8_e5m2_ | \
         thunder.dtypes.float8_e5m2fnuz | thunder.dtypes.float8_e5m2fnuz_:
      raise NotImplementedError(f"can't translate e5m2 {dt}")
    case thunder.dtypes.float8_e4m3fn | thunder.dtypes.float8_e4m3fn_ | \
         thunder.dtypes.float8_e4m3fnuz | thunder.dtypes.float8_e4m3fnuz_:
      raise NotImplementedError(f"can't translate e4m3 {dt}")
    case thunder.dtypes.bfloat16 | thunder.dtypes.bfloat16_: return "torch.bfloat16"
    case thunder.dtypes.float16 | thunder.dtypes.float16_: return "torch.float16"
    case thunder.dtypes.float32 | thunder.dtypes.float32_: return "torch.float32"
    case thunder.dtypes.float64 | thunder.dtypes.float64_: return "torch.float64"
    case thunder.dtypes.int8 | thunder.dtypes.int8_: return "torch.int8"
    case thunder.dtypes.int16 | thunder.dtypes.int16_: return "torch.int16"
    case thunder.dtypes.int32 | thunder.dtypes.int32_: return "torch.int32"
    case thunder.dtypes.int64 | thunder.dtypes.int64_: return "torch.int64"
    case thunder.dtypes.uint8 | thunder.dtypes.uint8_: return "torch.uint8"
    case thunder.dtypes.uint16 | thunder.dtypes.uint16_: return "torch.uint16"
    case thunder.dtypes.uint32 | thunder.dtypes.uint32_: return "torch.uint32"
    case thunder.dtypes.uint64 | thunder.dtypes.uint64_: return "torch.uint64"
    case _: raise

def arg_like_tensor_proxy_fp(f: typing.TextIO, arg: thunder.core.proxies.TensorProxy) -> None:
  """Creates a new argument like the given tensor"""
  meta = f"size={arg.shape},"
  meta = f"{meta} dtype={_torch_dtype(arg.dtype)},"
  # Thunder's TensorProxy does not have a 'layout' arg.
  #meta = f"{meta} dtype={arg.dtype}, layout={arg.layout},"
  meta = f"{meta} layout=torch.strided,"
  meta = f"{meta} device=\"{_torch_device(arg.device)}\","
  meta = f"{meta} requires_grad={arg.requires_grad}"
  sz = arg.shape
  strd = arg.stride()
  print(f"  torch.randn({meta}).as_strided({sz}, {strd}),", file=f)


def arg_like_tensor_proxy_integer(f: typing.TextIO, arg: thunder.core.proxies.TensorProxy) -> None:
  itypes = (
    thunder.dtypes.int8,
    thunder.dtypes.int8_,
    thunder.dtypes.int16,
    thunder.dtypes.int16_,
    thunder.dtypes.int32,
    thunder.dtypes.int32_,
    thunder.dtypes.int64,
    thunder.dtypes.int64_,
    thunder.dtypes.uint8,
    thunder.dtypes.uint8_,
    thunder.dtypes.bool8,
    thunder.dtypes.bool8_,
  )
  assert arg.dtype in itypes
  minmax: tuple[int, int] = (0, 127) # hack: choose a minmax that'll always fit
  if arg.dtype in (thunder.core.dtypes.bool8, thunder.core.dtypes.bool8_):
    minmax: tuple[int, int] = (0, 1)
  # Sometimes the tensor can just be all the same value, in which case
  # randint() will complain that there's no "range" to the low/high params.
  # So we use a completely different method for tensor generation, then.
  if minmax[0] == minmax[1]:
    meta = f"data={minmax[0]}, dtype={arg.dtype},"
    meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
    print(f"  torch.tensor({meta}).broadcast_to({arg.shape}),", file=f)
    return
  meta = f"low={minmax[0]}, high={minmax[1]},"
  meta = f"{meta} size={arg.shape},"
  meta = f"{meta} dtype={arg.dtype},"
  # Thunder's TensorProxy does not implement 'layout' yet, so we hardcode torch.strided.
  meta = f"{meta} layout=torch.strided,"
  meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
  print(f"  torch.randint({meta}),", file=f)

def arg_like_tensor_proxy(f: typing.TextIO, arg: thunder.core.proxies.TensorProxy) -> None:
  assert isinstance(arg, thunder.core.proxies.TensorProxy)
  itypes = (
    thunder.core.dtypes.int8,
    thunder.core.dtypes.int8_,
    thunder.core.dtypes.int16,
    thunder.core.dtypes.int16_,
    thunder.core.dtypes.int32,
    thunder.core.dtypes.int32_,
    thunder.core.dtypes.int64,
    thunder.core.dtypes.int64_,
    thunder.core.dtypes.uint8,
    thunder.core.dtypes.uint8_,
    thunder.core.dtypes.bool8,
    thunder.core.dtypes.bool8_,
  )
  if arg.dtype in itypes:
    arg_like_tensor_proxy_integer(f, arg)
  else:
    arg_like_tensor_proxy_fp(f, arg)

def arg_like(f: typing.TextIO, arg: typing.Any) -> None:
  """Creates a new argument that is similar to the given arg."""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  if isinstance(arg, torch.Tensor) and arg.dtype in itypes:
      arg_like_tensor_integer(arg, f)
  elif isinstance(arg, torch.Tensor):
      arg_like_tensor(arg, f)
  elif isinstance(arg, thunder.core.proxies.TensorProxy):
      arg_like_tensor_proxy(f, arg)
  else:
      # Assume it's a literal that we can just print directly.
      print(f"  {arg},", file=f)

def _standalone(f: typing.TextIO, fusion: typing.Any):
  #print("fusion is a:", type(fusion))
  # fusion is a FusionDefinitionWrapper from thunder
  if fusion.last_inputs is not None:
    pycode: str = fusion.last_used.repro_script_for(fusion.last_inputs)
    print(pycode, file=f)
  else:
    print("import torch", file=f)
    print("from nvfuser import DataType, FusionDefinition\n", file=f)
    print(fusion.last_used, file=f)

def _simplify_stride_order(sym: thunder.core.baseutils.BoundSymbolInterface) -> None:
  """
  The thunder stride_order primitive is not directly translatable to torch. This tries to rewrite
  the primitive when it appears as a subsymbol to something that *is* representable as torch
  functions.
  """
  def _compute_strides(shp: Sequence[int], order: Sequence[int]) -> list[int]:
    ordered_dims = sorted(zip(shp, order), key=lambda x: x[1])
    ordered_strides = [1]
    accum = ordered_dims[0][0]
    for dim_length, _ in ordered_dims[1:]:
        ordered_strides.append(accum)
        accum *= dim_length

    strides = tuple(ordered_strides[x] for x in order)
    return strides

  def _stride_order_printer(
    bsym: thunder.core.baseutils.BoundSymbolInterface,
    out_printables: Any,
    arg_printables: Sequence[Any],
    kwarg_printables: dict[str, Any]
  ) -> str | Iterable[str]:
    if arg_printables:
      arg_printables = list(arg_printables)
      assert isinstance(bsym.args[1], (list, tuple, Sequence))
      strides = _compute_strides(bsym.args[0].shape, bsym.args[1])
      arg_printables = (
        bsym.args[0],
        bsym.args[0].shape,
        strides
      )
    arg_str = (
        ""
        if (arg_printables is None or len(arg_printables) == 0)
        else ", ".join(thunder.core.codeutils.prettyprint(x) for x in arg_printables)
    )
    kwarg_str: str

    if len(kwarg_printables) == 0:
        kwarg_str = ""
    else:
        kwarg_str = ", ".join(f"{k}={thunder.core.codeutils.prettyprint(v)}" for k, v in kwarg_printables.items())

    result_str: str
    if bsym.output is None or (thunder.core.baseutils.is_collection(bsym.output) and len(bsym.output) == 0):
        result_str = ""
    else:
        result_str = f"{thunder.core.codeutils.prettyprint(out_printables, literals_as_underscores=True)} = "

    # Creates a comment describing the output
    comment_str = ""
    if isinstance(bsym.output, thunder.core.proxies.Proxy):
        comment_str = f"  # {thunder.core.codeutils.prettyprint(out_printables, with_type=True)}"

    s = f"{result_str}{torch.as_strided}({arg_str}{', ' if (len(arg_str) > 0 and len(kwarg_str) > 0) else ''}{kwarg_str}){comment_str}"

    if bsym.header:
        header_lines = (
            bsym.header
            if isinstance(bsym.header, Sequence) and not isinstance(bsym.header, str)
            else bsym.header.splitlines()
        )
        header_lines = (f"# {line}" for line in header_lines)
        return chain(header_lines, [s])

    return s

  for s in sym.subsymbols:
    print(f"{s.sym.name=}", flush=True)
    if s.sym.name == "torch_stride_order_prim_impl" or s.sym == thunder.core.prims.stride_order:
      import dataclasses
      s.sym = dataclasses.replace(s.sym, python_printer=_stride_order_printer)

def _tc_standalone(f: typing.TextIO, sym: thunder.core.baseutils.BoundSymbolInterface) -> None:
  """
  Generates a torch.compile program for the given symbol.
  """
  trc: thunder.core.TraceCtx = thunder.executors.torch_compile.to_trace(
    sym.subsymbols, sym.flat_args, sym.flat_outs
  )
  #print("trc:", trc)
  #for a in trc.args:
  #  print(f"arg: {type(a)}, {a=}")

  pgm: str = trc.python(print_depth=4294967296, include_decorators=True)
  # We use some torch_reshaps_prims_impl indirection that is no longer valid PyTorch code. Rewrite
  # it to fix things.
  pgm = pgm.replace("torch_prims_reshape_impl", "torch.reshape")
  pgm = pgm.replace(" = Tensor.to", " = torch.Tensor.to")
  print(pgm, file=f, end="\n\n")

  print("inputs = [", file=f)
  for arg in trc.args:
    arg_like(f, arg)
  print("]\n", file=f)
  print("tc = torch.compile(to_be_compiled)", file=f)
  print("tc(*inputs)", file=f)


#def to_be_compiled(t222, t1548, t_transformer_h_2_ln_1_weight, t_transformer_h_2_ln_1_bias):
#arg: <class 'thunder.core.proxies.TensorProxy'>, a=<TensorProxy(name="t222", dtype=thunder.dtypes.float32, shape=(4, 64, 768))>
#arg: <class 'thunder.core.proxies.TensorProxy'>, a=<TensorProxy(name="t1548", dtype=thunder.dtypes.float32, shape=(4, 64, 768))>
#arg: <class 'thunder.core.proxies.TensorProxy'>, a=<TensorProxy(name="t_transformer_h_2_ln_1_weight", dtype=thunder.dtypes.float32, shape=(768,))>
#arg: <class 'thunder.core.proxies.TensorProxy'>, a=<TensorProxy(name="t_transformer_h_2_ln_1_bias", dtype=thunder.dtypes.float32, shape=(768,))>

def _release_nvfuser_memory(bsym: thunder.core.symbol.BoundSymbol) -> None:
  nvfusion = bsym._call_ctx[bsym.sym.name]
  del nvfusion.last_inputs
  nvfusion.last_inputs = None
  del nvfusion.last_used


def _worse_than_tc(directory: str, prefix: str, trace: thunder.core.trace.TraceCtx) -> None:
  """
  Create output files for all the fusions where torch.compile is better than nvFuser.
  """
  for bsym in trace.bound_symbols:
    if bsym.sym.is_fusion and "nvFusion" in bsym.sym.name:
      try:
        data = _benchmark_fusion_region_with_nvfuser_and_torch_compile(bsym)
      except torch._dynamo.exc.Unsupported:
        # i.e. dynamo failed, in which case we cannot compare.
        continue
      _simplify_stride_order(bsym)
      filename: str | None = None
      if data.torch_compile_kernel_time <= data.nvfuser_kernel_time and \
         data.torch_compile_walltime.median <= data.nvfuser_walltime.median:
        filename: str = f"{directory}/worse-{prefix}-{bsym.sym.name}.py"
      elif data.torch_compile_kernel_time <= data.nvfuser_kernel_time:
        filename: str = f"{directory}/ktime-worse-{prefix}-{bsym.sym.name}.py"
      elif data.torch_compile_walltime.median <= data.nvfuser_walltime.median:
        filename: str = f"{directory}/walltime-worse-{prefix}-{bsym.sym.name}.py"

      if filename is None:
        continue

      nvf_callable = bsym._call_ctx[bsym.sym.name]
      with open(f"{filename}", "w") as f:
        print(f"writing {filename}")
        _standalone(f, nvf_callable)
        _tc_standalone(f, bsym)
      # To save memory, let the inputs die.
      _release_nvfuser_memory(bsym)


def _fusions_from(DIR_NAME: str, prefix: str, ctx: dict) -> None:
  for k in ctx:
    if k.startswith("nvFusion") or k.startswith("TorchCompile"):
      if ctx[k].last_used is not None:
        print(f"writing {DIR_NAME}/{prefix}-{k}.py")
        with open(f"{DIR_NAME}/{prefix}-{k}.py", "w") as f:
          _standalone(f, ctx[k])

def traces(directory: str, functor: typing.Callable):
    fwd: list = thunder.last_traces(functor)
    bwd: list = thunder.last_backward_traces(functor)
    if fwd is None and bwd is None:
      return
    os.makedirs(directory, exist_ok=True)

    if fwd is not None and len(fwd) > 0:
      _worse_than_tc(directory, "fwd", fwd[-1])
    del fwd

    if bwd is not None and len(bwd) > 0:
      _worse_than_tc(directory, "bwd", bwd[-1])
