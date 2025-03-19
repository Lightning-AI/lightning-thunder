import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


class ScaleTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, b, outer_size=None, outer_stride=None):
        if outer_size == None:
            outer_size = a.shape
        if outer_stride == None:
            outer_stride = a.stride()
        shape = outer_size
        stride = outer_stride
        kwargs = {}
        kwargs["strides"] = stride
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        return out

    def __init__(self, a, b, outer_size=None, outer_stride=None):
        self.a = a
        self.b = b

    def __repr__(self):
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"ScaleTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        return [
            "a",
        ], {"b": self.b}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size=None, outer_stride=None):
        a = inner_tensors["a"]
        b = meta["b"]
        return ScaleTensor(a, b, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(ScaleTensor, lambda x: x.a, args)
        args_b = pytree.tree_map_only(ScaleTensor, lambda x: x.b, args)

        kwargs_a = pytree.tree_map_only(ScaleTensor, lambda x: x.a, kwargs)
        kwargs_b = pytree.tree_map_only(ScaleTensor, lambda x: x.b, kwargs)
        out_a = func(*args_a, **kwargs_a)
        try:
            out_b = sum(args_b)
        except:
            out_b = 1

        if isinstance(out_a, (list, tuple)):
            return list(ScaleTensor(o, out_b) for o in out_a)
        return ScaleTensor(out_a, out_b)


# @torch.compile()
def f(x, y):
    return x * y  #  * 2
    # return x.split(2)
    # return torch.nn.functional.smooth_l1_loss(x, y)


x1_inner = torch.ones(4, 4)
x1 = ScaleTensor(x1_inner, 2)
x2 = ScaleTensor(x1_inner, 2)

out1 = f(x1, x2)

from thunder.dynamo import thunderfx
from thunder.core.transforms import Transform
from thunder.torch.tensor_subclass_utils import (
    trace_from_bsym_or_bsyms,
    make_trace_executable,
    get_fx_graph,
    decompose_into_aten_subsymbols,
)


class ATenTransform(Transform):
    def transform_traces_pre_prologue(self, prologue_trace, comp_trace, epi_trc, executors_list):
        from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
        from thunder.core.proxies import TensorProxy, ScaleTensorProxy, variableify
        from thunder.core.prims import unpack_trivial, python_return
        import thunder.core.prims as prims
        from thunder.executors.torch_compile import make_compiled as make_torch_compile_callable
        from thunder.core.trace import tracectx

        for bsym in comp_trace.bound_symbols:
            if bsym.sym in (unpack_trivial, python_return):
                continue

            bsym = decompose_into_aten_subsymbols(bsym, comp_trace, ScaleTensor)

        print(comp_trace)
        return prologue_trace, comp_trace, epi_trc


jfn = thunderfx(
    f,
    transforms=[
        ATenTransform(),
    ],
)
o = jfn(x1, x2)

# print(jfn.last_traces[-1])
