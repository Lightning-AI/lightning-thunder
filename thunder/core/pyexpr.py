from functools import partial
from typing import Any, Sequence

from thunder.core.trace import Variable
from thunder.core.pytree import tree_map, tree_flatten, tree_flatten_only, tree_unflatten
import thunder.core.dtypes as dtypes


_dtype_str_map = {
    dtypes.bool8: "b8",
    dtypes.bool8_: "b8",
    dtypes.uint8: "u8",
    dtypes.uint8_: "u8",
    dtypes.int8: "i8",
    dtypes.int8_: "i8",
    dtypes.int16: "i16",
    dtypes.int16_: "i16",
    dtypes.int32: "i32",
    dtypes.int32_: "i32",
    dtypes.int64: "i64",
    dtypes.int64_: "i64",
    dtypes.bfloat16: "bf16",
    dtypes.bfloat16_: "bf16",
    dtypes.float16: "f16",
    dtypes.float16_: "f16",
    dtypes.float32: "f32",
    dtypes.float32_: "f32",
    dtypes.float64: "f64",
    dtypes.float64_: "f64",
    dtypes.complex32: "c32",
    dtypes.complex32_: "c32",
    dtypes.complex64: "c64",
    dtypes.complex64_: "c64",
    dtypes.complex128: "c128",
    dtypes.complex128_: "c128",
}


class UnquotedStr:
    def __init__(self, s):
        self.str = s

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.str


class Expr:
    def __init__(self):
        self.fn_name_counter = 0

    def _get_dtype_str(self, dtype):
        return _dtype_str_map[dtype]

    def _build_arg_str(self, arg: Any, annotate=False):
        from thunder.core.proxies import NumberProxy, TensorProxy

        if isinstance(arg, Variable):
            name = arg.name[1:] if arg.name.startswith("__") else arg.name  # remove leading _

            if isinstance(arg.proxy, TensorProxy):
                if not annotate:
                    return UnquotedStr(name)

                dtype = arg.proxy.dtype
                shape = arg.proxy.shape
                dtype_str = self._get_dtype_str(dtype)
                shape_str = ",".join([str(el) for el in shape])
                return UnquotedStr(f'{name}: "{dtype_str}[{shape_str}]"')

            if isinstance(arg.proxy, NumberProxy):
                if not annotate:
                    return UnquotedStr(name)

                return UnquotedStr(f"{name}: {arg.proxy.value}")

            raise ValueError("Variable doesn't hold a proxy")

        if isinstance(arg, str):
            return UnquotedStr(repr(arg))

        if isinstance(arg, type):
            return UnquotedStr(f'"{arg.__name__}"')

        return UnquotedStr(repr(arg))

    def _new_fn_name(self):
        self.fn_name_counter += 1
        return f"fn{self.fn_name_counter}"


class PyExpr(Expr):
    def __init__(self, trace, annotate=True):
        """A pyexpr is a representation of a trace similar to a Jaxpr in spirit
        (it's an ANF representation of the program, https://en.wikipedia.org/wiki/A-normal_form)
        but it's valid Python, which means it can be invoked.

        Program transforms are preserved as calls to nested functions.

        Args:

            trace: the input trace to create the pyexpr from

            annotate: generate type annotations in the pyexpr
        """
        super().__init__()
        self.trace = trace
        self.annotate = annotate

    def _build_fn_kwargs_str(self, kwargs, annotate):
        kwarg_strs = tree_map(partial(self._build_arg_str, annotate=annotate), kwargs or {})
        kwarg_isvar = tree_map(lambda x: isinstance(x, Variable), kwargs or {})
        kwarg_strs = [
            f"{v}=None" if isvar else f"{k}={v}" for (isvar, (k, v)) in zip(kwarg_isvar.values(), kwarg_strs.items())
        ]
        return ", ".join(kwarg_strs)

    def _build_args_kwargs_str(self, args, kwargs, annotate):
        arg_strs = tree_map(partial(self._build_arg_str, annotate=annotate), args or [])
        kwarg_strs = tree_map(partial(self._build_arg_str, annotate=annotate), kwargs or {})
        arg_kwarg_strs = [str(el) for el in arg_strs] + [f"{k}={v}" for k, v in kwarg_strs.items()]
        return ", ".join(arg_kwarg_strs)

    def _build_sym_strs(self, sym):
        import thunder.core.transforms as transforms

        if isinstance(sym.op, transforms.Transforms):
            trace = sym.kwargs.pop("trace")
            lines = []

            fn_name = self._new_fn_name()
            self._build_pyexpr_r(trace, lines, fn_name, 0)

            args_kwargs_str = self._build_args_kwargs_str(sym.args, sym.kwargs, annotate=False)
            args_kwargs_str = f"{fn_name}, {args_kwargs_str}"
            out_strs = tree_map(partial(self._build_arg_str, annotate=self.annotate), sym.outputs or [])

            if len(out_strs) > 1:
                if self.annotate:
                    for out_str in out_strs:
                        lines.append(out_str.str)
                outs_str = ", ".join(self._arg_names(out_strs))
            else:
                outs_str = out_strs[0].str

            lines.append(f"{outs_str} = {sym.name}({args_kwargs_str})")

            sym.kwargs["trace"] = trace
            return lines

        args_kwargs_str = self._build_args_kwargs_str(sym.args, sym.kwargs, annotate=False)
        out_strs = tree_map(partial(self._build_arg_str, annotate=self.annotate), sym.outputs or [])
        outs_str = ", ".join(el.str for el in out_strs)

        return [f"{outs_str} = {sym.name}({args_kwargs_str})"]

    def _append_pyexpr_line(self, pyexpr_lines, line, level, inner_level):
        indent_spaces = 4
        base_indent = " " * indent_spaces * level
        inner_indent = " " * indent_spaces * inner_level
        pyexpr_lines.append(base_indent + inner_indent + line)

    def _append_blank_line(self, pyexpr_lines):
        self._append_pyexpr_line(pyexpr_lines, "", 0, 0)

    def _arg_names(self, arg_strs):
        return [el.str.split(":")[0].strip() for el in arg_strs]

    def _build_pyexpr_r(self, trace, pyexpr_lines, fn_name, level):
        fn_start = f"def {fn_name}("
        fn_end = "):"

        self._append_pyexpr_line(pyexpr_lines, fn_start, level, 0)

        # NOTE: we assume all args are variables here, while we filter kwargs
        flat_args, _ = tree_flatten(trace.args)
        flat_kwargs, _ = tree_flatten_only(trace.kwargs, lambda x: isinstance(x, Variable))

        var_args = flat_args + flat_kwargs
        flat_nonvar_kwargs, nonvar_spec = tree_flatten_only(trace.kwargs, lambda x: not isinstance(x, Variable))

        kwargs = tree_unflatten(flat_nonvar_kwargs, nonvar_spec)

        for arg in var_args:
            arg_str = self._build_arg_str(arg, annotate=self.annotate)
            line = f"{arg_str.str},"
            self._append_pyexpr_line(pyexpr_lines, line, level, 1)

        if kwargs:
            for k, v in kwargs.items():
                v_str = self._build_arg_str(v, annotate=self.annotate)
                line = f"{k}={v_str.str},"
                self._append_pyexpr_line(pyexpr_lines, line, level, 1)

        self._append_pyexpr_line(pyexpr_lines, fn_end, level, 0)

        for sym in trace.symbols:
            sym_strs = self._build_sym_strs(sym)
            for sym_str in sym_strs:
                self._append_pyexpr_line(pyexpr_lines, sym_str, level, 1)

        outputs = trace.outputs
        if not isinstance(outputs, Sequence):
            outputs = [outputs]

        output_strs = tree_map(partial(self._build_arg_str, annotate=False), outputs or [])
        outputs_str = ", ".join(el.str for el in output_strs)

        ret_str = f"return {outputs_str}"

        self._append_pyexpr_line(pyexpr_lines, ret_str, level, 1)
        self._append_blank_line(pyexpr_lines)

    def _build_pyexpr_str(self):
        pyexpr_lines = []
        fn_name = self._new_fn_name()
        self._build_pyexpr_r(self.trace, pyexpr_lines, fn_name, 0)
        pyexpr_str = "\n".join(pyexpr_lines)
        return pyexpr_str

    def __repr__(self):
        return self._build_pyexpr_str()


def print_pyexpr(trace):
    pyexpr = PyExpr(trace)
    print(pyexpr)
