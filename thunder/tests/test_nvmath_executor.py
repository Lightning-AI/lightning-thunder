from torch.testing import assert_close

from thunder.core import dtypes, prims
from thunder.core.devices import DeviceType
from thunder.executors.nvmathex import nvmath_matmul_ex
from thunder.tests.framework import ops, TestExecutor
from thunder.tests.opinfos import opinfos


class nvMathTestExecutor(TestExecutor):
    name = "nvmath"
    supported_devicetypes = (DeviceType.CUDA,)
    supported_dtypes = (*dtypes.all_dtypes,)

    def executors_list(self):
        return [nvmath_matmul_ex]


@ops((op for op in opinfos if op.name in ("matmul", "linear")), supported_executors=(nvMathTestExecutor(),))
def test(op, device, dtype, executor, comparator):
    for sample in op.sample_inputs(device, dtype, requires_grad=False):
        # prims do not support broadcasting
        if sample.args[0].shape[:-2] != sample.args[1].shape[:-2]:
            continue
        # ops with bias argument are not expected to be transformed
        if len(sample.args) == 3:
            continue
        # TODO: OperatorExecutor doesn't decompose to the supported level.
        # I need to create a bug report for this.
        prim_op = {"matmul": prims.matmul, "linear": prims.linear}[op.name]
        compiled_op = executor.make_callable(prim_op)
        expected = op.torch_reference(*sample.args, **sample.kwargs)
        args = sample.args
        if op.name == "linear" and len(sample.args) == 2:  # Since prims.linear requires 3 arguments
            args = [*sample.args, None]
        actual = compiled_op(*args, **sample.kwargs)
        execution_trace = compiled_op._lc_cs.last_traces[-1]
        expected_ops = ["nvmath_matmul", "nvmath_linear"]
        assert any(bsym.sym.name in expected_ops for bsym in execution_trace.bound_symbols)
        assert_close(actual, expected)
