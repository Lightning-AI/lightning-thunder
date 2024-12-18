from itertools import product

import pytest
import torch
import torch.nn as nn

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

import thunder
from thunder.distributed import column_parallel, row_parallel
import thunder.executors
from thunder.tests.distributed.helper import ToyModel, DistributedParallelTestCase
from thunder.tests.distributed.modules import ParallelMLP

from torch.testing._internal import common_utils

_COL = "column"
_ROW = "row"
_name_to_transform = {
    _COL: column_parallel,
    _ROW: row_parallel,
}


class TensorParallelTest(DistributedParallelTestCase):

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name,bias", product(tuple(_name_to_transform.keys()), (True, False)))
    def test_linear(self, name, bias):
        device = torch.device("cuda", self.rank)
        x = torch.randn(2, 12).to(device).requires_grad_()
        x_ref = x.clone().detach().requires_grad_()

        process_group = None
        ref_model = ToyModel(bias).to(device)

        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x_ref)

        transform = _name_to_transform[name]
        model = ToyModel(bias=bias).to(device)
        model.load_state_dict(ref_state_dict)
        jitted_model = thunder.jit(model)
        tp_jitted_model = transform(
            jitted_model,
            target_modules=("net1", "net2"),
            process_group=process_group,
        )
        y = tp_jitted_model(x)
        torch.testing.assert_close(expected=expected, actual=y)

        expected.mean().backward()
        y.mean().backward()

        dim = 1 if name == _ROW else 0
        for layer_name in ("net1", "net2"):
            param_name = f"{layer_name}.weight"
            expected_full_grad: torch.Tensor = ref_model.get_parameter(param_name).grad
            expected = torch.chunk(expected_full_grad, self.world_size, dim)[self.rank]
            torch.testing.assert_close(
                expected=expected,
                actual=tp_jitted_model.get_parameter(param_name).grad,
            )
            if bias:
                param_name = f"{layer_name}.bias"
                expected_bias_grad: torch.Tensor = ref_model.get_parameter(param_name).grad
                if name == _COL:
                    expected = torch.chunk(expected_bias_grad, self.world_size, 0)[self.rank]
                else:
                    expected = expected_bias_grad
                torch.testing.assert_close(
                    expected=expected,
                    actual=tp_jitted_model.get_parameter(param_name).grad,
                )
        torch.testing.assert_close(expected=x_ref.grad, actual=x.grad)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name", tuple(_name_to_transform.keys()))
    def test_embedding(self, name):
        num_embeddings = 128
        embedding_dim = 32

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(num_embeddings, embedding_dim)

            def forward(self, x):
                return self.embed(x)

        device = torch.device(f"cuda:{self.rank}")
        x = torch.randint(0, num_embeddings - 1, (16, 16), device=device)
        x_ref = x.clone().detach()

        process_group = None
        ref_model = Model().to(device)

        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x_ref)

        transform = _name_to_transform[name]
        model = Model().to(device)
        model.load_state_dict(ref_state_dict)
        jitted_model = thunder.jit(model)
        tp_jitted_model = transform(
            jitted_model,
            target_modules=("embed",),
            process_group=process_group,
        )
        y = tp_jitted_model(x)

        dim: int
        orig_size: int
        if name == _COL:
            dim = 0
            orig_size = num_embeddings
        else:
            dim = 1
            orig_size = embedding_dim
        torch.testing.assert_close(
            tp_jitted_model.get_parameter("embed.weight").size(dim),
            orig_size // self.world_size,
        )
        torch.testing.assert_close(expected=expected, actual=y)

        expected.mean().backward()
        y.mean().backward()

        torch.testing.assert_close(
            expected=ref_model.embed.weight.grad.chunk(self.world_size, dim)[self.rank],
            actual=tp_jitted_model.get_parameter("embed.weight").grad,
        )

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("bias", (True, False))
    def test_both_column_and_row(self, bias):
        num_embeddings = 128
        embedding_dim = 32
        n_hidden = 96

        class Model(nn.Module):
            def __init__(self, bias: bool = True):
                super().__init__()
                self.embed_1 = nn.Embedding(num_embeddings, embedding_dim)
                self.embed_2 = nn.Embedding(num_embeddings, embedding_dim)
                self.linear1_0 = nn.Linear(embedding_dim, n_hidden, bias=bias)
                self.linear1_1 = nn.Linear(n_hidden, n_hidden, bias=bias)

            def forward(self, x):
                feat_1 = self.embed_1(x)
                feat_2 = self.embed_2(x)
                sum_of_feat = feat_1 + feat_2
                h = self.linear1_1(torch.relu(self.linear1_0(sum_of_feat)))
                return h

        device = torch.device("cuda", self.rank)
        x = torch.randint(0, num_embeddings - 1, (16, 16), device=device)
        x_ref = x.clone().detach()

        process_group = None
        ref_model = Model(bias=bias).to(device)
        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x_ref)

        model = Model(bias=bias).to(device)
        model.load_state_dict(ref_state_dict)
        tp_model = thunder.jit(model)

        column_parallel_layers = ["embed_1", "linear1_0"]
        tp_model = column_parallel(tp_model, column_parallel_layers, process_group)
        row_parallel_layers = ["embed_2", "linear1_1"]
        tp_model = row_parallel(tp_model, row_parallel_layers, process_group)
        actual = tp_model(x)
        torch.testing.assert_close(actual=actual, expected=expected)

        with torch.no_grad():
            g_ref = torch.rand_like(expected)
            g = g_ref.clone().detach()
        expected.backward(g_ref)
        actual.backward(g)

        for l_name, layer in reversed(list(ref_model.named_modules())):
            dim = int(l_name in row_parallel_layers)
            is_tensor_parallel = l_name in row_parallel_layers or l_name in column_parallel_layers
            prefix = "row-parallel" if dim else "column-parallel"
            for p_name, p_ref in layer.named_parameters(recurse=False):
                param_fqn = f"{l_name}.{p_name}"
                ref_grad = p_ref.grad
                msg = lambda err_msg: f"[{prefix} {param_fqn}] {err_msg}"
                if is_tensor_parallel and (ref_grad.ndim > 1 or dim == 0):
                    ref_grad = ref_grad.chunk(self.world_size, dim)[self.rank]
                grad = tp_model.get_parameter(param_fqn).grad
                torch.testing.assert_close(actual=grad, expected=ref_grad, msg=msg)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("meta_init", (False, True))
    def test_parallel_mlp(self, meta_init):
        from thunder.distributed.prims import PrimIDs

        sequence_length: int = 32
        batch_size: int = 4
        hidden_size: int = 128
        ffn_hidden_size: int = 512
        device = torch.device("cuda", self.rank)

        ref_mlp = ParallelMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size).to(device)
        ref_state_dict = ref_mlp.state_dict()

        # TODO(crcrpar): Support checkpoint load/save
        if meta_init:
            with torch.device("meta"):
                mlp = ParallelMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        else:
            mlp = ParallelMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size).to(device)
            mlp.load_state_dict(ref_state_dict)
        tp_mlp = thunder.jit(mlp)
        tp_mlp = column_parallel(tp_mlp, ParallelMLP.COLUMN_WISE)
        tp_mlp = row_parallel(tp_mlp, ParallelMLP.ROW_WISE)

        # See https://github.com/NVIDIA/NeMo/blob/95ca2f4/nemo/collections/nlp/modules/common/megatron/mlp.py#L221 for the input shape.
        x_ref = torch.randn((sequence_length, batch_size, hidden_size), device=device, requires_grad=True)
        x = x_ref.clone().detach().requires_grad_(True)

        expected = ref_mlp(x_ref)
        actual = tp_mlp(x)

        if not meta_init:
            torch.testing.assert_close(actual=actual, expected=expected)

        grad = torch.rand_like(x_ref)
        expected.backward(grad)
        actual.backward(grad)

        if not meta_init:
            torch.testing.assert_close(actual=x.grad, expected=x_ref.grad)

        tp_syncs = {PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT, PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT}
        fwd_traces_with_tensor_parallel_syncs = list(
            filter(
                lambda trace: any(bsym.sym.id in tp_syncs for bsym in trace.bound_symbols),
                thunder.last_traces(tp_mlp),
            )
        )

        last_fwd_trace_with_tp_sync = fwd_traces_with_tensor_parallel_syncs[-1]
        bsyms_of_tp_sync = tuple(
            filter(lambda bsym: bsym.sym.id in tp_syncs, last_fwd_trace_with_tp_sync.bound_symbols)
        )
        msg = f"{bsyms_of_tp_sync=}"
        # Two bsyms are supposed to be
        # - preprocessing of column-wise parallel linear
        # - postprocessing of row-wise parallel linear
        self.assertEqual(len(bsyms_of_tp_sync), 2, msg=msg)

        state_dict = tp_mlp.original_state_dict()
        ref_state_dict = ref_mlp.state_dict()
        for name in state_dict:
            param = state_dict[name]
            ref_param = ref_state_dict[name]
            self.assertEqual(param.shape, ref_param.shape)

        tp_mlp.load_original_state_dict(ref_state_dict)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    def test_litgpt_causal_self_attention(self):
        from thunder.tests.litgpt_model import Config
        from thunder.tests.litgpt_model import CausalSelfAttention
        from thunder.tests.make_tensor import make_tensor
        from thunder.distributed.prims import PrimIDs

        device = torch.device(f"cuda:{self.rank}")
        dtype = torch.bfloat16

        batch_size: int = 4  # 4 is chosen arbitrarily.
        config_name: str = "Llama-2-13b-hf"
        config = Config.from_name(config_name)

        x_shape = (batch_size, config.block_size, config.n_embd)
        cos_shape = (config.block_size, config.rope_n_elem)
        sin_shape = (config.block_size, config.rope_n_elem)
        mask = None
        input_pos = None

        attention = CausalSelfAttention(config, 0).to(device=device, dtype=dtype)
        # Temporarily use only torchex due to https://github.com/NVIDIA/Fuser/issues/2390
        tp_attention = thunder.jit(attention, executors=[thunder.executors.get_torch_executor()])
        tp_attention = column_parallel(tp_attention, ["attn"])
        tp_attention = row_parallel(tp_attention, ["proj"])

        x = make_tensor(x_shape, device=device, dtype=dtype, requires_grad=True)
        cos = make_tensor(cos_shape, device=device, dtype=dtype, requires_grad=True)
        sin = make_tensor(sin_shape, device=device, dtype=dtype, requires_grad=True)

        # TODO(crcrpar): add numeircal check
        y = tp_attention(x, cos, sin, mask, input_pos)
        tp_syncs = {PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT, PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT}
        fwd_traces_with_tensor_parallel_syncs = list(
            filter(
                lambda trace: any(bsym.sym.id in tp_syncs for bsym in trace.bound_symbols),
                thunder.last_traces(tp_attention),
            )
        )

        last_fwd_trace_with_tp_sync = fwd_traces_with_tensor_parallel_syncs[-1]
        bsyms_of_tp_sync = tuple(
            filter(lambda bsym: bsym.sym.id in tp_syncs, last_fwd_trace_with_tp_sync.bound_symbols)
        )
        msg = f"{bsyms_of_tp_sync=}"
        # TODO(crcrpar): Fix the comm optimization path. Ideally, 2.
        # Though note this class' forward seems to depend on a hyperparam that could be affected by tensor parallel transform.
        # ref: https://github.com/Lightning-AI/litgpt/blob/8ca46d2f/litgpt/model.py#L218
        self.assertEqual(len(bsyms_of_tp_sync), 4, msg=msg)


common_utils.instantiate_parametrized_tests(TensorParallelTest)


if __name__ == "__main__":
    common_utils.run_tests()
