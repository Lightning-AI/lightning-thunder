from itertools import product

import pytest
import torch
import torch.nn as nn

import thunder
from thunder.distributed import column_parallel, row_parallel
import thunder.executors
from thunder.tests.distributed.helper import ToyModel, DataParallelTestCase

from torch.testing._internal import common_utils
from torch.distributed import distributed_c10d as c10d

_COL = "column"
_ROW = "row"
_name_to_transform = {
    _COL: column_parallel,
    _ROW: row_parallel,
}


class TensorParallelTest(DataParallelTestCase):

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name,bias", product(tuple(_name_to_transform.keys()), (True, False)))
    def test_tensor_parallel_linear(self, name, bias):
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
            target_modules=("net2",),
            process_group=process_group,
        )
        y = tp_jitted_model(x)
        torch.testing.assert_close(expected=expected, actual=y)

        expected.mean().backward()
        y.mean().backward()
        torch.testing.assert_close(expected=x_ref.grad, actual=x.grad)

        dim = 1 if name == "row" else 0
        expected_full_grad: torch.Tensor = ref_model.net2.weight.grad
        expected = torch.chunk(expected_full_grad, self.world_size, dim)[self.rank]
        torch.testing.assert_close(
            expected=expected,
            actual=tp_jitted_model.get_parameter("net2.weight").grad,
        )
        if bias:
            expected_bias_grad: torch.Tensor = ref_model.net2.bias.grad
            if name == _COL:
                expected = torch.chunk(expected_bias_grad, self.world_size, 0)[self.rank]
            else:
                expected = expected_bias_grad
            torch.testing.assert_close(
                expected=expected,
                actual=tp_jitted_model.get_parameter("net2.bias").grad,
                msg=f"{expected.shape=}, {tp_jitted_model.get_parameter('net2.bias').grad}",
            )
        torch.testing.assert_close(
            expected=(ref_model.net1.weight.grad,) + (ref_model.net1.bias.grad,) if bias else (),
            actual=(
                (tp_jitted_model.get_parameter("net1.weight").grad,)
                + (tp_jitted_model.get_parameter("net1.bias").grad,)
                if bias
                else ()
            ),
        )

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name", tuple(_name_to_transform.keys()))
    def test_tensor_parallel_embedding(self, name):
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
            tp_jitted_model.get_parameter("embed.weight").size(dim), orig_size // self.world_size
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
    def test_tensor_parallel_both_column_and_row(self, bias):
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
        actual.mean().backward()

        for col, orig_size in zip(column_parallel_layers, [num_embeddings, n_hidden]):
            weight = tp_model.get_parameter(f"{col}.weight")
            self.assertEqual(weight.size(0), orig_size // self.world_size)
        for row, orig_size in zip(row_parallel_layers, [embedding_dim, n_hidden]):
            weight = tp_model.get_parameter(f"{row}.weight")
            self.assertEqual(weight.size(1), orig_size // self.world_size)

        torch.testing.assert_close(actual=actual, expected=expected)
        torch.testing.assert_close(actual=x.grad, expected=x_ref.grad)


common_utils.instantiate_parametrized_tests(TensorParallelTest)


if __name__ == "__main__":
    common_utils.run_tests()
