import pytest
import torch
import torch.nn as nn

import thunder
from thunder.distributed import column_parallel, row_parallel
from thunder.tests.distributed.helper import ToyModel, DataParallelTestCase

from torch.testing._internal import common_utils
from torch.distributed import distributed_c10d as c10d


class TensorParallelTest(DataParallelTestCase):

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name", ("column", "row"))
    def test_tensor_parallel_linear(self, name):
        device = torch.device("cuda", self.rank)
        x = torch.randn(2, 12).to(device).requires_grad_()

        process_group = None
        ref_model = ToyModel().to(device)
        with c10d._coalescing_manager(async_ops=True) as cm:
            c10d.all_reduce(x)
            for p in ref_model.parameters():
                c10d.all_reduce(p)
        cm.wait()

        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x)

        transform = {
            "column": column_parallel,
            "row": row_parallel,
        }[name]
        model = ToyModel().to(device)
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

        dim = 1 if name == "row" else 0
        expected_full_grad: torch.Tensor = ref_model.net2.weight.grad
        expected = torch.chunk(expected_full_grad, self.world_size, dim)[self.rank]
        torch.testing.assert_close(
            expected=expected,
            actual=tp_jitted_model.get_parameter("net2.weight").grad,
        )

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    @common_utils.parametrize("name", ("column", "row"))
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

        process_group = None
        ref_model = Model().to(device)
        with c10d._coalescing_manager(async_ops=True) as cm:
            for p in ref_model.parameters():
                c10d.all_reduce(p)
        cm.wait()

        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x)

        transform = {
            "column": column_parallel,
            "row": row_parallel,
        }[name]
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
        if name == "column":
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

    # TODO(crcrpar): Activate numerical check
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="")
    def test_tensor_parallel_both_column_and_row(self):
        num_embeddings = 128
        embedding_dim = 32
        n_hidden = 96
        n_out = 16

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_1 = nn.Embedding(num_embeddings, embedding_dim)
                self.embed_2 = nn.Embedding(num_embeddings, embedding_dim)
                self.linear1_0 = nn.Linear(embedding_dim, n_hidden)
                self.linear1_1 = nn.Linear(n_hidden, n_hidden)
                self.linear2_0 = nn.Linear(n_hidden, n_hidden)
                self.linear2_1 = nn.Linear(n_hidden, n_out)

            def forward(self, x):
                feat_1 = self.embed_1(x)
                feat_2 = self.embed_2(x)
                sum_of_feat = feat_1 + feat_2
                h = self.linear1_1(self.linear1_0(sum_of_feat))
                return self.linear2_1(self.linear2_0(h))

        device = torch.device("cuda", self.rank)
        x = torch.randint(0, num_embeddings - 1, (16, 16), device=device)

        process_group = None
        ref_model = Model().to(device)
        ref_state_dict = ref_model.state_dict()
        expected = ref_model(x)

        model = Model().to(device)
        model.load_state_dict(ref_state_dict)
        jitted_model = thunder.jit(model)

        column_parallels = ["embed_1", "linear1_0", "linear2_1"]
        row_parallels = ["embed_2", "linear1_1", "linear2_0"]
        tp_jitted_model = row_parallel(
            column_parallel(
                jitted_model,
                column_parallels,
                process_group,
            ),
            row_parallels,
            process_group,
        )
        actual = tp_jitted_model(x)
        actual.mean().backward()

        for col, orig_size in zip(column_parallels, [num_embeddings, n_hidden, n_out]):
            weight = tp_jitted_model.get_parameter(f"{col}.weight")
            self.assertEqual(weight.size(0), orig_size // self.world_size)
        for row, orig_size in zip(row_parallels, [embedding_dim, n_hidden, n_hidden]):
            weight = tp_jitted_model.get_parameter(f"{row}.weight")
            self.assertEqual(weight.size(1), orig_size // self.world_size)


common_utils.instantiate_parametrized_tests(TensorParallelTest)


if __name__ == "__main__":
    common_utils.run_tests()
