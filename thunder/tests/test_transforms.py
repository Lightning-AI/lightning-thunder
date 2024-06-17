import thunder
from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform, nvtx_push, nvtx_pop

import torch


def test_nvtx_transform():
    DIM = 2

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(DIM, DIM)
            self.fc2 = torch.nn.Linear(DIM, DIM)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc2(x)
            return x

    model = Model()
    x = torch.randn(4, DIM)

    # Transform for profiling with NVTX markers.
    # This transform adds NVTX markers to all the computation symbols in the execution trace.

    nvtx_profile_transform = NvtxProfileTransform()

    jmodel = thunder.jit(
        model,
        post_optimization_transforms=[
            nvtx_profile_transform,
        ],
    )
    o = jmodel(x)
    o.sum().backward()

    fwd_trc = thunder.last_traces(jmodel)[-1]
    bwd_trc = thunder.last_backward_traces(jmodel)[-1]

    def _test_equal_nvtx_push_and_pop(trc):
        # This test verifies that we see equal number of `nvtx_push` and `nvtx_pop` symbols
        # and also order is `nvtx_push` followed by `nvtx_pop`

        # First NVTX symbol will always be push.
        EXPECT_PUSH = True
        push_cnt = 0
        pop_cnt = 0
        for bsym in trc.bound_symbols:
            if bsym.sym.id in (nvtx_push.id, nvtx_pop.id):
                if EXPECT_PUSH:
                    assert bsym.sym.id == nvtx_push.id
                    push_cnt += 1
                    # After `nvtx_push` we expect next the `nvtx` symbol to be `nvtx_pop`
                    EXPECT_PUSH = False
                else:
                    assert bsym.sym.id == nvtx_pop.id
                    pop_cnt += 1
                    # After `nvtx_pop` we expect next the `nvtx` symbol to be `nvtx_push`
                    EXPECT_PUSH = True

        assert push_cnt == pop_cnt

    _test_equal_nvtx_push_and_pop(fwd_trc)
    _test_equal_nvtx_push_and_pop(bwd_trc)
