import torch

import thunder
from thunder.tests.framework import requiresCUDA


@requiresCUDA
def test_warmup_runs_with_correct_buffers():
    """
    Tests whether newly-created buffers are being properly initialized.
    Otherwise we should expect failures because of incorrect values.
    """

    weights = torch.tensor([0, 10, 3, 0], device="cuda", dtype=torch.float)

    def f(x):
        return torch.multinomial(x, num_samples=3, replacement=True)

    jf = thunder.jit(f, use_cudagraphs=True)
    jf(weights)
    jf(weights)
