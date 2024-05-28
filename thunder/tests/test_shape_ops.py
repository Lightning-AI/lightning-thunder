import thunder
import torch
from thunder.tests.framework import JAX_AVAILABLE

if JAX_AVAILABLE:
    pass


def test_pad_cast_value_itof():
    """
    Pad should cast the given value to the type of tensor and pad that value.
    """

    def fqn():
        x = torch.tensor([2, 3], dtype=torch.int32)
        y = torch.nn.functional.pad(x, pad=(1, 2), value=6.4)
        return y

    th_fqn = thunder.jit(fqn)
    v = th_fqn()
    assert v[0] == 6
    assert v[1] == 2
    assert v[2] == 3
    assert v[3] == 6
    assert v[4] == 6


def test_pad_cast_value_ftoi():
    """
    Pad should cast the given value to the type of tensor and pad that value.
    """

    def fqn():
        x = torch.tensor([2.4, 3.8])
        y = torch.nn.functional.pad(x, pad=(1, 2), value=1)
        return y

    th_fqn = thunder.jit(fqn)
    v = th_fqn()
    assert v[0] == 1.0
    assert v[1] == 2.4
    assert v[2] == 3.8
    assert v[3] == 1.0
    assert v[4] == 1.0
