import re
from typing import Any
from functools import partial, wraps

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.tests.opinfos import get_opinfo
from thunder.tests.make_tensor import make_tensor
from thunder.core.transforms import grad


einops = pytest.importorskip("einops")


def skipIfNoCUDA(f):
    @wraps(f)
    def wrapped_test(device, *args, **kwargs):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        f(device, *args, **kwargs)

    return wrapped_test


@skipIfNoCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64], ids=("float32", "int64"))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_rearrange(device: str, dtype: torch.dtype):
    # shape, expression, kwargs
    cases = (
        # Test "flatten"-like functionality
        ((2, 3, 4, 5), "b c h w -> b (c h w)", {}),
        ((2, 3, 4), "h w c -> w h c", {}),
        ((2, 3, 4, 5), "b h w c -> (b h) w c", {}),
        ((2, 3, 4, 5), "b h w c -> h (b w) c", {}),
        ((2, 3, 4, 5), "b h w c -> (b h w c)", {}),
        # Test "unflatten"-like functionality
        ((6, 2, 1, 3), "(b1 b2) h w c -> b1 b2 h w c", dict(b1=2)),
        # Test both
        ((6, 2, 4, 6), "(b1 b2) h w c -> (b1 h) (b2 w) c", dict(b1=2)),
        ((6, 2, 4, 6), "(b1 b2) h w c -> (b2 h) (b1 w) c", dict(b1=2)),
        # Width-to-height rearrange
        ((2, 12, 32, 16), "b c h (w w2) -> b c (h w2) w", dict(w2=2)),
        # Dim insertion
        ((2, 3, 4, 5), "b h w c -> b 1 h w 1 c", {}),
    )

    def f(input, expr, **kwargs):
        return einops.rearrange(input, expr, **kwargs)

    fc = thunder.jit(f)

    for shape, expr, kwargs in cases:
        input = make_tensor(shape, dtype=dtype, device=device)

        res_thunder = fc(input, expr, **kwargs)
        res_einops = f(input, expr, **kwargs)

        assert_close(res_thunder, res_einops)


@skipIfNoCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64], ids=("float32", "int64"))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_repeat(device: str, dtype: torch.dtype):
    # shape, expression, kwargs
    cases = (
        ((2, 3, 4), "h w c -> h new_axis w c", dict(new_axis=5)),
        ((2, 3, 4), "h w c -> h 5 w c", {}),
        ((2, 3, 4), "h w c -> h (repeat w) c", dict(repeat=3)),
        ((2, 3, 4), "h w c -> (2 h) (2 w) c", {}),
        ((2, 3, 4), "h w c -> h (w repeat) c", dict(repeat=3)),
    )

    def f(input, expr, **kwargs):
        return einops.repeat(input, expr, **kwargs)

    fc = thunder.jit(f)

    for shape, expr, kwargs in cases:
        input = make_tensor(shape, dtype=dtype, device=device)

        res_thunder = fc(input, expr, **kwargs)
        res_einops = f(input, expr, **kwargs)

        assert_close(res_thunder, res_einops)


@skipIfNoCUDA
@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_reduce(device: str, dtype: torch.dtype):
    # shape, expression, kwargs
    cases = (
        ((2, 3, 4, 5), "b h w c -> h w c", dict(reduction="mean")),
        ((2, 3, 4, 5), "b h w c -> h w", dict(reduction="min")),
        ((2, 6, 6, 6), "b (h h2) (w w2) c -> h (b w) c", dict(reduction="mean", h2=2, w2=2)),
        ((2, 6, 6, 6), "b (h h2) (w w2) c -> h (b w) c", dict(reduction="max", h2=2, w2=2)),
        ((2, 6, 6, 6), "(b1 b2) h w c -> (b2 h) (b1 w)", dict(reduction="mean", b1=2)),
        ((2, 3, 4, 5), "b h w c -> b () () c", dict(reduction="max")),
        ((4, 6, 8, 8), "(b1 b2) h w c -> h (b2 w) c", dict(reduction="max", b1=2)),
        ((4, 6, 8, 8), "b (h 2) (w 2) c -> (c h) (b w)", dict(reduction="mean")),
        (
            (2, 8, 8, 8),
            "(b1 b2) (h1 h2 h3) (w1 w2 w3) c -> (h1 w1 h3) (b1 w2 h2 w3 b2) c",
            dict(reduction="mean", h2=2, w1=2, w3=2, h3=2, b2=2),
        ),
    )

    def f(input, expr, **kwargs):
        return einops.reduce(input, expr, **kwargs)

    # TODO(#1993): don't enforce `nv_enable_bookend` when #1993 is resolved.
    fc = thunder.jit(f, nv_enable_bookend=True)

    for shape, expr, kwargs in cases:
        input = make_tensor(shape, dtype=dtype, device=device)

        res_thunder = fc(input, expr, **kwargs)
        res_einops = f(input, expr, **kwargs)

        assert_close(res_thunder, res_einops)


@skipIfNoCUDA
@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_einsum(device: str, dtype: torch.dtype):
    op = get_opinfo("einsum")

    def f(expr, *operands):
        return einops.einsum(*operands, expr)

    def torch_f(expr, *operands):
        return torch.einsum(expr, *operands)

    fc = thunder.jit(f)
    torch_fc = thunder.jit(torch_f)

    for sample in op.sample_inputs(device, dtype, requires_grad=False):
        expr, *operands = sample.args

        # Einops pattern requires '->' and that each dim is space-separated.
        # Remove spaces
        expr = re.sub(r"\s", "", expr)
        # Replace '...' with '.' for easier string transform.
        expr = re.sub(r"\.\.\.", ".", expr)

        rhs, *lhs = expr.split("->")
        # Insert '->' is required by Einops,
        # and split dims with spaces.
        expr = " ".join(rhs) + " -> "
        if lhs:
            # If needed, space-separate dims in the output subexpression.
            expr = expr + " ".join(*lhs)
        # replace '.' with '...'.
        expr = re.sub(r"\.", "...", expr)

        einops_res = f(expr, *operands)
        einops_compiled_res = fc(expr, *operands)
        assert_close(einops_res, einops_compiled_res, atol=1e-4, rtol=1e-4)

        torch_compiled_res = torch_fc(expr, *operands)
        assert_close(einops_res, torch_compiled_res, atol=1e-4, rtol=1e-4)
