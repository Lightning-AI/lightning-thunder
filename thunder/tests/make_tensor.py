import collections.abc
import math
from typing import cast, List, Optional, Tuple, Union

import torch

# adapted from https://github.com/pytorch/pytorch/blob/master/torch/testing/_creation.py
# Changes:
#   bool generation respects low and high
#   when low and high would be a floating point type, and the requested type is integral, the
#       ceil of both low and high is taken, instead of the floor and ceil (this ensures a valid range)
# NOTE: these changes may not be reflected in the documentation

# Used by make_tensor for generating complex tensor.
complex_to_corresponding_float_type_map = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
float_to_corresponding_complex_type_map = {v: k for k, v in complex_to_corresponding_float_type_map.items()}


def _uniform_random(t: torch.Tensor, low: float, high: float):
    # uniform_ requires to-from <= std::numeric_limits<scalar_t>::max()
    # Work around this by scaling the range before and after the PRNG
    if high - low >= torch.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)


def make_tensor(
    *shape: int | torch.Size | list[int] | tuple[int, ...],
    dtype: torch.dtype,
    device: str | torch.device,
    low: float | None = None,
    high: float | None = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
    memory_format: torch.memory_format | None = None,
) -> torch.Tensor:
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.
    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
    finite values then they are clamped to the lowest or highest representable finite value, respectively.
    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
    which depend on :attr:`dtype`.
    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | signed integral types     | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    Args:
        shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.
        dtype (:class:`torch.dtype`): The data type of the returned tensor.
        device (Union[str, torch.device]): The device of the returned tensor.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is
            clamped to the least representable finite value of the given dtype. When ``None`` (default),
            this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is
            clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value
            is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
        noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is
            ignored if the constructed tensor has fewer than two elements.
        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
            point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
            :attr:`dtype`'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number
            whose real and imaginary parts are both the smallest positive normal number representable by the complex
            type. Default ``False``.
        memory_format (Optional[torch.memory_format]): The memory format of the returned tensor.  Incompatible
            with :attr:`noncontiguous`.
    Raises:
        ValueError: if ``requires_grad=True`` is passed for integral `dtype`
        ValueError: If ``low > high``.
        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
        TypeError: If :attr:`dtype` isn't supported by this function.
    """

    def _modify_low_high(low, high, lowest, highest, default_low, default_high, dtype):
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high) if required.
        """

        def clamp(a, l, h):
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Checks for error cases
        if low != low or high != high:
            raise ValueError("make_tensor: one of low or high was NaN!")
        if low > high:
            raise ValueError("make_tensor: low must be weakly less than high!")

        low = clamp(low, lowest, highest)
        high = clamp(high, lowest, highest)

        if dtype in [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            return math.ceil(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]  # type: ignore[assignment]
    shape = cast(tuple[int, ...], tuple(shape))

    _integral_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    _floating_8bit_types = [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]
    _floating_types = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    _complex_types = [torch.complex32, torch.complex64, torch.complex128]
    if (
        requires_grad
        and dtype not in _floating_types
        and dtype not in _floating_8bit_types
        and dtype not in _complex_types
    ):
        raise ValueError("make_tensor: requires_grad must be False for integral dtype")

    if dtype is torch.bool:
        low, high = cast(
            tuple[int, int],
            _modify_low_high(low, high, 0, 2, 0, 2, dtype),
        )
        if low == high:
            return torch.full(shape, low, device=device, dtype=dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype is torch.uint8:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = cast(
            tuple[int, int],
            _modify_low_high(low, high, ranges[0], ranges[1], 0, 10, dtype),
        )
        if low == high:
            return torch.full(shape, low, device=device, dtype=dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype in _integral_types:
        ranges = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
        low, high = _modify_low_high(low, high, ranges[0], ranges[1], -9, 10, dtype)
        if low == high:
            return torch.full(shape, low, device=device, dtype=dtype)
        result = torch.randint(low, high, shape, device=device, dtype=dtype)  # type: ignore[call-overload]
    elif dtype in _floating_types + _floating_8bit_types:
        ranges_floats = (torch.finfo(dtype).min, torch.finfo(dtype).max)
        m_low, m_high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        result = torch.empty(shape, device=device, dtype=dtype if dtype not in _floating_8bit_types else torch.float32)
        _uniform_random(result, m_low, m_high)
    elif dtype in _complex_types:
        float_dtype = complex_to_corresponding_float_type_map[dtype]
        ranges_floats = (torch.finfo(float_dtype).min, torch.finfo(float_dtype).max)
        m_low, m_high = _modify_low_high(low, high, ranges_floats[0], ranges_floats[1], -9, 9, dtype)
        result = torch.empty(shape, device=device, dtype=dtype)
        result_real = torch.view_as_real(result)
        _uniform_random(result_real, m_low, m_high)
    else:
        raise TypeError(
            f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor()."
            " To request support, file an issue at: https://github.com/pytorch/pytorch/issues"
        )

    assert not (noncontiguous and memory_format is not None)
    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]
    elif memory_format is not None:
        result = result.clone(memory_format=memory_format)

    if exclude_zero:
        if dtype in _integral_types or dtype is torch.bool:
            replace_with = torch.tensor(1, device=device, dtype=dtype)
        elif dtype in _floating_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        elif dtype in _floating_8bit_types:
            replace_with = torch.tensor(torch.finfo(dtype).tiny, device=device, dtype=torch.float32)
        else:  # dtype in _complex_types:
            float_dtype = complex_to_corresponding_float_type_map[dtype]
            float_eps = torch.tensor(torch.finfo(float_dtype).tiny, device=device, dtype=float_dtype)
            replace_with = torch.complex(float_eps, float_eps)
        result[result == 0] = replace_with

    if dtype in _floating_types + _complex_types:
        result.requires_grad = requires_grad

    # NOTE This is a workaround. There are so many not supported operations that,
    # even creating the test tensors is hard.
    if dtype in _floating_8bit_types:
        result = result.to(dtype)

    return result


def make_tensor_like(a, **kwargs):
    # type: (torch.Tensor) -> torch.Tensor
    """Returns a tensor with the same properties as the given tensor.

    Args:
        a (torch.Tensor): The tensor to copy properties from.
        kwargs (dict): Additional properties for `make_tensor`.

    Returns:
        torch.Tensor: A tensor with the same properties as :attr:`a`.
    """
    kwargs = kwargs | dict(device=a.device, dtype=a.dtype, requires_grad=a.requires_grad)
    return make_tensor(a.shape, **kwargs)
