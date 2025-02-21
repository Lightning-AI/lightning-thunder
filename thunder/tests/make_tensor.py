import torch
import thunder


def make_tensor(
    *shape: int | torch.Size | list[int] | tuple[int, ...],
    dtype: torch.dtype,
    device: str | torch.device | thunder.devices.Device,
    low: float | None = None,
    high: float | None = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
    memory_format: torch.memory_format | None = None,
) -> torch.Tensor:
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.
    Calls torch.testing.make_tensor and optionally torch.Tensor.fill_ to allow for low == high, as
    torch.testing.make_tensor enforces low < high.

    Args:
        shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.
        dtype (:class:`torch.dtype`): The data type of the returned tensor.
        device (Union[str, torch.device, thunder.devices.Device]): The device of the returned tensor.
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
    if isinstance(device, thunder.devices.Device):
        device = device.device_str()

    fill_value = None
    if low is not None and low == high:
        fill_value = low
        low = None
        high = None

    t = torch.testing.make_tensor(
        *shape,
        dtype=dtype,
        device=device,
        low=low,
        high=high,
        requires_grad=requires_grad,
        noncontiguous=noncontiguous,
        exclude_zero=exclude_zero,
        memory_format=memory_format,
    )

    if fill_value is not None:
        t.fill_(fill_value)

    return t


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
