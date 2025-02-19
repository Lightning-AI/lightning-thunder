from torch.testing import make_tensor


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
