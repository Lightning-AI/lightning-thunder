import torch


def is_output_differentiable(x):
    # grad_fn is set only if one of the input `requires_grad=True`
    # and the op is differentiable.
    # Example:
    # >>> x = torch.ones(3, requires_grad=True)
    # >>> y = torch.ones(3, requires_grad=False)
    # >>> (x + x).grad_fn  # <AddBackward0 object at 0x7f0502edcf40>
    # >>> (y + y).grad_fn  # None
    # >>> (y + x).grad_fn  # <AddBackward0 object at 0x7f0502e21060>
    # >>> (x < 1).grad_fn  # None (non-differentiable op)
    # Op with differentiable and non-differentiable outputs.
    # >>> torch.topk(x, k=2)
    # torch.return_types.topk(
    # values=tensor([1., 1.], grad_fn=<TopkBackward0>),
    # indices=tensor([0, 1]))
    # >>> torch.topk(torch.ones(3, requires_grad=False), k=2)
    # torch.return_types.topk(
    # values=tensor([1., 1.]),
    # indices=tensor([0, 1]))
    return x.grad_fn is not None or is_returning_self(x)


def is_returning_self(x):
    if x.is_leaf and x.requires_grad:
        return True
    return False


def filter_differentiable_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        # Otherwise `filter` below will
        # iterate over the Tensor data.
        outputs = [outputs]

    return list(filter(is_output_differentiable, outputs))
