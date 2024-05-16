from contextlib import contextmanager
from typing import Any

import torch as pytorch

from thunder.core.compile_data import get_compile_data


class ThunderModule(pytorch.nn.Module):
    """A wrapper nn.Module subclass.

    This wrapper is returned by ``thunder.jit``, you would typically not
    instantiate it manually.

    """

    def __init__(self, model, compiled_model_call):
        """"""
        super().__init__()
        self._model = model

        # We delete self.training in order for training to be used from
        # the model itself through `__getattr__`.
        del self.training

        self._forward_fn = compiled_model_call

        # overrides for parameters and buffers (see get_buffer/get_parameter)
        self._overrides = {}

        self._null = object()

    def get_buffer(self, name):
        p = self._overrides.get(name, self._null)
        if p is not self._null:
            return p
        return self._model.get_buffer(name)

    def get_parameter(self, name):
        p = self._overrides.get(name, self._null)
        if p is not self._null:
            return p
        return self._model.get_parameter(name)

    def get_submodule(self, name):
        return self._model.get_submodule(name)

    def forward(self, *args, **kwargs):
        res = self._forward_fn(*args, **kwargs)
        return res

    @contextmanager
    def no_sync(self):
        r"""Context manager to disable gradient synchronization in data parallel mode.

        This context manager is intended to be used in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel` to disable gradient
        synchronization in the backward pass. It will not have any effect when
        used with other modules.

        .. note::

            This could lead to different accumulated gradients with ``torch.nn.parallel.distributed.DistributedDataParallel.no_sync``.
            PyTorch's gradient synchronization is implemented by applying all-reduce to gradient buckets of ``torch.nn.Parameter.grad``.
            Thus the ``no_sync`` context leads to :math:`\text{AllReduce} \left( \sum_{i = 0}^{\text{ga_steps}} g_i \right)` where :math:`\text{ga_steps}` means the number of gradient accumulation steps.
            In contrast, this synchronizes accumulated gradients when exiting, leading to
            :math:`\text{AllReduce} \left( \sum_{i = 0}^{\text{ga_steps - 1}} g_i \right) + \text{AllReduce}(g_{\text{ga_steps}})`.

        .. warning::

            You must reuse this context manager in each group of gradient accumulation iterations since gradients will get synchronized
            on context manager exit.

            .. code-block:: python

                with model.no_sync():
                    for _ in range(len(gradient_accumulation_iters)):
                        loss(model(x)).backward()  # uses no-sync-backward trace
                loss(model(x)).backward()  # uses the regular backward trace
                optimizer.step()

        """
        from thunder.distributed import (
            set_skip_data_parallel_grad_sync,
            reset_skip_data_parallel_grad_sync,
            _sync_grads,
        )

        token = set_skip_data_parallel_grad_sync(True)
        try:
            yield
        finally:
            reset_skip_data_parallel_grad_sync(token)
            _sync_grads(self)

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            return self._modules["_model"]
        return getattr(self._model, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.load_state_dict(*args, **kwargs)


def get_thunder_module(model):
    cd = get_compile_data()

    # to not hold a reference to model in the _thunder_module_map dict, we index by id.
    # But this means that we need to check if it is actually the right model, which we do with the if below.
    tm = cd._thunder_module_map.get(id(model))
    if tm and tm._model is not model:
        tm = None

    if tm is None:
        # TODO: we would like to raise an error here, but this would require
        #       us wrapping models that are passed in closures etc.
        # raise RuntimeError("could not find ThunderModule")
        return model
    return tm
