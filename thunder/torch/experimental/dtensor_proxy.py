from typing import Optional

from thunder.core.proxies import TensorProxy, AnyProxy, _infer_tensor_properties
from torch.distributed._tensor import DTensor
from thunder.core.proxies import proxy
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes


# Inherit from TensorProxy as DTensor also supports
# Tensor methods like __add__, __div__, sin, etc.
class DTensorProxy(TensorProxy):
    def __init__(
        self,
        name=None,
        *,
        local_tensor_proxy=None,
        spec=None,
        like=None,
        shape=None,
        device=None,
        dtype=None,
        requires_grad=False,
        grad=None,
        prefix=None,
        distparallel_type=None,
        history=None,
        tags=None,
        thunder_fsdp_padding_size=None,
    ):
        super().__init__(
            name,
            like=like,
            shape=shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            grad=grad,
            prefix=prefix,
            distparallel_type=distparallel_type,
            history=history,
            tags=tags,
            thunder_fsdp_padding_size=thunder_fsdp_padding_size,
        )
        if like is not None:
            self._spec = like._spec
            self._local_tensor = like._local_tensor
        else:
            assert isinstance(spec, AnyProxy)
            self._spec = spec
            self._local_tensor = local_tensor_proxy

    def type_string(self):
        return f"DTensor {self.device.device_str()} {self.dtype.shortname()}{list(self._shape)}"

    def replace(self, **changes):
        r"""Return a copy of the TensorProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``shape``, ``dtype``, ``device``, ``requires_grad``, ``distparallel_type``,  ``thunder_fsdp_padding_size``.
        ``like`` is also a valid keyword and will take metadata from the tensor proxy argument
        in preference to the old values but overridable by keyword arguments.
        Note that the copy will use the current (environment) tracectx."""

        like = changes.get("like")
        (
            shape,
            device,
            dtype,
            true_dtype,
            numel,
            ndim,
            requires_grad,
            grad,
            distparallel_type,
            thunder_fsdp_padding_size,
        ) = _infer_tensor_properties(
            like,
            changes.get("shape", self._shape if like is None else None),
            changes.get("device", self._device if like is None else None),
            changes.get("dtype", self._dtype if like is None else None),
            changes.get("requires_grad", self._requires_grad if like is None else None),
            changes.get("grad", self._grad if like is None else None),
            changes.get("distparallel_type", self._distparallel_type if like is None else None),
            changes.get("thunder_fsdp_padding_size", self._thunder_fsdp_padding_size if like is None else None),
        )
        name = changes.get("name", self.name)
        history = changes.get("history", self.history)
        tags = changes.get("tags", self.tags)
        return DTensorProxy(
            name=name,
            local_tensor_proxy=self._local_tensor,
            spec=self._spec,
            tags=tags,
            shape=shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            distparallel_type=distparallel_type,
            thunder_fsdp_padding_size=thunder_fsdp_padding_size,
            history=history,
        )


def proxify_dtensor(x, name: str | None = None, history: None | tuple = None) -> DTensorProxy | None:
    if isinstance(x, DTensor):
        spec_proxy = AnyProxy(x._spec, history=history)
        t = x._local_tensor
        shape = x.shape
        device = devices.to_device(x.device)
        dtype = dtypes.to_dtype(x.dtype)
        grad = None
        distparallel_type = None
        _thunder_fsdp_padding_size = None
        local_tensor_proxy = proxy(t, history=history)
        return DTensorProxy(
            name,
            local_tensor_proxy=local_tensor_proxy,
            spec=spec_proxy,
            shape=tuple(shape),
            device=device,
            dtype=dtype,
            requires_grad=x.requires_grad,
            grad=grad,
            distparallel_type=distparallel_type,
            history=history,
            thunder_fsdp_padding_size=_thunder_fsdp_padding_size,
        )

    return None


def is_dtensor_proxy(x):
    return isinstance(x, DTensorProxy)
