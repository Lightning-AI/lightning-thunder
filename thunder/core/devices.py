# NOTE This delays annotation evaluation, allowing functions in the Device class
#   to be annotated with the Device type.
#   This feature is available in Python 3.7 and later.
#   This import (like all __future__ imports) must be at the beginning of the file.
from __future__ import annotations

from enum import Enum, auto
from collections.abc import Sequence

import torch
import thunder.core.baseutils as baseutils


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()


all_devicetypes = (DeviceType.CPU, DeviceType.CUDA)

_devicetype_prettyprint_map = {
    DeviceType.CPU: "cpu",
    DeviceType.CUDA: "cuda",
}
_inverse_devicetype_prettyprint_map = {v: k for k, v in _devicetype_prettyprint_map.items()}


def devicetype_string(devicetype: DeviceType) -> str:
    return _devicetype_prettyprint_map[devicetype]


def _parse_device_info(
    string_or_devicetype: str | DeviceType, index: None | int = None, /
) -> tuple[DeviceType, None | int]:
    _index = None

    if isinstance(string_or_devicetype, str):
        _devicetype, _index = _device_from_string_helper(string_or_devicetype)
    else:
        baseutils.check_type(string_or_devicetype, DeviceType)
        _devicetype = string_or_devicetype

    baseutils.check(
        index is None or _index is None or index == _index,
        lambda: f"Trying to parse a device but was given two distinct indices, {_index} and {index}",
    )

    if _index is None:
        _index = index

    if _devicetype is DeviceType.CUDA:
        if _index is None:
            _index = 0
    else:
        # _devicetype is DeviceType.CPU
        # NOTE That it's not an error to request cpu:1, but we ignore the index
        #   This is distinct from PyTorch, where `cpu:3` will have index 3, even though
        #   all cpu devices refer to the same physical device
        _index = None

    return _devicetype, _index


# A metaclass that ensures device objects are singletons. When a the Device constructor is called,
#   this may cause the constructor to return an existing object that already represents the device.
class DeviceMeta(type):
    def __call__(cls, *args, **kwargs):
        info = _parse_device_info(*args)
        cur = cls._cache.get(info, None)

        if cur is not None:
            return cur

        slf = cls.__new__(cls, *args, **kwargs)
        cls.__init__(slf, *args, **kwargs)
        cls._cache[info] = slf
        return slf

    def __init__(cls, name, bases, attributes):
        super().__init__(name, bases, attributes)
        cls._cache = {}


class Device(metaclass=DeviceMeta):
    _devicetype: DeviceType
    _index: None | int

    def __init__(self, string_or_devicetype: str | DeviceType, index: None | int = None, /):
        self._devicetype, self._index = _parse_device_info(string_or_devicetype, index)

        if self._devicetype is DeviceType.CUDA:
            baseutils.check_type(self._index, int)
            baseutils.check(self._index >= 0, lambda: f"Trying to create a device with invalid index {index}")
            baseutils.check(
                self._index < torch.cuda.device_count(),
                lambda: f"Trying to create a CUDA device with {index=}, but there are only {torch.cuda.device_count()} CUDA devices available",
            )

    @property
    def devicetype(self) -> DeviceType:
        return self._devicetype

    # Returns a string representation of the devicetype, consistent with PyTorch's device.type
    @property
    def type(self) -> str:
        return devicetype_string(self.devicetype)

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self) -> int:
        return id(self)

    # NOTE This representation is a valid PyTorch device string, which is currently relied upon when
    #   converting Thunder devices to PyTorch devices
    def __repr__(self) -> str:
        if self.devicetype == DeviceType.CPU:
            return devicetype_string(self.devicetype)

        # NOTE self.devicetype == DeviceType.CUDA
        return f"{devicetype_string(self.devicetype)}:{self.index}"

    # NOTE Because devices are singleton object, this has the luxury of using "is"
    def __eq__(self, other: Device) -> bool:
        return self is other


cpu = Device(DeviceType.CPU, None)


# Returns a tuple of available devices
def available_devices() -> tuple[Device]:
    available_devices = [cpu]

    # Short-circuits if there are no CUDA devices
    if not torch.cuda.is_available:
        return available_devices

    # NOTE torch.cuda.is_available, extends with CUDA devices
    cuda_devices = tuple(Device(DeviceType.CUDA, idx) for idx in range(torch.cuda.device_count()))
    available_devices.extend(cuda_devices)

    return tuple(available_devices)


def _device_from_string_helper(devicestr: str) -> tuple[DeviceType, None | int]:
    if devicestr == "cpu":
        return DeviceType.CPU, None

    if devicestr == "cuda":
        return DeviceType.CUDA, None

    devicetype, idx_str = devicestr.split(":")
    idx = int(idx_str)

    return _inverse_devicetype_prettyprint_map[devicetype], idx


# Translates strings of the form "cpu" or "cuda:x" (for a valid integer x) into a Device object
def device_from_string(devicestr: str) -> Device:
    devicetype, deviceno = _device_from_string_helper(devicestr)

    if devicetype is DeviceType.CPU:
        return cpu

    return Device(devicetype, deviceno)


# TODO Maybe allow acquiring a tensor's device this way, too
def to_device(x: None | str | torch.device | Device, /) -> None | Device:
    if x is None or isinstance(x, Device):
        return x

    baseutils.check_type(x, (str, torch.device))
    return device_from_string(str(x))


def to_torch_device(x: None | str | torch.device | Device, /) -> None | torch.device:
    if x is None or isinstance(x, torch.device):
        return x

    baseutils.check_type(x, (Device, str))
    return torch.device(str(x))
