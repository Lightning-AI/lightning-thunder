# NOTE This delays annotation evaluation, allowing functions in the Device class
#   to be annotated with the Device type.
#   This feature is available in Python 3.7 and later.
#   This import (like all __future__ imports) must be at the beginning of the file.
from __future__ import annotations

from enum import Enum, auto
from numbers import Number
from typing import Optional, Tuple, Union
from collections.abc import Sequence

import thunder.core.baseutils as baseutils
from thunder.core.langctx import get_default_langctx


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()


all_devicetypes = (DeviceType.CPU, DeviceType.CUDA)

_devicetype_prettyprint_map = {
    DeviceType.CPU: "cpu",
    DeviceType.CUDA: "cuda",
}


def devicetype_string(devicetype: DeviceType) -> str:
    return _devicetype_prettyprint_map[devicetype]


# A metaclass that ensures device objects are singletons. When a the Device constructor is called,
#   this may cause the constructor to return an existing object that already represents the device.
class DeviceMeta(type):
    def __call__(cls, *args, **kwargs):
        cur = cls._cache.get(args, None)

        if cur is not None:
            return cur

        slf = cls.__new__(cls, *args, **kwargs)
        cls.__init__(slf, *args, **kwargs)
        cls._cache[args] = slf
        return slf

    def __init__(cls, name, bases, attributes):
        super().__init__(name, bases, attributes)
        cls._cache = {}


class Device(metaclass=DeviceMeta):
    _devicetype: DeviceType
    _number: int

    def __init__(self, string_or_devicetype: str | DeviceType, number: None | int = None, /):
        _number = None

        if isinstance(string_or_devicetype, str):
            self._devicetype, _number = _device_from_string_helper(string_or_devicetype)
        else:
            baseutils.check_type(string_or_devicetype, DeviceType)
            self._devicetype = string_or_devicetype

        baseutils.check(
            number is None or _number is None or number == _number,
            lambda: f"Trying to create a device but the device has two numbers, {_number} and {number}",
        )

        if _number is None:
            _number = number

        if _number is None:
            _number = 0

        self._number = _number

        # NOTE While we don't consider it an error to request CPU:1, we also don't pretend
        #   like there are multiple CPU devices.
        if self._devicetype is DeviceType.CPU:
            self._number = 0

        baseutils.check_type(self._number, int)
        baseutils.check(self._number >= 0, lambda: f"Trying to create a device with invalid number {number}")

    @property
    def devicetype(self) -> DeviceType:
        return self._devicetype

    # Returns a string representation of the devicetype, consistent with PyTorch's device.type
    @property
    def type(self) -> str:
        return devicetype_string(self.devicetype)

    @property
    def number(self) -> int:
        return self._number

    def __hash__(self) -> int:
        return id(self)

    # NOTE This representation is a valid PyTorch device string, which is currently relied upon when
    #   converting Thunder devices to PyTorch devices
    def __repr__(self) -> str:
        if self.devicetype == DeviceType.CPU:
            return devicetype_string(self.devicetype)

        # NOTE self.devicetype == DeviceType.CUDA
        return f"{devicetype_string(self.devicetype)}:{self.number}"

    # NOTE Because devices are singleton object, this has the luxury of using "is"
    def __eq__(self, other: Device) -> bool:
        return self is other


def available_devices() -> Sequence[Device]:
    return get_default_langctx().available_devices()


cpu = Device(DeviceType.CPU, 0)


def _device_from_string_helper(devicestr: str) -> tuple[DeviceType, None | int]:
    if devicestr == "cpu":
        return DeviceType.CPU, None

    if devicestr == "cuda":
        return DeviceType.CUDA, None

    devicetype, deviceno = devicestr.split(":")
    deviceno = int(deviceno)

    baseutils.check(devicetype == "cuda", lambda: f"Unknown devicetype {devicetype}")

    return DeviceType.CUDA, deviceno


# Translates strings of the form "cpu" or "cuda:x" (for a valid integer x) into a Device object
def device_from_string(devicestr: str) -> Device:
    devicetype, deviceno = _device_from_string_helper(devicestr)

    if devicetype is DeviceType.CPU:
        return cpu

    return Device(devicetype, deviceno)


# TODO Maybe allow acquiring a tensor's device this way, too
def to_device(device_or_string: Device | str) -> Device:
    baseutils.check_type(device_or_string, (Device, str))

    if isinstance(device_or_string, str):
        return device_from_string(device_or_string)

    return device_or_string
