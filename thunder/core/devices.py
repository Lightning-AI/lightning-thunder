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


class Device:
    # TODO Simplify this so string_or_devicetype can be either a string or devicetype regardless of
    #   whether number is provided
    def __init__(self, string_or_devicetype: str | DeviceType, number: Number | None = None):
        if number is not None:
            baseutils.check_type(number, Number)
            baseutils.check_type(string_or_devicetype, DeviceType)
            baseutils.check(number >= 0, lambda: f"Invalid device {number=}")

            self.devicetype = string_or_devicetype
            self.number = number
        else:
            baseutils.check_type(string_or_devicetype, str)
            self.devicetype, self.number = _device_from_string_helper(string_or_devicetype)

        # NOTE While we don't consider it an error to request CPU:1, we also don't pretend
        #   like there are multiple CPU devices.
        if self.devicetype is DeviceType.CPU:
            self.number = 0

    # NOTE This representation is a valid PyTorch device string, which is currently relied upon when
    #   converting Thunder devices to PyTorch devices
    def __repr__(self) -> str:
        if self.devicetype == DeviceType.CPU:
            return devicetype_string(self.devicetype)

        # NOTE self.devicetype == DeviceType.CUDA
        return f"{devicetype_string(self.devicetype)}:{self.number}"

    # TODO Review this
    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other: Device) -> bool:
        if not isinstance(other, Device):
            return False
        return self.devicetype is other.devicetype and self.number == other.number


def available_devices() -> Sequence[Device]:
    return get_default_langctx().available_devices()


cpu = Device(DeviceType.CPU, 0)


def _device_from_string_helper(devicestr: str) -> tuple[DeviceType, int]:
    if devicestr == "cpu":
        return DeviceType.CPU, 0

    if devicestr == "cuda":
        return DeviceType.CUDA, 0

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
