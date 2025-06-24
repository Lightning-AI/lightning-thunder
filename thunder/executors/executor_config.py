from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
import json
from typing import NamedTuple
from typing import TYPE_CHECKING

from thunder.core.proxies import FutureTensorProxy
from thunder.core.proxies import TensorProxy
from thunder.core.pytree import tree_flatten
from thunder.extend import get_executor
from thunder.torch.experimental.dtensor_proxy import DTensorProxy


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from thunder.core.devices import Device
    from thunder.core.dtypes import dtype
    from thunder.core.proxies import AnyProxy
    from thunder.core.symbol import Symbol
    from thunder.extend import Executor


__all__ = [
    "ExecutorConfig",
]


class OpInputMetadata(NamedTuple):
    shapes: tuple[tuple[int, ...], ...]
    devices: tuple[Device, ...]
    dtypes: tuple[dtype, ...]
    dtensor_specs: tuple[AnyProxy, ...]
    constants: tuple[Any, ...]


@dataclass
class ExecutorConfig:
    _op_and_inputs_to_executor: dict = field(init=False, default_factory=dict)

    def _key_from_op_and_inputs(self, *args, **kwargs) -> OpInputMetadata:
        flat_args, _spec = tree_flatten((args, kwargs))
        shapes, devices, dtypes, dtensor_specs, constants = [[] for _ in range(5)]
        for t in flat_args:
            if isinstance(t, (TensorProxy, FutureTensorProxy, DTensorProxy)):
                shapes.append(t.shape)
                devices.append(t.device)
                dtypes.append(t.dtype)
                if isinstance(t, DTensorProxy):
                    dtensor_specs.append(t.spec)
            else:
                constants.append(t)

        return OpInputMetadata(
            tuple(shapes),
            tuple(devices),
            tuple(dtypes),
            tuple(dtensor_specs),
            tuple(constants),
        )

    def maybe_get_executor_for(
        self,
        symbol: Symbol,
        *args,
        **kwargs,
    ) -> Executor | None:
        key = self._key_from_op_and_inputs(symbol, *args, **kwargs)
        if (executor_name := self._op_and_inputs_to_executor.get(key)) is not None:
            return get_executor(executor_name)
        else:
            return None

    @staticmethod
    def from_executor_config_json(executor_config_file: str | Path) -> ExecutorConfig:
        with open(executor_config_file) as f:
            existing_config = json.load(f)
        executor_config = ExecutorConfig()
        executor_config._op_and_inputs_to_executor.update(existing_config)
        return executor_config
