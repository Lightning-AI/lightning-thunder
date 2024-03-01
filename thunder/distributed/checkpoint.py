import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
from typing import TypeGuard

import torch
from lightning_utilities import compare_version
from torch import Tensor
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import Module

from thunder.core.pytree import tree_map
from thunder.distributed import _shard_params, _unshard_params

_TORCH_GREATER_EQUAL_2_3 = compare_version("torch", operator.ge, "2.3.0", use_base_version=True)

__all__ = [
    "has_fsdp_modules",
    "StateDictOptions",
    "get_model_state_dict",
    "load_model_state_dict",
    "save",
    "load",
]


def has_fsdp_modules(module: object) -> TypeGuard[Module]:
    """Returns whether a module or submodule has been sharded."""
    return isinstance(module, Module) and any(getattr(m, "use_fsdp", False) for m in module.modules())


@dataclass
class StateDictOptions:
    """
    Represents checkpointing options. Based on :class:`torch.distributed.checkpoint.state_dict.StateDictOptions`

    Attributes:
        full_state_dict: Whether we are dealing with a full or sharded state dict. Used during saving or loading.
        strict: Whether to strictly enforce that the keys
            in the ``state_dict`` match the keys returned the module's
            :meth:`~torch.nn.Module.state_dict` function.
        cpu_offload: Whether to offload tensors to CPU.
        rank0_only: Whether to save or load on rank 0 only.
    """

    full_state_dict: bool = False
    strict: bool = True
    cpu_offload: bool = False
    rank0_only: bool = False


def get_model_state_dict(module: Module, options: StateDictOptions, rank: int) -> dict[str, Union[Tensor, "DTensor"]]:
    """Returns the state dict of the model. Based on :func:`torch.distributed.checkpoint.state_dict.get_model_state_dict`

    This supports regular and FSDP sharded models.

    If ``options.full_state_dict is True``, the state dict parameters will be unsharded tensors.
    If ``options.full_state_dict is False``, the state dict parameters will be returned`as sharded ``DTensor``s.

    .. code-block:: python

            from thunder.distributed.checkpoint import StateDictOptions, get_model_state_dict

            model = MyModel()
            sharded_model = thunder.distributed.fsdp(model)

            # Get a regular state_dict
            options = StateDictOptions(full_state_dict=True)
            state_dict = get_model_state_dict(model, options, rank)
            # Same result
            options = StateDictOptions(full_state_dict=True)
            state_dict = get_model_state_dict(sharded_model, options, rank)

            # Get a sharded state dict
            options = StateDictOptions(full_state_dict=False)
            state_dict = get_model_state_dict(sharded_model, options, self.rank)
    """

    cpu = torch.device("cpu")

    # If it's not an FSDP module, nothing special to do
    if not has_fsdp_modules(module):
        if not options.full_state_dict:
            raise ValueError("`full_state_dict=False` cannot be used with non-sharded checkpoints")

        state_dict = module.state_dict() if not options.rank0_only or rank == 0 else {}
        if options.cpu_offload:
            state_dict = tree_map(lambda t: t.to(device=cpu) if isinstance(t, Tensor) else t, state_dict)
        return state_dict

    if not options.full_state_dict and options.rank0_only:
        raise ValueError("`rank0_only=True` cannot be used with `full_state_dict=False`")

    # Get the local state dict. Only the parameters are sharded so everything else is assumed to be replicated
    params_state_dict, rest_state_dict = _split_state_dict(module)

    # CPU-offload the non-parameters
    if options.cpu_offload and (not options.rank0_only or rank == 0):
        for name, tensor in list(rest_state_dict.items()):
            if isinstance(tensor, Tensor):
                rest_state_dict[name] = tensor.to(device=cpu)

    if not hasattr(module, "process_group_for_ddp"):
        raise RuntimeError(f"Expected {module} to be FSDP transformed")
    process_group = module.process_group_for_ddp
    world_size = torch.distributed.get_world_size(group=process_group)
    device_mesh = init_device_mesh("cuda", (world_size,))
    placements = [Shard(0) for _ in range(device_mesh.ndim)]

    # Convert the params state dict to DTensors
    # we do this because it's what distributed checkpoint (DCP) supports to save sharded tensors
    params_state_dict = tree_map(lambda t: DTensor.from_local(t, device_mesh, placements), params_state_dict)

    # If a full state dict was requested, unshard the parameters
    if options.full_state_dict:

        def _full_tensor(tensor: DTensor) -> Tensor | None:
            # This calls collectives so every rank must call it
            full = DTensor.full_tensor(tensor)
            if not options.rank0_only or rank == 0:
                return full.to(cpu) if options.cpu_offload else full
            # Non-zero ranks return None to free the full tensor
            return None

        params_state_dict = tree_map(_full_tensor, params_state_dict)
        if options.rank0_only and rank != 0:
            return {}

    return params_state_dict | rest_state_dict


def load_model_state_dict(state_dict: dict[str, Any], module: Module, options: StateDictOptions, rank: int) -> None:
    """Lodas a state dict into a model.

    This supports regular and FSDP sharded models.

    If ``options.full_state_dict is True``, the state dict parameters are assumed to be unsharded tensors.
    If ``options.full_state_dict is False``, the state dict parameters are assumed to include sharded ``DTensor`` for parameters.

    .. code-block:: python

            from thunder.distributed.checkpoint import StateDictOptions, load_model_state_dict

            model = MyModel()
            sharded_model = thunder.distributed.fsdp(model)

            # Load a full checkpoint into a sharded model
            options = StateDictOptions(full_state_dict=True)
            state_dict = torch.load(..., checkpoint_path)
            load_model_state_dict(state_dict, sharded_model, options, rank)

            from thunder.distributed.checkpoint import get_model_state_dict, load

            # Load a sharded checkpoint into a sharded model
            options = StateDictOptions(full_state_dict=False)
            state_dict = get_model_state_dict(sharded_model, options, rank)
            load(state_dict, checkpoint_dir)
            load_model_state_dict(state_dict, sharded_model, options, rank)
    """

    if not has_fsdp_modules(module):
        if not options.full_state_dict:
            raise ValueError("`full_state_dict=False` cannot be used with non-sharded checkpoints")
        if not options.rank0_only or rank == 0:
            module.load_state_dict(state_dict, strict=options.strict)

    elif options.full_state_dict:
        if not hasattr(module, "process_group_for_ddp"):
            raise RuntimeError(f"Expected {module} to be FSDP transformed")
        process_group = module.process_group_for_ddp
        device = next(module.parameters()).device
        _unshard_params(module, process_group, options.cpu_offload)
        if not options.rank0_only or rank == 0:
            module.load_state_dict(state_dict, strict=options.strict)
        # with rank0_only enabled, it's useful to broadcast so that the other shards are still loaded as expected
        _shard_params(module, process_group, device, 0 if options.rank0_only else None)
    else:
        state_dict = tree_map(lambda t: DTensor.to_local(t) if isinstance(t, DTensor) else t, state_dict)
        module.load_state_dict(state_dict, strict=options.strict)


def save(converted_state: dict[str, Any], path: Path, **kwargs: Any) -> None:
    """Wrapper for backwards compatibility with :func:`torch.distributed.checkpoint.save`"""
    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint import save

        save(converted_state, checkpoint_id=path, **kwargs)
    else:  # deprecated
        from torch.distributed.checkpoint import FileSystemWriter, save

        writer = FileSystemWriter(path=path, single_file_per_rank=True)
        save(converted_state, writer, **kwargs)


def load(module_state: dict[str, Any], path: Path, **kwargs: Any) -> None:
    """Wrapper for backwards compatibility with :func:`torch.distributed.checkpoint.load`"""
    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint import load

        load(module_state, checkpoint_id=path, **kwargs)
    else:  # deprecated
        from torch.distributed.checkpoint import FileSystemReader, load

        reader = FileSystemReader(path=path)
        load(module_state, reader, **kwargs)


def _split_state_dict(module: Module) -> tuple[dict[str, Any], dict[str, Any]]:
    """A flavor of ``module.state_dict()`` that returns parameters separated to everything else."""
    params = {
        param_name: param.detach()
        for module_name, submodule in module.named_modules()
        for param_name, param in submodule.named_parameters(recurse=False, prefix=module_name)
    }
    rest = {k: v for k, v in module.state_dict().items() if k not in params}
    return params, rest
