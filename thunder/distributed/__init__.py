from typing import Optional, Any

import torch.distributed as tdist

import thunder as lc
import thunder.core.utils as utils


# TODO Add backward support (this currently just makes forward consistent)
# TODO Verify parameters are not partially initialized
# TODO Handle buffers
# TODO Improve initial broadcast logic
# Syncs a module's parameters across multiple processes
#   world, if specified, is a list of (rank, device) tuples
#   broadcast_from, if specified, is the rank to broadcast tensors from
#   At least one of world or broadcast_from must be specified so that we can
#       coordinate the broadcasting of parameters
def ddp(
        cmodel: lc.ThunderOptimizedModule, 
        rank: int,
        *, 
        world: Optional[Any] = None,
        broadcast_from: Optional[int] = None, 
        process_group: Optional[tdist.ProcessGroup] = None) -> lc.ThunderOptimizedModule:
    
    utils.check(tdist.is_available(), lambda: f"ddp requires torch distributed to be available (but it's not)")
    utils.check(world is not None or broadcast_from is not None, lambda: f"At least one of world_size or broadcast_from must be specified")
    
    # Infers device information from model
    # TODO Verify parameters are not partially initialized
    # TODO Handle buffers
    named_params = cmodel.named_parameters()
    _, first_param = next(named_params)
    device = first_param.device
    devicetype = device.type
    deviceindex = device.index
    for name, param in named_params:
        utils.check(param.device.type == devicetype, lambda: f"Trying to ddp a model with parameters on devices with different device types, including {devicetype} and {param.device.type}")
        utils.check(deviceindex == param.device.index, lambda: f"Trying to ddp a model with parameters on multiple devices, including devices {deviceindex} and {param.device.index}, but currently only models with all their parameters on one device are supported")

    # Validates world information, if available
    lowest_device_index = deviceindex
    if world is not None:
        found_broadcast_process = False
        for rank_, dev in world:
            utils.check(dev.type == devicetype, lambda: f"Found a world with multiple device types")
            if rank_ == rank:
                utils.check(dev == device, lambda: f"World entry ({rank_}, {dev}) disagrees with inferred device {device} of rank {rank}")
            lowest_device_index = min(lowest_device_index, dev.index)
            if rank_ == broadcast_from:
                found_broadcast_process = True
        
        utils.check(not broadcast_from or found_broadcast_process, lambda: f"Trying to broadcast from rank={broadcast_from}, but didn't find that rank in the world description")

    # Identifies which process to broadcast from
    broadcast_from = broadcast_from if broadcast_from is not None else lowest_device_index

    # Starts broadcasts
    # TODO Make these broadcast asyncs
    # TODO Perform up to two broadcasts at a time
        # https://github.com/Lightning-AI/lightning-thunder/issues/727
    # TODO "Bucket" small tensors together before broadcasting
    for name, param in cmodel.named_parameters():
        tdist.broadcast(
            param, 
            src=broadcast_from, 
            group=process_group, 
            async_op=False)

    return cmodel
