import thunder.core
import torch


def device_supports_bf16(device: None | str | torch.device | thunder.core.devices.Device, /) -> bool:
    """Torch has its own `torch.cuda.is_bf16_supported()`, but the contract of
    this API changed in December 2023 to be "can bf16 be represented?", which
    is true even for devices that existed before bf16 was invented.

    The contract for this API is that the device implements bf16 operations."""
    if not torch.cuda.is_available():
        return False

    dev: torch.device = thunder.core.devices.to_torch_device(device)

    if dev.type != "cuda":
        return False

    cuda_major: int
    cuda_minor: int
    cuda_major, cuda_minor = torch.cuda.get_device_capability(dev)
    return cuda_major >= 8
