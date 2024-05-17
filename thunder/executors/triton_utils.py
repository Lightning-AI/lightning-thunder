from lightning_utilities.core.imports import package_available
from looseversion import LooseVersion


def triton_version() -> None | str:
    if not package_available("triton"):
        return None

    import triton

    return triton.__version__


def is_triton_version_at_least(minimum_version: str) -> bool:
    version = triton_version()

    if version is None:
        return False

    return LooseVersion(version) >= minimum_version
