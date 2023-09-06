from looseversion import LooseVersion
from lightning_utilities.core.imports import package_available


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


def assert_triton_unavailable_or_too_old(msg: str) -> None:
    version = triton_version()

    assert version is not None, ""
