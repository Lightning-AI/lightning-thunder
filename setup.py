import os
import re
from importlib.util import module_from_spec, spec_from_file_location

from packaging.requirements import Requirement as parse_requirements
from setuptools import find_packages, setup


_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRES = os.path.join(_PATH_ROOT, "requirements")
# check if os env. variable is set to convert version to nightly
_CONVERT_VERSION = int(os.environ.get("CONVERT_VERSION2NIGHTLY", 0))


def _load_py_module(fname, pkg="thunder"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list[str]:
    """Parse requirements file -> list of strings"""
    reqs: list[str] = []
    with open(os.path.join(path_dir, file_name)) as f:
        for line in f:
            if line and not line.startswith("#"):
                reqs.append(line.strip())
    # Filter out requirements referring to local paths or specific URLs (if any)
    reqs = [str(parse_requirements(r)) for r in reqs if "@" not in r and "://" not in r]
    return reqs


def convert_version2nightly(about_file: str = "thunder/__about__.py") -> None:
    """Load the actual version and convert it to the nightly version."""
    from datetime import datetime

    # load the about file
    with open(about_file) as fo:
        lines = fo.readlines()
    idx = None
    # find the line with version
    for i, ln in enumerate(lines):
        if ln.startswith("__version__"):
            idx = i
            break
    if idx is None:
        raise ValueError("The version is not found in the `__about__.py` file.")
    # parse the version from variable assignment
    version = lines[idx].split("=")[1].strip().strip('"')
    # parse X.Y.Z version and prune any suffix
    vers = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    # create timestamp  YYYYMMDD
    timestamp = datetime.now().strftime("%Y%m%d")
    version = f"{'.'.join(vers.groups())}.dev{timestamp}"
    # print the new version
    lines[idx] = f'__version__ = "{version}"\n'
    # dump updated lines
    with open(about_file, "w") as fo:
        fo.writelines(lines)


if _CONVERT_VERSION:
    convert_version2nightly()

about = _load_py_module("__about__.py")

setup(
    version=about.__version__,
    packages=find_packages(exclude=["thunder/tests", "docs"]),
    install_requires=_load_requirements(_PATH_REQUIRES, file_name="base.txt"),
    include_package_data=True,
    zip_safe=False,
)
