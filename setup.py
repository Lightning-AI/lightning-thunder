import glob
import os
import re
from pathlib import Path
from setuptools import setup

_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIRES = os.path.join(_PATH_ROOT, "requirements")
# check if os env. variable is set to convert version to nightly
_CONVERT_VERSION = int(os.environ.get("CONVERT_VERSION2NIGHTLY", 0))


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


def _prepare_extras(
    requirements_dir: str = _PATH_REQUIRES, skip_files: tuple = ("base.txt", "devel.txt", "docs.txt")
) -> dict:
    # Define package extras from requirements files
    req_files = [Path(p) for p in glob.glob(os.path.join(requirements_dir, "*.txt"))]
    extras = {}

    for p in req_files:
        if p.name in skip_files:
            continue

        with open(p) as f:
            extras[p.stem] = [
                line.strip() for line in f if line.strip() and not line.strip().startswith("#") and "@" not in line
            ]

    extras = {name: sorted(set(reqs)) for name, reqs in extras.items()}
    print("The extras are: ", extras)
    return extras


if _CONVERT_VERSION:
    convert_version2nightly()

# Update pyproject.toml with dynamic extras
extras = _prepare_extras()
setup(extras_require=extras)
