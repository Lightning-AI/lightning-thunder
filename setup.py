#!/usr/bin/env python
import glob
import os
import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pkg_resources import parse_requirements
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


def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return [r for r in list(map(str, reqs)) if "@" not in r]


def _prepare_extras(
    requirements_dir: str = _PATH_REQUIRES, skip_files: tuple = ("base.txt", "devel.txt", "docs.txt")
) -> dict:
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install pytorch-lightning[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    req_files = [Path(p) for p in glob.glob(os.path.join(requirements_dir, "*.txt"))]
    extras = {
        p.stem: _load_requirements(file_name=p.name, path_dir=str(p.parent))
        for p in req_files
        if p.name not in skip_files
    }
    # todo: eventual some custom aggregations
    extras = {name: sorted(set(reqs)) for name, reqs in extras.items()}
    print("The extras are: ", extras)
    return extras


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion."""
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fp:
        text = fp.read()
    # https://github.com/Lightning-AI/lightning-thunder/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we replace some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")
    return text


if _CONVERT_VERSION:
    convert_version2nightly()

about = _load_py_module("__about__.py")

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file. it's not obvious
# what happens and to non-engineers they won't know to look in init.
setup(
    name="lightning-thunder",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Lightning-AI/lightning-thunder",
    license=about.__license__,
    packages=find_packages(exclude=["thunder/tests", "docs"]),
    long_description=_load_readme_description(
        path_dir=_PATH_ROOT, homepage=about.__homepage__, version=about.__version__
    ),
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "AI"],
    python_requires=">=3.10, <3.12",
    setup_requires=["wheel"],
    install_requires=_load_requirements(_PATH_REQUIRES, file_name="base.txt"),
    extras_require=_prepare_extras(),
    project_urls={
        "Bug Tracker": "https://github.com/Lightning-AI/lightning-thunder/issues",
        "Documentation": "https://lightning-thunder.rtfd.io/en/latest/",
        "Source Code": "https://github.com/Lightning-AI/lightning-thunder",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
