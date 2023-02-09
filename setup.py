#!/usr/bin/env python
import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)
_PATH_SOURCE = os.path.join(_PATH_ROOT, "src")
_PATH_REQUIRES = os.path.join(_PATH_ROOT, "requirements")


def _load_py_module(fname, pkg="thunder"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_SOURCE, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


def _prepare_extras(requirements_dir: str = _PATH_REQUIRES, skip_files: tuple = ("devel.txt", "docs.txt")) -> dict:
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


about = _load_py_module("__about__.py")

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="lightning-thunder",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Lightning-AI/lightning-thunder",
    license=about.__license__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=about.__long_doc__,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "AI"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=_load_requirements(_PATH_ROOT),
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
