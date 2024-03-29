"""Check that `scripts/build_from_source.sh` has produced a correct build."""

import os

BUILD_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "build"))


def main():
    import torch

    expected = os.path.join(BUILD_DIR, "pytorch", "torch", "__init__.py")
    actual = os.path.abspath(torch.__file__)
    assert expected == actual, f"{expected} vs. {actual}"

    import nvfuser
    from looseversion import LooseVersion

    assert LooseVersion(nvfuser.version()) >= LooseVersion("0.0.6"), nvfuser.version()

    import thunder

    expected = os.path.abspath(os.path.join(BUILD_DIR, "..", "..", "thunder", "__init__.py"))
    assert expected == thunder.__file__, f"{expected} vs. {thunder.__file__}"

    print("Build looks good!")


if __name__ == "__main__":
    main()
