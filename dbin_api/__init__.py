"""
dbin_api package

This package will contain the FastAPI-based API wrapper and moduleized internals
for the 3D bin-packing functionality originally implemented in `BinCore.py`.

Currently this initializer exposes a small, stable surface:
- __version__: package version string
- get_version(): helper to retrieve the version

Later modules (e.g. `api`, `core`, `models`, `services`) will be added under
this package. Keep this file minimal to avoid import-time side-effects.
"""

from typing import Final

__all__ = ["__version__", "get_version"]

__version__: Final[str] = "0.1.0"


def get_version() -> str:
    """
    Return the package version.

    Use this from external code to check the installed/refactored API version.
    """
    return __version__
