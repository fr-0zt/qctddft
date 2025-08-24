from __future__ import annotations
import logging
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("qctddft")
except PackageNotFoundError:
    __version__ = "1.0.0"

# Configure a logger for consistent output throughout the package
logger = logging.getLogger("qctddft")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

__all__ = ["__version__", "logger"]
