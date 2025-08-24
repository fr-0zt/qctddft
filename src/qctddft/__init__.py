from __future__ import annotations
import logging
from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("qctddft")
except PackageNotFoundError:
    __version__ = "1.0.0"

logger = logging.getLogger("qctddft")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

__all__ = ["__version__", "logger"]
