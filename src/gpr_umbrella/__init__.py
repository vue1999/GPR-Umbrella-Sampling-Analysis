"""Gaussian-process umbrella integration for one- and two-dimensional PMFs."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from gpr_umbrella_1d.gpr import gpr_umbrella_integration as reconstruct_pmf_1d
from .integration_2d import reconstruct_pmf_2d

__all__ = [
    "reconstruct_pmf_1d",
    "reconstruct_pmf_2d",
]

try:
    __version__ = _pkg_version("gpr-umbrella")
except PackageNotFoundError:
    __version__ = "0.1.0"
