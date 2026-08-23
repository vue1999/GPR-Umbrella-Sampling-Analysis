"""Gaussian-process umbrella integration for one- and two-dimensional PMFs."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .integration_1d import reconstruct_pmf_1d
from .integration_2d import reconstruct_pmf_2d
from .pathways import find_lowest_barrier_path
from .support import sampled_support_mask

__all__ = [
    "reconstruct_pmf_1d",
    "reconstruct_pmf_2d",
    "find_lowest_barrier_path",
    "sampled_support_mask",
]

try:
    __version__ = _pkg_version("gpr-umbrella")
except PackageNotFoundError:
    __version__ = "0.1.0"
