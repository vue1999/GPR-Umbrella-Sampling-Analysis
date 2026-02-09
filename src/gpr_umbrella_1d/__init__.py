"""GPR umbrella integration for 1D PLUMED umbrella sampling outputs."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .gpr import gpr_umbrella_integration, load_plumed_colvar_data, load_window_data

__all__ = ["gpr_umbrella_integration", "load_plumed_colvar_data", "load_window_data"]

try:
    __version__ = _pkg_version("gpr-umbrella-1d")
except PackageNotFoundError:
    __version__ = "0.1.0"
