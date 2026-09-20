"""Compatibility imports for the original 1D package."""
from gpr_umbrella import __version__
from .gpr import gpr_umbrella_integration, load_plumed_colvar_data, load_window_data

__all__ = ["gpr_umbrella_integration", "load_plumed_colvar_data", "load_window_data"]
