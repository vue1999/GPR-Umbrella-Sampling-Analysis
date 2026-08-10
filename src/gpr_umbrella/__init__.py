"""Free-energy reconstruction from umbrella and adaptive biased sampling."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from .integration_1d import reconstruct_pmf_1d
from .integration_2d import reconstruct_pmf_2d
from .pathways import find_lowest_barrier_path
from .gradient_gpr import (
    fit_gradient_gp,
    fit_icf_gp,
    posterior_covariance_gradient_gp,
    predict_gradient_gp,
)
from .icf import reconstruct_pmf_icf
from .metadynamics import analyze_metadynamics
from .opes import analyze_opes_1d
from .biased_sampling import (
    block_reweighted_pmf_1d,
    detect_hysteretic_transitions,
    pmf_landmarks_1d,
    region_weight_diagnostics,
)

__all__ = [
    "reconstruct_pmf_1d",
    "reconstruct_pmf_2d",
    "find_lowest_barrier_path",
    "fit_gradient_gp",
    "fit_icf_gp",
    "predict_gradient_gp",
    "posterior_covariance_gradient_gp",
    "reconstruct_pmf_icf",
    "analyze_opes_1d",
    "analyze_metadynamics",
    "block_reweighted_pmf_1d",
    "region_weight_diagnostics",
    "pmf_landmarks_1d",
    "detect_hysteretic_transitions",
]

try:
    __version__ = _pkg_version("gpr-umbrella")
except PackageNotFoundError:
    __version__ = "0.2.0"
