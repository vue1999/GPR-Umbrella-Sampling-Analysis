"""Optional structural provenance checks for NEB-initialised MD windows."""

import json
from pathlib import Path

import numpy as np


def audit_campaign(manifest, path_xyz, *, fixed_tolerance=1e-6):
    """Verify fixed geometry and cell, not just model names or temperatures.

    These checks cannot establish equilibration or equality of arbitrary
    unrecorded restraints. Small but nonzero changes in fixed atoms are not
    assumed energetically negligible; cross-Hamiltonian energies are required.
    """
    from ase.io import read

    meta = json.loads(Path(manifest).read_text())
    temperatures = [
        t
        for t in (
            meta.get("temperature_K"),
            meta.get("sampling", {}).get("temperature_K"),
        )
        if t is not None
    ]
    if (
        not temperatures
        or not np.all(np.isfinite(temperatures))
        or min(temperatures) <= 0
        or not np.allclose(temperatures, temperatures[0], rtol=0, atol=1e-8)
    ):
        raise ValueError(
            "Campaign temperature metadata is missing, invalid or inconsistent"
        )
    frames = read(path_xyz, ":")
    fixed = meta["selection"]["fixed_fe_indices"]
    cells = np.array([a.cell.array for a in frames])
    if len(frames) != meta["n_windows"]:
        raise ValueError("Campaign reference frame count does not match windows")
    same_atoms = all(
        a.get_chemical_symbols() == frames[0].get_chemical_symbols() for a in frames
    )
    same_cell = bool(np.allclose(cells, cells[0], atol=1e-8, rtol=0))
    positions = np.array([a.positions[fixed] for a in frames])
    delta = positions - positions[0]
    fractional = delta @ np.linalg.inv(cells[0])
    fractional[..., frames[0].pbc] -= np.round(fractional[..., frames[0].pbc])
    displacement = np.linalg.norm(fractional @ cells[0], axis=-1)
    maximum = float(np.max(displacement)) if displacement.size else 0.0
    issues = []
    if not same_atoms:
        issues.append("inconsistent_atom_identity")
    if not same_cell:
        issues.append("inconsistent_simulation_cell")
    if maximum > fixed_tolerance:
        issues.append("different_frozen_backgrounds_require_cross_hamiltonian_energies")
    model_hash = meta.get("inputs", {}).get("model_sha256", meta.get("model_sha256"))
    if not model_hash:
        issues.append("model_provenance_not_verified")
    return {
        "quality_issues": issues,
        "max_fixed_atom_displacement_A": maximum,
        "n_windows": meta["n_windows"],
        "temperature_K": float(temperatures[0]),
        "model_sha256": model_hash,
        "fixed_atom_tolerance_A": fixed_tolerance,
        "same_atom_order": same_atoms,
        "same_cell": same_cell,
        "manifest": str(manifest),
        "path_xyz": str(path_xyz),
    }
