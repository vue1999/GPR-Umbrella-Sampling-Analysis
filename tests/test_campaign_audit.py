import json

import pytest


def test_dense_metadata_and_different_fixed_backgrounds(tmp_path):
    ase = pytest.importorskip("ase")
    from ase.io import write

    from gpr_umbrella.campaign_audit import audit_campaign

    a = ase.Atoms("FeH", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10], pbc=True)
    b = a.copy()
    b.positions[0, 0] = 0.02
    geometry = tmp_path / "neb.xyz"
    write(geometry, [a, b])
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "temperature_K": 300.0,
                "n_windows": 2,
                "selection": {"fixed_fe_indices": [0]},
            }
        )
    )
    report = audit_campaign(manifest, geometry)
    assert report["temperature_K"] == 300.0
    assert report["max_fixed_atom_displacement_A"] == pytest.approx(0.02)
    assert (
        "different_frozen_backgrounds_require_cross_hamiltonian_energies"
        in report["quality_issues"]
    )
    assert "model_provenance_not_verified" in report["quality_issues"]


def test_conflicting_temperatures_fail_explicitly(tmp_path):
    pytest.importorskip("ase")
    from gpr_umbrella.campaign_audit import audit_campaign

    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps({"temperature_K": 300.0, "sampling": {"temperature_K": 200.0}})
    )
    with pytest.raises(ValueError, match="temperature metadata"):
        audit_campaign(manifest, tmp_path / "not_read.xyz")
