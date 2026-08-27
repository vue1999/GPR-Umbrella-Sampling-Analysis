"""Contract tests for the installed package and command-line surface."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

import gpr_umbrella
from gpr_umbrella import cli_1d, cli_2d


ROOT = Path(__file__).resolve().parents[1]


def _toml_section(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^\[{re.escape(name)}\]\s*$\n(.*?)(?=^\[|\Z)", text
    )
    assert match is not None, f"missing [{name}] section"
    return match.group(1)


def _quoted_assignment(section: str, key: str) -> str:
    match = re.search(
        rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', section
    )
    assert match is not None, f"missing {key!r} assignment"
    return match.group(1)


def test_public_api_is_exactly_the_supported_functions() -> None:
    expected = [
        "reconstruct_pmf_1d",
        "reconstruct_pmf_2d",
        "find_lowest_barrier_path",
        "sampled_support_mask",
    ]
    assert gpr_umbrella.__all__ == expected
    for name in expected:
        assert callable(getattr(gpr_umbrella, name))

def test_distribution_name_and_console_script_targets() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = _toml_section(pyproject, "project")
    scripts = _toml_section(pyproject, "project.scripts")

    assert _quoted_assignment(project, "name") == "gpr-umbrella"
    assert _quoted_assignment(scripts, "gpr-umbrella") == (
        "gpr_umbrella.cli_1d:main"
    )
    assert _quoted_assignment(scripts, "gpr-umbrella-2d") == (
        "gpr_umbrella.cli_2d:main"
    )


def test_cli_help_parses_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as one_dimensional_help:
        cli_1d.build_parser().parse_args(["--help"])
    assert one_dimensional_help.value.code == 0
    assert "usage:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as two_dimensional_help:
        cli_2d.main(["--help"])
    assert two_dimensional_help.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text
    assert "--find-lowest-barrier-path" in help_text
    assert "--path-aligned-marginal" in help_text
    assert "--thermal-energy" in help_text
    assert "--restrict-to-sampled-support" in help_text



def test_2d_path_options_are_minimal_and_find_mep_is_retired() -> None:
    parser = cli_2d.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--find-lowest-barrier-path",
        "--path-endpoints",
        "--path-mode",
        "--path-reference",
        "--path-corridor-radius",
        "--path-metric-scales",
        "--path-aligned-marginal",
        "--thermal-energy",
        "--restrict-to-sampled-support",
        "--support-radius",
    } <= option_strings
    assert {
        "--path-endpoint-radius",
        "--adjust-path-endpoints",
        "--no-adjust-path-endpoints",
        "--path-gradient-weight",
        "--path-uncertainty-weight",
        "--find-mep",
    }.isdisjoint(option_strings)

    parsed = parser.parse_args([
        "--colvar-dir", "unused",
        "--find-lowest-barrier-path",
        "--path-endpoints", "-1", "0", "1", "0",
        "--path-aligned-marginal",
        "--thermal-energy", "0.025",
    ])
    assert parsed.find_lowest_barrier_path is True
    assert parsed.path_endpoints == [-1.0, 0.0, 1.0, 0.0]
    assert parsed.path_aligned_marginal is True
    assert parsed.thermal_energy == pytest.approx(0.025)
    assert parsed.restrict_to_sampled_support is True
    assert parsed.support_radius == pytest.approx(0.5)
    assert parsed.path_mode == "search"

    corridor = parser.parse_args([
        "--colvar-dir", "unused",
        "--support-radius", "2.25",
        "--path-mode", "corridor",
        "--path-reference", "neb.dat",
        "--path-corridor-radius", "0.25",
    ])
    assert corridor.path_mode == "corridor"
    assert corridor.path_reference == "neb.dat"
    assert corridor.path_corridor_radius == pytest.approx(0.25)
    assert corridor.support_radius == pytest.approx(2.25)


def test_2d_cli_forwards_only_retained_support_and_path_options(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli_2d, "reconstruct_pmf_2d", lambda **kwargs: calls.append(kwargs)
    )
    assert cli_2d.main([
        "--colvar-dir", "unused",
        "--support-radius", "2.25",
        "--no-plot",
        "--no-diagnostics",
    ]) == 0

    assert len(calls) == 1
    assert calls[0]["restrict_to_sampled_support"] is True
    assert calls[0]["support_radius"] == pytest.approx(2.25)
    for removed in (
        "path_endpoint_radius", "adjust_path_endpoints",
        "path_gradient_weight", "path_uncertainty_weight",
    ):
        assert removed not in calls[0]


def test_2d_cli_requires_search_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(cli_2d, "reconstruct_pmf_2d", lambda **kwargs: {})
    with pytest.raises(SystemExit) as missing:
        cli_2d.main([
            "--colvar-dir", "unused", "--find-lowest-barrier-path"
        ])
    assert missing.value.code != 0


def test_2d_cli_loads_reference_trajectory(tmp_path, monkeypatch):
    reference_file = tmp_path / "neb.dat"
    reference_file.write_text("# HH RELZ\n0.0 1.0\n2.0 3.0\n")
    captured = {}

    def fake_reconstruct(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli_2d, "reconstruct_pmf_2d", fake_reconstruct)
    assert cli_2d.main([
        "--colvar-dir", "unused",
        "--find-lowest-barrier-path",
        "--path-mode", "fixed",
        "--path-reference", str(reference_file),
    ]) == 0
    np.testing.assert_allclose(
        captured["path_reference"], [[0.0, 1.0], [2.0, 3.0]]
    )
    assert captured["path_mode"] == "fixed"
    assert captured["path_corridor_radius"] is None
