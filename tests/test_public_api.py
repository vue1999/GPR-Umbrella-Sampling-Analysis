"""Contract tests for the installed package and command-line surface."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

import gpr_umbrella
from gpr_umbrella import cli_1d, cli_2d, cli_biased


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


def test_public_api_contains_the_supported_reconstruction_functions() -> None:
    expected = [
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
    assert _quoted_assignment(scripts, "gpr-biased") == (
        "gpr_umbrella.cli_biased:main"
    )


def test_readme_cites_the_adaptive_bias_icf_gpr_paper() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    icf_section = readme.split("## ICF/GPR for OPES or metadynamics", 1)[1]
    icf_section = icf_section.split("## Reading convergence evidence", 1)[0]
    assert "Mones, N. Bernstein, and G. Csányi" in icf_section
    assert "10.1021/acs.jctc.6b00553" in icf_section
    assert "../ct6b00553.pdf" in icf_section


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

    with pytest.raises(SystemExit) as biased_help:
        cli_biased.build_parser().parse_args(["--help"])
    assert biased_help.value.code == 0
    biased_help_text = capsys.readouterr().out
    assert "opes" in biased_help_text
    assert "metad" in biased_help_text
    assert "icf" in biased_help_text


def test_2d_path_options_are_exposed_and_find_mep_is_retired() -> None:
    parser = cli_2d.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--find-lowest-barrier-path",
        "--path-aligned-marginal",
        "--thermal-energy",
        "--restrict-to-sampled-support",
    } <= option_strings
    assert "--find-mep" not in option_strings

    parsed = parser.parse_args([
        "--colvar-dir", "unused",
        "--find-lowest-barrier-path",
        "--path-aligned-marginal",
        "--thermal-energy", "0.025",
    ])
    assert parsed.find_lowest_barrier_path is True
    assert parsed.path_aligned_marginal is True
    assert parsed.thermal_energy == pytest.approx(0.025)
    assert parsed.restrict_to_sampled_support is False

    restricted = parser.parse_args([
        "--colvar-dir", "unused",
        "--restrict-to-sampled-support",
        "--support-radius", "2.25",
    ])
    assert restricted.restrict_to_sampled_support is True
    assert restricted.support_radius == pytest.approx(2.25)

    with pytest.raises(SystemExit) as retired_option:
        parser.parse_args(["--colvar-dir", "unused", "--find-mep"])
    assert retired_option.value.code != 0


def test_2d_cli_forwards_sampled_support_options(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        cli_2d,
        "reconstruct_pmf_2d",
        lambda **kwargs: calls.append(kwargs),
    )
    assert cli_2d.main([
        "--colvar-dir", "unused",
        "--restrict-to-sampled-support",
        "--support-radius", "2.25",
        "--no-plot",
        "--no-diagnostics",
    ]) == 0

    assert len(calls) == 1
    assert calls[0]["restrict_to_sampled_support"] is True
    assert calls[0]["support_radius"] == pytest.approx(2.25)
