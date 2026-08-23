"""Contract tests for the installed package and command-line surface."""

from __future__ import annotations

from pathlib import Path
import re

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
        "--path-endpoint-radius",
        "--adjust-path-endpoints",
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
    assert parsed.restrict_to_sampled_support is True
    assert parsed.support_radius == pytest.approx(1.0)
    assert parsed.adjust_path_endpoints is True

    restricted = parser.parse_args([
        "--colvar-dir", "unused",
        "--restrict-to-sampled-support",
        "--support-radius", "2.25",
        "--path-endpoint-radius", "0.75",
        "--no-adjust-path-endpoints",
    ])
    assert restricted.restrict_to_sampled_support is True
    assert restricted.support_radius == pytest.approx(2.25)
    assert restricted.path_endpoint_radius == pytest.approx(0.75)
    assert restricted.adjust_path_endpoints is False

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
        "--path-endpoint-radius", "0.75",
        "--no-adjust-path-endpoints",
        "--no-plot",
        "--no-diagnostics",
    ]) == 0

    assert len(calls) == 1
    assert calls[0]["restrict_to_sampled_support"] is True
    assert calls[0]["support_radius"] == pytest.approx(2.25)
    assert calls[0]["path_endpoint_radius"] == pytest.approx(0.75)
    assert calls[0]["adjust_path_endpoints"] is False
