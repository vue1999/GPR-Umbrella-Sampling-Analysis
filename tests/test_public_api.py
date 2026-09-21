"""Behavior-level contracts for supported imports and CLI routes."""
from importlib import import_module
from pathlib import Path

import pytest

import gpr_umbrella
from gpr_umbrella import cli_2d
from gpr_umbrella_1d import cli as cli_1d, gpr


@pytest.mark.parametrize("name", ["reconstruct_pmf_1d", "reconstruct_pmf_2d"])
def test_public_functions_are_importable(name):
    assert callable(getattr(gpr_umbrella, name))


@pytest.mark.parametrize("module", ["gpr_umbrella_1d", "gpr_umbrella_1d.gpr"])
def test_documented_legacy_1d_imports_are_compatible(module):
    legacy = import_module(module)
    for name in ("gpr_umbrella_integration", "load_plumed_colvar_data", "load_window_data"):
        assert getattr(legacy, name) is getattr(gpr, name)
    assert gpr_umbrella.reconstruct_pmf_1d is gpr.gpr_umbrella_integration


@pytest.mark.parametrize("cli", [cli_1d, cli_2d])
def test_cli_help_succeeds(cli, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(["--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_distribution_retains_basic_console_scripts():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    for name, target in [("gpr-umbrella", "gpr_umbrella_1d.cli:main"),
                         ("gpr-umbrella-2d", "gpr_umbrella.cli_2d:main")]:
        assert f'{name} = "{target}"' in text


def test_2d_cli_forwards_sampling_and_support_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_2d, "reconstruct_pmf_2d",
                        lambda **kwargs: captured.update(kwargs) or {})
    assert cli_2d.main(["--colvar-dir", "unused", "--support-radius", "2.25",
                       "--covariance-block-size", "4000", "--fit-extra-noise",
                       "--no-plot", "--no-diagnostics"]) == 0
    assert captured["restrict_to_sampled_support"] is True
    assert captured["support_radius"] == pytest.approx(2.25)
    assert captured["covariance_block_size"] == 4000
    assert captured["fit_extra_noise"] is True
    assert not any(key.startswith("path_") for key in captured)
