"""Behavior-level contracts for supported imports and CLI routes."""
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import gpr_umbrella
from gpr_umbrella import cli_1d, cli_2d, integration_1d


@pytest.mark.parametrize("name", ["reconstruct_pmf_1d", "reconstruct_pmf_2d",
                                 "find_lowest_barrier_path", "sampled_support_mask"])
def test_public_functions_are_importable(name):
    assert callable(getattr(gpr_umbrella, name))


@pytest.mark.parametrize("module", ["gpr_umbrella_1d", "gpr_umbrella_1d.gpr"])
def test_documented_legacy_1d_imports_are_compatible(module):
    legacy = import_module(module)
    for old, current in [("gpr_umbrella_integration", "reconstruct_pmf_1d"),
                         ("load_plumed_colvar_data", "load_plumed_colvar_data"),
                         ("load_window_data", "load_window_data")]:
        assert getattr(legacy, old) is getattr(integration_1d, current)


@pytest.mark.parametrize("cli", [cli_1d, cli_2d])
def test_cli_help_succeeds(cli, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(["--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_distribution_retains_basic_console_scripts():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    for name, target in [("gpr-umbrella", "gpr_umbrella.cli_1d:main"),
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


def test_2d_cli_requires_search_endpoints(monkeypatch):
    monkeypatch.setattr(cli_2d, "reconstruct_pmf_2d", lambda **kwargs: {})
    with pytest.raises(SystemExit) as missing:
        cli_2d.main(["--colvar-dir", "unused", "--find-lowest-barrier-path"])
    assert missing.value.code != 0


def test_cli_parses_fixed_reference_and_corridor_modes():
    parser = cli_2d.build_parser()
    for mode, extra in [("fixed", []), ("corridor", ["--path-corridor-radius", "0.25"])]:
        parsed = parser.parse_args(["--colvar-dir", "unused", "--find-lowest-barrier-path",
                                   "--path-mode", mode, "--path-reference", "neb.dat", *extra])
        assert parsed.path_mode == mode and parsed.path_reference == "neb.dat"
        if extra:
            assert parsed.path_corridor_radius == pytest.approx(0.25)


def test_2d_cli_runs_fixed_reference_as_separate_analysis(tmp_path, monkeypatch):
    reference = tmp_path / "neb.dat"
    np.savetxt(reference, [[0., 1.], [2., 3.]], header="coordinate0 coordinate1")
    results = {"pmf_path": str(tmp_path / "test_pmf_2d.dat")}
    captured, saved = {}, []
    monkeypatch.setattr(cli_2d, "reconstruct_pmf_2d", lambda **kwargs: results)
    def find(result, **kwargs):
        assert result is results
        captured.update(kwargs)
        return {"test": "path"}
    monkeypatch.setattr(cli_2d, "find_lowest_barrier_path", find)
    monkeypatch.setattr(cli_2d, "save_lowest_barrier_path", lambda *args: saved.append(args))
    assert cli_2d.main(["--colvar-dir", "unused", "--find-lowest-barrier-path",
                       "--path-mode", "fixed", "--path-reference", str(reference),
                       "--no-plot", "--no-diagnostics", "--quiet"]) == 0
    np.testing.assert_allclose(captured["reference_path"], [[0., 1.], [2., 3.]])
    assert captured["path_mode"] == "fixed" and captured["corridor_radius"] is None
    assert saved[0][0] == {"test": "path"}
    assert str(saved[0][1]).endswith("test_lowest_barrier_path.dat")
