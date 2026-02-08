from gpr_umbrella_1d import gpr_umbrella_integration, load_plumed_colvar_data, load_window_data


def test_import_only():
    assert callable(gpr_umbrella_integration)
    assert callable(load_plumed_colvar_data)
    assert callable(load_window_data)
