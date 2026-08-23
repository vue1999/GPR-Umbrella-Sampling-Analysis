"""Tests for restart-aware PLUMED table parsing."""

from __future__ import annotations

import numpy as np
import pytest

from gpr_umbrella.plumed_io import read_plumed_table


def test_repeated_reordered_headers_and_restart_duplicates(tmp_path):
    first = tmp_path / "COLVAR.first"
    first.write_text(
        """#! FIELDS time path.s opes.bias
0 1 -1
5 2 -2
#! FIELDS path.s opes.bias time
3 -3 10
bad partial
nan -4 12
""",
        encoding="utf-8",
    )
    second = tmp_path / "COLVAR.restart"
    second.write_text(
        """#! FIELDS time path.s opes.bias
#! SET biasfactor 10
5 2.5 -2.5
15 4 -4
""",
        encoding="utf-8",
    )

    table = read_plumed_table(
        (first, second),
        required_fields=("time", "path.s"),
        deduplicate_field="time",
    )

    assert table["fields"] == ("time", "path.s", "opes.bias")
    np.testing.assert_allclose(table["columns"]["time"], [0, 5, 10, 15])
    np.testing.assert_allclose(table["columns"]["path.s"], [1, 2.5, 3, 4])
    assert table["fields_headers_seen"] == 3
    assert table["records_read"] == 5
    assert table["n_records"] == 4
    assert table["malformed_records_ignored"] == 2
    assert table["duplicate_records_replaced"] == 1
    assert table["settings"]["biasfactor"] == "10"


def test_schema_changes_are_rejected(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text(
        """#! FIELDS time x bias
0 1 2
#! FIELDS time y bias
1 2 3
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema changed"):
        read_plumed_table(path)


def test_strict_mode_rejects_nonfinite_records(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time x\n0 nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        read_plumed_table(path, strict=True)
    with pytest.raises(ValueError, match="strict must be Boolean"):
        read_plumed_table(path, strict="false")


def test_missing_required_and_deduplication_fields_are_reported(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time x\n0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required"):
        read_plumed_table(path, required_fields="opes.bias")
    with pytest.raises(ValueError, match="Deduplication field"):
        read_plumed_table(path, deduplicate_field="step")


def test_duplicate_policy_error_is_available(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time x\n0 1\n0 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate time"):
        read_plumed_table(
            path, deduplicate_field="time", duplicate_policy="error"
        )


def test_time_must_be_monotonic_after_restart_deduplication(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time x\n1 1\n0 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_plumed_table(path, deduplicate_field="time")
