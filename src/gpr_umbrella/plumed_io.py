"""Small, dependency-free readers for PLUMED text tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _normalise_paths(
    paths: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    """Return one or more input paths as a non-empty tuple."""
    if isinstance(paths, (str, Path)):
        result = (Path(paths),)
    else:
        result = tuple(Path(path) for path in paths)
    if not result:
        raise ValueError("At least one PLUMED table path is required")
    return result


def _normalise_fields(fields: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        return (fields,)
    return tuple(str(field) for field in fields)


def read_plumed_table(
    paths: str | Path | Iterable[str | Path],
    *,
    required_fields: str | Iterable[str] = (),
    deduplicate_field: str | None = None,
    duplicate_policy: str = "last",
    strict: bool = False,
) -> dict:
    """Read one or more PLUMED ``#! FIELDS`` text tables.

    Repeated headers, including headers emitted after a restart, are allowed.
    A repeated header may reorder the same fields; every numeric row is mapped
    back to the order of the first header.  A header that changes the field set
    is rejected because silently combining different schemas is unsafe.

    Parameters
    ----------
    paths : path or iterable of paths
        Tables to concatenate in the supplied order.
    required_fields : str or iterable of str
        Named fields that must be present.
    deduplicate_field : str, optional
        Field used to identify restart-overlap records, commonly ``"time"``.
        No deduplication is performed when omitted.
    duplicate_policy : {"last", "first", "error"}
        How duplicate values of *deduplicate_field* are handled.  ``"last"``
        replaces the earlier row in place, matching restart continuation data.
    strict : bool
        Raise on malformed numeric rows.  By default they are counted and
        ignored, which also tolerates an interrupted final write.

    Returns
    -------
    dict
        A NumPy-based table with ``fields``, ``data``, named ``columns``,
        parsed ``#! SET`` metadata, file provenance, and parser counters.
    """
    input_paths = _normalise_paths(paths)
    required = _normalise_fields(required_fields)
    if not isinstance(strict, (bool, np.bool_)):
        raise ValueError("strict must be Boolean")
    if duplicate_policy not in {"last", "first", "error"}:
        raise ValueError(
            "duplicate_policy must be 'last', 'first', or 'error'"
        )

    canonical_fields: tuple[str, ...] | None = None
    current_fields: tuple[str, ...] | None = None
    reorder: np.ndarray | None = None
    rows: list[np.ndarray] = []
    settings: dict[str, str] = {}
    setting_history: dict[str, list[str]] = {}
    headers_seen = 0
    malformed = 0

    def malformed_row(path: Path, line_number: int, message: str) -> None:
        nonlocal malformed
        if strict:
            raise ValueError(f"{path}:{line_number}: {message}")
        malformed += 1

    for path in input_paths:
        current_fields = None
        reorder = None
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#! FIELDS"):
                    fields = tuple(stripped.split()[2:])
                    headers_seen += 1
                    if not fields:
                        raise ValueError(
                            f"{path}:{line_number}: empty #! FIELDS header"
                        )
                    if len(set(fields)) != len(fields):
                        raise ValueError(
                            f"{path}:{line_number}: duplicate field names in header"
                        )
                    if canonical_fields is None:
                        canonical_fields = fields
                    elif set(fields) != set(canonical_fields):
                        missing = sorted(set(canonical_fields) - set(fields))
                        added = sorted(set(fields) - set(canonical_fields))
                        raise ValueError(
                            f"{path}:{line_number}: PLUMED field schema changed; "
                            f"missing={missing}, added={added}"
                        )
                    current_fields = fields
                    field_index = {name: index for index, name in enumerate(fields)}
                    reorder = np.array(
                        [field_index[name] for name in canonical_fields], dtype=int
                    )
                    continue
                if stripped.startswith("#! SET"):
                    parts = stripped.split(maxsplit=3)
                    if len(parts) < 4:
                        malformed_row(path, line_number, "malformed #! SET line")
                        continue
                    key, value = parts[2], parts[3].strip()
                    settings[key] = value
                    setting_history.setdefault(key, []).append(value)
                    continue
                if stripped.startswith("#"):
                    continue
                if current_fields is None or reorder is None:
                    malformed_row(
                        path,
                        line_number,
                        "numeric record appears before a #! FIELDS header",
                    )
                    continue
                parts = stripped.split()
                if len(parts) != len(current_fields):
                    malformed_row(
                        path,
                        line_number,
                        f"expected {len(current_fields)} values, got {len(parts)}",
                    )
                    continue
                try:
                    values = np.asarray(parts, dtype=float)
                except ValueError:
                    malformed_row(path, line_number, "record is not fully numeric")
                    continue
                if np.any(~np.isfinite(values)):
                    malformed_row(path, line_number, "record contains nonfinite values")
                    continue
                rows.append(values[reorder])

    if canonical_fields is None:
        joined = ", ".join(str(path) for path in input_paths)
        raise ValueError(f"No #! FIELDS header found in: {joined}")

    missing_required = [
        field for field in required if field not in canonical_fields
    ]
    if missing_required:
        raise ValueError(
            "Missing required PLUMED fields: " + ", ".join(missing_required)
        )

    if rows:
        data = np.vstack(rows)
    else:
        data = np.empty((0, len(canonical_fields)), dtype=float)
    records_read = len(data)
    duplicates = 0

    if deduplicate_field is not None:
        if deduplicate_field not in canonical_fields:
            raise ValueError(
                f"Deduplication field {deduplicate_field!r} is not present"
            )
        key_index = canonical_fields.index(deduplicate_field)
        selected: list[np.ndarray] = []
        positions: dict[float, int] = {}
        for row in data:
            key = float(row[key_index])
            if not np.isfinite(key):
                raise ValueError(
                    f"Deduplication field {deduplicate_field!r} must be finite"
                )
            if key not in positions:
                positions[key] = len(selected)
                selected.append(row)
                continue
            duplicates += 1
            if duplicate_policy == "error":
                raise ValueError(
                    f"Duplicate {deduplicate_field} value encountered: {key}"
                )
            if duplicate_policy == "last":
                selected[positions[key]] = row
        if selected:
            data = np.vstack(selected)
        else:
            data = np.empty((0, len(canonical_fields)), dtype=float)
        if (
            deduplicate_field == "time"
            and len(data) > 1
            and np.any(np.diff(data[:, key_index]) <= 0)
        ):
            raise ValueError(
                "time values must be strictly increasing after restart "
                "deduplication"
            )

    columns = {
        field: data[:, index]
        for index, field in enumerate(canonical_fields)
    }
    return {
        "fields": canonical_fields,
        "data": data,
        "columns": columns,
        "settings": settings,
        "setting_history": setting_history,
        "files": tuple(str(path) for path in input_paths),
        "fields_headers_seen": headers_seen,
        "records_read": records_read,
        "n_records": len(data),
        "malformed_records_ignored": malformed,
        "duplicate_records_replaced": duplicates,
        "deduplicate_field": deduplicate_field,
    }
