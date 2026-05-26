"""Arrow is PipeDream's single in-memory table representation.

The DuckDB executor returns a :class:`pyarrow.Table`; the CLI and eval harness
consume Arrow directly. DuckDB is Arrow-native (zero-copy, and more type-faithful
than the pandas path), and the vectorized ``classify`` UDF already speaks Arrow,
so there is one columnar format end to end and no pandas dependency. These
helpers keep the few presentation/IO touch-points in one place.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv


def read_csv(path: str | Path) -> pa.Table:
    return pacsv.read_csv(str(path))


def write_csv(table: pa.Table, path: str | Path) -> None:
    # stdlib csv (QUOTE_MINIMAL) rather than pyarrow, which always quotes header
    # names; this keeps plain headers/values bare for readable diffs.
    names = table.column_names
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in table.to_pylist():
            writer.writerow(["" if row[n] is None else row[n] for n in names])


def head(table: pa.Table, n: int) -> pa.Table:
    return table.slice(0, max(0, n))


def rows(table: pa.Table) -> list[dict]:
    return table.to_pylist()


def render(table: pa.Table) -> str:
    """A compact, right-aligned text rendering for console output."""
    names = table.column_names
    if not names:
        return ""
    str_cols = [[_cell(v) for v in table.column(n).to_pylist()] for n in names]
    widths = [
        max([len(names[i])] + [len(c) for c in str_cols[i]])
        for i in range(len(names))
    ]
    lines = ["  ".join(names[i].rjust(widths[i]) for i in range(len(names)))]
    for r in range(table.num_rows):
        lines.append(
            "  ".join(str_cols[i][r].rjust(widths[i]) for i in range(len(names)))
        )
    return "\n".join(lines)


def _cell(value: object) -> str:
    return "" if value is None else str(value)
