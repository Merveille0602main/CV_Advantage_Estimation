"""Output helpers shared by synthetic experiments."""
import csv
from pathlib import Path


def write_csv(path, rows, columns=None):
    """Write records, or a header-only file when empty columns are supplied."""
    if not rows and not columns:
        return
    fieldnames = list(rows[0]) if rows else list(columns)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
