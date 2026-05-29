from __future__ import annotations

from collections.abc import Mapping


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def print_metrics(metrics: Mapping[str, float]) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def print_table(rows: list[Mapping[str, object]], columns: list[str]) -> None:
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
