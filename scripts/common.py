from __future__ import annotations

from collections.abc import Mapping
from unicodedata import east_asian_width


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def print_metrics(metrics: Mapping[str, float]) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def print_table(rows: list[Mapping[str, object]], columns: list[str]) -> None:
    # 日本語は多くのターミナルで2文字分の幅になるので、その幅でそろえる。
    def display_width(value: object) -> int:
        return sum(2 if east_asian_width(char) in {"F", "W"} else 1 for char in str(value))

    def pad(value: object, width: int) -> str:
        return str(value) + " " * (width - display_width(value))

    widths = {
        column: max(display_width(column), *(display_width(row[column]) for row in rows))
        for column in columns
    }
    header = " | ".join(pad(column, widths[column]) for column in columns)
    print(header)
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(pad(row[column], widths[column]) for column in columns))
