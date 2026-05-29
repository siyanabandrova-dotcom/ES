"""Plot validation learning curve (Figure 4b style) from validation.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_figure_4b(
    validation_csv: Path,
    output_path: Path | None = None,
    title: str = "Countdown validation score (EGGROLL)",
) -> Path:
    df = pd.read_csv(validation_csv)
    required = {"epoch", "validation_score", "time_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{validation_csv} missing columns: {sorted(missing)}")

    x_hours = df["time_seconds"] / 3600.0
    y = df["validation_score"]

    if output_path is None:
        output_path = validation_csv.with_name("figure_4b.png")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x_hours, y, linewidth=2, label="EGGROLL")
    plt.xlabel("Wall-clock time (hours)")
    plt.ylabel("Validation score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 4b from validation.csv")
    parser.add_argument(
        "validation_csv",
        type=Path,
        nargs="?",
        default=Path("validation.csv"),
        help="Path to validation.csv (default: ./validation.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: figure_4b.png next to CSV)",
    )
    args = parser.parse_args()

    out = plot_figure_4b(args.validation_csv, args.output)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
