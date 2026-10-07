#!/usr/bin/env python3
"""Plot performance graphs from results CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    return pd.read_csv(file_path)


def plot_execution_time(timing_df: pd.DataFrame, output_path: Path) -> None:
    required_columns = {"implementation", "N", "time_ms"}

    if not required_columns.issubset(timing_df.columns):
        raise ValueError(
            f"Timing CSV missing required columns: {required_columns}"
        )

    timing_df = timing_df.copy()
    timing_df["N"] = timing_df["N"].astype(int)
    timing_df = timing_df.sort_values(["implementation", "N"])

    plt.figure(figsize=(10, 6))

    for impl, group in timing_df.groupby("implementation"):
        plt.plot(
            group["N"],
            group["time_ms"],
            marker="o",
            label=impl
        )

    plt.title("Execution Time Comparison")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Execution Time (ms)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_speedup(speedup_df: pd.DataFrame, output_path: Path) -> None:
    required_columns = {"implementation", "N", "speedup_vs_cpu"}

    if not required_columns.issubset(speedup_df.columns):
        raise ValueError(
            f"Speedup CSV missing required columns: {required_columns}"
        )

    speedup_df = speedup_df.copy()
    speedup_df["N"] = speedup_df["N"].astype(int)
    speedup_df = speedup_df.sort_values(["implementation", "N"])

    plt.figure(figsize=(10, 6))

    for impl, group in speedup_df.groupby("implementation"):
        plt.plot(
            group["N"],
            group["speedup_vs_cpu"],
            marker="o",
            label=impl
        )

    plt.title("Speedup vs CPU Baseline")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Speedup Factor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot matrix multiplication result graphs."
    )

    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "results"

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=results_path,
        help="Path to the results folder containing CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=results_path,
        help="Directory where graph images will be saved.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the graphs after generating them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timing_path = args.results_dir / "timing_results.csv"
    speedup_path = args.results_dir / "speedup_results.csv"

    timing_df = load_csv(timing_path)
    speedup_df = load_csv(speedup_path)

    output_dir = args.output_dir

    time_output = output_dir / "execution_time_comparison.png"
    speedup_output = output_dir / "speedup_comparison.png"

    plot_execution_time(timing_df, time_output)
    plot_speedup(speedup_df, speedup_output)

    print(f"Saved execution time graph to: {time_output}")
    print(f"Saved speedup graph to: {speedup_output}")

    if args.show:
        print("Opening graphs for display...")

        for path in (time_output, speedup_output):
            img = plt.imread(path)

            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(path.name)

        plt.show()


if __name__ == "__main__":
    main()