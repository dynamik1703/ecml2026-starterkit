#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tools.train_counterfactual_gate import (
    choose_feature_columns,
    filter_feature_columns,
    outcome_category,
    safe_float,
)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def parse_comparison(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("comparison must look like good:bad")
    return parts[0], parts[1]


def category_mask(categories: list[str], category: str) -> np.ndarray:
    if category == "rest":
        return np.ones(len(categories), dtype=bool)
    return np.array([item == category for item in categories], dtype=bool)


def finite_values(rows: list[dict[str, str]], column: str, mask: np.ndarray) -> np.ndarray:
    values = [
        safe_float(row.get(column))
        for row, selected in zip(rows, mask)
        if selected
    ]
    finite = [value for value in values if math.isfinite(value)]
    return np.asarray(finite, dtype=np.float64)


def stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def contrast_row(
    rows: list[dict[str, str]],
    categories: list[str],
    column: str,
    left_category: str,
    right_category: str,
) -> dict[str, float | str]:
    left_mask = category_mask(categories, left_category)
    right_mask = (
        np.logical_not(category_mask(categories, left_category))
        if right_category == "rest"
        else category_mask(categories, right_category)
    )
    left = finite_values(rows, column, left_mask)
    right = finite_values(rows, column, right_mask)
    left_stats = stats(left)
    right_stats = stats(right)
    if left_stats["count"] == 0 or right_stats["count"] == 0:
        effect = float("nan")
        mean_delta = float("nan")
        median_delta = float("nan")
    else:
        mean_delta = left_stats["mean"] - right_stats["mean"]
        median_delta = left_stats["median"] - right_stats["median"]
        pooled = math.sqrt(
            (
                float(left_stats["std"]) ** 2
                + float(right_stats["std"]) ** 2
            )
            / 2.0
        )
        effect = mean_delta / pooled if pooled > 1e-12 else 0.0
    return {
        "feature": column,
        "left": left_category,
        "right": right_category,
        "left_count": left_stats["count"],
        "right_count": right_stats["count"],
        "left_mean": left_stats["mean"],
        "right_mean": right_stats["mean"],
        "mean_delta": mean_delta,
        "left_median": left_stats["median"],
        "right_median": right_stats["median"],
        "median_delta": median_delta,
        "effect_size": effect,
        "abs_effect_size": abs(effect) if math.isfinite(effect) else float("nan"),
    }


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6g}"


def print_rows(rows: list[dict[str, float | str]], top_k: int) -> None:
    print(
        "feature,left_mean,right_mean,mean_delta,left_median,"
        "right_median,median_delta,effect_size"
    )
    for row in rows[:top_k]:
        print(
            "{feature},{left_mean},{right_mean},{mean_delta},"
            "{left_median},{right_median},{median_delta},{effect_size}".format(
                feature=row["feature"],
                left_mean=format_float(float(row["left_mean"])),
                right_mean=format_float(float(row["right_mean"])),
                mean_delta=format_float(float(row["mean_delta"])),
                left_median=format_float(float(row["left_median"])),
                right_median=format_float(float(row["right_median"])),
                median_delta=format_float(float(row["median_delta"])),
                effect_size=format_float(float(row["effect_size"])),
            )
        )


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank counterfactual feature contrasts between outcome classes."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--comparison",
        type=parse_comparison,
        default=("good", "bad"),
        help=(
            "Outcome groups to compare as left:right. Use right=rest to compare "
            "left against all other rows."
        ),
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--include-feature-regex",
        help="Only use feature columns whose names match this regular expression.",
    )
    parser.add_argument(
        "--exclude-feature-regex",
        help="Drop feature columns whose names match this regular expression.",
    )
    parser.add_argument(
        "--drop-prefix-features",
        action="store_true",
        help="Drop route-prefix conflict features for ablation runs.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError("No rows found")
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
    counts = Counter(categories)
    features = filter_feature_columns(choose_feature_columns(rows), args)
    left_category, right_category = args.comparison
    contrasts = [
        contrast_row(rows, categories, feature, left_category, right_category)
        for feature in features
    ]
    contrasts = [
        row
        for row in contrasts
        if int(row["left_count"]) >= args.min_count
        and int(row["right_count"]) >= args.min_count
        and math.isfinite(float(row["abs_effect_size"]))
    ]
    contrasts.sort(key=lambda row: float(row["abs_effect_size"]), reverse=True)

    print(
        "Summary: "
        f"rows={len(rows)} good={counts['good']} neutral={counts['neutral']} "
        f"bad={counts['bad']} features={len(features)} "
        f"comparison={left_category}:{right_category}"
    )
    print_rows(contrasts, args.top_k)

    if args.output_csv is not None:
        write_csv(args.output_csv, contrasts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
