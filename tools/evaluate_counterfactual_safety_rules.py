#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tools.train_counterfactual_gate import outcome_category, safe_float


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def value(row: dict[str, str], column: str, default: float = 0.0) -> float:
    result = safe_float(row.get(column))
    return result if math.isfinite(result) else default


def is_safe(row: dict[str, str], rule: dict[str, float]) -> bool:
    if value(row, "forced_prefix_conflict_agents") > rule["max_conflict_agents"]:
        return False
    if (
        value(row, "forced_prefix_conflict_agents")
        - value(row, "baseline_prefix_conflict_agents")
        > rule["max_conflict_delta"]
    ):
        return False
    if (
        value(row, "forced_prefix_min_intersection_eta_gap", 999.0)
        < rule["min_intersection_eta_gap"]
    ):
        return False
    if (
        value(row, "forced_prefix_min_head_on_eta_gap", 999.0)
        < rule["min_head_on_eta_gap"]
    ):
        return False
    if (
        value(row, "forced_prefix_head_on_edge_conflicts")
        > rule["max_head_on_edge_conflicts"]
    ):
        return False
    if (
        value(row, "forced_prefix_opposing_direction_intersections")
        > rule["max_opposing_intersections"]
    ):
        return False
    if value(row, "future_head_on_risk_delta") > rule["max_future_risk_delta"]:
        return False
    return True


def grid(args: argparse.Namespace) -> list[dict[str, float]]:
    rules = []
    for values in itertools.product(
        args.max_conflict_agents,
        args.max_conflict_delta,
        args.min_intersection_eta_gap,
        args.min_head_on_eta_gap,
        args.max_head_on_edge_conflicts,
        args.max_opposing_intersections,
        args.max_future_risk_delta,
    ):
        rules.append(
            {
                "max_conflict_agents": values[0],
                "max_conflict_delta": values[1],
                "min_intersection_eta_gap": values[2],
                "min_head_on_eta_gap": values[3],
                "max_head_on_edge_conflicts": values[4],
                "max_opposing_intersections": values[5],
                "max_future_risk_delta": values[6],
            }
        )
    return rules


def evaluate_rule(
    rows: list[dict[str, str]],
    categories: list[str],
    rule: dict[str, float],
) -> dict[str, Any]:
    accepted = [is_safe(row, rule) for row in rows]
    accepted_categories = [
        category
        for category, keep in zip(categories, accepted)
        if keep
    ]
    rejected_categories = [
        category
        for category, keep in zip(categories, accepted)
        if not keep
    ]
    total_good = categories.count("good")
    total_bad = categories.count("bad")
    accepted_good = accepted_categories.count("good")
    accepted_bad = accepted_categories.count("bad")
    return {
        **rule,
        "accepted": len(accepted_categories),
        "accepted_good": accepted_good,
        "accepted_neutral": accepted_categories.count("neutral"),
        "accepted_bad": accepted_bad,
        "rejected_good": rejected_categories.count("good"),
        "rejected_neutral": rejected_categories.count("neutral"),
        "rejected_bad": rejected_categories.count("bad"),
        "good_recall": accepted_good / total_good if total_good else 0.0,
        "bad_leak_rate": accepted_bad / total_bad if total_bad else 0.0,
    }


def format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def print_results(results: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "rejected_good",
        "rejected_bad",
        "good_recall",
        "bad_leak_rate",
        "max_conflict_agents",
        "max_conflict_delta",
        "min_intersection_eta_gap",
        "min_head_on_eta_gap",
        "max_head_on_edge_conflicts",
        "max_opposing_intersections",
        "max_future_risk_delta",
    ]
    print(",".join(columns))
    for row in results[:top_k]:
        values = []
        for column in columns:
            item = row[column]
            values.append(format_float(float(item)) if isinstance(item, float) else str(item))
        print(",".join(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prefix/ETA safety rules on counterfactual labels."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--max-conflict-agents",
        type=float,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--max-conflict-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
    )
    parser.add_argument(
        "--min-intersection-eta-gap",
        type=float,
        nargs="+",
        default=[0, 2, 5, 10, 20, 50],
    )
    parser.add_argument(
        "--min-head-on-eta-gap",
        type=float,
        nargs="+",
        default=[0, 2, 5, 10, 20, 50],
    )
    parser.add_argument(
        "--max-head-on-edge-conflicts",
        type=float,
        nargs="+",
        default=[0, 1, 5, 10, 20],
    )
    parser.add_argument(
        "--max-opposing-intersections",
        type=float,
        nargs="+",
        default=[0, 1, 5, 10, 20],
    )
    parser.add_argument(
        "--max-future-risk-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 1, 5, 20, 999],
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError("No rows found")
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
    counts = Counter(categories)
    results = [evaluate_rule(rows, categories, rule) for rule in grid(args)]
    results.sort(
        key=lambda row: (
            int(row["accepted_bad"]),
            -int(row["accepted_good"]),
            -float(row["good_recall"]),
            int(row["accepted_neutral"]),
        )
    )
    zero_bad = [row for row in results if int(row["accepted_bad"]) == 0]
    print(
        "Summary: "
        f"rows={len(rows)} good={counts['good']} neutral={counts['neutral']} "
        f"bad={counts['bad']} rules={len(results)} zero_bad_rules={len(zero_bad)}"
    )
    print_results(results, args.top_k)
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
