#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.train_counterfactual_gate import outcome_category, safe_float


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def values(rows: list[dict[str, str]], column: str, default: float = 0.0) -> np.ndarray:
    result = []
    for row in rows:
        value = safe_float(row.get(column))
        result.append(value if math.isfinite(value) else default)
    return np.asarray(result, dtype=np.float64)


def feature_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    columns = [
        "baseline_prefix_conflict_agents",
        "baseline_future_head_on_risk",
        "forced_prefix_conflict_agents",
        "forced_prefix_conflict_agents_delta",
        "forced_prefix_min_intersection_eta_gap_delta",
        "forced_prefix_min_head_on_eta_gap_delta",
        "forced_prefix_head_on_edge_conflicts_delta",
        "forced_prefix_opposing_direction_intersections_delta",
        "future_head_on_risk_delta",
        "policy_candidate_minus_baseline_logit",
        "forced_distance_delta",
    ]
    return {column: values(rows, column) for column in columns}


def grid(args: argparse.Namespace) -> list[dict[str, float]]:
    rules = []
    for items in itertools.product(
        args.min_baseline_conflict_agents,
        args.min_baseline_future_risk,
        args.max_forced_conflict_agents,
        args.max_conflict_delta,
        args.min_intersection_eta_gap_delta,
        args.min_head_on_eta_gap_delta,
        args.max_head_on_edge_delta,
        args.max_opposing_intersections_delta,
        args.max_future_risk_delta,
        args.min_logit_delta,
        args.max_forced_distance_delta,
    ):
        rules.append(
            {
                "min_baseline_conflict_agents": items[0],
                "min_baseline_future_risk": items[1],
                "max_forced_conflict_agents": items[2],
                "max_conflict_delta": items[3],
                "min_intersection_eta_gap_delta": items[4],
                "min_head_on_eta_gap_delta": items[5],
                "max_head_on_edge_delta": items[6],
                "max_opposing_intersections_delta": items[7],
                "max_future_risk_delta": items[8],
                "min_logit_delta": items[9],
                "max_forced_distance_delta": items[10],
            }
        )
    return rules


def accepted_mask(features: dict[str, np.ndarray], rule: dict[str, float]) -> np.ndarray:
    return (
        (
            features["baseline_prefix_conflict_agents"]
            >= rule["min_baseline_conflict_agents"]
        )
        & (features["baseline_future_head_on_risk"] >= rule["min_baseline_future_risk"])
        & (features["forced_prefix_conflict_agents"] <= rule["max_forced_conflict_agents"])
        & (features["forced_prefix_conflict_agents_delta"] <= rule["max_conflict_delta"])
        & (
            features["forced_prefix_min_intersection_eta_gap_delta"]
            >= rule["min_intersection_eta_gap_delta"]
        )
        & (
            features["forced_prefix_min_head_on_eta_gap_delta"]
            >= rule["min_head_on_eta_gap_delta"]
        )
        & (
            features["forced_prefix_head_on_edge_conflicts_delta"]
            <= rule["max_head_on_edge_delta"]
        )
        & (
            features["forced_prefix_opposing_direction_intersections_delta"]
            <= rule["max_opposing_intersections_delta"]
        )
        & (features["future_head_on_risk_delta"] <= rule["max_future_risk_delta"])
        & (
            features["policy_candidate_minus_baseline_logit"]
            >= rule["min_logit_delta"]
        )
        & (features["forced_distance_delta"] <= rule["max_forced_distance_delta"])
    )


def evaluate_rule(
    categories: np.ndarray,
    features: dict[str, np.ndarray],
    rule: dict[str, float],
) -> dict[str, Any]:
    accepted = accepted_mask(features, rule)
    good = categories == "good"
    neutral = categories == "neutral"
    bad = categories == "bad"
    accepted_good = int(np.logical_and(accepted, good).sum())
    accepted_neutral = int(np.logical_and(accepted, neutral).sum())
    accepted_bad = int(np.logical_and(accepted, bad).sum())
    total_good = int(good.sum())
    total_bad = int(bad.sum())
    return {
        **rule,
        "accepted": int(accepted.sum()),
        "accepted_good": accepted_good,
        "accepted_neutral": accepted_neutral,
        "accepted_bad": accepted_bad,
        "rejected_good": total_good - accepted_good,
        "rejected_neutral": int(neutral.sum()) - accepted_neutral,
        "rejected_bad": total_bad - accepted_bad,
        "good_recall": accepted_good / total_good if total_good else 0.0,
        "bad_leak_rate": accepted_bad / total_bad if total_bad else 0.0,
    }


def sort_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
    return (
        int(row["accepted_bad"]),
        -int(row["accepted_good"]),
        -float(row["good_recall"]),
        int(row["accepted_neutral"]),
    )


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
        "min_baseline_conflict_agents",
        "min_baseline_future_risk",
        "max_forced_conflict_agents",
        "max_conflict_delta",
        "min_intersection_eta_gap_delta",
        "min_head_on_eta_gap_delta",
        "max_head_on_edge_delta",
        "max_opposing_intersections_delta",
        "max_future_risk_delta",
        "min_logit_delta",
        "max_forced_distance_delta",
    ]
    print(",".join(columns))
    for row in results[:top_k]:
        print(
            ",".join(
                format_float(float(row[column]))
                if isinstance(row[column], float)
                else str(row[column])
                for column in columns
            )
        )


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
        description="Evaluate monotone rescue rules on counterfactual labels."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--min-baseline-conflict-agents",
        type=float,
        nargs="+",
        default=[0, 1],
    )
    parser.add_argument(
        "--min-baseline-future-risk",
        type=float,
        nargs="+",
        default=[0, 5],
    )
    parser.add_argument(
        "--max-forced-conflict-agents",
        type=float,
        nargs="+",
        default=[1, 2, 999],
    )
    parser.add_argument(
        "--max-conflict-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
    )
    parser.add_argument(
        "--min-intersection-eta-gap-delta",
        type=float,
        nargs="+",
        default=[-999, 0, 50],
    )
    parser.add_argument(
        "--min-head-on-eta-gap-delta",
        type=float,
        nargs="+",
        default=[-999, 0, 50],
    )
    parser.add_argument(
        "--max-head-on-edge-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
    )
    parser.add_argument(
        "--max-opposing-intersections-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
    )
    parser.add_argument(
        "--max-future-risk-delta",
        type=float,
        nargs="+",
        default=[-1, 0, 999],
    )
    parser.add_argument(
        "--min-logit-delta",
        type=float,
        nargs="+",
        default=[-999, 0, 1],
    )
    parser.add_argument(
        "--max-forced-distance-delta",
        type=float,
        nargs="+",
        default=[0, 999],
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError("No rows found")
    category_list = [outcome_category(row, args.reward_epsilon) for row in rows]
    categories = np.asarray(category_list, dtype=object)
    counts = Counter(category_list)
    features = feature_arrays(rows)
    rules = grid(args)
    results = [evaluate_rule(categories, features, rule) for rule in rules]
    results.sort(key=sort_key)
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
