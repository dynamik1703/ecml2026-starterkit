#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_success_rescue_classifier import format_float, json_default, metric_row


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_planner_rows(path: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("split_seed", "")), str(row.get("row_index", "")))
            rows_by_key.setdefault(key, row)
    return list(rows_by_key.values())


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("seed", "")),
        str(row.get("prefix_len", "")),
        str(row.get("forced_applied", "")),
        str(row.get("events", "")),
        round(safe_float(row.get("reward_delta")), 9),
        round(safe_float(row.get("success_delta")), 9),
    )


def accepted_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = np.asarray([row.get("outcome_category", "") for row in rows], dtype=object)
    reward_delta = np.asarray(
        [safe_float(row.get("reward_delta")) for row in rows],
        dtype=np.float32,
    )
    success_delta = np.asarray(
        [safe_float(row.get("success_delta")) for row in rows],
        dtype=np.float32,
    )
    failed_delta = np.asarray(
        [safe_float(row.get("failed_agents_delta")) for row in rows],
        dtype=np.float32,
    )
    return metric_row(
        categories=categories,
        reward_delta=reward_delta,
        success_delta=success_delta,
        failed_delta=failed_delta,
        accepted=np.ones(len(rows), dtype=bool),
        min_success_probability=0.0,
        max_unsafe_probability=0.0,
    )


def planner_score(row: dict[str, Any], unsafe_weight: float) -> float:
    return safe_float(row.get("success_probability_lcb")) - unsafe_weight * safe_float(
        row.get("unsafe_probability_ucb")
    )


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for unsafe_weight in args.planner_unsafe_weight:
        for score_threshold in args.planner_score_threshold:
            accepted_by_candidate: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                score = planner_score(row, unsafe_weight)
                if score < score_threshold:
                    continue
                candidate = dict(row)
                candidate["planner_score"] = score
                accepted_by_candidate[candidate_key(row)].append(candidate)

            for min_splits in args.consensus_min_splits:
                candidates: list[dict[str, Any]] = []
                for candidate_rows in accepted_by_candidate.values():
                    split_set = {str(row.get("split_seed", "")) for row in candidate_rows}
                    if len(split_set) < min_splits:
                        continue
                    representative = dict(candidate_rows[0])
                    representative["consensus_splits"] = len(split_set)
                    representative["planner_score_mean"] = sum(
                        safe_float(row.get("planner_score")) for row in candidate_rows
                    ) / len(candidate_rows)
                    candidates.append(representative)

                best_by_seed: dict[str, dict[str, Any]] = {}
                for candidate in candidates:
                    seed = str(candidate.get("seed", ""))
                    current = best_by_seed.get(seed)
                    candidate_rank = (
                        safe_float(candidate.get("planner_score_mean")),
                        safe_float(candidate.get("success_delta")),
                        safe_float(candidate.get("reward_delta")),
                        -safe_float(candidate.get("prefix_len")),
                    )
                    current_rank = (
                        safe_float(current.get("planner_score_mean")),
                        safe_float(current.get("success_delta")),
                        safe_float(current.get("reward_delta")),
                        -safe_float(current.get("prefix_len")),
                    ) if current is not None else None
                    if current is None or candidate_rank > current_rank:
                        best_by_seed[seed] = candidate

                accepted_rows = list(best_by_seed.values())
                metric = accepted_metric(accepted_rows)
                metric.update(
                    {
                        "planner_unsafe_weight": float(unsafe_weight),
                        "planner_score_threshold": float(score_threshold),
                        "consensus_min_splits": int(min_splits),
                        "accepted_unique_seeds": len(accepted_rows),
                        "accepted_unique_candidates": len(candidates),
                    }
                )
                metrics.append(metric)

    metrics.sort(
        key=lambda row: (
            int(row["accepted_success_negative"]),
            int(row["accepted_bad"]),
            -float(row["accepted_success_delta_sum"]),
            -float(row["accepted_reward_delta_sum"]),
            -int(row["accepted_success_positive"]),
        )
    )
    return metrics


def print_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "planner_unsafe_weight",
        "planner_score_threshold",
        "consensus_min_splits",
        "accepted_unique_seeds",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "accepted_success_negative",
        "accepted_reward_positive",
        "accepted_reward_negative",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
    ]
    print(",".join(columns))
    for row in metrics[:top_k]:
        print(
            ",".join(
                format_float(float(row[column]))
                if isinstance(row.get(column), float)
                else str(row.get(column, ""))
                for column in columns
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize prefix-planner audit rows as deployment-style decisions: "
            "score candidates per split, require split consensus, and keep at most "
            "one prefix per seed."
        )
    )
    parser.add_argument("--planner-audit-csv", required=True, type=Path)
    parser.add_argument(
        "--planner-unsafe-weight",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--planner-score-threshold",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--consensus-min-splits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_planner_rows(args.planner_audit_csv)
    metrics = summarize(rows, args)
    print(f"Data: planner_rows={len(rows)}")
    print_metrics(metrics, args.top_k)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "planner_rows": len(rows),
                        "planner_audit_csv": str(args.planner_audit_csv),
                        "args": vars(args),
                    },
                    "metrics": metrics,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
