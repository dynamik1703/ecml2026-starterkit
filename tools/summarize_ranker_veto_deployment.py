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


def join_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("split_seed", "")),
        str(row.get("seed", "")),
        str(row.get("prefix_len", "")),
        str(row.get("forced_applied", "")),
        str(row.get("events", "")),
        round(safe_float(row.get("reward_delta")), 9),
        round(safe_float(row.get("success_delta")), 9),
    )


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("seed", "")),
        str(row.get("prefix_len", "")),
        str(row.get("forced_applied", "")),
        str(row.get("events", "")),
        round(safe_float(row.get("reward_delta")), 9),
        round(safe_float(row.get("success_delta")), 9),
    )


def read_risk_rows(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = join_key(row)
            current = rows_by_key.get(key)
            if current is None or risk_probability_ucb(row) > risk_probability_ucb(current):
                rows_by_key[key] = row
    return rows_by_key


def risk_probability_ucb(row: dict[str, Any]) -> float:
    if "risk_probability_ucb" in row:
        return safe_float(row.get("risk_probability_ucb"))
    return safe_float(row.get("unsafe_probability_ucb"))


def read_ranker_rows(path: Path) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("split_seed", "")), str(row.get("row_index", "")))
            rows_by_key.setdefault(key, row)
    return list(rows_by_key.values())


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


def summarize(
    ranker_rows: list[dict[str, Any]],
    risk_rows_by_key: dict[tuple[Any, ...], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows = [
        row
        for row in ranker_rows
        if safe_int(row.get("selected_best_for_seed")) == 1
        and safe_int(row.get("is_baseline_candidate")) == 0
    ]
    split_metrics: list[dict[str, Any]] = []
    consensus_metrics: list[dict[str, Any]] = []

    for score_threshold in args.ranker_score_threshold:
        score_rows = [
            row
            for row in selected_rows
            if safe_float(row.get("rank_score_lcb")) >= score_threshold
        ]
        rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in score_rows:
            rows_by_split[str(row.get("split_seed", ""))].append(row)

        for max_risk_probability in args.veto_max_risk_probability:
            accepted_by_candidate: dict[tuple[Any, ...], dict[str, Any]] = {}
            accepted_splits_by_candidate: dict[tuple[Any, ...], set[str]] = defaultdict(set)

            for split_seed, split_rows in rows_by_split.items():
                accepted_rows = []
                missing_risk_rows = 0
                for row in split_rows:
                    risk_row = risk_rows_by_key.get(join_key(row))
                    if risk_row is None:
                        missing_risk_rows += 1
                        continue
                    if risk_probability_ucb(risk_row) <= max_risk_probability:
                        accepted_rows.append(row)
                        key = candidate_key(row)
                        accepted_by_candidate.setdefault(key, row)
                        accepted_splits_by_candidate[key].add(split_seed)

                metric = accepted_metric(accepted_rows)
                metric.update(
                    {
                        "split_seed": split_seed,
                        "ranker_score_threshold": float(score_threshold),
                        "veto_max_risk_probability": float(max_risk_probability),
                        "ranker_selected": len(split_rows),
                        "missing_risk_rows": missing_risk_rows,
                        "vetoed": len(split_rows) - len(accepted_rows) - missing_risk_rows,
                    }
                )
                split_metrics.append(metric)

            for min_splits in args.consensus_min_splits:
                consensus_rows = [
                    accepted_by_candidate[key]
                    for key, split_set in accepted_splits_by_candidate.items()
                    if len(split_set) >= min_splits
                ]
                metric = accepted_metric(consensus_rows)
                metric.update(
                    {
                        "ranker_score_threshold": float(score_threshold),
                        "veto_max_risk_probability": float(max_risk_probability),
                        "consensus_min_splits": int(min_splits),
                        "accepted_unique_candidates": len(consensus_rows),
                    }
                )
                consensus_metrics.append(metric)

    split_metrics.sort(
        key=lambda row: (
            int(row["accepted_success_negative"]),
            int(row["accepted_bad"]),
            -float(row["accepted_success_delta_sum"]),
            -float(row["accepted_reward_delta_sum"]),
            -int(row["accepted_success_positive"]),
        )
    )
    consensus_metrics.sort(
        key=lambda row: (
            int(row["accepted_success_negative"]),
            int(row["accepted_bad"]),
            -float(row["accepted_success_delta_sum"]),
            -float(row["accepted_reward_delta_sum"]),
            -int(row["accepted_success_positive"]),
        )
    )
    return split_metrics, consensus_metrics


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str], top_k: int) -> None:
    print(title)
    print(",".join(columns))
    for row in rows[:top_k]:
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
            "Summarize a ranker+veto audit as deployment-style per-model metrics "
            "instead of cross-split aggregate rescue counts."
        )
    )
    parser.add_argument("--ranker-audit-csv", required=True, type=Path)
    parser.add_argument("--risk-audit-csv", required=True, type=Path)
    parser.add_argument(
        "--ranker-score-threshold",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--veto-max-risk-probability",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--consensus-min-splits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ranker_rows = read_ranker_rows(args.ranker_audit_csv)
    risk_rows_by_key = read_risk_rows(args.risk_audit_csv)
    split_metrics, consensus_metrics = summarize(ranker_rows, risk_rows_by_key, args)

    print(
        "Data: "
        f"ranker_rows={len(ranker_rows)} risk_rows={len(risk_rows_by_key)} "
        f"splits={len({str(row.get('split_seed', '')) for row in ranker_rows})}"
    )
    print_table(
        "Per-split deployment metrics:",
        split_metrics,
        [
            "split_seed",
            "ranker_score_threshold",
            "veto_max_risk_probability",
            "ranker_selected",
            "vetoed",
            "missing_risk_rows",
            "accepted_good",
            "accepted_neutral",
            "accepted_bad",
            "accepted_success_positive",
            "accepted_success_negative",
            "accepted_reward_delta_sum",
            "accepted_success_delta_sum",
            "accepted_failed_delta_sum",
        ],
        args.top_k,
    )
    print_table(
        "Consensus unique-candidate metrics:",
        consensus_metrics,
        [
            "ranker_score_threshold",
            "veto_max_risk_probability",
            "consensus_min_splits",
            "accepted_unique_candidates",
            "accepted_good",
            "accepted_neutral",
            "accepted_bad",
            "accepted_success_positive",
            "accepted_success_negative",
            "accepted_reward_delta_sum",
            "accepted_success_delta_sum",
            "accepted_failed_delta_sum",
        ],
        args.top_k,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "ranker_rows": len(ranker_rows),
                        "risk_rows": len(risk_rows_by_key),
                        "ranker_audit_csv": str(args.ranker_audit_csv),
                        "risk_audit_csv": str(args.risk_audit_csv),
                        "args": vars(args),
                    },
                    "split_metrics": split_metrics,
                    "consensus_metrics": consensus_metrics,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
