#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_sequence_rescue_planner import read_json_rows
from tools.evaluate_success_rescue_classifier import format_float, json_default, metric_row
from tools.train_counterfactual_gate import outcome_category, seed_split, standardize


IDENTITY_COLUMNS = {
    "event_agent_id",
    "event_seed",
    "row_index",
    "seed",
}

TARGET_COLUMNS = {
    "failed_agents_delta",
    "good_label",
    "risk_label",
    "reward_delta",
    "success_delta",
    "target_value",
}


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def finite_or_zero(value: Any) -> float:
    result = safe_float(value)
    return result if math.isfinite(result) else 0.0


def flatten_events(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        details = row.get("event_details") or []
        if not isinstance(details, list):
            continue
        reward_delta = finite_or_zero(row.get("reward_delta"))
        success_delta = finite_or_zero(row.get("success_delta"))
        failed_delta = finite_or_zero(row.get("failed_agents_delta"))
        target_value = (
            args.success_weight * success_delta
            + args.reward_weight * reward_delta
            - args.reward_loss_penalty * max(-reward_delta, 0.0)
            - args.success_loss_penalty * max(-success_delta, 0.0)
            - args.failure_penalty * max(failed_delta, 0.0)
        )
        good_label = float(
            success_delta > 1e-9
            or (
                reward_delta > args.reward_epsilon
                and success_delta >= -1e-9
                and failed_delta <= 0.0
            )
        )
        reward_loss = reward_delta < -args.reward_loss_epsilon
        if args.reward_loss_mode == "non_success":
            reward_loss = reward_loss and success_delta <= 1e-9
        risk_label = float(
            success_delta < -1e-9
            or failed_delta > 0.0
            or reward_loss
            or target_value < args.target_risk_threshold
        )
        category = outcome_category(
            {
                "reward_delta": str(reward_delta),
                "success_delta": str(success_delta),
                "failed_agents_delta": str(failed_delta),
            },
            args.reward_epsilon,
        )
        for event in details:
            if not isinstance(event, dict):
                continue
            record: dict[str, Any] = {
                "row_index": row_index,
                "seed": int(float(row.get("seed", event.get("seed", -1)))),
                "prefix_len": finite_or_zero(row.get("prefix_len", len(details))),
                "prefix_event_count": len(details),
                "reward_delta": reward_delta,
                "success_delta": success_delta,
                "failed_agents_delta": failed_delta,
                "target_value": target_value,
                "good_label": good_label,
                "risk_label": risk_label,
                "outcome_category": category,
            }
            for key, value in event.items():
                record[f"event_{key}"] = value
            events.append(record)
    return events


def choose_feature_columns(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    columns = sorted({key for row in rows for key in row})
    feature_columns = []
    for column in columns:
        if column in IDENTITY_COLUMNS or column in TARGET_COLUMNS:
            continue
        if column == "outcome_category":
            continue
        if args.drop_candidate_source_features and "candidate_source" in column:
            continue
        if args.drop_prefix_index_features and column in {
            "event_prefix_index",
            "prefix_event_count",
            "prefix_len",
        }:
            continue
        values = [safe_float(row.get(column)) for row in rows]
        if any(math.isfinite(value) for value in values):
            feature_columns.append(column)
    if args.include_feature_regex:
        include = re.compile(args.include_feature_regex)
        feature_columns = [column for column in feature_columns if include.search(column)]
    if args.exclude_feature_regex:
        exclude = re.compile(args.exclude_feature_regex)
        feature_columns = [
            column for column in feature_columns if not exclude.search(column)
        ]
    return feature_columns


def feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            matrix[row_index, column_index] = finite_or_zero(row.get(column))
    return matrix


def train_member(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    positive_weight: float,
    negative_weight: float,
    args: argparse.Namespace,
) -> BinaryMLP:
    rng = np.random.default_rng(seed)
    if args.bootstrap:
        indices = rng.integers(0, len(x), size=len(x))
    else:
        indices = np.arange(len(x))
    x_tensor = torch.as_tensor(x[indices], dtype=torch.float32)
    y_tensor = torch.as_tensor(y[indices], dtype=torch.float32)
    weights = torch.where(
        y_tensor > 0.5,
        torch.full_like(y_tensor, float(positive_weight)),
        torch.full_like(y_tensor, float(negative_weight)),
    )

    torch.manual_seed(seed)
    model = BinaryMLP(input_dim=x.shape[1], hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = (loss_fn(logits, y_tensor) * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_ensemble(
    split_seed: int,
    x: np.ndarray,
    good_y: np.ndarray,
    risk_y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[BinaryMLP], list[BinaryMLP]]:
    good_models = []
    risk_models = []
    for member in range(args.ensemble_size):
        seed = args.torch_seed + split_seed * 1000 + member
        good_models.append(
            train_member(
                seed,
                x,
                good_y,
                args.good_positive_weight,
                args.good_negative_weight,
                args,
            )
        )
        risk_models.append(
            train_member(
                seed + 100_000,
                x,
                risk_y,
                args.risk_positive_weight,
                args.risk_negative_weight,
                args,
            )
        )
    return good_models, risk_models


def predict(models: list[BinaryMLP], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    tensor = torch.as_tensor(x, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(torch.sigmoid(model(tensor)).cpu().numpy())
    stack = np.stack(probs, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("seed", "")),
        str(row.get("event_env_time", "")),
        str(row.get("event_agent_id", "")),
        str(row.get("event_candidate_action", "")),
        round(finite_or_zero(row.get("reward_delta")), 9),
        round(finite_or_zero(row.get("success_delta")), 9),
    )


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        by_key.setdefault(event_key(row), row)
    return list(by_key.values())


def accepted_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = np.asarray([row.get("outcome_category", "") for row in rows], dtype=object)
    reward_delta = np.asarray(
        [finite_or_zero(row.get("reward_delta")) for row in rows],
        dtype=np.float32,
    )
    success_delta = np.asarray(
        [finite_or_zero(row.get("success_delta")) for row in rows],
        dtype=np.float32,
    )
    failed_delta = np.asarray(
        [finite_or_zero(row.get("failed_agents_delta")) for row in rows],
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


def evaluate_split(
    split_seed: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_x_raw = feature_matrix(train_rows, feature_columns)
    val_x_raw = feature_matrix(validation_rows, feature_columns)
    _, mean, std = standardize(train_x_raw, train_x_raw)
    train_x = (train_x_raw - mean) / std
    val_x = (val_x_raw - mean) / std
    good_y = np.asarray([finite_or_zero(row.get("good_label")) for row in train_rows])
    risk_y = np.asarray([finite_or_zero(row.get("risk_label")) for row in train_rows])
    good_models, risk_models = train_ensemble(split_seed, train_x, good_y, risk_y, args)
    good_mean, good_std = predict(good_models, val_x)
    risk_mean, risk_std = predict(risk_models, val_x)
    good_lcb = good_mean - args.good_std_coef * good_std
    risk_ucb = risk_mean + args.risk_std_coef * risk_std

    metrics = []
    for min_good in args.min_good_probability:
        for max_risk in args.max_risk_probability:
            accepted_mask = np.logical_and(good_lcb >= min_good, risk_ucb <= max_risk)
            accepted_rows = [
                row for row, accepted in zip(validation_rows, accepted_mask) if accepted
            ]
            unique_accepted = unique_rows(accepted_rows)
            row = accepted_metric(accepted_rows)
            unique_row = accepted_metric(unique_accepted)
            row.update(
                {
                    "split_seed": split_seed,
                    "min_good_probability": float(min_good),
                    "max_risk_probability": float(max_risk),
                    "unique_accepted": int(unique_row["accepted"]),
                    "unique_accepted_good": int(unique_row["accepted_good"]),
                    "unique_accepted_neutral": int(unique_row["accepted_neutral"]),
                    "unique_accepted_bad": int(unique_row["accepted_bad"]),
                    "unique_accepted_success_positive": int(
                        unique_row["accepted_success_positive"]
                    ),
                    "unique_accepted_success_negative": int(
                        unique_row["accepted_success_negative"]
                    ),
                    "unique_accepted_reward_delta_sum": float(
                        unique_row["accepted_reward_delta_sum"]
                    ),
                    "unique_accepted_success_delta_sum": float(
                        unique_row["accepted_success_delta_sum"]
                    ),
                    "unique_accepted_failed_delta_sum": float(
                        unique_row["accepted_failed_delta_sum"]
                    ),
                }
            )
            metrics.append(row)

    audit_rows = []
    if args.output_audit_csv:
        for row, good_p, good_s, risk_p, risk_s in zip(
            validation_rows,
            good_mean,
            good_std,
            risk_mean,
            risk_std,
        ):
            audit_rows.append(
                {
                    "split_seed": split_seed,
                    "seed": row.get("seed", ""),
                    "event_env_time": row.get("event_env_time", ""),
                    "event_agent_id": row.get("event_agent_id", ""),
                    "event_baseline_action": row.get("event_baseline_action", ""),
                    "event_candidate_action": row.get("event_candidate_action", ""),
                    "event_candidate_action_name": row.get(
                        "event_candidate_action_name", ""
                    ),
                    "prefix_len": row.get("prefix_len", ""),
                    "event_prefix_index": row.get("event_prefix_index", ""),
                    "outcome_category": row.get("outcome_category", ""),
                    "reward_delta": row.get("reward_delta", 0.0),
                    "success_delta": row.get("success_delta", 0.0),
                    "failed_agents_delta": row.get("failed_agents_delta", 0.0),
                    "good_probability_mean": float(good_p),
                    "good_probability_std": float(good_s),
                    "good_probability_lcb": float(
                        good_p - args.good_std_coef * good_s
                    ),
                    "risk_probability_mean": float(risk_p),
                    "risk_probability_std": float(risk_s),
                    "risk_probability_ucb": float(
                        risk_p + args.risk_std_coef * risk_s
                    ),
                }
            )
    metrics.sort(
        key=lambda row: (
            int(row["unique_accepted_success_negative"]),
            int(row["unique_accepted_bad"]),
            -float(row["unique_accepted_success_delta_sum"]),
            -float(row["unique_accepted_reward_delta_sum"]),
            -int(row["unique_accepted_success_positive"]),
        )
    )
    return metrics, audit_rows


def aggregate_metrics(split_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[float, float], dict[str, Any]] = {}
    sum_keys = [
        "accepted",
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
        "unique_accepted",
        "unique_accepted_good",
        "unique_accepted_neutral",
        "unique_accepted_bad",
        "unique_accepted_success_positive",
        "unique_accepted_success_negative",
        "unique_accepted_reward_delta_sum",
        "unique_accepted_success_delta_sum",
        "unique_accepted_failed_delta_sum",
    ]
    for row in split_metrics:
        key = (row["min_good_probability"], row["max_risk_probability"])
        bucket = buckets.setdefault(
            key,
            {
                "min_good_probability": key[0],
                "max_risk_probability": key[1],
                "splits": 0,
            },
        )
        bucket["splits"] += 1
        for item in sum_keys:
            bucket[item] = bucket.get(item, 0) + row[item]
    aggregate = list(buckets.values())
    aggregate.sort(
        key=lambda row: (
            int(row["unique_accepted_success_negative"]),
            int(row["unique_accepted_bad"]),
            -float(row["unique_accepted_success_delta_sum"]),
            -float(row["unique_accepted_reward_delta_sum"]),
            -int(row["unique_accepted_success_positive"]),
        )
    )
    return aggregate


def print_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "min_good_probability",
        "max_risk_probability",
        "splits",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "accepted_success_negative",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "unique_accepted_good",
        "unique_accepted_neutral",
        "unique_accepted_bad",
        "unique_accepted_success_positive",
        "unique_accepted_success_negative",
        "unique_accepted_reward_delta_sum",
        "unique_accepted_success_delta_sum",
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
        description="Evaluate an event-level online action value/risk head prototype."
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--validation-json", nargs="+", type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--reward-loss-epsilon", type=float, default=1e-6)
    parser.add_argument("--reward-loss-mode", choices=["all", "non_success"], default="all")
    parser.add_argument("--target-risk-threshold", type=float, default=0.0)
    parser.add_argument("--success-weight", type=float, default=6.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-penalty", type=float, default=3.0)
    parser.add_argument("--success-loss-penalty", type=float, default=8.0)
    parser.add_argument("--failure-penalty", type=float, default=3.0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-candidate-source-features", action="store_true")
    parser.add_argument("--drop-prefix-index-features", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=313)
    parser.add_argument("--good-positive-weight", type=float, default=8.0)
    parser.add_argument("--good-negative-weight", type=float, default=1.0)
    parser.add_argument("--risk-positive-weight", type=float, default=16.0)
    parser.add_argument("--risk-negative-weight", type=float, default=1.0)
    parser.add_argument("--good-std-coef", type=float, default=0.5)
    parser.add_argument("--risk-std-coef", type=float, default=1.0)
    parser.add_argument(
        "--min-good-probability",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--max-risk-probability",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25],
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-audit-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_prefix_rows = read_json_rows(args.json)
    validation_prefix_rows = (
        read_json_rows(args.validation_json) if args.validation_json else []
    )
    train_rows = flatten_events(train_prefix_rows, args)
    validation_rows = flatten_events(validation_prefix_rows, args)
    all_rows = train_rows + validation_rows
    if not train_rows:
        raise ValueError("No event rows found")
    feature_columns = choose_feature_columns(all_rows, args)
    if not feature_columns:
        raise ValueError("No feature columns found")

    split_metrics = []
    audit_rows = []
    for split_seed in args.split_seeds:
        if validation_rows:
            split_train_rows = train_rows
            split_validation_rows = validation_rows
        else:
            train_mask, validation_mask = seed_split(
                train_rows,
                args.val_fraction,
                split_seed,
                ordered=False,
            )
            split_train_rows = [
                row for row, is_train in zip(train_rows, train_mask) if is_train
            ]
            split_validation_rows = [
                row
                for row, is_validation in zip(train_rows, validation_mask)
                if is_validation
            ]
        metrics, split_audit = evaluate_split(
            split_seed,
            split_train_rows,
            split_validation_rows,
            feature_columns,
            args,
        )
        split_metrics.extend(metrics)
        audit_rows.extend(split_audit)

    aggregate = aggregate_metrics(split_metrics)
    categories = Counter(row["outcome_category"] for row in train_rows)
    validation_categories = Counter(row["outcome_category"] for row in validation_rows)
    print(
        "Data: "
        f"train_events={len(train_rows)} validation_events={len(validation_rows)} "
        f"train_categories={dict(categories)} "
        f"validation_categories={dict(validation_categories)} "
        f"features={len(feature_columns)} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print_metrics(aggregate, args.top_k)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "train_events": len(train_rows),
                        "validation_events": len(validation_rows),
                        "train_categories": dict(categories),
                        "validation_categories": dict(validation_categories),
                        "feature_columns": feature_columns,
                        "features": len(feature_columns),
                        "args": vars(args),
                    },
                    "aggregate_metrics": aggregate,
                    "split_metrics": split_metrics,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    if args.output_audit_csv:
        args.output_audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_audit_csv.open("w", newline="") as handle:
            fieldnames = sorted({key for row in audit_rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
