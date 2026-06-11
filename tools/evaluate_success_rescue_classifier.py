#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tools.train_counterfactual_gate import (
    choose_feature_columns,
    feature_matrix,
    filter_feature_columns,
    outcome_category,
    read_rows,
    safe_float,
    seed_split,
    standardize,
)


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def row_arrays(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    reward_delta[~np.isfinite(reward_delta)] = 0.0
    success_delta[~np.isfinite(success_delta)] = 0.0
    failed_delta[~np.isfinite(failed_delta)] = 0.0
    success_labels = (success_delta > 1e-9).astype(np.float32)
    reward_negative = reward_delta < -args.reward_epsilon
    if args.reward_negative_unsafe_mode == "non_success":
        reward_negative = reward_negative & (success_delta <= 1e-9)
    unsafe_labels = (
        (success_delta < -1e-9)
        | reward_negative
        | (failed_delta > 0.0)
    ).astype(np.float32)
    return reward_delta, success_delta, failed_delta, success_labels, unsafe_labels


def train_binary_member(
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    positive_weight: float,
    negative_weight: float,
    args: argparse.Namespace,
) -> BinaryMLP:
    rng = np.random.default_rng(seed)
    if args.bootstrap:
        sample_indices = rng.integers(0, len(train_x), size=len(train_x))
    else:
        sample_indices = np.arange(len(train_x))

    x = torch.as_tensor(train_x[sample_indices], dtype=torch.float32)
    y = torch.as_tensor(train_y[sample_indices], dtype=torch.float32)
    weights = torch.where(
        y > 0.5,
        torch.full_like(y, float(positive_weight)),
        torch.full_like(y, float(negative_weight)),
    )

    torch.manual_seed(seed)
    model = BinaryMLP(input_dim=train_x.shape[1], hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = (loss_fn(logits, y) * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_ensemble(models: list[BinaryMLP], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def train_ensembles(
    split_seed: int,
    train_x: np.ndarray,
    train_success: np.ndarray,
    train_unsafe: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[BinaryMLP], list[BinaryMLP]]:
    success_models = []
    unsafe_models = []
    for member in range(args.ensemble_size):
        base_seed = args.torch_seed + split_seed * 1000 + member
        success_models.append(
            train_binary_member(
                seed=base_seed,
                train_x=train_x,
                train_y=train_success,
                positive_weight=args.success_positive_weight,
                negative_weight=args.success_negative_weight,
                args=args,
            )
        )
        unsafe_models.append(
            train_binary_member(
                seed=base_seed + 100_000,
                train_x=train_x,
                train_y=train_unsafe,
                positive_weight=args.unsafe_positive_weight,
                negative_weight=args.unsafe_negative_weight,
                args=args,
            )
        )
    return success_models, unsafe_models


def metric_row(
    categories: np.ndarray,
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    accepted: np.ndarray,
    min_success_probability: float,
    max_unsafe_probability: float,
) -> dict[str, Any]:
    success_positive = success_delta > 1e-9
    success_negative = success_delta < -1e-9
    reward_positive = reward_delta > 1e-6
    reward_negative = reward_delta < -1e-6
    good = categories == "good"
    neutral = categories == "neutral"
    bad = categories == "bad"
    return {
        "min_success_probability": float(min_success_probability),
        "max_unsafe_probability": float(max_unsafe_probability),
        "accepted": int(accepted.sum()),
        "accepted_good": int(np.logical_and(accepted, good).sum()),
        "accepted_neutral": int(np.logical_and(accepted, neutral).sum()),
        "accepted_bad": int(np.logical_and(accepted, bad).sum()),
        "accepted_success_positive": int(
            np.logical_and(accepted, success_positive).sum()
        ),
        "accepted_success_negative": int(
            np.logical_and(accepted, success_negative).sum()
        ),
        "accepted_reward_positive": int(np.logical_and(accepted, reward_positive).sum()),
        "accepted_reward_negative": int(np.logical_and(accepted, reward_negative).sum()),
        "rejected_success_positive": int(
            np.logical_and(~accepted, success_positive).sum()
        ),
        "rejected_success_negative": int(
            np.logical_and(~accepted, success_negative).sum()
        ),
        "rejected_bad": int(np.logical_and(~accepted, bad).sum()),
        "success_positive_recall": (
            float(np.logical_and(accepted, success_positive).sum())
            / float(success_positive.sum())
            if success_positive.sum()
            else 0.0
        ),
        "success_negative_leak_rate": (
            float(np.logical_and(accepted, success_negative).sum())
            / float(success_negative.sum())
            if success_negative.sum()
            else 0.0
        ),
        "bad_leak_rate": (
            float(np.logical_and(accepted, bad).sum()) / float(bad.sum())
            if bad.sum()
            else 0.0
        ),
        "accepted_reward_delta_sum": float(reward_delta[accepted].sum()),
        "accepted_success_delta_sum": float(success_delta[accepted].sum()),
        "accepted_failed_delta_sum": float(failed_delta[accepted].sum()),
    }


def evaluate_split(
    split_seed: int,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_x_raw = feature_matrix(train_rows, feature_columns)
    validation_x_raw = feature_matrix(validation_rows, feature_columns)
    (
        _train_reward_delta,
        _train_success_delta,
        _train_failed_delta,
        train_success_labels,
        train_unsafe_labels,
    ) = row_arrays(train_rows, args)
    (
        validation_reward_delta,
        validation_success_delta,
        validation_failed_delta,
        _validation_success_labels,
        _validation_unsafe_labels,
    ) = row_arrays(validation_rows, args)
    validation_categories = np.asarray(
        [outcome_category(row, args.reward_epsilon) for row in validation_rows],
        dtype=object,
    )

    _, mean, std = standardize(train_x_raw, train_x_raw)
    train_x = (train_x_raw - mean) / std
    validation_x = (validation_x_raw - mean) / std
    success_models, unsafe_models = train_ensembles(
        split_seed=split_seed,
        train_x=train_x,
        train_success=train_success_labels,
        train_unsafe=train_unsafe_labels,
        args=args,
    )
    success_mean, success_std = predict_ensemble(success_models, validation_x)
    unsafe_mean, unsafe_std = predict_ensemble(unsafe_models, validation_x)
    success_lcb = success_mean - args.success_std_coef * success_std
    unsafe_ucb = unsafe_mean + args.unsafe_std_coef * unsafe_std

    metrics = []
    audit_rows = []
    for min_success_probability in args.min_success_probability:
        for max_unsafe_probability in args.max_unsafe_probability:
            accepted = (success_lcb >= min_success_probability) & (
                unsafe_ucb <= max_unsafe_probability
            )
            row = metric_row(
                categories=validation_categories,
                reward_delta=validation_reward_delta,
                success_delta=validation_success_delta,
                failed_delta=validation_failed_delta,
                accepted=accepted,
                min_success_probability=min_success_probability,
                max_unsafe_probability=max_unsafe_probability,
            )
            row["split_seed"] = split_seed
            row["val_rows"] = len(validation_rows)
            metrics.append(row)
    if args.output_audit_csv:
        for row_index, row in enumerate(validation_rows):
            base = {
                "split_seed": split_seed,
                "row_index": row_index,
                "seed": row.get("seed", ""),
                "prefix_len": row.get("prefix_len", ""),
                "forced_applied": row.get("forced_applied", ""),
                "outcome_category": validation_categories[row_index],
                "reward_delta": float(validation_reward_delta[row_index]),
                "success_delta": float(validation_success_delta[row_index]),
                "failed_agents_delta": float(validation_failed_delta[row_index]),
                "success_probability_mean": float(success_mean[row_index]),
                "success_probability_std": float(success_std[row_index]),
                "success_probability_lcb": float(success_lcb[row_index]),
                "unsafe_probability_mean": float(unsafe_mean[row_index]),
                "unsafe_probability_std": float(unsafe_std[row_index]),
                "unsafe_probability_ucb": float(unsafe_ucb[row_index]),
                "events": row.get("events", ""),
            }
            for min_success_probability in args.min_success_probability:
                for max_unsafe_probability in args.max_unsafe_probability:
                    accepted = int(
                        success_lcb[row_index] >= min_success_probability
                        and unsafe_ucb[row_index] <= max_unsafe_probability
                    )
                    audit_rows.append(
                        {
                            **base,
                            "min_success_probability": min_success_probability,
                            "max_unsafe_probability": max_unsafe_probability,
                            "accepted": accepted,
                        }
                    )
    return metrics, audit_rows


def aggregate_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summed_keys = [
        "accepted",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "accepted_success_negative",
        "accepted_reward_positive",
        "accepted_reward_negative",
        "rejected_success_positive",
        "rejected_success_negative",
        "rejected_bad",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
        "val_rows",
    ]
    buckets: dict[tuple[float, float], dict[str, Any]] = {}
    for row in results:
        key = (row["min_success_probability"], row["max_unsafe_probability"])
        bucket = buckets.setdefault(
            key,
            {
                "min_success_probability": row["min_success_probability"],
                "max_unsafe_probability": row["max_unsafe_probability"],
                "splits": 0,
            },
        )
        bucket["splits"] += 1
        for item in summed_keys:
            bucket[item] = bucket.get(item, 0) + row[item]
    aggregate = []
    for bucket in buckets.values():
        total_success_positive = (
            bucket["accepted_success_positive"]
            + bucket["rejected_success_positive"]
        )
        total_success_negative = (
            bucket["accepted_success_negative"]
            + bucket["rejected_success_negative"]
        )
        total_bad = bucket["accepted_bad"] + bucket["rejected_bad"]
        bucket["success_positive_recall"] = (
            bucket["accepted_success_positive"] / total_success_positive
            if total_success_positive
            else 0.0
        )
        bucket["success_negative_leak_rate"] = (
            bucket["accepted_success_negative"] / total_success_negative
            if total_success_negative
            else 0.0
        )
        bucket["bad_leak_rate"] = (
            bucket["accepted_bad"] / total_bad if total_bad else 0.0
        )
        aggregate.append(bucket)
    aggregate.sort(
        key=lambda row: (
            int(row["accepted_success_negative"]),
            int(row["accepted_bad"]),
            -int(row["accepted_success_positive"]),
            int(row["accepted_reward_negative"]),
            -float(row["accepted_success_delta_sum"]),
        )
    )
    return aggregate


def format_float(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6g}"


def print_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "min_success_probability",
        "max_unsafe_probability",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "accepted_success_negative",
        "accepted_reward_positive",
        "accepted_reward_negative",
        "success_positive_recall",
        "success_negative_leak_rate",
        "bad_leak_rate",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
    ]
    print(",".join(columns))
    for row in metrics[:top_k]:
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
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a dedicated Success-rescue classifier plus unsafe "
            "classifier on counterfactual prefix rows."
        )
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--validation-csv", nargs="+", type=Path)
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-prefix-features", action="store_true")
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=31)
    parser.add_argument("--success-positive-weight", type=float, default=40.0)
    parser.add_argument("--success-negative-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-positive-weight", type=float, default=12.0)
    parser.add_argument("--unsafe-negative-weight", type=float, default=1.0)
    parser.add_argument("--success-std-coef", type=float, default=1.0)
    parser.add_argument("--unsafe-std-coef", type=float, default=1.0)
    parser.add_argument(
        "--reward-negative-unsafe-mode",
        choices=["all", "non_success"],
        default="all",
        help=(
            "Whether reward-negative rows are always unsafe or only unsafe "
            "when they are not Success-positive. Use non_success for a pure "
            "Success-rescue head that may tolerate modest reward loss."
        ),
    )
    parser.add_argument(
        "--min-success-probability",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--max-unsafe-probability",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.1],
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-audit-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_rows(args.csv)
    validation_rows = read_rows(args.validation_csv) if args.validation_csv else []
    rows_for_features = train_rows + validation_rows
    if not train_rows:
        raise ValueError("No rows found")
    feature_columns = filter_feature_columns(choose_feature_columns(rows_for_features), args)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    split_results = []
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
                row for row, is_validation in zip(train_rows, validation_mask) if is_validation
            ]
        metrics, split_audit = evaluate_split(
            split_seed=split_seed,
            train_rows=split_train_rows,
            validation_rows=split_validation_rows,
            feature_columns=feature_columns,
            args=args,
        )
        split_results.extend(metrics)
        audit_rows.extend(split_audit)

    aggregate = aggregate_metrics(split_results)
    categories = Counter(outcome_category(row, args.reward_epsilon) for row in train_rows)
    _, _, _, success_labels, unsafe_labels = row_arrays(train_rows, args)
    success_rows = int(success_labels.sum())
    unsafe_rows = int(unsafe_labels.sum())
    print(
        "Data: "
        f"rows={len(train_rows) + len(validation_rows)} train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} good={categories['good']} "
        f"neutral={categories['neutral']} bad={categories['bad']} "
        f"success_positive_rows={success_rows} unsafe_rows={unsafe_rows} "
        f"features={len(feature_columns)} splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print_metrics(aggregate, args.top_k)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "rows": len(train_rows) + len(validation_rows),
                        "train_rows": len(train_rows),
                        "validation_rows": len(validation_rows),
                        "categories": dict(categories),
                        "success_positive_rows": success_rows,
                        "unsafe_rows": unsafe_rows,
                        "features": len(feature_columns),
                        "feature_columns": feature_columns,
                        "args": vars(args),
                    },
                    "aggregate_metrics": aggregate,
                    "split_metrics": split_results,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    if args.output_audit_csv:
        write_csv(args.output_audit_csv, audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
