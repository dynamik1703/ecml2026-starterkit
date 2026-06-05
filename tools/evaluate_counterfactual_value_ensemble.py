#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


class ValueRiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            self.shared = nn.Identity()
            shared_dim = input_dim
        else:
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            shared_dim = hidden_size
        self.value_head = nn.Linear(shared_dim, 1)
        self.bad_head = nn.Linear(shared_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.value_head(hidden).squeeze(-1), self.bad_head(hidden).squeeze(-1)


def utility_target(row: dict[str, str], args: argparse.Namespace) -> float:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    if not np.isfinite(reward_delta):
        reward_delta = 0.0
    if not np.isfinite(success_delta):
        success_delta = 0.0
    if not np.isfinite(failed_delta):
        failed_delta = 0.0
    return float(
        reward_delta
        + args.success_weight * success_delta
        - args.failed_weight * max(0.0, failed_delta)
    )


def sample_weight(category: str, args: argparse.Namespace) -> float:
    if category == "good":
        return args.good_weight
    if category == "bad":
        return args.bad_weight
    return args.neutral_weight


def train_member(
    seed: int,
    train_x: np.ndarray,
    train_value: np.ndarray,
    train_bad: np.ndarray,
    train_categories: list[str],
    args: argparse.Namespace,
) -> ValueRiskMLP:
    rng = np.random.default_rng(seed)
    if args.bootstrap:
        sample_indices = rng.integers(0, len(train_x), size=len(train_x))
    else:
        sample_indices = np.arange(len(train_x))

    x = torch.as_tensor(train_x[sample_indices], dtype=torch.float32)
    value = torch.as_tensor(train_value[sample_indices], dtype=torch.float32)
    bad = torch.as_tensor(train_bad[sample_indices], dtype=torch.float32)
    weights = torch.as_tensor(
        [sample_weight(train_categories[index], args) for index in sample_indices],
        dtype=torch.float32,
    )

    torch.manual_seed(seed)
    model = ValueRiskMLP(input_dim=train_x.shape[1], hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    value_loss_fn = nn.SmoothL1Loss(reduction="none")
    bad_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        value_pred, bad_logits = model(x)
        value_loss = value_loss_fn(value_pred, value)
        bad_loss = bad_loss_fn(bad_logits, bad)
        loss = (
            (value_loss * weights).sum()
            + args.bad_loss_weight * (bad_loss * weights).sum()
        ) / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_ensemble(
    models: list[ValueRiskMLP],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    tensor = torch.as_tensor(x, dtype=torch.float32)
    values = []
    bad_probs = []
    with torch.no_grad():
        for model in models:
            value_pred, bad_logits = model(tensor)
            values.append(value_pred.cpu().numpy())
            bad_probs.append(torch.sigmoid(bad_logits).cpu().numpy())
    value_stack = np.stack(values, axis=0)
    bad_stack = np.stack(bad_probs, axis=0)
    return (
        value_stack.mean(axis=0),
        value_stack.std(axis=0),
        bad_stack.mean(axis=0),
        bad_stack.std(axis=0),
    )


def metric_row(
    categories: np.ndarray,
    utility: np.ndarray,
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    accepted: np.ndarray,
    min_utility: float,
    max_bad_probability: float,
) -> dict[str, Any]:
    good = categories == "good"
    neutral = categories == "neutral"
    bad = categories == "bad"
    accepted_good = int(np.logical_and(accepted, good).sum())
    accepted_neutral = int(np.logical_and(accepted, neutral).sum())
    accepted_bad = int(np.logical_and(accepted, bad).sum())
    total_good = int(good.sum())
    total_bad = int(bad.sum())
    return {
        "min_utility": float(min_utility),
        "max_bad_probability": float(max_bad_probability),
        "accepted": int(accepted.sum()),
        "accepted_good": accepted_good,
        "accepted_neutral": accepted_neutral,
        "accepted_bad": accepted_bad,
        "rejected_good": total_good - accepted_good,
        "rejected_neutral": int(neutral.sum()) - accepted_neutral,
        "rejected_bad": total_bad - accepted_bad,
        "good_recall": accepted_good / total_good if total_good else 0.0,
        "bad_leak_rate": accepted_bad / total_bad if total_bad else 0.0,
        "accepted_utility_sum": float(utility[accepted].sum()),
        "accepted_reward_delta_sum": float(reward_delta[accepted].sum()),
        "accepted_success_delta_sum": float(success_delta[accepted].sum()),
        "accepted_failed_delta_sum": float(failed_delta[accepted].sum()),
    }


def split_metrics(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    split_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    x_raw = feature_matrix(rows, feature_columns)
    utility = np.asarray([utility_target(row, args) for row in rows], dtype=np.float32)
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

    categories_list = [outcome_category(row, args.reward_epsilon) for row in rows]
    categories = np.asarray(categories_list, dtype=object)
    bad_labels = np.asarray([category == "bad" for category in categories_list], dtype=np.float32)
    train_mask, val_mask = seed_split(
        rows,
        args.val_fraction,
        split_seed,
        ordered=False,
    )
    x, _, _ = standardize(x_raw[train_mask], x_raw)
    train_categories = [
        category for category, is_train in zip(categories_list, train_mask) if is_train
    ]
    models = [
        train_member(
            seed=args.torch_seed + split_seed * 1000 + member,
            train_x=x[train_mask],
            train_value=utility[train_mask],
            train_bad=bad_labels[train_mask],
            train_categories=train_categories,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]
    value_mean, value_std, bad_mean, bad_std = predict_ensemble(models, x[val_mask])
    value_lcb = value_mean - args.value_std_coef * value_std
    bad_ucb = bad_mean + args.bad_std_coef * bad_std

    split_results = []
    for min_utility in args.min_utility:
        for max_bad_probability in args.max_bad_probability:
            accepted = (value_lcb >= min_utility) & (bad_ucb <= max_bad_probability)
            row = metric_row(
                categories=categories[val_mask],
                utility=utility[val_mask],
                reward_delta=reward_delta[val_mask],
                success_delta=success_delta[val_mask],
                failed_delta=failed_delta[val_mask],
                accepted=accepted,
                min_utility=min_utility,
                max_bad_probability=max_bad_probability,
            )
            row["split_seed"] = split_seed
            row["val_rows"] = int(val_mask.sum())
            row["val_seeds"] = sorted(
                {int(float(row["seed"])) for row, is_val in zip(rows, val_mask) if is_val}
            )
            split_results.append(row)
    return split_results


def row_arrays(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    utility = np.asarray([utility_target(row, args) for row in rows], dtype=np.float32)
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
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
    bad_labels = np.asarray([category == "bad" for category in categories], dtype=np.float32)
    return (
        utility,
        reward_delta,
        success_delta,
        failed_delta,
        np.asarray(categories, dtype=object),
        categories,
        bad_labels,
    )


def explicit_validation_metrics(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    feature_columns: list[str],
    split_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    train_x_raw = feature_matrix(train_rows, feature_columns)
    validation_x_raw = feature_matrix(validation_rows, feature_columns)
    train_utility, _, _, _, _, train_categories, train_bad_labels = row_arrays(
        train_rows,
        args,
    )
    (
        validation_utility,
        validation_reward_delta,
        validation_success_delta,
        validation_failed_delta,
        validation_categories,
        _,
        _,
    ) = row_arrays(validation_rows, args)

    _, mean, std = standardize(train_x_raw, train_x_raw)
    train_x = (train_x_raw - mean) / std
    validation_x = (validation_x_raw - mean) / std

    models = [
        train_member(
            seed=args.torch_seed + split_seed * 1000 + member,
            train_x=train_x,
            train_value=train_utility,
            train_bad=train_bad_labels,
            train_categories=train_categories,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]
    value_mean, value_std, bad_mean, bad_std = predict_ensemble(models, validation_x)
    value_lcb = value_mean - args.value_std_coef * value_std
    bad_ucb = bad_mean + args.bad_std_coef * bad_std

    split_results = []
    for min_utility in args.min_utility:
        for max_bad_probability in args.max_bad_probability:
            accepted = (value_lcb >= min_utility) & (bad_ucb <= max_bad_probability)
            row = metric_row(
                categories=validation_categories,
                utility=validation_utility,
                reward_delta=validation_reward_delta,
                success_delta=validation_success_delta,
                failed_delta=validation_failed_delta,
                accepted=accepted,
                min_utility=min_utility,
                max_bad_probability=max_bad_probability,
            )
            row["split_seed"] = split_seed
            row["val_rows"] = len(validation_rows)
            row["val_seeds"] = sorted({int(float(row["seed"])) for row in validation_rows})
            split_results.append(row)
    return split_results


def aggregate_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[float, float], dict[str, Any]] = {}
    summed_keys = [
        "accepted",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "rejected_good",
        "rejected_neutral",
        "rejected_bad",
        "accepted_utility_sum",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
        "val_rows",
    ]
    for row in results:
        key = (row["min_utility"], row["max_bad_probability"])
        bucket = buckets.setdefault(
            key,
            {
                "min_utility": row["min_utility"],
                "max_bad_probability": row["max_bad_probability"],
                "splits": 0,
            },
        )
        bucket["splits"] += 1
        for item in summed_keys:
            bucket[item] = bucket.get(item, 0) + row[item]

    aggregate = []
    for bucket in buckets.values():
        total_good = bucket["accepted_good"] + bucket["rejected_good"]
        total_bad = bucket["accepted_bad"] + bucket["rejected_bad"]
        bucket["good_recall"] = (
            bucket["accepted_good"] / total_good if total_good else 0.0
        )
        bucket["bad_leak_rate"] = (
            bucket["accepted_bad"] / total_bad if total_bad else 0.0
        )
        aggregate.append(bucket)
    aggregate.sort(
        key=lambda row: (
            int(row["accepted_bad"]),
            -int(row["accepted_good"]),
            -float(row["accepted_utility_sum"]),
            int(row["accepted_neutral"]),
        )
    )
    return aggregate


def format_float(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6g}"


def print_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "min_utility",
        "max_bad_probability",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "good_recall",
        "bad_leak_rate",
        "accepted_utility_sum",
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
            "Evaluate a conservative value/risk ensemble on counterfactual "
            "action rows."
        )
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--validation-csv",
        nargs="+",
        type=Path,
        help="Use these CSV rows as explicit validation rows instead of random seed splits.",
    )
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
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=17)
    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--failed-weight", type=float, default=0.75)
    parser.add_argument("--bad-loss-weight", type=float, default=1.5)
    parser.add_argument("--good-weight", type=float, default=6.0)
    parser.add_argument("--neutral-weight", type=float, default=0.5)
    parser.add_argument("--bad-weight", type=float, default=8.0)
    parser.add_argument("--value-std-coef", type=float, default=1.0)
    parser.add_argument("--bad-std-coef", type=float, default=1.0)
    parser.add_argument(
        "--min-utility",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05, 0.1],
    )
    parser.add_argument(
        "--max-bad-probability",
        nargs="+",
        type=float,
        default=[0.02, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_rows(args.csv)
    validation_rows = read_rows(args.validation_csv) if args.validation_csv else []
    rows = train_rows + validation_rows
    if not train_rows:
        raise ValueError("No rows found")
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
    counts = Counter(categories)
    feature_columns = filter_feature_columns(choose_feature_columns(rows), args)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    split_results = []
    for split_seed in args.split_seeds:
        if validation_rows:
            split_results.extend(
                explicit_validation_metrics(
                    train_rows,
                    validation_rows,
                    feature_columns,
                    split_seed,
                    args,
                )
            )
        else:
            split_results.extend(split_metrics(rows, feature_columns, split_seed, args))
    aggregate = aggregate_metrics(split_results)

    print(
        "Data: "
        f"rows={len(rows)} train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} "
        f"good={counts['good']} neutral={counts['neutral']} "
        f"bad={counts['bad']} features={len(feature_columns)} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print_metrics(aggregate, args.top_k)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "rows": len(rows),
                        "train_rows": len(train_rows),
                        "validation_rows": len(validation_rows),
                        "categories": dict(counts),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
