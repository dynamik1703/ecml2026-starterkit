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
        self.success_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return (
            self.value_head(hidden).squeeze(-1),
            self.bad_head(hidden).squeeze(-1),
            self.success_head(hidden).squeeze(-1),
        )


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


def success_sample_weight(success_label: float, category: str, args: argparse.Namespace) -> float:
    if success_label > 0.5:
        return args.success_positive_weight
    if category == "bad":
        return args.success_bad_weight
    return args.success_negative_weight


def train_member(
    seed: int,
    train_x: np.ndarray,
    train_value: np.ndarray,
    train_bad: np.ndarray,
    train_success: np.ndarray,
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
    success = torch.as_tensor(train_success[sample_indices], dtype=torch.float32)
    weights = torch.as_tensor(
        [sample_weight(train_categories[index], args) for index in sample_indices],
        dtype=torch.float32,
    )
    success_weights = torch.as_tensor(
        [
            success_sample_weight(
                float(train_success[index]),
                train_categories[index],
                args,
            )
            for index in sample_indices
        ],
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
        value_pred, bad_logits, success_logits = model(x)
        value_loss = value_loss_fn(value_pred, value)
        bad_loss = bad_loss_fn(bad_logits, bad)
        success_loss = bad_loss_fn(success_logits, success)
        loss = (
            (value_loss * weights).sum()
            + args.bad_loss_weight * (bad_loss * weights).sum()
            + args.success_loss_weight * (success_loss * success_weights).sum()
        ) / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_ensemble(
    models: list[ValueRiskMLP],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    tensor = torch.as_tensor(x, dtype=torch.float32)
    values = []
    bad_probs = []
    success_probs = []
    with torch.no_grad():
        for model in models:
            value_pred, bad_logits, success_logits = model(tensor)
            values.append(value_pred.cpu().numpy())
            bad_probs.append(torch.sigmoid(bad_logits).cpu().numpy())
            success_probs.append(torch.sigmoid(success_logits).cpu().numpy())
    value_stack = np.stack(values, axis=0)
    bad_stack = np.stack(bad_probs, axis=0)
    success_stack = np.stack(success_probs, axis=0)
    return (
        value_stack.mean(axis=0),
        value_stack.std(axis=0),
        bad_stack.mean(axis=0),
        bad_stack.std(axis=0),
        success_stack.mean(axis=0),
        success_stack.std(axis=0),
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
    min_success_probability: float | None = None,
) -> dict[str, Any]:
    good = categories == "good"
    neutral = categories == "neutral"
    bad = categories == "bad"
    success_positive = success_delta > 1e-9
    accepted_good = int(np.logical_and(accepted, good).sum())
    accepted_neutral = int(np.logical_and(accepted, neutral).sum())
    accepted_bad = int(np.logical_and(accepted, bad).sum())
    accepted_success_positive = int(np.logical_and(accepted, success_positive).sum())
    total_good = int(good.sum())
    total_bad = int(bad.sum())
    total_success_positive = int(success_positive.sum())
    return {
        "min_utility": float(min_utility),
        "max_bad_probability": float(max_bad_probability),
        "min_success_probability": min_success_probability,
        "accepted": int(accepted.sum()),
        "accepted_good": accepted_good,
        "accepted_neutral": accepted_neutral,
        "accepted_bad": accepted_bad,
        "accepted_success_positive": accepted_success_positive,
        "rejected_good": total_good - accepted_good,
        "rejected_neutral": int(neutral.sum()) - accepted_neutral,
        "rejected_bad": total_bad - accepted_bad,
        "rejected_success_positive": total_success_positive - accepted_success_positive,
        "good_recall": accepted_good / total_good if total_good else 0.0,
        "bad_leak_rate": accepted_bad / total_bad if total_bad else 0.0,
        "success_positive_recall": (
            accepted_success_positive / total_success_positive
            if total_success_positive
            else 0.0
        ),
        "accepted_utility_sum": float(utility[accepted].sum()),
        "accepted_reward_delta_sum": float(reward_delta[accepted].sum()),
        "accepted_success_delta_sum": float(success_delta[accepted].sum()),
        "accepted_failed_delta_sum": float(failed_delta[accepted].sum()),
    }


def success_thresholds(args: argparse.Namespace) -> list[float | None]:
    return list(args.min_success_probability) if args.min_success_probability else [None]


def acceptance_mask(
    value_lcb: np.ndarray,
    bad_ucb: np.ndarray,
    success_lcb: np.ndarray,
    min_utility: float,
    max_bad_probability: float,
    min_success_probability: float | None,
) -> np.ndarray:
    utility_accept = value_lcb >= min_utility
    if min_success_probability is None:
        rescue_accept = np.zeros_like(utility_accept, dtype=bool)
    else:
        rescue_accept = success_lcb >= min_success_probability
    return (bad_ucb <= max_bad_probability) & (utility_accept | rescue_accept)


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
    success_labels = (success_delta > 1e-9).astype(np.float32)
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
            train_success=success_labels[train_mask],
            train_categories=train_categories,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]
    value_mean, value_std, bad_mean, bad_std, success_mean, success_std = (
        predict_ensemble(models, x[val_mask])
    )
    value_lcb = value_mean - args.value_std_coef * value_std
    bad_ucb = bad_mean + args.bad_std_coef * bad_std
    success_lcb = success_mean - args.success_std_coef * success_std

    split_results = []
    for min_utility in args.min_utility:
        for max_bad_probability in args.max_bad_probability:
            for min_success_probability in success_thresholds(args):
                accepted = acceptance_mask(
                    value_lcb=value_lcb,
                    bad_ucb=bad_ucb,
                    success_lcb=success_lcb,
                    min_utility=min_utility,
                    max_bad_probability=max_bad_probability,
                    min_success_probability=min_success_probability,
                )
                row = metric_row(
                    categories=categories[val_mask],
                    utility=utility[val_mask],
                    reward_delta=reward_delta[val_mask],
                    success_delta=success_delta[val_mask],
                    failed_delta=failed_delta[val_mask],
                    accepted=accepted,
                    min_utility=min_utility,
                    max_bad_probability=max_bad_probability,
                    min_success_probability=min_success_probability,
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
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
    success_labels = (success_delta > 1e-9).astype(np.float32)
    return (
        utility,
        reward_delta,
        success_delta,
        failed_delta,
        np.asarray(categories, dtype=object),
        categories,
        bad_labels,
        success_labels,
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
    (
        train_utility,
        _,
        _,
        _,
        _,
        train_categories,
        train_bad_labels,
        train_success_labels,
    ) = row_arrays(train_rows, args)
    (
        validation_utility,
        validation_reward_delta,
        validation_success_delta,
        validation_failed_delta,
        validation_categories,
        _,
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
            train_success=train_success_labels,
            train_categories=train_categories,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]
    value_mean, value_std, bad_mean, bad_std, success_mean, success_std = (
        predict_ensemble(models, validation_x)
    )
    value_lcb = value_mean - args.value_std_coef * value_std
    bad_ucb = bad_mean + args.bad_std_coef * bad_std
    success_lcb = success_mean - args.success_std_coef * success_std

    split_results = []
    for min_utility in args.min_utility:
        for max_bad_probability in args.max_bad_probability:
            for min_success_probability in success_thresholds(args):
                accepted = acceptance_mask(
                    value_lcb=value_lcb,
                    bad_ucb=bad_ucb,
                    success_lcb=success_lcb,
                    min_utility=min_utility,
                    max_bad_probability=max_bad_probability,
                    min_success_probability=min_success_probability,
                )
                row = metric_row(
                    categories=validation_categories,
                    utility=validation_utility,
                    reward_delta=validation_reward_delta,
                    success_delta=validation_success_delta,
                    failed_delta=validation_failed_delta,
                    accepted=accepted,
                    min_utility=min_utility,
                    max_bad_probability=max_bad_probability,
                    min_success_probability=min_success_probability,
                )
                row["split_seed"] = split_seed
                row["val_rows"] = len(validation_rows)
                row["val_seeds"] = sorted({int(float(row["seed"])) for row in validation_rows})
                split_results.append(row)
    return split_results


def validation_audit_rows(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    feature_columns: list[str],
    split_seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    train_x_raw = feature_matrix(train_rows, feature_columns)
    validation_x_raw = feature_matrix(validation_rows, feature_columns)
    (
        train_utility,
        _,
        _,
        _,
        _,
        train_categories,
        train_bad_labels,
        train_success_labels,
    ) = row_arrays(train_rows, args)
    (
        validation_utility,
        validation_reward_delta,
        validation_success_delta,
        validation_failed_delta,
        validation_categories,
        _,
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
            train_success=train_success_labels,
            train_categories=train_categories,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]
    value_mean, value_std, bad_mean, bad_std, success_mean, success_std = (
        predict_ensemble(models, validation_x)
    )
    value_lcb = value_mean - args.value_std_coef * value_std
    bad_ucb = bad_mean + args.bad_std_coef * bad_std
    success_lcb = success_mean - args.success_std_coef * success_std

    audit_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(validation_rows):
        base = {
            "split_seed": split_seed,
            "row_index": row_index,
            "seed": row.get("seed", ""),
            "prefix_len": row.get("prefix_len", ""),
            "forced_applied": row.get("forced_applied", ""),
            "outcome_category": validation_categories[row_index],
            "utility": float(validation_utility[row_index]),
            "reward_delta": float(validation_reward_delta[row_index]),
            "success_delta": float(validation_success_delta[row_index]),
            "failed_agents_delta": float(validation_failed_delta[row_index]),
            "value_mean": float(value_mean[row_index]),
            "value_std": float(value_std[row_index]),
            "value_lcb": float(value_lcb[row_index]),
            "bad_probability_mean": float(bad_mean[row_index]),
            "bad_probability_std": float(bad_std[row_index]),
            "bad_probability_ucb": float(bad_ucb[row_index]),
            "success_probability_mean": float(success_mean[row_index]),
            "success_probability_std": float(success_std[row_index]),
            "success_probability_lcb": float(success_lcb[row_index]),
            "events": row.get("events", ""),
        }
        for min_utility in args.min_utility:
            for max_bad_probability in args.max_bad_probability:
                for min_success_probability in success_thresholds(args):
                    accepted = acceptance_mask(
                        value_lcb=value_lcb[row_index : row_index + 1],
                        bad_ucb=bad_ucb[row_index : row_index + 1],
                        success_lcb=success_lcb[row_index : row_index + 1],
                        min_utility=min_utility,
                        max_bad_probability=max_bad_probability,
                        min_success_probability=min_success_probability,
                    )[0]
                    audit_rows.append(
                        {
                            **base,
                            "min_utility": min_utility,
                            "max_bad_probability": max_bad_probability,
                            "min_success_probability": min_success_probability,
                            "accepted": int(accepted),
                        }
                    )
    return audit_rows


def train_export_ensemble(
    train_rows: list[dict[str, str]],
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[ValueRiskMLP], np.ndarray, np.ndarray]:
    train_x_raw = feature_matrix(train_rows, feature_columns)
    (
        train_utility,
        _,
        _,
        _,
        _,
        train_categories,
        train_bad_labels,
        train_success_labels,
    ) = row_arrays(train_rows, args)
    _, mean, std = standardize(train_x_raw, train_x_raw)
    train_x = (train_x_raw - mean) / std

    models = []
    for split_seed in args.split_seeds:
        for member in range(args.ensemble_size):
            models.append(
                train_member(
                    seed=args.torch_seed + split_seed * 1000 + member,
                    train_x=train_x,
                    train_value=train_utility,
                    train_bad=train_bad_labels,
                    train_success=train_success_labels,
                    train_categories=train_categories,
                    args=args,
                )
            )
    return models, mean, std


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [
                str(item) if isinstance(item, Path) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def write_model_checkpoint(
    path: Path,
    train_rows: list[dict[str, str]],
    feature_columns: list[str],
    args: argparse.Namespace,
) -> None:
    models, mean, std = train_export_ensemble(train_rows, feature_columns, args)
    categories = [outcome_category(row, args.reward_epsilon) for row in train_rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "ValueRiskMLP",
            "model_state_dicts": [model.state_dict() for model in models],
            "feature_columns": feature_columns,
            "feature_mean": torch.as_tensor(mean, dtype=torch.float32),
            "feature_std": torch.as_tensor(std, dtype=torch.float32),
            "input_dim": len(feature_columns),
            "hidden_size": args.hidden_size,
            "split_seeds": list(args.split_seeds),
            "ensemble_size": args.ensemble_size,
            "config": serializable_args(args),
            "train_summary": {
                "rows": len(train_rows),
                "categories": dict(Counter(categories)),
                "seeds": sorted({int(float(row["seed"])) for row in train_rows}),
            },
        },
        path,
    )


def aggregate_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[float, float, float | None], dict[str, Any]] = {}
    summed_keys = [
        "accepted",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "rejected_good",
        "rejected_neutral",
        "rejected_bad",
        "rejected_success_positive",
        "accepted_utility_sum",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
        "val_rows",
    ]
    for row in results:
        key = (
            row["min_utility"],
            row["max_bad_probability"],
            row.get("min_success_probability"),
        )
        bucket = buckets.setdefault(
            key,
            {
                "min_utility": row["min_utility"],
                "max_bad_probability": row["max_bad_probability"],
                "min_success_probability": row.get("min_success_probability"),
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
        total_success_positive = (
            bucket["accepted_success_positive"]
            + bucket["rejected_success_positive"]
        )
        bucket["good_recall"] = (
            bucket["accepted_good"] / total_good if total_good else 0.0
        )
        bucket["bad_leak_rate"] = (
            bucket["accepted_bad"] / total_bad if total_bad else 0.0
        )
        bucket["success_positive_recall"] = (
            bucket["accepted_success_positive"] / total_success_positive
            if total_success_positive
            else 0.0
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
        "min_success_probability",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "good_recall",
        "bad_leak_rate",
        "success_positive_recall",
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


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--success-std-coef", type=float, default=1.0)
    parser.add_argument(
        "--success-loss-weight",
        type=float,
        help=(
            "Weight for the auxiliary success-improvement classifier. Defaults "
            "to 1.0 when --min-success-probability is used, otherwise 0.0."
        ),
    )
    parser.add_argument("--success-positive-weight", type=float, default=12.0)
    parser.add_argument("--success-negative-weight", type=float, default=1.0)
    parser.add_argument("--success-bad-weight", type=float, default=8.0)
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
    parser.add_argument(
        "--min-success-probability",
        nargs="+",
        type=float,
        default=[],
        help=(
            "Optional Success-rescue probability LCB thresholds. When set, "
            "rows are accepted if either utility LCB or success probability LCB "
            "passes while the bad-risk UCB remains below the bad threshold."
        ),
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--output-audit-csv",
        type=Path,
        help=(
            "With --validation-csv, write per-validation-row predictions and "
            "acceptance decisions for each threshold combination."
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help=(
            "Train a final ensemble on the training CSV rows and save model "
            "weights, feature columns, normalization statistics, and config."
        ),
    )
    args = parser.parse_args()
    if args.success_loss_weight is None:
        args.success_loss_weight = 1.0 if args.min_success_probability else 0.0
    return args


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
    audit_rows: list[dict[str, Any]] = []
    if validation_rows and args.output_audit_csv is not None:
        for split_seed in args.split_seeds:
            audit_rows.extend(
                validation_audit_rows(
                    train_rows,
                    validation_rows,
                    feature_columns,
                    split_seed,
                    args,
                )
            )

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
    if args.output_audit_csv is not None:
        write_audit_csv(args.output_audit_csv, audit_rows)
    if args.output_model is not None:
        write_model_checkpoint(
            args.output_model,
            train_rows=train_rows,
            feature_columns=feature_columns,
            args=args,
        )
        print(f"saved_model={args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
