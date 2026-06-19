#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


OUTCOME_COLUMNS = {
    "baseline_reward",
    "forced_reward",
    "reward_delta",
    "baseline_success",
    "forced_success",
    "success_delta",
    "normalized_reward",
    "success_rate",
    "baseline_env_time",
    "forced_env_time",
    "env_time_delta",
    "baseline_failed_agents",
    "forced_failed_agents",
    "failed_agents_delta",
    "baseline_failed_agent_ids",
    "forced_failed_agent_ids",
    "forced_applied",
    "pairwise_label",
}

NON_NUMERIC_COLUMNS = {
    "baseline_action_name",
    "baseline_target",
    "forced_action_name",
    "forced_target",
    "position",
    "scene",
    "state",
}

IDENTITY_COLUMNS = {
    "seed",
    "decision_index",
    "agent_id",
}


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int = 1):
        super().__init__()
        if hidden_size <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.net(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def label_row(row: dict[str, str], reward_epsilon: float) -> int:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    if success_delta > 1e-9:
        return 1
    if success_delta < -1e-9 or failed_delta > 0:
        return 0
    return int(reward_delta > reward_epsilon)


def outcome_category(row: dict[str, str], reward_epsilon: float) -> str:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    if success_delta > 1e-9:
        return "good"
    if success_delta < -1e-9 or failed_delta > 0:
        return "bad"
    if reward_delta > reward_epsilon:
        return "good"
    if reward_delta < -reward_epsilon:
        return "bad"
    return "neutral"


def choose_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    columns = sorted({key for row in rows for key in row})
    feature_columns = []
    excluded = OUTCOME_COLUMNS | NON_NUMERIC_COLUMNS | IDENTITY_COLUMNS
    for column in columns:
        if column in excluded:
            continue
        values = [safe_float(row.get(column)) for row in rows]
        finite_count = sum(math.isfinite(value) for value in values)
        if finite_count:
            feature_columns.append(column)
    return feature_columns


def filter_feature_columns(
    feature_columns: list[str],
    args: argparse.Namespace,
) -> list[str]:
    include_pattern = getattr(args, "include_feature_regex", None)
    exclude_pattern = getattr(args, "exclude_feature_regex", None)
    drop_prefix_features = bool(getattr(args, "drop_prefix_features", False))
    if include_pattern:
        include_regex = re.compile(include_pattern)
        feature_columns = [
            column for column in feature_columns if include_regex.search(column)
        ]
    if exclude_pattern:
        exclude_regex = re.compile(exclude_pattern)
        feature_columns = [
            column for column in feature_columns if not exclude_regex.search(column)
        ]
    if drop_prefix_features:
        feature_columns = [
            column
            for column in feature_columns
            if not (
                column.startswith("baseline_prefix_")
                or column.startswith("forced_prefix_")
            )
        ]
    return feature_columns


def feature_matrix(
    rows: list[dict[str, str]],
    feature_columns: list[str],
) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(feature_columns):
            value = safe_float(row.get(column))
            matrix[row_index, column_index] = value if math.isfinite(value) else 0.0
    return matrix


def seed_split(
    rows: list[dict[str, str]],
    val_fraction: float,
    split_seed: int,
    ordered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = sorted({int(float(row["seed"])) for row in rows})
    val_count = max(1, int(round(len(seeds) * val_fraction))) if len(seeds) > 1 else 0
    if ordered:
        split_seeds = seeds
    else:
        rng = np.random.default_rng(split_seed)
        split_seeds = list(seeds)
        rng.shuffle(split_seeds)
    val_seeds = set(split_seeds[-val_count:]) if val_count else set()
    train_mask = np.array([int(float(row["seed"])) not in val_seeds for row in rows])
    val_mask = ~train_mask
    if not train_mask.any():
        train_mask[:] = True
        val_mask[:] = False
    return train_mask, val_mask


def standardize(
    train_x: np.ndarray,
    all_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (all_x - mean) / std, mean, std


def metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    threshold: float,
    bad_probabilities: np.ndarray | None = None,
    max_bad_probability: float | None = None,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    if bad_probabilities is not None and max_bad_probability is not None:
        predictions = np.logical_and(
            predictions,
            bad_probabilities <= max_bad_probability,
        )
    positives = labels == 1
    tp = int(np.logical_and(predictions, positives).sum())
    fp = int(np.logical_and(predictions, ~positives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())
    tn = int(np.logical_and(~predictions, ~positives).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    accepted_categories = [
        category
        for category, prediction in zip(categories, predictions)
        if prediction
    ]
    return {
        "threshold": threshold,
        "max_bad_probability": max_bad_probability,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "accepted": len(accepted_categories),
        "accepted_good": accepted_categories.count("good"),
        "accepted_neutral": accepted_categories.count("neutral"),
        "accepted_bad": accepted_categories.count("bad"),
    }


def category_weight(category: str, args: argparse.Namespace) -> float:
    if category == "good":
        return args.positive_weight
    if category == "bad":
        return args.bad_negative_weight
    return args.neutral_negative_weight


def category_index(category: str) -> int:
    if category == "good":
        return 1
    if category == "bad":
        return 2
    return 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    train_rows_input = read_rows(args.csv)
    validation_rows = read_rows(args.validation_csv) if args.validation_csv else []
    rows = train_rows_input + validation_rows
    if not train_rows_input:
        raise ValueError("No rows loaded")
    feature_columns = filter_feature_columns(choose_feature_columns(rows), args)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    x_raw = feature_matrix(rows, feature_columns)
    y = np.asarray([label_row(row, args.reward_epsilon) for row in rows], dtype=np.float32)
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
    if validation_rows:
        train_mask = np.array(
            [index < len(train_rows_input) for index in range(len(rows))],
            dtype=bool,
        )
        val_mask = ~train_mask
    else:
        train_mask, val_mask = seed_split(
            rows,
            args.val_fraction,
            args.split_seed,
            args.ordered_seed_split,
        )
    x, mean, std = standardize(x_raw[train_mask], x_raw)

    train_x = torch.as_tensor(x[train_mask], dtype=torch.float32)
    train_y = torch.as_tensor(y[train_mask], dtype=torch.float32)
    val_x = torch.as_tensor(x[val_mask], dtype=torch.float32)
    val_y = y[val_mask]
    train_categories = [category for category, is_train in zip(categories, train_mask) if is_train]
    val_categories = [category for category, is_val in zip(categories, val_mask) if is_val]
    train_weights = torch.as_tensor(
        [category_weight(category, args) for category in train_categories],
        dtype=torch.float32,
    )

    torch.manual_seed(args.torch_seed)
    output_dim = 3 if args.objective == "multiclass" else 1
    model = GateMLP(
        input_dim=train_x.shape[1],
        hidden_size=args.hidden_size,
        output_dim=output_dim,
    )
    if args.objective == "multiclass":
        train_y_class = torch.as_tensor(
            [category_index(category) for category in train_categories],
            dtype=torch.long,
        )
        class_weights = torch.as_tensor(
            [
                args.neutral_negative_weight,
                args.positive_weight,
                args.bad_negative_weight,
            ],
            dtype=torch.float32,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_y_class = None
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        if args.objective == "multiclass":
            loss = loss_fn(logits, train_y_class)
        else:
            raw_loss = loss_fn(logits, train_y)
            loss = (raw_loss * train_weights).sum() / train_weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        val_logits = model(val_x) if len(val_x) else None
        if args.objective == "multiclass":
            train_probs = torch.softmax(train_logits, dim=-1).cpu().numpy()
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy() if val_logits is not None else np.empty((0, 3))
            train_prob = train_probs[:, 1]
            train_bad_prob = train_probs[:, 2]
            val_prob = val_probs[:, 1]
            val_bad_prob = val_probs[:, 2]
        else:
            train_prob = torch.sigmoid(train_logits).cpu().numpy()
            val_prob = torch.sigmoid(val_logits).cpu().numpy() if val_logits is not None else np.array([])
            train_bad_prob = None
            val_bad_prob = None

    train_metrics = [
        metrics(
            train_prob,
            y[train_mask],
            train_categories,
            threshold,
            train_bad_prob,
            args.max_bad_probability if args.objective == "multiclass" else None,
        )
        for threshold in args.thresholds
    ]
    val_metrics = [
        metrics(
            val_prob,
            val_y,
            val_categories,
            threshold,
            val_bad_prob,
            args.max_bad_probability if args.objective == "multiclass" else None,
        )
        for threshold in args.thresholds
    ] if len(val_y) else []
    summary = {
        "rows": len(rows),
        "input_rows": len(train_rows_input),
        "validation_input_rows": len(validation_rows),
        "features": len(feature_columns),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
        "train_positive_rows": int(y[train_mask].sum()),
        "val_positive_rows": int(y[val_mask].sum()) if val_mask.any() else 0,
        "val_seeds": sorted(
            {int(float(row["seed"])) for row, is_val in zip(rows, val_mask) if is_val}
        ),
        "feature_columns": feature_columns,
        "include_feature_regex": getattr(args, "include_feature_regex", None),
        "exclude_feature_regex": getattr(args, "exclude_feature_regex", None),
        "drop_prefix_features": bool(getattr(args, "drop_prefix_features", False)),
        "thresholds": list(args.thresholds),
        "objective": args.objective,
        "max_bad_probability": args.max_bad_probability,
        "weights": {
            "good": args.positive_weight,
            "neutral": args.neutral_negative_weight,
            "bad": args.bad_negative_weight,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    if args.output_checkpoint is not None:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "input_dim": train_x.shape[1],
                    "hidden_size": args.hidden_size,
                    "output_dim": output_dim,
                    "objective": args.objective,
                    "reward_epsilon": args.reward_epsilon,
                    "max_bad_probability": args.max_bad_probability,
                    "feature_columns": feature_columns,
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                },
                "summary": summary,
            },
            args.output_checkpoint,
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small classifier for counterfactual gate labels."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--validation-csv",
        nargs="+",
        type=Path,
        help="Use these CSV rows as an explicit validation set instead of a seed split.",
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
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument(
        "--ordered-seed-split",
        action="store_true",
        help="Use the highest seeds as validation instead of shuffling seed groups.",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument(
        "--objective",
        choices=["binary", "multiclass"],
        default="binary",
        help=(
            "Use binary good-vs-rest training or three-class "
            "neutral/good/bad training."
        ),
    )
    parser.add_argument(
        "--max-bad-probability",
        type=float,
        default=0.05,
        help=(
            "For multiclass training, accept an action only if its predicted "
            "bad probability stays below this value."
        ),
    )
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=8.0,
        help="Per-sample loss weight for good counterfactual actions.",
    )
    parser.add_argument(
        "--neutral-negative-weight",
        type=float,
        default=0.25,
        help="Per-sample loss weight for neutral counterfactual actions.",
    )
    parser.add_argument(
        "--bad-negative-weight",
        type=float,
        default=8.0,
        help="Per-sample loss weight for bad counterfactual actions.",
    )
    parser.add_argument("--torch-seed", type=int, default=7)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.9, 0.95, 0.97],
    )
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    summary = train(parse_args())
    print(
        "Summary: "
        f"rows={summary['rows']} positives={summary['positive_rows']} "
        f"features={summary['features']} train={summary['train_rows']} "
        f"val={summary['val_rows']}"
    )
    print("Train metrics:")
    for metric in summary["train_metrics"]:
        print(
            "  threshold={threshold:.2f} precision={precision:.3f} "
            "recall={recall:.3f} accuracy={accuracy:.3f} "
            "tp/fp/fn/tn={tp}/{fp}/{fn}/{tn} "
            "accepted(g/n/b)={accepted_good}/{accepted_neutral}/"
            "{accepted_bad}".format(**metric)
        )
    if summary["val_metrics"]:
        print("Validation metrics:")
        for metric in summary["val_metrics"]:
            print(
                "  threshold={threshold:.2f} precision={precision:.3f} "
                "recall={recall:.3f} accuracy={accuracy:.3f} "
                "tp/fp/fn/tn={tp}/{fp}/{fn}/{tn} "
                "accepted(g/n/b)={accepted_good}/{accepted_neutral}/"
                "{accepted_bad}".format(**metric)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
