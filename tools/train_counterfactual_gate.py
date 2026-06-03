#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
    "baseline_env_time",
    "forced_env_time",
    "env_time_delta",
    "baseline_failed_agents",
    "forced_failed_agents",
    "failed_agents_delta",
    "baseline_failed_agent_ids",
    "forced_failed_agent_ids",
    "forced_applied",
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
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
    if success_delta < -1e-9 or failed_delta > 0 or reward_delta < -reward_epsilon:
        return "bad"
    if success_delta > 1e-9 or reward_delta > reward_epsilon:
        return "good"
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
) -> dict[str, float]:
    predictions = probabilities >= threshold
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError("No rows loaded")
    feature_columns = choose_feature_columns(rows)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    x_raw = feature_matrix(rows, feature_columns)
    y = np.asarray([label_row(row, args.reward_epsilon) for row in rows], dtype=np.float32)
    categories = [outcome_category(row, args.reward_epsilon) for row in rows]
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

    torch.manual_seed(args.torch_seed)
    model = GateMLP(input_dim=train_x.shape[1], hidden_size=args.hidden_size)
    positives = float(train_y.sum().item())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(train_x)).cpu().numpy()
        val_prob = torch.sigmoid(model(val_x)).cpu().numpy() if len(val_x) else np.array([])

    train_categories = [category for category, is_train in zip(categories, train_mask) if is_train]
    val_categories = [category for category, is_val in zip(categories, val_mask) if is_val]
    train_metrics = [
        metrics(train_prob, y[train_mask], train_categories, threshold)
        for threshold in args.thresholds
    ]
    val_metrics = [
        metrics(val_prob, val_y, val_categories, threshold)
        for threshold in args.thresholds
    ] if len(val_y) else []
    summary = {
        "rows": len(rows),
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
        "thresholds": list(args.thresholds),
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
                    "reward_epsilon": args.reward_epsilon,
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
    parser.add_argument("--torch-seed", type=int, default=7)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.9],
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
