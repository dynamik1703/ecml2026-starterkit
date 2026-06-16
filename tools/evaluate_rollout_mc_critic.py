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


IDENTITY_COLUMNS = {
    "agent_id",
    "action_name",
    "position",
    "scene",
    "seed",
    "state",
}

TARGET_COLUMNS = {
    "agent_failure",
    "agent_success",
    "episode_env_time",
    "episode_normalized_reward",
    "failed_agents_count",
    "immediate_reward",
    "team_success",
}


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def choose_feature_columns(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    excluded = set(IDENTITY_COLUMNS) | set(TARGET_COLUMNS)
    columns = sorted({key for row in rows for key in row})
    include_pattern = re.compile(args.include_feature_regex) if args.include_feature_regex else None
    exclude_pattern = re.compile(args.exclude_feature_regex) if args.exclude_feature_regex else None
    feature_columns = []
    for column in columns:
        if column in excluded:
            continue
        if args.drop_observation_features and column.startswith("obs_"):
            continue
        if include_pattern and not include_pattern.search(column):
            continue
        if exclude_pattern and exclude_pattern.search(column):
            continue
        values = [safe_float(row.get(column)) for row in rows]
        if any(math.isfinite(value) for value in values):
            feature_columns.append(column)
    return feature_columns


def feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            matrix[row_index, column_index] = finite_or_zero(row.get(column))
    return matrix


def target_values(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target == "team_failure":
        return np.asarray(
            [1.0 - finite_or_zero(row.get("team_success")) for row in rows],
            dtype=np.float32,
        )
    return np.asarray([finite_or_zero(row.get(target)) for row in rows], dtype=np.float32)


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
) -> BinaryMLP:
    torch.manual_seed(args.torch_seed)
    model = BinaryMLP(train_x.shape[1], args.hidden_size, args.dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    weights = torch.where(
        y_tensor > 0.5,
        torch.full_like(y_tensor, args.positive_weight),
        torch.full_like(y_tensor, args.negative_weight),
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(args.epochs):
        permutation = torch.randperm(x_tensor.shape[0])
        for start in range(0, x_tensor.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            logits = model(x_tensor[batch])
            loss = (
                loss_fn(logits, y_tensor[batch]) * weights[batch]
            ).sum() / weights[batch].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
    model.eval()
    return model


def predict(model: BinaryMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_ranks = ranks[labels > 0.5]
    return float(
        (positive_ranks.sum() - len(positives) * (len(positives) + 1) / 2)
        / (len(positives) * len(negatives))
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int((labels > 0.5).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order] > 0.5
    hits = 0
    precision_sum = 0.0
    for index, label in enumerate(sorted_labels, start=1):
        if label:
            hits += 1
            precision_sum += hits / index
    return float(precision_sum / positives)


def top_fraction_metrics(labels: np.ndarray, scores: np.ndarray, fractions: list[float]) -> list[dict[str, Any]]:
    order = np.argsort(-scores)
    positives = int((labels > 0.5).sum())
    rows = []
    for fraction in fractions:
        count = max(1, int(round(len(labels) * fraction)))
        selected = order[:count]
        selected_positive = int((labels[selected] > 0.5).sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "selected": int(count),
                "selected_positive": selected_positive,
                "precision": selected_positive / count,
                "recall": selected_positive / max(1, positives),
            }
        )
    return rows


def evaluate(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_rows = train_rows + validation_rows
    feature_columns = choose_feature_columns(all_rows, args)
    if not feature_columns:
        raise ValueError("No feature columns selected")
    train_x_raw = feature_matrix(train_rows, feature_columns)
    val_x_raw = feature_matrix(validation_rows, feature_columns)
    train_x, val_x, mean, std = standardize(train_x_raw, val_x_raw)
    train_y = target_values(train_rows, args.target)
    val_y = target_values(validation_rows, args.target)
    model = train_model(train_x, train_y, args)
    scores = predict(model, val_x)
    predictions = scores >= args.threshold
    true_positive = int(np.logical_and(predictions, val_y > 0.5).sum())
    false_positive = int(np.logical_and(predictions, val_y <= 0.5).sum())
    false_negative = int(np.logical_and(~predictions, val_y > 0.5).sum())
    return {
        "summary": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "features": len(feature_columns),
            "feature_columns": feature_columns,
            "target": args.target,
            "train_positive": int((train_y > 0.5).sum()),
            "validation_positive": int((val_y > 0.5).sum()),
            "validation_auc": roc_auc(val_y, scores),
            "validation_average_precision": average_precision(val_y, scores),
            "threshold": args.threshold,
            "threshold_true_positive": true_positive,
            "threshold_false_positive": false_positive,
            "threshold_false_negative": false_negative,
            "threshold_precision": true_positive / max(1, true_positive + false_positive),
            "threshold_recall": true_positive / max(1, true_positive + false_negative),
            "top_fraction_metrics": top_fraction_metrics(
                val_y,
                scores,
                args.top_fraction,
            ),
        },
        "validation_scores": [
            {
                "seed": row.get("seed", ""),
                "env_time": row.get("env_time", ""),
                "agent_id": row.get("agent_id", ""),
                "action": row.get("action", ""),
                "action_name": row.get("action_name", ""),
                "target": float(label),
                "score": float(score),
            }
            for row, label, score in zip(validation_rows, val_y, scores)
        ],
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Monte-Carlo rollout action risk critic."
    )
    parser.add_argument("--train-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-csv", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--target",
        choices=["agent_failure", "team_failure"],
        default="agent_failure",
    )
    parser.add_argument("--drop-observation-features", action="store_true")
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--torch-seed", type=int, default=727)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--top-fraction",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_rows(args.train_csv)
    validation_rows = read_rows(args.validation_csv)
    result = evaluate(train_rows, validation_rows, args)
    summary = result["summary"]
    print(
        "mc_critic "
        f"train_rows={summary['train_rows']} "
        f"validation_rows={summary['validation_rows']} "
        f"features={summary['features']} "
        f"target={summary['target']} "
        f"train_positive={summary['train_positive']} "
        f"validation_positive={summary['validation_positive']} "
        f"auc={summary['validation_auc']:.6g} "
        f"ap={summary['validation_average_precision']:.6g} "
        f"precision={summary['threshold_precision']:.6g} "
        f"recall={summary['threshold_recall']:.6g}"
    )
    for item in summary["top_fraction_metrics"]:
        print(
            "top "
            f"fraction={item['top_fraction']:.3g} "
            f"selected={item['selected']} "
            f"precision={item['precision']:.6g} "
            f"recall={item['recall']:.6g}"
        )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump({"config": vars(args), **result}, handle, indent=2, default=json_default)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
