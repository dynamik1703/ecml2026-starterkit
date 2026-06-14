#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tools.evaluate_success_rescue_classifier import format_float, json_default
from tools.train_counterfactual_gate import safe_float, seed_split, standardize


IDENTITY_COLUMNS = {
    "agent_id",
    "position",
    "seed",
    "state",
}

NON_FEATURE_COLUMNS = {
    "direction",
    "has_diff",
    "label",
    "label_name",
    "source_file",
}


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


def read_csv_rows(paths: list[Path], label: int, label_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                item: dict[str, Any] = dict(row)
                item["label"] = int(label)
                item["label_name"] = label_name
                item["source_file"] = str(path)
                rows.append(item)
    return rows


def action_transition(row: dict[str, Any]) -> str:
    baseline = str(row.get("baseline_action_name", ""))
    candidate = str(row.get("candidate_action_name", ""))
    return f"{baseline}->{candidate}"


def choose_feature_columns(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    columns = sorted({key for row in rows for key in row})
    features = []
    excluded = set(IDENTITY_COLUMNS) | set(NON_FEATURE_COLUMNS)
    if args.drop_action_id_features:
        excluded.update(
            {
                "baseline_action",
                "baseline_raw_action",
                "candidate_action",
                "candidate_raw_action",
            }
        )
    for column in columns:
        if column in excluded:
            continue
        if column.endswith("_name") or column.endswith("_target"):
            continue
        if args.drop_logit_features and "logit" in column:
            continue
        if args.drop_prefix_features and "_prefix_" in column:
            continue
        if args.drop_obs_features and column.startswith("obs_"):
            continue
        values = [safe_float(row.get(column)) for row in rows]
        if any(math.isfinite(value) for value in values):
            features.append(column)
    if args.include_feature_regex:
        include = re.compile(args.include_feature_regex)
        features = [column for column in features if include.search(column)]
    if args.exclude_feature_regex:
        exclude = re.compile(args.exclude_feature_regex)
        features = [column for column in features if not exclude.search(column)]
    return features


def feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            value = safe_float(row.get(column))
            matrix[row_index, column_index] = value if math.isfinite(value) else 0.0
    return matrix


def train_member(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
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
        torch.full_like(y_tensor, float(args.positive_weight)),
        torch.full_like(y_tensor, float(args.negative_weight)),
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
    y: np.ndarray,
    args: argparse.Namespace,
) -> list[BinaryMLP]:
    models = []
    for member in range(args.ensemble_size):
        seed = args.torch_seed + split_seed * 1000 + member
        models.append(train_member(seed, x, y, args))
    return models


def predict(models: list[BinaryMLP], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty
    tensor = torch.as_tensor(x, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(torch.sigmoid(model(tensor)).cpu().numpy())
    stack = np.stack(predictions, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def split_rows(
    rows: list[dict[str, Any]],
    split_seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_mask, validation_mask = seed_split(
        rows,
        args.val_fraction,
        split_seed,
        ordered=args.ordered_split,
    )
    train_rows = [row for row, keep in zip(rows, train_mask) if keep]
    validation_rows = [row for row, keep in zip(rows, validation_mask) if keep]
    return train_rows, validation_rows


def evaluate_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
    split_seed: int,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    accepted = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    accepted_positive = int(np.logical_and(accepted, positives).sum())
    accepted_negative = int(np.logical_and(accepted, negatives).sum())
    accepted_rows = [row for row, keep in zip(rows, accepted) if keep]
    accepted_positive_seeds = {
        int(float(row["seed"])) for row in accepted_rows if int(row["label"]) == 1
    }
    accepted_negative_seeds = {
        int(float(row["seed"])) for row in accepted_rows if int(row["label"]) == 0
    }
    accepted_transitions = Counter(action_transition(row) for row in accepted_rows)
    precision = (
        accepted_positive / (accepted_positive + accepted_negative)
        if accepted_positive + accepted_negative
        else 0.0
    )
    recall = accepted_positive / max(1, int(positives.sum()))
    negative_accept_rate = accepted_negative / max(1, int(negatives.sum()))
    return {
        "split_seed": split_seed,
        "threshold": float(threshold),
        "validation_rows": len(rows),
        "validation_positive": int(positives.sum()),
        "validation_negative": int(negatives.sum()),
        "accepted": int(accepted.sum()),
        "accepted_positive": accepted_positive,
        "accepted_negative": accepted_negative,
        "accepted_positive_seeds": len(accepted_positive_seeds),
        "accepted_negative_seeds": len(accepted_negative_seeds),
        "precision": precision,
        "recall": recall,
        "negative_accept_rate": negative_accept_rate,
        "top_accepted_transitions": dict(accepted_transitions.most_common(8)),
    }


def aggregate_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[float, dict[str, Any]] = {}
    sum_keys = [
        "validation_rows",
        "validation_positive",
        "validation_negative",
        "accepted",
        "accepted_positive",
        "accepted_negative",
        "accepted_positive_seeds",
        "accepted_negative_seeds",
    ]
    for row in metrics:
        threshold = float(row["threshold"])
        bucket = buckets.setdefault(
            threshold,
            {"threshold": threshold, "splits": 0},
        )
        bucket["splits"] += 1
        for key in sum_keys:
            bucket[key] = bucket.get(key, 0) + int(row[key])
    aggregate = []
    for row in buckets.values():
        accepted = int(row.get("accepted", 0))
        accepted_positive = int(row.get("accepted_positive", 0))
        accepted_negative = int(row.get("accepted_negative", 0))
        validation_positive = int(row.get("validation_positive", 0))
        validation_negative = int(row.get("validation_negative", 0))
        row["precision"] = (
            accepted_positive / accepted if accepted else 0.0
        )
        row["recall"] = accepted_positive / max(1, validation_positive)
        row["negative_accept_rate"] = accepted_negative / max(1, validation_negative)
        aggregate.append(row)
    aggregate.sort(
        key=lambda row: (
            int(row["accepted_negative"]),
            -int(row["accepted_positive"]),
            -float(row["precision"]),
            -float(row["recall"]),
        )
    )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            "Evaluate whether policy-diff event features can separate "
            "candidate-win events from candidate-loss events on seed holdouts."
        )
    )
    parser.add_argument("--positive-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--negative-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.35)
    parser.add_argument("--ordered-split", action="store_true")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-action-id-features", action="store_true")
    parser.add_argument("--drop-logit-features", action="store_true")
    parser.add_argument("--drop-prefix-features", action="store_true")
    parser.add_argument("--drop-obs-features", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=919)
    parser.add_argument("--std-coef", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-audit-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.positive_csv, 1, "positive") + read_csv_rows(
        args.negative_csv,
        0,
        "negative",
    )
    if not rows:
        raise ValueError("No rows loaded")
    feature_columns = choose_feature_columns(rows, args)
    if not feature_columns:
        raise ValueError("No numeric feature columns selected")

    all_metrics: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for split_seed in args.split_seeds:
        train_rows, validation_rows = split_rows(rows, split_seed, args)
        train_x_raw = feature_matrix(train_rows, feature_columns)
        validation_x_raw = feature_matrix(validation_rows, feature_columns)
        _, mean, std = standardize(train_x_raw, train_x_raw)
        train_x = (train_x_raw - mean) / std
        validation_x = (validation_x_raw - mean) / std
        train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.float32)
        models = train_ensemble(split_seed, train_x, train_y, args)
        prob_mean, prob_std = predict(models, validation_x)
        score = prob_mean - args.std_coef * prob_std
        for threshold in args.thresholds:
            all_metrics.append(
                evaluate_threshold(validation_rows, score, threshold, split_seed)
            )
        if args.output_audit_csv:
            for row, mean_value, std_value, score_value in zip(
                validation_rows,
                prob_mean,
                prob_std,
                score,
            ):
                audit_rows.append(
                    {
                        "split_seed": split_seed,
                        "seed": row.get("seed", ""),
                        "env_time": row.get("env_time", ""),
                        "agent_id": row.get("agent_id", ""),
                        "label": row.get("label", ""),
                        "label_name": row.get("label_name", ""),
                        "baseline_action_name": row.get("baseline_action_name", ""),
                        "candidate_action_name": row.get("candidate_action_name", ""),
                        "transition": action_transition(row),
                        "probability_mean": float(mean_value),
                        "probability_std": float(std_value),
                        "score_lcb": float(score_value),
                    }
                )

    aggregate = aggregate_metrics(all_metrics)
    labels = Counter(row["label_name"] for row in rows)
    transitions = Counter(action_transition(row) for row in rows)
    print(
        "Data: "
        f"rows={len(rows)} labels={dict(labels)} features={len(feature_columns)} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print("Top transitions:", dict(transitions.most_common(8)))
    columns = [
        "threshold",
        "splits",
        "accepted",
        "accepted_positive",
        "accepted_negative",
        "accepted_positive_seeds",
        "accepted_negative_seeds",
        "precision",
        "recall",
        "negative_accept_rate",
    ]
    print(",".join(columns))
    for row in aggregate[: args.top_k]:
        print(
            ",".join(
                format_float(float(row[column]))
                if isinstance(row.get(column), float)
                else str(row.get(column, ""))
                for column in columns
            )
        )

    if args.output_audit_csv:
        write_csv(args.output_audit_csv, audit_rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "rows": len(rows),
                        "labels": dict(labels),
                        "feature_columns": feature_columns,
                        "features": len(feature_columns),
                        "transitions": dict(transitions),
                        "args": vars(args),
                    },
                    "aggregate_metrics": aggregate,
                    "split_metrics": all_metrics,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
