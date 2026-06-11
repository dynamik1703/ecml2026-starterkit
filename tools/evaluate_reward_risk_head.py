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

from tools.evaluate_success_rescue_classifier import (
    BinaryMLP,
    format_float,
    json_default,
    metric_row,
    predict_ensemble,
    write_csv,
)
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


def outcome_arrays(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return reward_delta, success_delta, failed_delta


def utility_target(
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return (
        args.success_weight * success_delta
        + args.reward_weight * reward_delta
        - args.reward_loss_penalty * np.maximum(-reward_delta, 0.0)
        - args.success_loss_penalty * np.maximum(-success_delta, 0.0)
        - args.failure_penalty * np.maximum(failed_delta, 0.0)
    ).astype(np.float32)


def risk_labels(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reward_delta, success_delta, failed_delta = outcome_arrays(rows)
    target = utility_target(reward_delta, success_delta, failed_delta, args)
    reward_loss = reward_delta < -args.reward_loss_epsilon
    if args.reward_loss_mode == "non_success":
        reward_loss = reward_loss & (success_delta <= 1e-9)
    labels = (
        (target < args.target_risk_threshold)
        | (success_delta < -1e-9)
        | (failed_delta > 0.0)
        | reward_loss
    ).astype(np.float32)
    return reward_delta, success_delta, failed_delta, target, labels


def train_risk_member(
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
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
        torch.full_like(y, float(args.risk_positive_weight)),
        torch.full_like(y, float(args.risk_negative_weight)),
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


def train_risk_ensemble(
    split_seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
) -> list[BinaryMLP]:
    models = []
    for member in range(args.ensemble_size):
        models.append(
            train_risk_member(
                seed=args.torch_seed + split_seed * 1000 + member,
                train_x=train_x,
                train_y=train_y,
                args=args,
            )
        )
    return models


def risk_metric_row(
    categories: np.ndarray,
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    max_risk_probability: float,
) -> dict[str, Any]:
    false_negative = np.logical_and(labels > 0.5, ~predictions)
    false_positive = np.logical_and(labels <= 0.5, predictions)
    row = metric_row(
        categories=categories,
        reward_delta=reward_delta,
        success_delta=success_delta,
        failed_delta=failed_delta,
        accepted=~predictions,
        min_success_probability=0.0,
        max_unsafe_probability=max_risk_probability,
    )
    row.update(
        {
            "max_risk_probability": float(max_risk_probability),
            "risk_rows": int(labels.sum()),
            "predicted_risk": int(predictions.sum()),
            "risk_true_positive": int(np.logical_and(labels > 0.5, predictions).sum()),
            "risk_false_negative": int(false_negative.sum()),
            "risk_false_positive": int(false_positive.sum()),
            "risk_recall": (
                float(np.logical_and(labels > 0.5, predictions).sum())
                / float(labels.sum())
                if labels.sum()
                else 0.0
            ),
            "risk_precision": (
                float(np.logical_and(labels > 0.5, predictions).sum())
                / float(predictions.sum())
                if predictions.sum()
                else 0.0
            ),
        }
    )
    return row


def join_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("split_seed", "")),
        str(row.get("seed", "")),
        str(row.get("prefix_len", "")),
        str(row.get("forced_applied", "")),
        str(row.get("events", "")),
        round(float(row.get("reward_delta", 0.0)), 9),
        round(float(row.get("success_delta", 0.0)), 9),
    )


def ranker_veto_metrics(
    ranker_audit_csv: Path,
    risk_audit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    risk_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in risk_audit_rows:
        key = join_key(row)
        current = risk_by_key.get(key)
        if current is None or float(row["risk_probability_ucb"]) > float(
            current["risk_probability_ucb"]
        ):
            risk_by_key[key] = row
    ranker_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with ranker_audit_csv.open() as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("split_seed", "")), str(row.get("row_index", "")))
            ranker_rows_by_key.setdefault(key, row)
    selected_rows = [
        row
        for row in ranker_rows_by_key.values()
        if int(float(row.get("selected_best_for_seed", "0") or 0)) == 1
        and int(float(row.get("is_baseline_candidate", "0") or 0)) == 0
    ]

    metrics = []
    for score_threshold in args.ranker_score_threshold:
        score_rows = [
            row
            for row in selected_rows
            if safe_float(row.get("rank_score_lcb")) >= score_threshold
        ]
        for max_risk_probability in args.veto_max_risk_probability:
            accepted_rows = []
            missing = 0
            for row in score_rows:
                risk_row = risk_by_key.get(join_key(row))
                if risk_row is None:
                    missing += 1
                    continue
                if float(risk_row["risk_probability_ucb"]) <= max_risk_probability:
                    accepted_rows.append(row)
            reward_delta = np.asarray(
                [safe_float(row.get("reward_delta")) for row in accepted_rows],
                dtype=np.float32,
            )
            success_delta = np.asarray(
                [safe_float(row.get("success_delta")) for row in accepted_rows],
                dtype=np.float32,
            )
            failed_delta = np.asarray(
                [safe_float(row.get("failed_agents_delta")) for row in accepted_rows],
                dtype=np.float32,
            )
            categories = np.asarray(
                [row.get("outcome_category", "") for row in accepted_rows],
                dtype=object,
            )
            accepted = np.ones(len(accepted_rows), dtype=bool)
            row = metric_row(
                categories=categories,
                reward_delta=reward_delta,
                success_delta=success_delta,
                failed_delta=failed_delta,
                accepted=accepted,
                min_success_probability=score_threshold,
                max_unsafe_probability=max_risk_probability,
            )
            row.update(
                {
                    "ranker_score_threshold": float(score_threshold),
                    "veto_max_risk_probability": float(max_risk_probability),
                    "ranker_selected": len(score_rows),
                    "missing_risk_rows": missing,
                    "vetoed": len(score_rows) - len(accepted_rows) - missing,
                }
            )
            metrics.append(row)
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
        _train_target,
        train_labels,
    ) = risk_labels(train_rows, args)
    (
        validation_reward_delta,
        validation_success_delta,
        validation_failed_delta,
        validation_target,
        validation_labels,
    ) = risk_labels(validation_rows, args)
    validation_categories = np.asarray(
        [outcome_category(row, args.reward_epsilon) for row in validation_rows],
        dtype=object,
    )
    _, mean, std = standardize(train_x_raw, train_x_raw)
    train_x = (train_x_raw - mean) / std
    validation_x = (validation_x_raw - mean) / std
    models = train_risk_ensemble(
        split_seed=split_seed,
        train_x=train_x,
        train_y=train_labels,
        args=args,
    )
    risk_mean, risk_std = predict_ensemble(models, validation_x)
    risk_ucb = risk_mean + args.risk_std_coef * risk_std

    metrics = []
    audit_rows = []
    for max_risk_probability in args.max_risk_probability:
        predicted_risk = risk_ucb > max_risk_probability
        row = risk_metric_row(
            categories=validation_categories,
            reward_delta=validation_reward_delta,
            success_delta=validation_success_delta,
            failed_delta=validation_failed_delta,
            labels=validation_labels,
            predictions=predicted_risk,
            max_risk_probability=max_risk_probability,
        )
        row["split_seed"] = split_seed
        row["val_rows"] = len(validation_rows)
        metrics.append(row)
    if args.output_audit_csv or args.ranker_audit_csv:
        for row_index, row in enumerate(validation_rows):
            audit_rows.append(
                {
                    "split_seed": split_seed,
                    "row_index": row_index,
                    "seed": row.get("seed", ""),
                    "prefix_len": row.get("prefix_len", ""),
                    "forced_applied": row.get("forced_applied", ""),
                    "outcome_category": validation_categories[row_index],
                    "reward_delta": float(validation_reward_delta[row_index]),
                    "success_delta": float(validation_success_delta[row_index]),
                    "failed_agents_delta": float(validation_failed_delta[row_index]),
                    "target_value": float(validation_target[row_index]),
                    "risk_label": int(validation_labels[row_index]),
                    "risk_probability_mean": float(risk_mean[row_index]),
                    "risk_probability_std": float(risk_std[row_index]),
                    "risk_probability_ucb": float(risk_ucb[row_index]),
                    "events": row.get("events", ""),
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
        "risk_rows",
        "predicted_risk",
        "risk_true_positive",
        "risk_false_negative",
        "risk_false_positive",
        "val_rows",
    ]
    buckets: dict[float, dict[str, Any]] = {}
    for row in results:
        key = row["max_risk_probability"]
        bucket = buckets.setdefault(
            key,
            {"max_risk_probability": key, "splits": 0},
        )
        bucket["splits"] += 1
        for item in summed_keys:
            bucket[item] = bucket.get(item, 0) + row[item]
    aggregate = []
    for bucket in buckets.values():
        bucket["risk_recall"] = (
            bucket["risk_true_positive"] / bucket["risk_rows"]
            if bucket["risk_rows"]
            else 0.0
        )
        bucket["risk_precision"] = (
            bucket["risk_true_positive"] / bucket["predicted_risk"]
            if bucket["predicted_risk"]
            else 0.0
        )
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
            int(row["risk_false_negative"]),
            int(row["risk_false_positive"]),
            -float(row["risk_recall"]),
        )
    )
    return aggregate


def print_risk_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "max_risk_probability",
        "risk_rows",
        "predicted_risk",
        "risk_true_positive",
        "risk_false_negative",
        "risk_false_positive",
        "risk_recall",
        "risk_precision",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
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


def print_veto_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
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
                if isinstance(row[column], float)
                else str(row[column])
                for column in columns
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a dedicated reward-loss/risk veto head."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--validation-csv", nargs="+", type=Path)
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-prefix-features", action="store_true")
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--reward-loss-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--reward-loss-mode",
        choices=["all", "non_success"],
        default="non_success",
    )
    parser.add_argument("--target-risk-threshold", type=float, default=0.0)
    parser.add_argument("--success-weight", type=float, default=6.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-penalty", type=float, default=3.0)
    parser.add_argument("--success-loss-penalty", type=float, default=8.0)
    parser.add_argument("--failure-penalty", type=float, default=3.0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=127)
    parser.add_argument("--risk-positive-weight", type=float, default=30.0)
    parser.add_argument("--risk-negative-weight", type=float, default=1.0)
    parser.add_argument("--risk-std-coef", type=float, default=1.0)
    parser.add_argument(
        "--max-risk-probability",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    parser.add_argument("--ranker-audit-csv", type=Path)
    parser.add_argument(
        "--ranker-score-threshold",
        nargs="+",
        type=float,
        default=[1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--veto-max-risk-probability",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
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
                row
                for row, is_validation in zip(train_rows, validation_mask)
                if is_validation
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
    reward_delta, success_delta, failed_delta, target, labels = risk_labels(train_rows, args)
    veto_metrics = (
        ranker_veto_metrics(args.ranker_audit_csv, audit_rows, args)
        if args.ranker_audit_csv
        else []
    )
    print(
        "Data: "
        f"rows={len(train_rows) + len(validation_rows)} train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} good={categories['good']} "
        f"neutral={categories['neutral']} bad={categories['bad']} "
        f"risk_rows={int(labels.sum())} target_min={float(target.min()):.6g} "
        f"target_max={float(target.max()):.6g} features={len(feature_columns)} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print("Risk metrics:")
    print_risk_metrics(aggregate, args.top_k)
    if veto_metrics:
        print("Ranker veto metrics:")
        print_veto_metrics(veto_metrics, args.top_k)

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
                        "risk_rows": int(labels.sum()),
                        "target_min": float(target.min()),
                        "target_max": float(target.max()),
                        "features": len(feature_columns),
                        "feature_columns": feature_columns,
                        "args": vars(args),
                    },
                    "aggregate_metrics": aggregate,
                    "veto_metrics": veto_metrics,
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
