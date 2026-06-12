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

from tools.evaluate_success_rescue_classifier import (
    aggregate_metrics,
    format_float,
    json_default,
    metric_row,
    row_arrays,
    write_csv,
)
from tools.train_counterfactual_gate import (
    choose_feature_columns,
    filter_feature_columns,
    outcome_category,
    safe_float,
    seed_split,
    standardize,
)


EXTRA_OUTCOME_COLUMNS = {
    "candidate_reward",
    "candidate_success",
    "candidate_reward_delta",
    "candidate_success_delta",
    "candidate_failed_agent_ids",
    "outcome_category",
}

EVENT_EXCLUDE_COLUMNS = {
    "seed",
    "has_diff",
    "agent_id",
    "position",
    "baseline_target",
    "candidate_target",
    "baseline_action_name",
    "candidate_action_name",
    "baseline_raw_action_name",
    "candidate_raw_action_name",
    "state",
}

EVENT_CATEGORICAL_FIELDS = (
    "baseline_action_name",
    "candidate_action_name",
    "baseline_raw_action_name",
    "candidate_raw_action_name",
    "state",
)

FEATURE_VALUE_CLIP = 1_000.0


def bounded_feature_value(value: Any) -> float:
    numeric_value = safe_float(value)
    if not math.isfinite(numeric_value):
        return 0.0
    return float(np.clip(numeric_value, -FEATURE_VALUE_CLIP, FEATURE_VALUE_CLIP))


class SequenceRescueNet(nn.Module):
    def __init__(
        self,
        static_dim: int,
        event_dim: int,
        hidden_size: int,
        event_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.static_encoder = (
            nn.Sequential(nn.Linear(static_dim, hidden_size), nn.ReLU())
            if static_dim > 0
            else None
        )
        self.event_encoder = (
            nn.Sequential(nn.Linear(event_dim, event_hidden_size), nn.ReLU())
            if event_dim > 0
            else None
        )
        self.event_gru = (
            nn.GRU(event_hidden_size, hidden_size, batch_first=True)
            if event_dim > 0
            else None
        )
        combined_dim = hidden_size * (1 + int(static_dim > 0))
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        static_features: torch.Tensor,
        event_features: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        if self.event_encoder is not None and self.event_gru is not None:
            encoded_events = self.event_encoder(event_features)
            output, _hidden = self.event_gru(encoded_events)
            lengths = event_mask.sum(dim=1).long().clamp_min(1)
            indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
            event_summary = output.gather(dim=1, index=indices).squeeze(1)
            event_summary = torch.where(
                (event_mask.sum(dim=1, keepdim=True) > 0),
                event_summary,
                torch.zeros_like(event_summary),
            )
            parts.append(event_summary)
        if self.static_encoder is not None:
            parts.append(self.static_encoder(static_features))
        if not parts:
            raise RuntimeError("SequenceRescueNet needs static or event features")
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def read_json_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            path_rows = payload.get("rows", [])
        else:
            path_rows = payload
        if not isinstance(path_rows, list):
            raise ValueError(f"{path} does not contain a rows list")
        for row in path_rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(row)
    return rows


def choose_static_feature_columns(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    columns = filter_feature_columns(choose_feature_columns(rows), args)
    filtered = []
    for column in columns:
        if column in EXTRA_OUTCOME_COLUMNS:
            continue
        if args.drop_aggregate_event_features and column.startswith("event_"):
            continue
        filtered.append(column)
    return filtered


def static_feature_matrix(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(feature_columns):
            matrix[row_index, column_index] = bounded_feature_value(row.get(column))
    return matrix


def all_event_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        details = row.get("event_details") or []
        if not isinstance(details, list):
            continue
        events.extend(event for event in details if isinstance(event, dict))
    return events


def regex_filter(names: list[str], include: str | None, exclude: str | None) -> list[str]:
    if include:
        include_regex = re.compile(include)
        names = [name for name in names if include_regex.search(name)]
    if exclude:
        exclude_regex = re.compile(exclude)
        names = [name for name in names if not exclude_regex.search(name)]
    return names


def choose_event_feature_columns(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    events = all_event_details(rows)
    numeric_columns = []
    for column in sorted({key for event in events for key in event}):
        if column in EVENT_EXCLUDE_COLUMNS:
            continue
        values = [safe_float(event.get(column)) for event in events]
        if any(math.isfinite(value) for value in values):
            numeric_columns.append(f"num:{column}")

    categorical_columns = []
    if not args.drop_event_categorical_features:
        for field in EVENT_CATEGORICAL_FIELDS:
            values = sorted(
                {
                    str(event.get(field))
                    for event in events
                    if event.get(field) not in (None, "")
                }
            )
            categorical_columns.extend(f"cat:{field}:{value}" for value in values)
        transitions = sorted(
            {
                f"{event.get('baseline_action_name')}->{event.get('candidate_action_name')}"
                for event in events
                if event.get("baseline_action_name") not in (None, "")
                and event.get("candidate_action_name") not in (None, "")
            }
        )
        categorical_columns.extend(f"transition:{value}" for value in transitions)

    columns = numeric_columns + categorical_columns
    columns = regex_filter(
        columns,
        args.include_event_feature_regex,
        args.exclude_event_feature_regex,
    )
    if args.exclude_feature_regex:
        columns = regex_filter(columns, None, args.exclude_feature_regex)
    return columns


def event_feature_value(event: dict[str, Any], feature: str) -> float:
    if feature.startswith("num:"):
        return bounded_feature_value(event.get(feature[4:]))
    if feature.startswith("cat:"):
        _prefix, field, value = feature.split(":", 2)
        return float(str(event.get(field)) == value)
    if feature.startswith("transition:"):
        value = feature[len("transition:") :]
        transition = f"{event.get('baseline_action_name')}->{event.get('candidate_action_name')}"
        return float(transition == value)
    raise ValueError(f"Unknown event feature kind: {feature}")


def sequence_feature_matrix(
    rows: list[dict[str, Any]],
    event_feature_columns: list[str],
    max_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros(
        (len(rows), max_events, len(event_feature_columns)),
        dtype=np.float32,
    )
    mask = np.zeros((len(rows), max_events), dtype=np.float32)
    for row_index, row in enumerate(rows):
        details = row.get("event_details") or []
        if not isinstance(details, list):
            details = []
        for event_index, event in enumerate(details[:max_events]):
            if not isinstance(event, dict):
                continue
            mask[row_index, event_index] = 1.0
            for feature_index, feature in enumerate(event_feature_columns):
                matrix[row_index, event_index, feature_index] = event_feature_value(
                    event,
                    feature,
                )
    return matrix, mask


def standardize_sequences(
    train_events: np.ndarray,
    train_mask: np.ndarray,
    all_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_events.shape[-1] == 0:
        mean = np.zeros((0,), dtype=np.float32)
        std = np.ones((0,), dtype=np.float32)
        return all_events, mean, std
    flat = train_events[train_mask.astype(bool)]
    if flat.size == 0:
        flat = train_events.reshape(-1, train_events.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return (all_events - mean) / std, mean, std


def train_binary_member(
    seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    train_y: np.ndarray,
    positive_weight: float,
    negative_weight: float,
    args: argparse.Namespace,
) -> SequenceRescueNet:
    rng = np.random.default_rng(seed)
    if args.bootstrap:
        sample_indices = rng.integers(0, len(train_y), size=len(train_y))
    else:
        sample_indices = np.arange(len(train_y))

    static_tensor = torch.as_tensor(train_static[sample_indices], dtype=torch.float32)
    event_tensor = torch.as_tensor(train_events[sample_indices], dtype=torch.float32)
    event_mask_tensor = torch.as_tensor(train_event_mask[sample_indices], dtype=torch.float32)
    y = torch.as_tensor(train_y[sample_indices], dtype=torch.float32)
    weights = torch.where(
        y > 0.5,
        torch.full_like(y, float(positive_weight)),
        torch.full_like(y, float(negative_weight)),
    )

    torch.manual_seed(seed)
    model = SequenceRescueNet(
        static_dim=train_static.shape[1],
        event_dim=train_events.shape[2],
        hidden_size=args.hidden_size,
        event_hidden_size=args.event_hidden_size,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(static_tensor, event_tensor, event_mask_tensor)
        loss = (loss_fn(logits, y) * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    model.eval()
    return model


def predict_ensemble(
    models: list[SequenceRescueNet],
    static_x: np.ndarray,
    event_x: np.ndarray,
    event_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(static_x) == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    static_tensor = torch.as_tensor(static_x, dtype=torch.float32)
    event_tensor = torch.as_tensor(event_x, dtype=torch.float32)
    event_mask_tensor = torch.as_tensor(event_mask, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(
                torch.sigmoid(
                    model(static_tensor, event_tensor, event_mask_tensor)
                )
                .cpu()
                .numpy()
            )
    stack = np.stack(probs, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def train_ensembles(
    split_seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    train_success: np.ndarray,
    train_unsafe: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[SequenceRescueNet], list[SequenceRescueNet]]:
    success_models = []
    unsafe_models = []
    for member in range(args.ensemble_size):
        base_seed = args.torch_seed + split_seed * 1000 + member
        success_models.append(
            train_binary_member(
                seed=base_seed,
                train_static=train_static,
                train_events=train_events,
                train_event_mask=train_event_mask,
                train_y=train_success,
                positive_weight=args.success_positive_weight,
                negative_weight=args.success_negative_weight,
                args=args,
            )
        )
        unsafe_models.append(
            train_binary_member(
                seed=base_seed + 100_000,
                train_static=train_static,
                train_events=train_events,
                train_event_mask=train_event_mask,
                train_y=train_unsafe,
                positive_weight=args.unsafe_positive_weight,
                negative_weight=args.unsafe_negative_weight,
                args=args,
            )
        )
    return success_models, unsafe_models


def evaluate_split(
    split_seed: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    static_feature_columns: list[str],
    event_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_static_raw = static_feature_matrix(train_rows, static_feature_columns)
    validation_static_raw = static_feature_matrix(validation_rows, static_feature_columns)
    train_events_raw, train_event_mask = sequence_feature_matrix(
        train_rows,
        event_feature_columns,
        args.max_events,
    )
    validation_events_raw, validation_event_mask = sequence_feature_matrix(
        validation_rows,
        event_feature_columns,
        args.max_events,
    )
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

    _, static_mean, static_std = standardize(train_static_raw, train_static_raw)
    train_static = (train_static_raw - static_mean) / static_std
    validation_static = (validation_static_raw - static_mean) / static_std
    combined_events = np.concatenate([train_events_raw, validation_events_raw], axis=0)
    combined_mask = np.concatenate([train_event_mask, validation_event_mask], axis=0)
    standardized_events, _event_mean, _event_std = standardize_sequences(
        train_events=train_events_raw,
        train_mask=train_event_mask,
        all_events=combined_events,
    )
    train_events = standardized_events[: len(train_rows)]
    validation_events = standardized_events[len(train_rows) :]
    train_event_mask = combined_mask[: len(train_rows)]
    validation_event_mask = combined_mask[len(train_rows) :]

    success_models, unsafe_models = train_ensembles(
        split_seed=split_seed,
        train_static=train_static,
        train_events=train_events,
        train_event_mask=train_event_mask,
        train_success=train_success_labels,
        train_unsafe=train_unsafe_labels,
        args=args,
    )
    success_mean, success_std = predict_ensemble(
        success_models,
        validation_static,
        validation_events,
        validation_event_mask,
    )
    unsafe_mean, unsafe_std = predict_ensemble(
        unsafe_models,
        validation_static,
        validation_events,
        validation_event_mask,
    )
    success_lcb = success_mean - args.success_std_coef * success_std
    unsafe_ucb = unsafe_mean + args.unsafe_std_coef * unsafe_std

    metrics = []
    planner_metrics = []
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

    for unsafe_weight in args.planner_unsafe_weight:
        planner_score = success_lcb - unsafe_weight * unsafe_ucb
        best_by_seed: dict[str, int] = {}
        for row_index, row in enumerate(validation_rows):
            seed_key = str(row.get("seed", row_index))
            current = best_by_seed.get(seed_key)
            if current is None or planner_score[row_index] > planner_score[current]:
                best_by_seed[seed_key] = row_index
        for score_threshold in args.planner_score_threshold:
            accepted = np.zeros(len(validation_rows), dtype=bool)
            for row_index in best_by_seed.values():
                if planner_score[row_index] >= score_threshold:
                    accepted[row_index] = True
            row = metric_row(
                categories=validation_categories,
                reward_delta=validation_reward_delta,
                success_delta=validation_success_delta,
                failed_delta=validation_failed_delta,
                accepted=accepted,
                min_success_probability=score_threshold,
                max_unsafe_probability=unsafe_weight,
            )
            row["split_seed"] = split_seed
            row["val_rows"] = len(validation_rows)
            row["selection_groups"] = len(best_by_seed)
            row["planner_score_threshold"] = score_threshold
            row["planner_unsafe_weight"] = unsafe_weight
            planner_metrics.append(row)

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
                "event_count": int(validation_event_mask[row_index].sum()),
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
    return metrics, planner_metrics, audit_rows


def aggregate_planner_metrics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        "selection_groups",
    ]
    buckets: dict[tuple[float, float], dict[str, Any]] = {}
    for row in results:
        key = (row["planner_unsafe_weight"], row["planner_score_threshold"])
        bucket = buckets.setdefault(
            key,
            {
                "planner_unsafe_weight": row["planner_unsafe_weight"],
                "planner_score_threshold": row["planner_score_threshold"],
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
            -float(row["accepted_success_delta_sum"]),
            -float(row["accepted_reward_delta_sum"]),
            -int(row["accepted_success_positive"]),
        )
    )
    return aggregate


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


def print_planner_metrics(metrics: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "planner_unsafe_weight",
        "planner_score_threshold",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an ordered prefix-event sequence model for counterfactual "
            "Success-rescue selection."
        )
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--validation-json", nargs="+", type=Path)
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-prefix-features", action="store_true")
    parser.add_argument(
        "--drop-aggregate-event-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop flattened top-level event_* aggregate features from static input.",
    )
    parser.add_argument("--include-event-feature-regex")
    parser.add_argument("--exclude-event-feature-regex")
    parser.add_argument(
        "--drop-event-categorical-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--event-hidden-size", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=73)
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
    )
    parser.add_argument(
        "--min-success-probability",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.85, 0.95],
    )
    parser.add_argument(
        "--max-unsafe-probability",
        nargs="+",
        type=float,
        default=[0.0005, 0.001, 0.0025, 0.005, 0.01],
    )
    parser.add_argument(
        "--planner-unsafe-weight",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
        help="Weights for best-by-seed planner score success_lcb - weight * unsafe_ucb.",
    )
    parser.add_argument(
        "--planner-score-threshold",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75],
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-audit-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_json_rows(args.json)
    validation_rows = read_json_rows(args.validation_json) if args.validation_json else []
    rows_for_features = train_rows + validation_rows
    if not train_rows:
        raise ValueError("No rows found")
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive")

    static_feature_columns = choose_static_feature_columns(rows_for_features, args)
    event_feature_columns = choose_event_feature_columns(rows_for_features, args)
    if not static_feature_columns and not event_feature_columns:
        raise ValueError("No static or event feature columns found")

    split_results = []
    planner_results = []
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
        metrics, split_planner_metrics, split_audit = evaluate_split(
            split_seed=split_seed,
            train_rows=split_train_rows,
            validation_rows=split_validation_rows,
            static_feature_columns=static_feature_columns,
            event_feature_columns=event_feature_columns,
            args=args,
        )
        split_results.extend(metrics)
        planner_results.extend(split_planner_metrics)
        audit_rows.extend(split_audit)

    aggregate = aggregate_metrics(split_results)
    planner_aggregate = aggregate_planner_metrics(planner_results)
    categories = Counter(outcome_category(row, args.reward_epsilon) for row in train_rows)
    _, _, _, success_labels, unsafe_labels = row_arrays(train_rows, args)
    success_rows = int(success_labels.sum())
    unsafe_rows = int(unsafe_labels.sum())
    event_counts = [
        len(row.get("event_details") or [])
        for row in train_rows + validation_rows
        if isinstance(row.get("event_details") or [], list)
    ]
    print(
        "Data: "
        f"rows={len(train_rows) + len(validation_rows)} train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} good={categories['good']} "
        f"neutral={categories['neutral']} bad={categories['bad']} "
        f"success_positive_rows={success_rows} unsafe_rows={unsafe_rows} "
        f"static_features={len(static_feature_columns)} "
        f"event_features={len(event_feature_columns)} "
        f"max_events={args.max_events} "
        f"observed_max_events={max(event_counts) if event_counts else 0} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
    )
    print("Threshold metrics:")
    print_metrics(aggregate, args.top_k)
    print("Planner metrics:")
    print_planner_metrics(planner_aggregate, args.top_k)

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
                        "static_features": len(static_feature_columns),
                        "event_features": len(event_feature_columns),
                        "static_feature_columns": static_feature_columns,
                        "event_feature_columns": event_feature_columns,
                        "args": vars(args),
                    },
                    "aggregate_metrics": aggregate,
                    "planner_aggregate_metrics": planner_aggregate,
                    "split_metrics": split_results,
                    "planner_split_metrics": planner_results,
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
