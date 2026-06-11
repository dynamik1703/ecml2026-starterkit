#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.evaluate_sequence_rescue_planner import (
    SequenceRescueNet,
    choose_event_feature_columns,
    choose_static_feature_columns,
    read_json_rows,
    sequence_feature_matrix,
    standardize_sequences,
    static_feature_matrix,
)
from tools.evaluate_sequence_rescue_ranker import (
    add_baseline_candidates,
    is_baseline_candidate,
    outcome_arrays,
    seed_key,
    target_values,
)
from tools.evaluate_success_rescue_classifier import (
    format_float,
    json_default,
    metric_row,
    write_csv,
)
from tools.train_counterfactual_gate import (
    outcome_category,
    seed_split,
    standardize,
)


def group_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[seed_key(row, row_index)].append(row_index)
    return groups


def choose_group_label(
    indices: list[int],
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> int:
    baseline_indices = [
        row_index for row_index in indices if is_baseline_candidate(rows[row_index])
    ]
    baseline_index = baseline_indices[0] if baseline_indices else None
    baseline_target = float(targets[baseline_index]) if baseline_index is not None else 0.0
    best_index = max(indices, key=lambda row_index: float(targets[row_index]))
    best_target = float(targets[best_index])

    if baseline_index is not None:
        if best_target <= baseline_target + args.noop_tie_margin:
            return baseline_index
        if best_target < args.min_winner_target:
            return baseline_index
    return best_index


def build_training_groups(
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[list[int]], list[int], np.ndarray]:
    groups = []
    labels = []
    weights = []
    for indices in group_indices(rows).values():
        if len(indices) < 2:
            continue
        label_index = choose_group_label(indices, rows, targets, args)
        local_label = indices.index(label_index)
        target_values_for_group = [float(targets[row_index]) for row_index in indices]
        baseline_targets = [
            float(targets[row_index])
            for row_index in indices
            if is_baseline_candidate(rows[row_index])
        ]
        baseline_target = baseline_targets[0] if baseline_targets else 0.0
        best_gap = max(target_values_for_group) - baseline_target
        worst_gap = baseline_target - min(target_values_for_group)
        weight = min(
            args.max_group_weight,
            args.base_group_weight
            + args.group_weight_scale * max(abs(best_gap), abs(worst_gap)),
        )
        groups.append(indices)
        labels.append(local_label)
        weights.append(weight)
    return groups, labels, np.asarray(weights, dtype=np.float32)


def train_group_member(
    seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    targets: np.ndarray,
    train_groups: list[list[int]],
    train_labels: list[int],
    group_weights: np.ndarray,
    args: argparse.Namespace,
) -> SequenceRescueNet:
    rng = np.random.default_rng(seed)
    if args.bootstrap and train_groups:
        sample_indices = rng.integers(0, len(train_groups), size=len(train_groups))
    else:
        sample_indices = np.arange(len(train_groups))

    static_tensor = torch.as_tensor(train_static, dtype=torch.float32)
    event_tensor = torch.as_tensor(train_events, dtype=torch.float32)
    event_mask_tensor = torch.as_tensor(train_event_mask, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)

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

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(static_tensor, event_tensor, event_mask_tensor)
        losses = []
        for sampled_index in sample_indices:
            indices = train_groups[int(sampled_index)]
            label = train_labels[int(sampled_index)]
            group_score = scores[torch.as_tensor(indices, dtype=torch.long)]
            ce_loss = F.cross_entropy(
                group_score.unsqueeze(0),
                torch.as_tensor([label], dtype=torch.long),
                reduction="none",
            )[0]
            losses.append(ce_loss * float(group_weights[int(sampled_index)]))
        if losses:
            group_loss = torch.stack(losses).sum() / torch.as_tensor(
                group_weights[sample_indices],
                dtype=torch.float32,
            ).sum().clamp_min(1.0)
        else:
            group_loss = torch.zeros((), dtype=torch.float32)
        point_loss = F.mse_loss(scores, target_tensor)
        loss = group_loss + args.pointwise_weight * point_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    model.eval()
    return model


def train_group_ensemble(
    split_seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    targets: np.ndarray,
    train_groups: list[list[int]],
    train_labels: list[int],
    group_weights: np.ndarray,
    args: argparse.Namespace,
) -> list[SequenceRescueNet]:
    return [
        train_group_member(
            seed=args.torch_seed + split_seed * 1000 + member,
            train_static=train_static,
            train_events=train_events,
            train_event_mask=train_event_mask,
            targets=targets,
            train_groups=train_groups,
            train_labels=train_labels,
            group_weights=group_weights,
            args=args,
        )
        for member in range(args.ensemble_size)
    ]


def predict_scores(
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
    scores = []
    with torch.no_grad():
        for model in models:
            scores.append(model(static_tensor, event_tensor, event_mask_tensor).cpu().numpy())
    stack = np.stack(scores, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def group_metric_row(
    validation_rows: list[dict[str, Any]],
    validation_categories: np.ndarray,
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    accepted: np.ndarray,
    margin_threshold: float,
) -> dict[str, Any]:
    row = metric_row(
        categories=validation_categories,
        reward_delta=reward_delta,
        success_delta=success_delta,
        failed_delta=failed_delta,
        accepted=accepted,
        min_success_probability=margin_threshold,
        max_unsafe_probability=0.0,
    )
    row["margin_threshold"] = margin_threshold
    row["selection_groups"] = len(
        {seed_key(row, row_index) for row_index, row in enumerate(validation_rows)}
    )
    return row


def select_by_group_margin(
    rows: list[dict[str, Any]],
    score_mean: np.ndarray,
    score_std: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, int | None], dict[str, float]]:
    selected: dict[str, int | None] = {}
    margins: dict[str, float] = {}
    for key, indices in group_indices(rows).items():
        baseline_indices = [
            row_index for row_index in indices if is_baseline_candidate(rows[row_index])
        ]
        baseline_score = 0.0
        baseline_uncertainty = 0.0
        if baseline_indices:
            baseline_index = baseline_indices[0]
            baseline_score = float(score_mean[baseline_index])
            baseline_uncertainty = float(score_std[baseline_index])

        candidate_indices = [
            row_index
            for row_index in indices
            if not is_baseline_candidate(rows[row_index])
        ]
        if not candidate_indices:
            selected[key] = None
            margins[key] = -math.inf
            continue

        best_candidate = max(
            candidate_indices,
            key=lambda row_index: float(score_mean[row_index])
            - args.score_std_coef * float(score_std[row_index]),
        )
        margin = (
            float(score_mean[best_candidate])
            - args.score_std_coef * float(score_std[best_candidate])
            - baseline_score
            - args.baseline_std_coef * baseline_uncertainty
        )
        selected[key] = best_candidate
        margins[key] = margin
    return selected, margins


def evaluate_split(
    split_seed: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    static_feature_columns: list[str],
    event_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    targets = target_values(train_rows, args)
    validation_targets = target_values(validation_rows, args)
    train_groups, train_labels, group_weights = build_training_groups(
        train_rows,
        targets,
        args,
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

    models = train_group_ensemble(
        split_seed=split_seed,
        train_static=train_static,
        train_events=train_events,
        train_event_mask=train_event_mask,
        targets=targets,
        train_groups=train_groups,
        train_labels=train_labels,
        group_weights=group_weights,
        args=args,
    )
    score_mean, score_std = predict_scores(
        models,
        validation_static,
        validation_events,
        validation_event_mask,
    )
    selected_by_group, margin_by_group = select_by_group_margin(
        validation_rows,
        score_mean,
        score_std,
        args,
    )

    reward_delta, success_delta, failed_delta = outcome_arrays(validation_rows)
    validation_categories = np.asarray(
        [outcome_category(row, args.reward_epsilon) for row in validation_rows],
        dtype=object,
    )
    selected_baseline_count = 0
    selected_nonbaseline_count = 0
    for key, selected_index in selected_by_group.items():
        if selected_index is None:
            selected_baseline_count += 1
        elif margin_by_group[key] >= 0.0:
            selected_nonbaseline_count += 1
        else:
            selected_baseline_count += 1

    metrics = []
    for margin_threshold in args.margin_threshold:
        accepted = np.zeros(len(validation_rows), dtype=bool)
        for key, row_index in selected_by_group.items():
            if row_index is not None and margin_by_group[key] >= margin_threshold:
                accepted[row_index] = True
        row = group_metric_row(
            validation_rows=validation_rows,
            validation_categories=validation_categories,
            reward_delta=reward_delta,
            success_delta=success_delta,
            failed_delta=failed_delta,
            accepted=accepted,
            margin_threshold=margin_threshold,
        )
        row["split_seed"] = split_seed
        row["val_rows"] = len(validation_rows)
        row["train_groups"] = len(train_groups)
        row["selected_baseline"] = selected_baseline_count
        row["selected_nonbaseline"] = selected_nonbaseline_count
        metrics.append(row)

    audit_rows = []
    if args.output_audit_csv:
        selected_indices = {
            row_index
            for row_index in selected_by_group.values()
            if row_index is not None
        }
        for row_index, row in enumerate(validation_rows):
            key = seed_key(row, row_index)
            is_selected = row_index in selected_indices
            base = {
                "split_seed": split_seed,
                "row_index": row_index,
                "seed": row.get("seed", ""),
                "prefix_len": row.get("prefix_len", ""),
                "forced_applied": row.get("forced_applied", ""),
                "outcome_category": validation_categories[row_index],
                "target_value": float(validation_targets[row_index]),
                "reward_delta": float(reward_delta[row_index]),
                "success_delta": float(success_delta[row_index]),
                "failed_agents_delta": float(failed_delta[row_index]),
                "group_score_mean": float(score_mean[row_index]),
                "group_score_std": float(score_std[row_index]),
                "rank_score_mean": float(score_mean[row_index]),
                "rank_score_std": float(score_std[row_index]),
                "rank_score_lcb": float(margin_by_group.get(key, -math.inf)),
                "selected_best_for_seed": int(is_selected),
                "selected_margin_lcb": float(margin_by_group.get(key, -math.inf)),
                "is_baseline_candidate": int(is_baseline_candidate(row)),
                "event_count": int(validation_event_mask[row_index].sum()),
                "events": row.get("events", ""),
            }
            for margin_threshold in args.margin_threshold:
                audit_rows.append(
                    {
                        **base,
                        "margin_threshold": margin_threshold,
                        "score_threshold": margin_threshold,
                        "accepted": int(
                            is_selected
                            and not is_baseline_candidate(row)
                            and margin_by_group.get(key, -math.inf) >= margin_threshold
                        ),
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
        "selection_groups",
        "train_groups",
        "selected_baseline",
        "selected_nonbaseline",
    ]
    buckets: dict[float, dict[str, Any]] = {}
    for row in results:
        key = row["margin_threshold"]
        bucket = buckets.setdefault(key, {"margin_threshold": key, "splits": 0})
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
        "margin_threshold",
        "selected_baseline",
        "selected_nonbaseline",
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
            "Evaluate listwise seed-level rescue selection against baseline/no-op."
        )
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--validation-json", nargs="+", type=Path)
    parser.add_argument(
        "--add-baseline-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add one synthetic no-op/baseline candidate per seed group.",
    )
    parser.add_argument("--include-feature-regex")
    parser.add_argument("--exclude-feature-regex")
    parser.add_argument("--drop-prefix-features", action="store_true")
    parser.add_argument(
        "--drop-aggregate-event-features",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-seed", type=int, default=211)
    parser.add_argument("--success-weight", type=float, default=6.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-penalty", type=float, default=3.0)
    parser.add_argument("--success-loss-penalty", type=float, default=8.0)
    parser.add_argument("--failure-penalty", type=float, default=3.0)
    parser.add_argument("--min-winner-target", type=float, default=0.01)
    parser.add_argument("--noop-tie-margin", type=float, default=0.01)
    parser.add_argument("--base-group-weight", type=float, default=1.0)
    parser.add_argument("--group-weight-scale", type=float, default=1.0)
    parser.add_argument("--max-group-weight", type=float, default=8.0)
    parser.add_argument("--pointwise-weight", type=float, default=0.1)
    parser.add_argument("--score-std-coef", type=float, default=0.5)
    parser.add_argument("--baseline-std-coef", type=float, default=0.5)
    parser.add_argument(
        "--margin-threshold",
        nargs="+",
        type=float,
        default=[-1.0, -0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-audit-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_json_rows(args.json)
    validation_rows = read_json_rows(args.validation_json) if args.validation_json else []
    if args.add_baseline_candidate:
        train_rows = add_baseline_candidates(train_rows)
        if validation_rows:
            validation_rows = add_baseline_candidates(validation_rows)
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
            static_feature_columns=static_feature_columns,
            event_feature_columns=event_feature_columns,
            args=args,
        )
        split_results.extend(metrics)
        audit_rows.extend(split_audit)

    aggregate = aggregate_metrics(split_results)
    categories = Counter(outcome_category(row, args.reward_epsilon) for row in train_rows)
    targets = target_values(train_rows, args)
    train_groups, _labels, _weights = build_training_groups(train_rows, targets, args)
    event_counts = [
        len(row.get("event_details") or [])
        for row in train_rows + validation_rows
        if isinstance(row.get("event_details") or [], list)
    ]
    print(
        "Data: "
        f"rows={len(train_rows) + len(validation_rows)} train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} train_groups={len(train_groups)} "
        f"good={categories['good']} neutral={categories['neutral']} "
        f"bad={categories['bad']} target_min={float(targets.min()):.6g} "
        f"target_max={float(targets.max()):.6g} "
        f"static_features={len(static_feature_columns)} "
        f"event_features={len(event_feature_columns)} "
        f"max_events={args.max_events} "
        f"observed_max_events={max(event_counts) if event_counts else 0} "
        f"splits={','.join(str(seed) for seed in args.split_seeds)}"
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
                        "train_groups": len(train_groups),
                        "categories": dict(categories),
                        "target_min": float(targets.min()),
                        "target_max": float(targets.max()),
                        "static_features": len(static_feature_columns),
                        "event_features": len(event_feature_columns),
                        "static_feature_columns": static_feature_columns,
                        "event_feature_columns": event_feature_columns,
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
