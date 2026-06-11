#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tools.evaluate_sequence_rescue_planner import (
    SequenceRescueNet,
    choose_event_feature_columns,
    choose_static_feature_columns,
    read_json_rows,
    sequence_feature_matrix,
    standardize_sequences,
    static_feature_matrix,
)
from tools.evaluate_success_rescue_classifier import (
    format_float,
    json_default,
    metric_row,
    write_csv,
)
from tools.train_counterfactual_gate import (
    outcome_category,
    safe_float,
    seed_split,
    standardize,
)


def outcome_arrays(
    rows: list[dict[str, Any]],
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


def is_baseline_candidate(row: dict[str, Any]) -> bool:
    value = safe_float(row.get("is_baseline_candidate"))
    return math.isfinite(value) and value > 0.5


def add_baseline_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        groups.setdefault(seed_key(row, row_index), row)

    augmented = list(rows)
    for seed, template in sorted(groups.items(), key=lambda item: item[0]):
        baseline_reward = safe_float(template.get("baseline_reward"))
        baseline_success = safe_float(template.get("baseline_success"))
        baseline_failed_agents = safe_float(template.get("baseline_failed_agents"))
        baseline = {
            "seed": template.get("seed", seed),
            "scene": template.get("scene", ""),
            "num_agents": template.get("num_agents", ""),
            "line_length": template.get("line_length", ""),
            "prefix_len": 0,
            "forced_applied": 0,
            "baseline_reward": baseline_reward if math.isfinite(baseline_reward) else "",
            "forced_reward": baseline_reward if math.isfinite(baseline_reward) else "",
            "reward_delta": 0.0,
            "baseline_success": baseline_success if math.isfinite(baseline_success) else "",
            "forced_success": baseline_success if math.isfinite(baseline_success) else "",
            "success_delta": 0.0,
            "baseline_failed_agents": (
                baseline_failed_agents if math.isfinite(baseline_failed_agents) else ""
            ),
            "forced_failed_agents": (
                baseline_failed_agents if math.isfinite(baseline_failed_agents) else ""
            ),
            "failed_agents_delta": 0.0,
            "events": "",
            "event_details": [],
            "event_count": 0,
            "event_unique_agents": 0,
            "is_baseline_candidate": 1.0,
            "candidate_source_baseline_noop": 1.0,
        }
        augmented.append(baseline)
    return augmented


def target_values(rows: list[dict[str, Any]], args: argparse.Namespace) -> np.ndarray:
    reward_delta, success_delta, failed_delta = outcome_arrays(rows)
    return (
        args.success_weight * success_delta
        + args.reward_weight * reward_delta
        - args.reward_loss_penalty * np.maximum(-reward_delta, 0.0)
        - args.success_loss_penalty * np.maximum(-success_delta, 0.0)
        - args.failure_penalty * np.maximum(failed_delta, 0.0)
    ).astype(np.float32)


def seed_key(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("seed", row_index))


def grouped_pair_indices(
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[seed_key(row, row_index)].append(row_index)

    winners = []
    losers = []
    weights = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for left, right in combinations(indices, 2):
            delta = float(targets[left] - targets[right])
            if abs(delta) < args.min_target_delta:
                continue
            if delta > 0:
                winner, loser = left, right
            else:
                winner, loser = right, left
            winners.append(winner)
            losers.append(loser)
            weights.append(
                min(
                    args.max_pair_weight,
                    args.base_pair_weight + args.pair_weight_scale * abs(delta),
                )
            )
    return (
        np.asarray(winners, dtype=np.int64),
        np.asarray(losers, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def train_rank_member(
    seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    targets: np.ndarray,
    pair_winners: np.ndarray,
    pair_losers: np.ndarray,
    pair_weights: np.ndarray,
    args: argparse.Namespace,
) -> SequenceRescueNet:
    rng = np.random.default_rng(seed)
    if args.bootstrap and len(pair_winners):
        sample_indices = rng.integers(0, len(pair_winners), size=len(pair_winners))
    else:
        sample_indices = np.arange(len(pair_winners))

    static_tensor = torch.as_tensor(train_static, dtype=torch.float32)
    event_tensor = torch.as_tensor(train_events, dtype=torch.float32)
    event_mask_tensor = torch.as_tensor(train_event_mask, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    winner_tensor = torch.as_tensor(pair_winners[sample_indices], dtype=torch.long)
    loser_tensor = torch.as_tensor(pair_losers[sample_indices], dtype=torch.long)
    pair_weight_tensor = torch.as_tensor(pair_weights[sample_indices], dtype=torch.float32)

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
    pair_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    point_loss_fn = nn.MSELoss()
    ones = torch.ones(len(winner_tensor), dtype=torch.float32)

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(static_tensor, event_tensor, event_mask_tensor)
        if len(winner_tensor):
            pair_logits = scores[winner_tensor] - scores[loser_tensor]
            pair_loss = (
                pair_loss_fn(pair_logits, ones) * pair_weight_tensor
            ).sum() / pair_weight_tensor.sum().clamp_min(1.0)
        else:
            pair_loss = torch.zeros((), dtype=torch.float32)
        point_loss = point_loss_fn(scores, target_tensor)
        loss = pair_loss + args.pointwise_weight * point_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    model.eval()
    return model


def train_rank_ensemble(
    split_seed: int,
    train_static: np.ndarray,
    train_events: np.ndarray,
    train_event_mask: np.ndarray,
    targets: np.ndarray,
    pair_winners: np.ndarray,
    pair_losers: np.ndarray,
    pair_weights: np.ndarray,
    args: argparse.Namespace,
) -> list[SequenceRescueNet]:
    models = []
    for member in range(args.ensemble_size):
        models.append(
            train_rank_member(
                seed=args.torch_seed + split_seed * 1000 + member,
                train_static=train_static,
                train_events=train_events,
                train_event_mask=train_event_mask,
                targets=targets,
                pair_winners=pair_winners,
                pair_losers=pair_losers,
                pair_weights=pair_weights,
                args=args,
            )
        )
    return models


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


def ranker_metric_row(
    validation_rows: list[dict[str, Any]],
    validation_categories: np.ndarray,
    reward_delta: np.ndarray,
    success_delta: np.ndarray,
    failed_delta: np.ndarray,
    accepted: np.ndarray,
    score_threshold: float,
) -> dict[str, Any]:
    row = metric_row(
        categories=validation_categories,
        reward_delta=reward_delta,
        success_delta=success_delta,
        failed_delta=failed_delta,
        accepted=accepted,
        min_success_probability=score_threshold,
        max_unsafe_probability=0.0,
    )
    row["score_threshold"] = score_threshold
    row["selection_groups"] = len(
        {seed_key(row, row_index) for row_index, row in enumerate(validation_rows)}
    )
    return row


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
    pair_winners, pair_losers, pair_weights = grouped_pair_indices(
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

    models = train_rank_ensemble(
        split_seed=split_seed,
        train_static=train_static,
        train_events=train_events,
        train_event_mask=train_event_mask,
        targets=targets,
        pair_winners=pair_winners,
        pair_losers=pair_losers,
        pair_weights=pair_weights,
        args=args,
    )
    score_mean, score_std = predict_scores(
        models,
        validation_static,
        validation_events,
        validation_event_mask,
    )
    score_lcb = score_mean - args.score_std_coef * score_std
    reward_delta, success_delta, failed_delta = outcome_arrays(validation_rows)
    validation_categories = np.asarray(
        [outcome_category(row, args.reward_epsilon) for row in validation_rows],
        dtype=object,
    )

    best_by_seed: dict[str, int] = {}
    for row_index, row in enumerate(validation_rows):
        key = seed_key(row, row_index)
        current = best_by_seed.get(key)
        if current is None or score_lcb[row_index] > score_lcb[current]:
            best_by_seed[key] = row_index
    selected_baseline_count = sum(
        int(is_baseline_candidate(validation_rows[row_index]))
        for row_index in best_by_seed.values()
    )
    selected_nonbaseline_count = len(best_by_seed) - selected_baseline_count

    metrics = []
    for score_threshold in args.score_threshold:
        accepted = np.zeros(len(validation_rows), dtype=bool)
        for row_index in best_by_seed.values():
            if (
                score_lcb[row_index] >= score_threshold
                and not is_baseline_candidate(validation_rows[row_index])
            ):
                accepted[row_index] = True
        row = ranker_metric_row(
            validation_rows=validation_rows,
            validation_categories=validation_categories,
            reward_delta=reward_delta,
            success_delta=success_delta,
            failed_delta=failed_delta,
            accepted=accepted,
            score_threshold=score_threshold,
        )
        row["split_seed"] = split_seed
        row["val_rows"] = len(validation_rows)
        row["train_pairs"] = len(pair_winners)
        row["selected_baseline"] = selected_baseline_count
        row["selected_nonbaseline"] = selected_nonbaseline_count
        metrics.append(row)

    audit_rows = []
    if args.output_audit_csv:
        for row_index, row in enumerate(validation_rows):
            key = seed_key(row, row_index)
            is_best = best_by_seed.get(key) == row_index
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
                "rank_score_mean": float(score_mean[row_index]),
                "rank_score_std": float(score_std[row_index]),
                "rank_score_lcb": float(score_lcb[row_index]),
                "selected_best_for_seed": int(is_best),
                "is_baseline_candidate": int(is_baseline_candidate(row)),
                "event_count": int(validation_event_mask[row_index].sum()),
                "events": row.get("events", ""),
            }
            for score_threshold in args.score_threshold:
                audit_rows.append(
                    {
                        **base,
                        "score_threshold": score_threshold,
                        "accepted": int(
                            is_best
                            and score_lcb[row_index] >= score_threshold
                            and not is_baseline_candidate(row)
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
        "train_pairs",
        "selected_baseline",
        "selected_nonbaseline",
    ]
    buckets: dict[float, dict[str, Any]] = {}
    for row in results:
        key = row["score_threshold"]
        bucket = buckets.setdefault(
            key,
            {
                "score_threshold": key,
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
        "score_threshold",
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
            "Evaluate seed-level ranking for counterfactual rescue prefix selection."
        )
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--validation-json", nargs="+", type=Path)
    parser.add_argument(
        "--add-baseline-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--torch-seed", type=int, default=109)
    parser.add_argument("--success-weight", type=float, default=6.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-penalty", type=float, default=3.0)
    parser.add_argument("--success-loss-penalty", type=float, default=8.0)
    parser.add_argument("--failure-penalty", type=float, default=3.0)
    parser.add_argument("--min-target-delta", type=float, default=0.01)
    parser.add_argument("--base-pair-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight-scale", type=float, default=1.0)
    parser.add_argument("--max-pair-weight", type=float, default=5.0)
    parser.add_argument("--pointwise-weight", type=float, default=0.25)
    parser.add_argument("--score-std-coef", type=float, default=0.5)
    parser.add_argument(
        "--score-threshold",
        nargs="+",
        type=float,
        default=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
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
        f"target_min={float(targets.min()):.6g} target_max={float(targets.max()):.6g} "
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
