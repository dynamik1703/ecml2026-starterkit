#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from tools.train_counterfactual_gate import (
    outcome_category,
    read_rows,
    train,
)


PREFIX_FEATURE_REGEX = r"^(baseline_prefix_|forced_prefix_)"


def feature_filter(feature_set: str) -> tuple[str | None, str | None, bool]:
    if feature_set == "all":
        return None, None, False
    if feature_set == "without_prefix":
        return None, PREFIX_FEATURE_REGEX, False
    if feature_set == "only_prefix":
        return PREFIX_FEATURE_REGEX, None, False
    raise ValueError(f"Unsupported feature set: {feature_set}")


def aggregate_metrics(
    csvs: list[Path],
    args: argparse.Namespace,
    objective: str,
    feature_set: str,
) -> dict[str, Any]:
    include_regex, exclude_regex, drop_prefix_features = feature_filter(feature_set)
    aggregate = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "accepted_good": 0,
            "accepted_neutral": 0,
            "accepted_bad": 0,
        }
        for threshold in args.thresholds
    }
    summaries = []
    for split_seed in args.split_seeds:
        summary = train(
            argparse.Namespace(
                csv=csvs,
                include_feature_regex=include_regex,
                exclude_feature_regex=exclude_regex,
                drop_prefix_features=drop_prefix_features,
                reward_epsilon=args.reward_epsilon,
                val_fraction=args.val_fraction,
                split_seed=split_seed,
                ordered_seed_split=False,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                objective=objective,
                max_bad_probability=args.max_bad_probability,
                positive_weight=args.positive_weight,
                neutral_negative_weight=args.neutral_negative_weight,
                bad_negative_weight=args.bad_negative_weight,
                torch_seed=args.torch_seed,
                thresholds=args.thresholds,
                output_checkpoint=None,
                output_json=None,
            )
        )
        summaries.append(summary)
        for metric in summary["val_metrics"]:
            bucket = aggregate[metric["threshold"]]
            for key in bucket:
                bucket[key] += metric[key]

    metrics = []
    for threshold, bucket in aggregate.items():
        tp = bucket["tp"]
        fp = bucket["fp"]
        fn = bucket["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.append(
            {
                "threshold": threshold,
                **bucket,
                "precision": precision,
                "recall": recall,
            }
        )
    return {
        "objective": objective,
        "feature_set": feature_set,
        "features": summaries[0]["features"] if summaries else 0,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate counterfactual gate classifiers across seed splits."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=["binary", "multiclass"],
        default=["binary", "multiclass"],
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["all", "without_prefix", "only_prefix"],
        default=["all", "without_prefix", "only_prefix"],
    )
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[1, 3, 5, 7, 11])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.75, 0.9, 0.95])
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-bad-probability", type=float, default=0.01)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--neutral-negative-weight", type=float, default=0.25)
    parser.add_argument("--bad-negative-weight", type=float, default=8.0)
    parser.add_argument("--torch-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv)
    categories = Counter(outcome_category(row, args.reward_epsilon) for row in rows)
    print(
        "Data: "
        f"rows={len(rows)} good={categories['good']} "
        f"neutral={categories['neutral']} bad={categories['bad']}"
    )
    print(
        "Validation: "
        f"splits={','.join(str(seed) for seed in args.split_seeds)} "
        f"thresholds={','.join(str(threshold) for threshold in args.thresholds)}"
    )

    for objective in args.objectives:
        for feature_set in args.feature_sets:
            result = aggregate_metrics(args.csv, args, objective, feature_set)
            print(
                f"\nobjective={objective} feature_set={feature_set} "
                f"features={result['features']}"
            )
            for metric in result["metrics"]:
                print(
                    "  threshold={threshold:.3f} precision={precision:.3f} "
                    "recall={recall:.3f} tp/fp/fn/tn={tp}/{fp}/{fn}/{tn} "
                    "accepted(g/n/b)={accepted_good}/{accepted_neutral}/"
                    "{accepted_bad}".format(**metric)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
