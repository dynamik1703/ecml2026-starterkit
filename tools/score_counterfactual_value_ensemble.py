#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.evaluate_counterfactual_value_ensemble import (
    ValueRiskMLP,
    acceptance_mask,
    json_default,
    metric_row,
    predict_ensemble,
    row_arrays,
)
from tools.train_counterfactual_gate import feature_matrix, read_rows


def first_config_value(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, list):
        return float(value[0]) if value else default
    if value is None:
        return default
    return float(value)


def load_models(checkpoint: dict[str, Any]) -> list[ValueRiskMLP]:
    models = []
    for state_dict in checkpoint["model_state_dicts"]:
        model = ValueRiskMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_size=int(checkpoint["hidden_size"]),
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


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
        description="Score counterfactual prefix rows with an exported value/risk ensemble."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--min-utility", type=float)
    parser.add_argument("--max-bad-probability", type=float)
    parser.add_argument("--min-success-probability", type=float)
    parser.add_argument("--value-std-coef", type=float)
    parser.add_argument("--bad-std-coef", type=float)
    parser.add_argument("--success-std-coef", type=float)
    parser.add_argument("--reward-epsilon", type=float)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint.get("config", {})
    rows = read_rows(args.csv)
    if not rows:
        raise ValueError("No rows found")

    feature_columns = list(checkpoint["feature_columns"])
    x_raw = feature_matrix(rows, feature_columns)
    mean = checkpoint["feature_mean"].detach().cpu().numpy()
    std = checkpoint["feature_std"].detach().cpu().numpy()
    x = (x_raw - mean) / std

    scoring_args = argparse.Namespace(
        reward_epsilon=(
            args.reward_epsilon
            if args.reward_epsilon is not None
            else first_config_value(config, "reward_epsilon", 1e-6)
        ),
        success_weight=first_config_value(config, "success_weight", 2.0),
        failed_weight=first_config_value(config, "failed_weight", 0.75),
    )
    (
        utility,
        reward_delta,
        success_delta,
        failed_delta,
        categories,
        _,
        _,
        _,
    ) = row_arrays(rows, scoring_args)

    models = load_models(checkpoint)
    value_mean, value_std, bad_mean, bad_std, success_mean, success_std = (
        predict_ensemble(models, x)
    )
    value_std_coef = (
        args.value_std_coef
        if args.value_std_coef is not None
        else first_config_value(config, "value_std_coef", 1.0)
    )
    bad_std_coef = (
        args.bad_std_coef
        if args.bad_std_coef is not None
        else first_config_value(config, "bad_std_coef", 1.0)
    )
    success_std_coef = (
        args.success_std_coef
        if args.success_std_coef is not None
        else first_config_value(config, "success_std_coef", 1.0)
    )
    min_utility = (
        args.min_utility
        if args.min_utility is not None
        else first_config_value(config, "min_utility", 0.0)
    )
    max_bad_probability = (
        args.max_bad_probability
        if args.max_bad_probability is not None
        else first_config_value(config, "max_bad_probability", 0.02)
    )
    min_success_probability = (
        args.min_success_probability
        if args.min_success_probability is not None
        else first_config_value(config, "min_success_probability", float("nan"))
    )
    if not np.isfinite(min_success_probability):
        min_success_probability = None

    value_lcb = value_mean - value_std_coef * value_std
    bad_ucb = bad_mean + bad_std_coef * bad_std
    success_lcb = success_mean - success_std_coef * success_std
    accepted = acceptance_mask(
        value_lcb=value_lcb,
        bad_ucb=bad_ucb,
        success_lcb=success_lcb,
        min_utility=min_utility,
        max_bad_probability=max_bad_probability,
        min_success_probability=min_success_probability,
    )

    summary = metric_row(
        categories=categories,
        utility=utility,
        reward_delta=reward_delta,
        success_delta=success_delta,
        failed_delta=failed_delta,
        accepted=accepted,
        min_utility=min_utility,
        max_bad_probability=max_bad_probability,
        min_success_probability=min_success_probability,
    )
    audit_rows = []
    for row_index, row in enumerate(rows):
        audit_rows.append(
            {
                "row_index": row_index,
                "seed": row.get("seed", ""),
                "prefix_len": row.get("prefix_len", ""),
                "outcome_category": categories[row_index],
                "accepted": int(accepted[row_index]),
                "utility": float(utility[row_index]),
                "reward_delta": float(reward_delta[row_index]),
                "success_delta": float(success_delta[row_index]),
                "failed_agents_delta": float(failed_delta[row_index]),
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
        )

    print(
        "Score summary: "
        f"rows={len(rows)} accepted={summary['accepted']} "
        f"good/neutral/bad={summary['accepted_good']}/"
        f"{summary['accepted_neutral']}/{summary['accepted_bad']} "
        f"success_positive={summary['accepted_success_positive']} "
        f"thresholds={min_utility}/{max_bad_probability}/"
        f"{min_success_probability}"
    )
    if args.output_csv is not None:
        write_csv(args.output_csv, audit_rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": summary,
                    "model": str(args.model),
                    "rows": len(rows),
                    "feature_count": len(feature_columns),
                    "audit_rows": audit_rows,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
