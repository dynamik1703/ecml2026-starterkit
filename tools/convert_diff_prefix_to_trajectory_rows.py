#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tools.evaluate_sequence_rescue_planner import read_json_rows
from tools.train_counterfactual_gate import outcome_category, safe_float


LEAKAGE_COLUMNS = {
    "candidate_reward",
    "candidate_success",
    "candidate_reward_delta",
    "candidate_success_delta",
    "candidate_failed_agent_ids",
}


def finite_or_zero(value: Any) -> float:
    result = safe_float(value)
    return result if math.isfinite(result) else 0.0


def trajectory_failed_delta(row: dict[str, Any], success_delta: float) -> float:
    baseline_failed = safe_float(row.get("baseline_failed_agents"))
    candidate_failed_ids = str(row.get("candidate_failed_agent_ids", "")).strip()
    if candidate_failed_ids:
        candidate_failed = len(
            [item for item in candidate_failed_ids.split(",") if item.strip()]
        )
        if math.isfinite(baseline_failed):
            return float(candidate_failed) - float(baseline_failed)
    num_agents = safe_float(row.get("num_agents"))
    if math.isfinite(num_agents):
        return -float(num_agents) * success_delta
    return 0.0


def convert_row(row: dict[str, Any], reward_epsilon: float) -> dict[str, Any]:
    reward_delta = finite_or_zero(row.get("candidate_reward_delta"))
    success_delta = finite_or_zero(row.get("candidate_success_delta"))
    failed_delta = trajectory_failed_delta(row, success_delta)

    converted = {
        key: value
        for key, value in row.items()
        if key not in LEAKAGE_COLUMNS and key != "event_details"
    }
    baseline_reward = safe_float(row.get("baseline_reward"))
    baseline_success = safe_float(row.get("baseline_success"))
    baseline_failed = safe_float(row.get("baseline_failed_agents"))
    converted["reward_delta"] = reward_delta
    converted["success_delta"] = success_delta
    converted["failed_agents_delta"] = failed_delta
    converted["forced_reward"] = (
        baseline_reward + reward_delta if math.isfinite(baseline_reward) else ""
    )
    converted["forced_success"] = (
        baseline_success + success_delta if math.isfinite(baseline_success) else ""
    )
    converted["forced_failed_agents"] = (
        baseline_failed + failed_delta if math.isfinite(baseline_failed) else ""
    )
    converted["trajectory_label_source"] = "candidate_full_rollout"
    converted["outcome_category"] = outcome_category(converted, reward_epsilon)
    return converted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert diff-prefix feature rows into full-candidate trajectory "
            "label rows for value/risk diagnostics."
        )
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--prefix-len", nargs="+", type=int)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_json_rows(args.json)
    prefix_filter = set(args.prefix_len or [])
    converted = [
        convert_row(row, args.reward_epsilon)
        for row in rows
        if not prefix_filter or int(float(row.get("prefix_len", -1))) in prefix_filter
    ]
    write_csv(args.output_csv, converted)
    categories = Counter(
        row.get("outcome_category", "") for row in converted
    )
    summary = {
        "input_rows": len(rows),
        "output_rows": len(converted),
        "prefix_len": sorted(prefix_filter),
        "categories": dict(categories),
        "unique_seeds": len({row.get("seed") for row in converted}),
        "output_csv": str(args.output_csv),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(
        "converted "
        f"input_rows={summary['input_rows']} "
        f"output_rows={summary['output_rows']} "
        f"unique_seeds={summary['unique_seeds']} "
        f"categories={summary['categories']} "
        f"output_csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
