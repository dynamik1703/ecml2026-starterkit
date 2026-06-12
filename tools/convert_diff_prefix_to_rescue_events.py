#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.evaluate_sequence_rescue_planner import read_json_rows
from tools.train_counterfactual_gate import safe_float


ACTION_NAME_TO_ID = {
    "DO_NOTHING": 0,
    "MOVE_LEFT": 1,
    "MOVE_FORWARD": 2,
    "MOVE_RIGHT": 3,
    "STOP_MOVING": 4,
}


def action_id(value: Any, name: Any) -> int:
    numeric = safe_float(value)
    if numeric == numeric:
        return int(numeric)
    return ACTION_NAME_TO_ID.get(str(name), -1)


def positive_prefix(row: dict[str, Any], reward_epsilon: float) -> bool:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    reward_delta = reward_delta if reward_delta == reward_delta else 0.0
    success_delta = success_delta if success_delta == success_delta else 0.0
    failed_delta = failed_delta if failed_delta == failed_delta else 0.0
    if success_delta < -1e-9 or failed_delta > 0:
        return False
    return success_delta > 1e-9 or reward_delta > reward_epsilon


def row_utility(row: dict[str, Any], success_weight: float, failed_weight: float) -> float:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    reward_delta = reward_delta if reward_delta == reward_delta else 0.0
    success_delta = success_delta if success_delta == success_delta else 0.0
    failed_delta = failed_delta if failed_delta == failed_delta else 0.0
    return float(
        reward_delta
        + success_weight * success_delta
        - failed_weight * max(0.0, failed_delta)
    )


def negative_prefix(row: dict[str, Any], reward_epsilon: float) -> bool:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    reward_delta = reward_delta if reward_delta == reward_delta else 0.0
    success_delta = success_delta if success_delta == success_delta else 0.0
    failed_delta = failed_delta if failed_delta == failed_delta else 0.0
    if success_delta < -1e-9 or failed_delta > 0:
        return True
    return success_delta <= 1e-9 and reward_delta < -reward_epsilon


def prefix_conflict_value(event: dict[str, Any]) -> float:
    values = [
        safe_float(event.get("forced_prefix_cell_intersections")),
        safe_float(event.get("forced_prefix_head_on_edge_conflicts")),
        safe_float(event.get("forced_prefix_same_edge_conflicts")),
        safe_float(event.get("candidate_prefix_cell_intersections")),
        safe_float(event.get("candidate_prefix_head_on_edge_conflicts")),
        safe_float(event.get("candidate_prefix_same_edge_conflicts")),
    ]
    finite = [value for value in values if value == value]
    return max(finite, default=0.0)


def row_prefix_conflict_value(row: dict[str, Any]) -> float:
    details = row.get("event_details") or []
    if not isinstance(details, list):
        return 0.0
    return max(
        (
            prefix_conflict_value(event)
            for event in details
            if isinstance(event, dict)
        ),
        default=0.0,
    )


def source_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("candidate_source_")
        }
    )
    return columns


def event_rows(
    rows: list[dict[str, Any]],
    reward_epsilon: float,
    include_negative_baseline: bool,
    best_prefix_per_seed: bool,
    success_weight: float,
    failed_weight: float,
    positive_min_prefix_conflicts: float,
) -> list[dict[str, Any]]:
    if best_prefix_per_seed:
        best_positive_by_seed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not positive_prefix(row, reward_epsilon):
                continue
            if row_prefix_conflict_value(row) < positive_min_prefix_conflicts:
                continue
            seed = str(row.get("seed", ""))
            previous = best_positive_by_seed.get(seed)
            if previous is None or row_utility(
                row,
                success_weight,
                failed_weight,
            ) > row_utility(previous, success_weight, failed_weight):
                best_positive_by_seed[seed] = row
        selected_positive_rows = set(map(id, best_positive_by_seed.values()))
    else:
        selected_positive_rows = None

    sources = source_columns(rows)
    output = []
    seen = set()
    positive_event_keys = set()
    for row in rows:
        if selected_positive_rows is not None and id(row) not in selected_positive_rows:
            continue
        if not positive_prefix(row, reward_epsilon):
            continue
        if row_prefix_conflict_value(row) < positive_min_prefix_conflicts:
            continue
        details = row.get("event_details") or []
        if not isinstance(details, list):
            continue
        for event in details:
            if not isinstance(event, dict):
                continue
            forced_action = action_id(
                event.get("candidate_action"),
                event.get("candidate_action_name"),
            )
            if forced_action < 0:
                continue
            positive_event_keys.add(
                (
                    int(float(row.get("seed", event.get("seed", -1)))),
                    int(float(event.get("env_time", -1))),
                    int(float(event.get("agent_id", -1))),
                    forced_action,
                )
            )
    for row in rows:
        is_positive = positive_prefix(row, reward_epsilon) and (
            selected_positive_rows is None or id(row) in selected_positive_rows
        )
        if (
            is_positive
            and row_prefix_conflict_value(row) < positive_min_prefix_conflicts
        ):
            is_positive = False
        is_negative = include_negative_baseline and negative_prefix(row, reward_epsilon)
        if not is_positive and not is_negative:
            continue
        details = row.get("event_details") or []
        if not isinstance(details, list):
            continue
        reward_delta = safe_float(row.get("reward_delta"))
        success_delta = safe_float(row.get("success_delta"))
        failed_delta = safe_float(row.get("failed_agents_delta"))
        utility = row_utility(
            row,
            success_weight=success_weight,
            failed_weight=failed_weight,
        )
        for index, event in enumerate(details):
            if not isinstance(event, dict):
                continue
            forced_action = action_id(
                event.get("candidate_action"),
                event.get("candidate_action_name"),
            )
            baseline_action = action_id(
                event.get("baseline_action"),
                event.get("baseline_action_name"),
            )
            if forced_action < 0 or baseline_action < 0:
                continue
            target_action = forced_action if is_positive else baseline_action
            key = (
                int(float(row.get("seed", event.get("seed", -1)))),
                int(float(event.get("env_time", -1))),
                int(float(event.get("agent_id", -1))),
                target_action,
            )
            if key in seen:
                continue
            candidate_key = (
                key[0],
                key[1],
                key[2],
                forced_action,
            )
            if is_negative and candidate_key in positive_event_keys:
                continue
            seen.add(key)
            record = {
                "seed": key[0],
                "env_time": key[1],
                "agent_id": key[2],
                "baseline_action": baseline_action,
                "baseline_action_name": event.get("baseline_action_name", ""),
                "forced_action": target_action,
                "forced_action_name": (
                    event.get("candidate_action_name", "")
                    if is_positive
                    else event.get("baseline_action_name", "")
                ),
                "candidate_action": forced_action,
                "candidate_action_name": event.get("candidate_action_name", ""),
                "event_kind": "positive_rescue" if is_positive else "negative_baseline",
                "forced_applied": True,
                "reward_delta": reward_delta if reward_delta == reward_delta else 0.0,
                "success_delta": success_delta if success_delta == success_delta else 0.0,
                "failed_agents_delta": (
                    failed_delta if failed_delta == failed_delta else 0.0
                ),
                "prefix_len": row.get("prefix_len", len(details)),
                "prefix_event_index": index + 1,
                "prefix_event_count": len(details),
                "utility": utility,
                "events": row.get("events", ""),
            }
            for source in sources:
                record[source] = row.get(source, 0.0)
            output.append(record)
    return output


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
            "Convert positive diff-prefix JSON rows into event-level rescue "
            "labels for train_rescue_behavior_clone.py."
        )
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--best-prefix-per-seed",
        action="store_true",
        help="Emit positive events only from the highest-utility positive prefix per seed.",
    )
    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--failed-weight", type=float, default=0.75)
    parser.add_argument(
        "--positive-min-prefix-conflicts",
        type=float,
        default=0.0,
        help=(
            "Keep positive prefixes only when at least one event has this many "
            "forced/candidate prefix cell, same-edge, or head-on conflicts."
        ),
    )
    parser.add_argument(
        "--include-negative-baseline",
        action="store_true",
        help=(
            "Also emit events from harmful prefixes as baseline-action labels. "
            "Events that also appear in a positive prefix are skipped to avoid "
            "contradictory labels."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_json_rows(args.json)
    converted = event_rows(
        rows,
        reward_epsilon=args.reward_epsilon,
        include_negative_baseline=args.include_negative_baseline,
        best_prefix_per_seed=args.best_prefix_per_seed,
        success_weight=args.success_weight,
        failed_weight=args.failed_weight,
        positive_min_prefix_conflicts=args.positive_min_prefix_conflicts,
    )
    write_csv(args.output_csv, converted)
    summary = {
        "input_rows": len(rows),
        "event_rows": len(converted),
        "unique_seeds": len({row["seed"] for row in converted}),
        "positive_rescue_events": sum(
            1 for row in converted if row.get("event_kind") == "positive_rescue"
        ),
        "negative_baseline_events": sum(
            1 for row in converted if row.get("event_kind") == "negative_baseline"
        ),
        "positive_min_prefix_conflicts": args.positive_min_prefix_conflicts,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(
        "converted "
        f"input_rows={summary['input_rows']} "
        f"event_rows={summary['event_rows']} "
        f"unique_seeds={summary['unique_seeds']} "
        f"output_csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
