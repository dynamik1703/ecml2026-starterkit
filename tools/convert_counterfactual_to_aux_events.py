#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


def int_value(value: Any) -> int:
    numeric = safe_float(value)
    if numeric == numeric:
        return int(numeric)
    return -1


def finite_or_zero(value: Any) -> float:
    result = safe_float(value)
    return result if result == result else 0.0


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def positive_event(row: dict[str, Any], reward_epsilon: float) -> bool:
    reward_delta = finite_or_zero(row.get("reward_delta"))
    success_delta = finite_or_zero(row.get("success_delta"))
    failed_delta = finite_or_zero(row.get("failed_agents_delta"))
    if success_delta < -1e-9 or failed_delta > 0:
        return False
    return success_delta > 1e-9 or reward_delta > reward_epsilon


def negative_event(
    row: dict[str, Any],
    reward_epsilon: float,
    include_reward_only_negatives: bool,
) -> bool:
    reward_delta = finite_or_zero(row.get("reward_delta"))
    success_delta = finite_or_zero(row.get("success_delta"))
    failed_delta = finite_or_zero(row.get("failed_agents_delta"))
    if success_delta < -1e-9 or failed_delta > 0:
        return True
    return bool(include_reward_only_negatives and reward_delta < -reward_epsilon)


def utility(row: dict[str, Any], success_weight: float, failed_weight: float) -> float:
    reward_delta = finite_or_zero(row.get("reward_delta"))
    success_delta = finite_or_zero(row.get("success_delta"))
    failed_delta = finite_or_zero(row.get("failed_agents_delta"))
    return reward_delta + success_weight * success_delta - failed_weight * max(
        0.0,
        failed_delta,
    )


def event_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    positive_candidate_keys = set()
    for row in rows:
        if not positive_event(row, args.reward_epsilon):
            continue
        seed = int_value(row.get("seed"))
        env_time = int_value(row.get("env_time"))
        agent_id = int_value(row.get("agent_id"))
        forced_action = action_id(
            row.get("forced_action", row.get("candidate_action")),
            row.get("forced_action_name", row.get("candidate_action_name")),
        )
        positive_candidate_keys.add((seed, env_time, agent_id, forced_action))

    for row in rows:
        if str(row.get("forced_applied", "True")).lower() == "false":
            continue
        seed = int_value(row.get("seed"))
        env_time = int_value(row.get("env_time"))
        agent_id = int_value(row.get("agent_id"))
        baseline_action = action_id(
            row.get("baseline_action"),
            row.get("baseline_action_name"),
        )
        candidate_action = action_id(
            row.get("forced_action", row.get("candidate_action")),
            row.get("forced_action_name", row.get("candidate_action_name")),
        )
        if min(seed, env_time, agent_id, baseline_action, candidate_action) < 0:
            continue

        is_positive = positive_event(row, args.reward_epsilon)
        is_negative = negative_event(
            row,
            args.reward_epsilon,
            args.include_reward_only_negatives,
        )
        if not is_positive and not is_negative:
            continue
        if is_negative and (seed, env_time, agent_id, candidate_action) in positive_candidate_keys:
            continue

        event_kind = "positive_rescue" if is_positive else "negative_baseline"
        target_action = candidate_action if is_positive else baseline_action
        candidate_action_name = (
            row.get("forced_action_name")
            or row.get("candidate_action_name")
            or ""
        )
        target_action_name = (
            candidate_action_name
            if is_positive
            else row.get("baseline_action_name", "")
        )
        key = (seed, env_time, agent_id, target_action, event_kind)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "scene": str(row.get("scene", "")).strip() or args.scene or "",
                "seed": seed,
                "env_time": env_time,
                "agent_id": agent_id,
                "baseline_action": baseline_action,
                "baseline_action_name": row.get("baseline_action_name", ""),
                "candidate_action": candidate_action,
                "candidate_action_name": candidate_action_name,
                "forced_action": target_action,
                "forced_action_name": target_action_name,
                "event_kind": event_kind,
                "forced_applied": True,
                "reward_delta": finite_or_zero(row.get("reward_delta")),
                "success_delta": finite_or_zero(row.get("success_delta")),
                "failed_agents_delta": finite_or_zero(row.get("failed_agents_delta")),
                "prefix_len": 1,
                "prefix_event_index": 1,
                "prefix_event_count": 1,
                "utility": utility(row, args.success_weight, args.failed_weight),
                "events": (
                    f"{event_kind}@{env_time}:a{agent_id}:"
                    f"{row.get('baseline_action_name', '')}->"
                    f"{row.get('forced_action_name', '')}"
                ),
                "candidate_source_counterfactual": 1.0,
            }
        )
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
        description="Convert counterfactual single-action rows into Aux-BC events."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        help="Scene label to use for input CSVs created before scene was emitted.",
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--failed-weight", type=float, default=0.75)
    parser.add_argument(
        "--include-reward-only-negatives",
        action="store_true",
        help="Emit negative_baseline labels for reward-only losses as well.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.csv)
    converted = event_rows(rows, args)
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
        "include_reward_only_negatives": args.include_reward_only_negatives,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(
        "converted_counterfactual "
        f"input_rows={summary['input_rows']} "
        f"event_rows={summary['event_rows']} "
        f"positive={summary['positive_rescue_events']} "
        f"negative={summary['negative_baseline_events']} "
        f"output_csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
