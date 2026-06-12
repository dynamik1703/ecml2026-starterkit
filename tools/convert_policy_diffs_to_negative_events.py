#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.train_counterfactual_gate import safe_float


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".json":
            payload = json.load(path.open())
            if not isinstance(payload, list):
                raise ValueError(f"{path} does not contain a row list")
            rows.extend(row for row in payload if isinstance(row, dict))
        else:
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def action_id(row: dict[str, Any], key: str) -> int:
    value = safe_float(row.get(key))
    if value == value:
        return int(value)
    return -1


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def event_rows(
    rows: list[dict[str, Any]],
    max_events_per_seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    counts_by_seed: dict[int, int] = {}
    seen: set[tuple[int, int, int, int]] = set()
    for row in rows:
        if not bool_value(row.get("has_diff")):
            continue
        seed = action_id(row, "seed")
        env_time = action_id(row, "env_time")
        agent_id = action_id(row, "agent_id")
        baseline_action = action_id(row, "baseline_action")
        candidate_action = action_id(row, "candidate_action")
        if min(seed, env_time, agent_id, baseline_action, candidate_action) < 0:
            continue
        if max_events_per_seed > 0 and counts_by_seed.get(seed, 0) >= max_events_per_seed:
            continue
        key = (seed, env_time, agent_id, baseline_action)
        if key in seen:
            continue
        seen.add(key)
        counts_by_seed[seed] = counts_by_seed.get(seed, 0) + 1
        output.append(
            {
                "seed": seed,
                "env_time": env_time,
                "agent_id": agent_id,
                "baseline_action": baseline_action,
                "baseline_action_name": row.get("baseline_action_name", ""),
                "candidate_action": candidate_action,
                "candidate_action_name": row.get("candidate_action_name", ""),
                "forced_action": baseline_action,
                "forced_action_name": row.get("baseline_action_name", ""),
                "event_kind": "negative_baseline",
                "forced_applied": True,
                "reward_delta": 0.0,
                "success_delta": 0.0,
                "failed_agents_delta": 0.0,
                "prefix_len": 1,
                "prefix_event_index": 1,
                "prefix_event_count": 1,
                "utility": 0.0,
                "events": (
                    f"neg@{env_time}:a{agent_id}:"
                    f"{row.get('candidate_action_name', '')}->"
                    f"{row.get('baseline_action_name', '')}"
                ),
                "candidate_source_policy_diff_negative": 1.0,
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
        description=(
            "Convert baseline-vs-candidate policy diff rows into negative "
            "baseline-action labels for auxiliary PPO BC."
        )
    )
    parser.add_argument("diff", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--max-events-per-seed",
        type=int,
        default=0,
        help="Limit converted diffs per seed. Use 0 for all rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.diff)
    converted = event_rows(rows, args.max_events_per_seed)
    write_csv(args.output_csv, converted)
    summary = {
        "input_rows": len(rows),
        "event_rows": len(converted),
        "unique_seeds": len({row["seed"] for row in converted}),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(
        "converted_policy_diffs "
        f"input_rows={summary['input_rows']} "
        f"event_rows={summary['event_rows']} "
        f"unique_seeds={summary['unique_seeds']} "
        f"output_csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
