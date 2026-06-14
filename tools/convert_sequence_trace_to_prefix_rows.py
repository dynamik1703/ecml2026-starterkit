#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.evaluate_success_rescue_classifier import json_default
from tools.mine_diff_prefix_dataset import aggregate_event_features, candidate_source_features
from tools.train_counterfactual_gate import outcome_category, safe_float


SCORE_KEYS = {
    "accepted",
    "accepted_event_count_before",
    "bad_ucb",
    "listwise_baseline_lcb",
    "listwise_baseline_mean",
    "listwise_baseline_std",
    "listwise_baseline_ucb",
    "listwise_candidate_lcb",
    "listwise_candidate_mean",
    "listwise_candidate_std",
    "listwise_margin",
    "reject_reason",
    "success_lcb",
    "success_regression_ucb",
    "value_lcb",
}


def read_compare_rows(path: Path) -> dict[int, dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        result[seed] = row
    return result


def read_trace_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def candidate_source_from_trace(row: dict[str, Any]) -> dict[str, float]:
    checkpoint = str(row.get("candidate_checkpoint", ""))
    if not checkpoint:
        return {}
    return candidate_source_features(row.get("candidate_policy", ""), checkpoint)


def event_detail_from_trace(row: dict[str, Any], prefix_index: int) -> dict[str, Any]:
    detail = {}
    for key, value in row.items():
        if key in SCORE_KEYS:
            continue
        if key.startswith("event_"):
            continue
        if key in {"candidate_checkpoint"}:
            continue
        detail[key] = value
    detail["prefix_index"] = prefix_index
    detail.update(candidate_source_from_trace(row))
    return detail


def failed_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    return len([item for item in str(value).split(",") if item.strip()])


def output_row(
    compare_row: dict[str, Any],
    events: list[dict[str, Any]],
    final_trace_row: dict[str, Any],
) -> dict[str, Any]:
    baseline_reward = safe_float(compare_row.get("baseline_reward"))
    candidate_reward = safe_float(compare_row.get("candidate_reward"))
    baseline_success = safe_float(compare_row.get("baseline_success"))
    candidate_success = safe_float(compare_row.get("candidate_success"))
    baseline_failed = failed_count(compare_row.get("baseline_failed_agent_ids"))
    candidate_failed = failed_count(compare_row.get("candidate_failed_agent_ids"))

    reward_delta = candidate_reward - baseline_reward
    success_delta = candidate_success - baseline_success
    if not math.isfinite(reward_delta):
        reward_delta = safe_float(compare_row.get("reward_delta"))
    if not math.isfinite(success_delta):
        success_delta = safe_float(compare_row.get("success_delta"))

    row = {
        "seed": int(compare_row["seed"]),
        "scene": compare_row.get("scene", ""),
        "num_agents": compare_row.get("num_agents", ""),
        "line_length": compare_row.get("line_length", ""),
        "prefix_len": len(events),
        "forced_applied": len(events),
        "baseline_reward": baseline_reward,
        "forced_reward": candidate_reward,
        "reward_delta": reward_delta,
        "baseline_success": baseline_success,
        "forced_success": candidate_success,
        "success_delta": success_delta,
        "baseline_failed_agents": baseline_failed,
        "forced_failed_agents": candidate_failed,
        "failed_agents_delta": candidate_failed - baseline_failed,
        "baseline_failed_agent_ids": compare_row.get("baseline_failed_agent_ids", ""),
        "forced_failed_agent_ids": compare_row.get("candidate_failed_agent_ids", ""),
        "events": " ".join(
            "{prefix_index}@{env_time}:a{agent_id}:{baseline_action_name}->{candidate_action_name}".format(
                **event
            )
            for event in events
        ),
        "event_details": events,
        "outcome_category": outcome_category(
            {
                "reward_delta": reward_delta,
                "success_delta": success_delta,
            },
            1e-9,
        ),
    }
    row.update(aggregate_event_features(events))
    row.update(candidate_source_from_trace(final_trace_row))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row if key != "event_details"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SequenceSuccessPolicy accepted-event traces plus "
            "compare_policies results into prefix rows for sequence ranker training."
        )
    )
    parser.add_argument("--compare-json", type=Path, action="append", required=True)
    parser.add_argument("--trace-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--final-prefix-only",
        action="store_true",
        help="Write only the final accepted prefix per seed instead of every prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_rows: dict[int, dict[str, Any]] = {}
    for path in args.compare_json:
        compare_rows.update(read_compare_rows(path))

    traces_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_trace_rows(args.trace_jsonl):
        if not row.get("accepted", False):
            continue
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        traces_by_seed[seed].append(row)

    output_rows = []
    for seed, trace_rows in sorted(traces_by_seed.items()):
        compare_row = compare_rows.get(seed)
        if compare_row is None:
            continue
        trace_rows = sorted(
            trace_rows,
            key=lambda row: (
                int(row.get("env_time", 0)),
                int(row.get("agent_id", 0)),
            ),
        )
        events = [
            event_detail_from_trace(row, prefix_index=index + 1)
            for index, row in enumerate(trace_rows)
        ]
        prefix_indices = [len(events)] if args.final_prefix_only else range(1, len(events) + 1)
        for prefix_len in prefix_indices:
            output_rows.append(
                output_row(
                    compare_row=compare_row,
                    events=events[:prefix_len],
                    final_trace_row=trace_rows[prefix_len - 1],
                )
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(
            {
                "rows": output_rows,
                "summary": {
                    "rows": len(output_rows),
                    "seeds": len({row["seed"] for row in output_rows}),
                    "good": sum(row["outcome_category"] == "good" for row in output_rows),
                    "neutral": sum(row["outcome_category"] == "neutral" for row in output_rows),
                    "bad": sum(row["outcome_category"] == "bad" for row in output_rows),
                },
            },
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    if args.output_csv is not None:
        write_csv(args.output_csv, output_rows)
    print(
        "rows={rows} seeds={seeds} good={good} neutral={neutral} bad={bad}".format(
            rows=len(output_rows),
            seeds=len({row["seed"] for row in output_rows}),
            good=sum(row["outcome_category"] == "good" for row in output_rows),
            neutral=sum(row["outcome_category"] == "neutral" for row in output_rows),
            bad=sum(row["outcome_category"] == "bad" for row in output_rows),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
