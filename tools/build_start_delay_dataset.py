#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_compare_rows(paths: list[Path]) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                scene = str(row.get("scene", "scene_5") or "scene_5")
                seed = safe_int(row.get("seed"))
                rows[(scene, seed)] = row
    return rows


def read_trace_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("candidate_source") != "start_delay_guard":
                    continue
                if not bool(row.get("accepted", False)):
                    continue
                rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("scene", "scene_5")),
            safe_int(row.get("seed")),
            safe_int(row.get("env_time")),
            safe_int(row.get("agent_id")),
        )
    )
    return rows


def first_event_per_agent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (
            str(row.get("scene", "scene_5")),
            safe_int(row.get("seed")),
            safe_int(row.get("agent_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def outcome_category(row: dict[str, Any], reward_epsilon: float) -> str:
    reward_delta = safe_float(row.get("reward_delta"), 0.0)
    success_delta = safe_float(row.get("success_delta"), 0.0)
    if success_delta > 1e-9:
        return "good"
    if success_delta < -1e-9:
        return "bad"
    if reward_delta > reward_epsilon:
        return "good"
    if reward_delta < -reward_epsilon:
        return "bad"
    return "neutral"


def build_rows(
    trace_rows: list[dict[str, Any]],
    compare_rows: dict[tuple[str, int], dict[str, str]],
    reward_epsilon: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_per_seed_agent: Counter[tuple[str, int, int]] = Counter()
    seen_per_seed: Counter[tuple[str, int]] = Counter()
    total_per_seed: Counter[tuple[str, int]] = Counter(
        (str(row.get("scene", "scene_5")), safe_int(row.get("seed")))
        for row in trace_rows
    )
    total_per_seed_agent: Counter[tuple[str, int, int]] = Counter(
        (
            str(row.get("scene", "scene_5")),
            safe_int(row.get("seed")),
            safe_int(row.get("agent_id")),
        )
        for row in trace_rows
    )
    for trace in trace_rows:
        scene = str(trace.get("scene", "scene_5") or "scene_5")
        seed = safe_int(trace.get("seed"))
        agent_id = safe_int(trace.get("agent_id"))
        compare = compare_rows.get((scene, seed))
        if compare is None:
            continue

        key = (scene, seed)
        agent_key = (scene, seed, agent_id)
        seed_index = seen_per_seed[key]
        agent_index = seen_per_seed_agent[agent_key]
        seen_per_seed[key] += 1
        seen_per_seed_agent[agent_key] += 1

        baseline_reward = safe_float(compare.get("baseline_reward"), 0.0)
        forced_reward = safe_float(
            compare.get("candidate_reward", compare.get("forced_reward")),
            0.0,
        )
        reward_delta = safe_float(compare.get("reward_delta"), forced_reward - baseline_reward)
        baseline_success = safe_float(compare.get("baseline_success"), 0.0)
        forced_success = safe_float(
            compare.get("candidate_success", compare.get("forced_success")),
            0.0,
        )
        success_delta = safe_float(
            compare.get("success_delta"),
            forced_success - baseline_success,
        )
        num_agents = safe_float(
            compare.get("candidate_num_agents", compare.get("baseline_num_agents")),
            0.0,
        )
        row: dict[str, Any] = {
            "scene": scene,
            "seed": seed,
            "decision_index": seed_index,
            "agent_id": agent_id,
            "env_time": safe_int(trace.get("env_time")),
            "baseline_action": safe_int(trace.get("baseline_action")),
            "baseline_action_name": trace.get("baseline_action_name", ""),
            "forced_action": safe_int(trace.get("candidate_action")),
            "forced_action_name": trace.get("candidate_action_name", ""),
            "candidate_source": "start_delay_guard",
            "baseline_reward": baseline_reward,
            "forced_reward": forced_reward,
            "reward_delta": reward_delta,
            "baseline_success": baseline_success,
            "forced_success": forced_success,
            "success_delta": success_delta,
            "baseline_env_time": safe_float(compare.get("baseline_env_time"), 0.0),
            "forced_env_time": safe_float(compare.get("candidate_env_time"), 0.0),
            "env_time_delta": safe_float(compare.get("env_time_delta"), 0.0),
            "failed_agents_delta": -success_delta * num_agents,
            "seed_event_count": total_per_seed[key],
            "agent_event_count": total_per_seed_agent[agent_key],
            "event_index_fraction": (
                seed_index / max(1, total_per_seed[key] - 1)
                if total_per_seed[key] > 1
                else 0.0
            ),
            "agent_event_index": agent_index,
            "agent_event_index_fraction": (
                agent_index / max(1, total_per_seed_agent[agent_key] - 1)
                if total_per_seed_agent[agent_key] > 1
                else 0.0
            ),
        }
        for key_name, value in trace.items():
            if key_name in row:
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                row[f"trace_{key_name}"] = value
        row["outcome_category"] = outcome_category(row, reward_epsilon)
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join accepted start-delay guard trace events with seed-level A/B "
            "outcomes into a classifier/ranker dataset."
        )
    )
    parser.add_argument("--trace-jsonl", nargs="+", required=True, type=Path)
    parser.add_argument("--compare-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--first-event-per-agent",
        action="store_true",
        help=(
            "Keep only the first accepted delay per scene/seed/agent. This "
            "turns repeated per-step holds into one sequence-start example."
        ),
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_rows = read_compare_rows(args.compare_csv)
    trace_rows = read_trace_rows(args.trace_jsonl)
    raw_trace_rows = len(trace_rows)
    if args.first_event_per_agent:
        trace_rows = first_event_per_agent(trace_rows)
    rows = build_rows(trace_rows, compare_rows, args.reward_epsilon)
    summary = {
        "raw_trace_rows": raw_trace_rows,
        "trace_rows": len(trace_rows),
        "compare_rows": len(compare_rows),
        "rows": len(rows),
        "seeds": sorted({safe_int(row["seed"]) for row in rows}),
        "category_counts": dict(Counter(row["outcome_category"] for row in rows)),
    }
    write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, rows, summary)
    print(
        "start_delay_dataset "
        f"trace_rows={summary['trace_rows']} rows={summary['rows']} "
        f"categories={summary['category_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
