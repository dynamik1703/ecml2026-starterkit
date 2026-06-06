#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.load(path.open())
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"JSON file has no row list: {path}")
    return [normalize_row(row) for row in rows]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    if "candidate_reward" in row or "candidate_success" in row:
        reward = safe_float(row.get("candidate_reward"))
        success = safe_float(row.get("candidate_success"))
        env_time = safe_int(row.get("candidate_env_time"))
        failed_agents = safe_int(row.get("candidate_failed_agents"))
        failed_ids = str(row.get("candidate_failed_agent_ids", ""))
        candidate = str(row.get("candidate", "candidate"))
    else:
        reward = safe_float(row.get("normalized_reward"))
        success = safe_float(row.get("success_rate"))
        env_time = safe_int(row.get("env_time"))
        failed_agents = safe_int(row.get("failed_agents"))
        failed_ids = str(row.get("failed_agent_ids", ""))
        candidate = str(row.get("candidate", "candidate"))

    reward_delta = row.get("reward_delta", "")
    success_delta = row.get("success_delta", "")
    return {
        "seed": safe_int(row.get("seed")),
        "candidate": candidate,
        "reward": reward,
        "success": success,
        "env_time": env_time,
        "failed_agents": failed_agents,
        "failed_agent_ids": failed_ids,
        "reward_delta": (
            safe_float(reward_delta)
            if reward_delta not in ("", None)
            else ""
        ),
        "success_delta": (
            safe_float(success_delta)
            if success_delta not in ("", None)
            else ""
        ),
    }


def parse_candidate_names(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    names = set()
    for value in values:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.add(name)
    return names


def dedupe_seed_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        current = by_seed.get(seed)
        if current is None:
            by_seed[seed] = row
            continue
        if ranking_key(row, mode) < ranking_key(current, mode):
            by_seed[seed] = row
    return list(by_seed.values())


def is_candidate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    reward = float(row["reward"])
    success = float(row["success"])
    failed_agents = int(row["failed_agents"])
    if args.mode == "failures":
        if success < args.success_threshold:
            return True
        if failed_agents >= args.min_failed_agents:
            return True
        if args.max_reward is not None and reward <= args.max_reward:
            return True
        return False
    if args.mode == "full-success-low-reward":
        if success < 1.0 or failed_agents > 0:
            return False
        return args.max_reward is None or reward <= args.max_reward
    if args.mode == "reward-regressions":
        return row["reward_delta"] != "" and float(row["reward_delta"]) < -args.epsilon
    if args.mode == "success-regressions":
        return row["success_delta"] != "" and float(row["success_delta"]) < -args.epsilon
    raise ValueError(f"Unknown mode: {args.mode}")


def ranking_key(row: dict[str, Any], mode: str) -> tuple[float, ...]:
    reward = float(row["reward"])
    success = float(row["success"])
    failed_agents = int(row["failed_agents"])
    env_time = float(row["env_time"])
    if mode == "full-success-low-reward":
        return (reward, -env_time, float(row["seed"]))
    if mode == "reward-regressions":
        reward_delta = safe_float(row.get("reward_delta"), 0.0)
        return (reward_delta, success, reward, float(row["seed"]))
    if mode == "success-regressions":
        success_delta = safe_float(row.get("success_delta"), 0.0)
        return (success_delta, reward, float(row["seed"]))
    return (-failed_agents, success, reward, -env_time, float(row["seed"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select target seeds from policy_scoreboard or validate_policy_gate JSON "
            "rows for the next mining/training loop."
        )
    )
    parser.add_argument("json", type=Path, nargs="+")
    parser.add_argument(
        "--mode",
        choices=[
            "failures",
            "full-success-low-reward",
            "reward-regressions",
            "success-regressions",
        ],
        default="failures",
    )
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--min-failed-agents", type=int, default=1)
    parser.add_argument("--max-reward", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument(
        "--candidate",
        action="append",
        help=(
            "Optional candidate name filter. Can be repeated or comma-separated; "
            "useful for policy_scoreboard JSONs with multiple candidates."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.json:
        rows.extend(load_json_rows(path))
    candidate_names = parse_candidate_names(args.candidate)
    if candidate_names:
        rows = [row for row in rows if row["candidate"] in candidate_names]
    candidates = [row for row in rows if is_candidate(row, args)]
    candidates = dedupe_seed_rows(candidates, args.mode)
    selected = sorted(
        candidates,
        key=lambda row: ranking_key(row, args.mode),
    )[: max(0, args.top_k)]
    summary = {
        "mode": args.mode,
        "input_rows": len(rows),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_seeds": [int(row["seed"]) for row in selected],
    }

    print(
        "Summary: "
        f"mode={summary['mode']} input_rows={summary['input_rows']} "
        f"candidate_rows={summary['candidate_rows']} "
        f"selected_rows={summary['selected_rows']}"
    )
    if selected:
        print("seed,reward,success,failed_agents,failed_agent_ids,reward_delta,success_delta")
        for row in selected:
            failed_agent_ids = str(row["failed_agent_ids"]).replace(",", ";")
            print(
                f"{row['seed']},{row['reward']:.6g},{row['success']:.6g},"
                f"{row['failed_agents']},{failed_agent_ids},"
                f"{row['reward_delta']},{row['success_delta']}"
            )
        print("seeds=" + ",".join(str(row["seed"]) for row in selected))

    if args.output_csv is not None:
        write_csv(args.output_csv, selected)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump({"summary": summary, "rows": selected}, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
