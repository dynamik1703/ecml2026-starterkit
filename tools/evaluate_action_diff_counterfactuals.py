#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.analyze_policy_action_diffs import (
    done_all,
    instantiate_policy,
    observation_list,
    policy_actions,
)
from tools.counterfactual_decision_eval import final_result, make_env


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def event_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(float(row["seed"])),
        int(float(row["env_time"])),
        int(float(row["agent_id"])),
        int(float(row["candidate_action"])),
    )


def run_episode(
    args: argparse.Namespace,
    seed: int,
    forced_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env, _obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    forced_applied = False

    force_step = None
    force_handle = None
    force_action = None
    if forced_event is not None:
        force_step = int(float(forced_event["env_time"]))
        force_handle = int(float(forced_event["agent_id"]))
        force_action = int(float(forced_event["candidate_action"]))

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = policy_actions(policy, handles, obs_list)
        if (
            force_step is not None
            and int(env._elapsed_steps) == force_step
            and force_handle in actions
        ):
            actions[force_handle] = force_action
            forced_applied = True

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(int(action))
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    result["forced_applied"] = forced_applied
    return result


def annotate(
    row: dict[str, Any],
    baseline: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    baseline_failed = len(baseline["failed_agents"])
    forced_failed = len(forced["failed_agents"])
    reward_delta = forced["normalized_reward"] - baseline["normalized_reward"]
    success_delta = forced["success_rate"] - baseline["success_rate"]
    success_epsilon = 1e-9
    return {
        **row,
        "baseline_reward": baseline["normalized_reward"],
        "forced_reward": forced["normalized_reward"],
        "reward_delta": reward_delta,
        "baseline_success": baseline["success_rate"],
        "forced_success": forced["success_rate"],
        "success_delta": success_delta,
        "baseline_env_time": baseline["env_time"],
        "forced_env_time": forced["env_time"],
        "env_time_delta": forced["env_time"] - baseline["env_time"],
        "baseline_failed_agents": baseline_failed,
        "forced_failed_agents": forced_failed,
        "failed_agents_delta": forced_failed - baseline_failed,
        "baseline_failed_agent_ids": baseline["failed_agent_ids"],
        "forced_failed_agent_ids": forced["failed_agent_ids"],
        "forced_applied": forced["forced_applied"],
        "pairwise_label": float(
            success_delta > success_epsilon
            or (abs(success_delta) <= success_epsilon and reward_delta > 1e-9)
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "reward_wins": sum(float(row["reward_delta"]) > 1e-9 for row in rows),
        "reward_losses": sum(float(row["reward_delta"]) < -1e-9 for row in rows),
        "reward_ties": sum(abs(float(row["reward_delta"])) <= 1e-9 for row in rows),
        "success_wins": sum(float(row["success_delta"]) > 1e-9 for row in rows),
        "success_losses": sum(float(row["success_delta"]) < -1e-9 for row in rows),
        "success_ties": sum(abs(float(row["success_delta"])) <= 1e-9 for row in rows),
        "forced_not_applied": sum(not row["forced_applied"] for row in rows),
    }


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
            "Evaluate exact one-step counterfactual outcomes for candidate "
            "actions from analyze_policy_action_diffs.py rows."
        )
    )
    from tools.evaluate_sampled import DEFAULT_BASE_STATE, DEFAULT_OBS_BUILDER, DEFAULT_REWARDS, repo_root

    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--baseline-policy", default="submission.sequence_success_policy.MyPolicy")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--action-diff-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_rows = [
        row
        for row in read_rows(args.action_diff_csv)
        if str(row.get("has_diff", "True")).lower() != "false"
    ]
    unique_rows: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in raw_rows:
        unique_rows.setdefault(event_key(row), row)
    rows = list(unique_rows.values())
    rows.sort(key=lambda item: event_key(item))
    if args.max_events > 0:
        rows = rows[: args.max_events]

    baseline_by_seed: dict[int, dict[str, Any]] = {}
    output_rows = []
    for index, row in enumerate(rows, start=1):
        seed = int(float(row["seed"]))
        if seed not in baseline_by_seed:
            baseline_by_seed[seed] = run_episode(args, seed)
        forced = run_episode(args, seed, row)
        output_row = annotate(row, baseline_by_seed[seed], forced)
        output_rows.append(output_row)
        print(
            "event={index}/{total} seed={seed} step={step} agent={agent} "
            "{baseline}->{candidate} reward_delta={reward:.6g} "
            "success_delta={success:.6g} applied={applied}".format(
                index=index,
                total=len(rows),
                seed=seed,
                step=int(float(row["env_time"])),
                agent=int(float(row["agent_id"])),
                baseline=row.get("baseline_action_name", row.get("baseline_action")),
                candidate=row.get("candidate_action_name", row.get("candidate_action")),
                reward=output_row["reward_delta"],
                success=output_row["success_delta"],
                applied=output_row["forced_applied"],
            ),
            flush=True,
        )

    summary = summarize(output_rows)
    print(
        "summary rows={rows} reward={rw}/{rl}/{rt} success={sw}/{sl}/{st} "
        "forced_not_applied={fna}".format(
            rows=summary["rows"],
            rw=summary["reward_wins"],
            rl=summary["reward_losses"],
            rt=summary["reward_ties"],
            sw=summary["success_wins"],
            sl=summary["success_losses"],
            st=summary["success_ties"],
            fna=summary["forced_not_applied"],
        ),
        flush=True,
    )
    if args.output_csv:
        write_csv(args.output_csv, output_rows)
    if args.output_json:
        write_json(args.output_json, output_rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
